"""Full-document comparison of shared, own-slot and per-source proposal predictors."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_placement_runtime_readback as runtime
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import legal_placement_comparison as measure
from model import native_placement_efficient as efficient
from model.native_population_training import cpu, fingerprint
from model.native_rq_packed import PackedNativeRQ
from model.native_target_cache import FrozenLabels
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need

SOURCE = ('legal-placement-runtime-source-v2-20261023', 'a5976a125b4a41e42af703a2c12d10d835531250476fb460792aa900b07e2a97')
INPUT = ('legal-input-reuse-v1-20261022', '7dd93b1996f79768ead1bb419068b7eb62e3b15d3cf5f529e17c610d5279a02e')
REFS = dict(
    qualification=('legal-cached-target-reader-qualification-v1-20261023','8e0cf620b79cda62c22c9fa77b8f8918088ba7b9040fceeea9be0668dbcbc2b8'),
    cold=('legal-cached-target-reader-cold-v1-20261023','00180a88c30a013cafa65f9bc2558cc4e6c7eb608f23a7f8f4d8060dafc9096b'),
    cache=('legal-target-cache-full-full-v2-20261023','77319fe3eaac3ea9db2ede167519a418151133b5c4641ef19ce6c901d2d5fc72'),
    cache_cold=('legal-target-cache-full-cold-v2-20261023','d4498a44902db8e642ab368e1e1ead435e7e87caed44cae274245215b2a9a77f'),
    schedule=('legal-breadth-schedule-publication-v1-20261023','eada66a4f4e29e95ddecc0e41e2e20fcc9ca3c206d0088eb608c3408314c8b3a'),
    schedule_cold=('legal-breadth-schedule-cold-v1-20261023','f8e76e2e91b576d34dec76daea299e1b54f9d2093c6f1890e06dc82def24c39c'))


def pinned(root, pin):
    return runtime.pinned(root/'report.json', pin)


def code(source):
    root = Path(__file__).resolve().parents[1]
    # Existing full-source authentication/reader code is included without rewriting it.
    from audit import legal_efficient_predictor_training as existing
    names = set(source['code']) | set(existing.code()) | {
        'model/legal_placement_comparison.py', 'audit/legal_placement_comparison_training.py',
        'audit/legal_placement_comparison_readback.py'}
    return {n:file_sha(root/n) for n in sorted(names)}


def physical(root, report):
    for name, pin in report['files'].items(): need(file_sha(root/name) == pin, 'PLACEMENT_COMPARISON_PROOF_BYTES')


def source_bundle(root):
    source = pinned(root/SOURCE[0], SOURCE[1]); base = root/SOURCE[0]
    need(source['complete'] and file_sha(base/'bundle.pt') == source['bundle_sha256'], 'PLACEMENT_COMPARISON_SOURCE')
    for n, pin in source['code'].items():
        need(file_sha(Path(__file__).resolve().parents[1]/n) == pin, 'PLACEMENT_COMPARISON_PRIOR_CODE')
    bundle = torch.load(base/'bundle.pt', map_location='cpu', weights_only=True)
    need(fingerprint(bundle) == source['bundle_state'], 'PLACEMENT_COMPARISON_BUNDLE')
    return source, bundle


def small_metadata(root, items):
    source = pinned(root/INPUT[0], INPUT[1]); ref = source['corpus']; path = Path(ref['path'])
    report = pinned(path, ref['pin']); entry = report['files']['documents.jsonl']
    need(file_sha(path/'documents.jsonl') == entry['sha256'], 'PLACEMENT_COMPARISON_METADATA')
    lookup = {i['document']:i for i in items}
    with (path/'documents.jsonl').open() as stream:
        for ordinal, line in enumerate(stream):
            if ordinal not in lookup: continue
            row = json.loads(line); item = lookup[ordinal]
            need((row['atom_start'],row['atom_end']) == (item['start'],item['stop']), 'PLACEMENT_COMPARISON_BOUNDARY')
            item['source_collection'] = row['source_path']
    need(all('source_collection' in i for i in items), 'PLACEMENT_COMPARISON_SOURCE_MISSING')


def load_data(a, bundle):
    if a.scope == 'small':
        target = PackedNativeRQ(bundle['dimensions'], bundle['config'], bundle['sizes'],
                                'no_encoder_quantizer', bundle['statistics'])
        target.load_state_dict(bundle['target'], strict=True); target.eval().requires_grad_(False)
        labels = FrozenLabels(target); items = bundle['items']; small_metadata(a.root, items)
        fitting = [i for i in items if i['side']=='train']
        codes=torch.cat([i['codes'] for i in fitting]); modal=torch.tensor([
            torch.bincount(codes[:,level],minlength=size).argmax() for level,size in enumerate(target.sizes)])
        for item in items:
            current,_=labels.encode(item['frames'][:-1],canonical=True)
            item['baselines']=dict(copy_current=torch.tensor(current.astype('int64')),
                train_marginal_mode=modal[None].expand(len(item['codes']),-1).clone())
        return dict(labels=labels, items=items, train=[i['document'] for i in fitting],
            read=lambda n:next(i for i in fitting if i['document']==n), close=lambda:labels.fresh(),
            binding=dict(source=SOURCE[1], metadata=INPUT[1]), expected_rows=66, expected_windows=None)
    from audit import legal_efficient_predictor_training as existing
    from model.legal_native_resident_stream import ResidentNativeFrames
    from model.legal_cached_documents import read_document, categorical_baselines
    args = argparse.Namespace(root=a.root, scope=a.scope, arm='breadth')
    for name, (path, pin) in REFS.items(): setattr(args,name,a.root/path); setattr(args,name+'_pin',pin)
    data, continuation, old_reader, labels, cache, meta = existing.breadth.prepare(args)
    schedule, receipt = existing.load_schedule(args, data, cache, meta)
    plan = existing.selection(args, data, meta, schedule)
    old_reader.close()
    proof = existing.small.qualification.training.DEFAULTS['audit']
    reader = ResidentNativeFrames(data, a.root/proof[0], proof[1])
    def read(number, side='train', fit=True):
        item, timing = read_document(data, reader, cache, number, side=side, fit=fit)
        item.update(component=int(data.components[number]), source_collection=meta.rows[number]['source_path'])
        item['read_timings'] = timing
        return item
    items = []; physical_bytes = 0
    for row in plan['evaluations']:
        item = read(row['document'], row['side'], False)
        physical_bytes += sum(x.numel()*x.element_size() for f in item['frames'] for x in f.values())
        # This is an in-memory evaluation cache, not a cap on training work.
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        need(physical_bytes < available//2, 'PLACEMENT_COMPARISON_EVALUATION_CACHE_REQUIRES_TILING')
        baseline, _ = categorical_baselines(cache, continuation, item['uids'][:-1])
        item['baselines'] = {k:torch.from_numpy(v.copy()) for k,v in baseline.items()}; items.append(item)
    def close():
        cache.close(); reader.close(); continuation.fresh(); meta.fresh(); labels.fresh()
    return dict(labels=labels, items=items, train=plan['train_documents'], read=read, close=close,
        binding=dict(source=data.binding, cache=cache.binding, metadata=meta.binding, schedule=REFS['schedule'][1], plan=plan),
        expected_rows=receipt['summary']['targets'] if a.scope=='full' else 4104,
        expected_windows=receipt['summary']['native_windows'] if a.scope=='full' else None)


def checkpoint(path, model, optimizer, binding, step, rows, chain, policy):
    efficient.validate_optimizer(model, optimizer, policy, step)
    value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), binding=binding,
                 step=step, rows=rows, chain=chain, noise_cursor=policy['noise_seed']+step)
    return measure.persist(path, value)


def restore(path, reference, model, optimizer, binding, step, rows, chain, policy):
    need(file_sha(path) == reference['pin'], 'PLACEMENT_COMPARISON_CHECKPOINT_PIN')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(saved) == reference['state'], 'PLACEMENT_COMPARISON_CHECKPOINT_STATE')
    need(saved['binding'] == binding and saved['step'] == step and saved['rows'] == rows
         and saved['chain'] == chain and saved['noise_cursor'] == policy['noise_seed']+step,
         'PLACEMENT_COMPARISON_CHECKPOINT_CURSOR')
    model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    efficient.validate_optimizer(model, optimizer, policy, step)
    need(fingerprint((model.state_dict(),optimizer.state_dict())) == fingerprint((saved['model'],saved['optimizer'])),
         'PLACEMENT_COMPARISON_RESTORE_STATE')


def run(a):
    began = tick = time.perf_counter(); process = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    need(not a.cpu_proof or a.scope == 'small', 'PLACEMENT_COMPARISON_CPU_SCOPE')
    device = 'cpu' if a.cpu_proof else 'cuda'
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    if device == 'cuda':
        take_lease('29','legal-placement-comparison')
        need(torch.cuda.is_available() and not subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'PLACEMENT_COMPARISON_GPU_BUSY')
        need(a.runtime and a.runtime_cold, 'PLACEMENT_COMPARISON_RUNTIME_REQUIRED')
        q, c = pinned(a.runtime,a.runtime_pin), pinned(a.runtime_cold,a.runtime_cold_pin)
        need(q['gpu_qualified'] and c['gpu_qualified'] and q['all_updates_exact'] and c['all_updates_exact']
             and q['binding'] == c['binding'] and c['previous_pin'] == a.runtime_pin
             and q['results_pin'] == c['results_pin'], 'PLACEMENT_COMPARISON_GPU_RUNTIME')
        physical(a.runtime,q); physical(a.runtime_cold,c)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    source, bundle = source_bundle(a.root); pins = code(source)
    for name, pin in pins.items():
        f=a.output/'source'/name; f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes()); need(file_sha(f)==pin,'PLACEMENT_COMPARISON_CODE_COPY')
    policy = dict(bundle['policy'], device=device)
    if a.scope == 'full':
        need(a.qualification and a.qualification_cold, 'PLACEMENT_COMPARISON_FULL_REQUIRES_PROOF')
        q,c = pinned(a.qualification,a.qualification_pin), pinned(a.qualification_cold,a.qualification_cold_pin)
        need(q['complete'] and c['complete'] and q['binding']==c['binding'] and q['binding']['scope']=='qualify'
             and q['binding']['device']=='cuda' and q['binding']['code']==pins and q['binding']['policy']==policy
             and c['previous_pin']==a.qualification_pin and q['results_pin']==c['results_pin'], 'PLACEMENT_COMPARISON_FULL_GATE')
        physical(a.qualification,q); physical(a.qualification_cold,c)
    data = load_data(a,bundle); labels=data['labels']; items=data['items']
    target=deepcopy(labels.model).to(device).eval().requires_grad_(False)
    need(fingerprint(target.state_dict())==bundle['target_fingerprint'], 'PLACEMENT_COMPARISON_FROZEN_TARGET')
    metadata=measure.metadata(items); donors=measure.donor_plan(metadata); measure.validate_donors(metadata,donors)
    donor_pin=write_committed(a.output/'donors.json',dict(metadata=metadata,donors=donors))
    binding=dict(code=pins,policy=policy,source=data['binding'],target=bundle['target_ref'],scope=a.scope,
                 device=device,torch_version=str(torch.__version__),donors=donor_pin)
    previous=pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'PLACEMENT_COMPARISON_COLD_BINDING')
        physical(a.previous,previous)
    phases=dict(authentication_and_selection=time.perf_counter()-tick); results={}; timings={}
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    for name in runtime.original.ARCHITECTURES:
        tick=time.perf_counter(); out=a.output/name; out.mkdir(); model,optimizer=runtime.make(bundle,target,name,device)
        branch=dict(binding,architecture=name); chain=digest(branch); rows=0; windows={n:0 for n in target.roster}
        initial=checkpoint(out/'checkpoint-0.pt',model,optimizer,branch,0,0,chain,policy)
        records, payloads=measure.evaluate(model,target,labels,items,donors,bundle['statistics'],out/'initial')
        initial_metric=measure.summary(records,target.roster); initial_state=fingerprint(records)
        del records,payloads
        phases[name+'_initial']=time.perf_counter()-tick; tick=time.perf_counter(); cps={'0':initial}; updates=0
        old=(a.previous/name/'steps.jsonl').open() if previous else None
        timing=dict(native_read=0.,updates=0.,checkpoint=0.)
        with (out/'steps.jsonl').open('x') as ledger:
            for step, number in enumerate(data['train']):
                start=time.perf_counter(); item=data['read'](number); timing['native_read']+=time.perf_counter()-start
                sync(); start=time.perf_counter()
                update,gradients,noise=efficient.update(model,optimizer,item,step,policy,target=target)
                sync(); timing['updates']+=time.perf_counter()-start; rows+=update['targets']; updates=step+1
                for n in windows:windows[n]+=sum(len(f[n]) for f in item['frames'])
                record=dict(parent=chain,cursor=step,update=update); chain=digest(record);record['hash']=chain
                if old:need(record==json.loads(old.readline()),'PLACEMENT_COMPARISON_COLD_UPDATE')
                if a.scope=='small' and device=='cpu':
                    need(update==bundle['cpu_reference'][name]['updates'][step]['record'],'PLACEMENT_COMPARISON_PRIOR_UPDATE')
                ledger.write(canonical(record).decode()+'\n'); ledger.flush(); os.fsync(ledger.fileno())
                if a.scope!='full' or updates==1 or updates%1024==0 or updates==len(data['train']):
                    start=time.perf_counter(); path=out/f'checkpoint-{updates}.pt'
                    cps[str(updates)]=checkpoint(path,model,optimizer,branch,updates,rows,chain,policy)
                    if previous:
                        need(cps[str(updates)]['state']==previous['results'][name]['checkpoints'][str(updates)]['state'],
                             'PLACEMENT_COMPARISON_COLD_CHECKPOINT')
                    # Continue from the physical full checkpoint, never weights alone.
                    model,optimizer=runtime.make(bundle,target,name,device)
                    restore(path,cps[str(updates)],model,optimizer,branch,updates,rows,chain,policy)
                    timing['checkpoint']+=time.perf_counter()-start
                del gradients,noise
                if updates%128==0 or a.scope!='full':print(json.dumps(dict(architecture=name,step=updates,rows=rows,times=timing)),flush=True)
        if old:need(not old.read(),'PLACEMENT_COMPARISON_EXTRA_UPDATE');old.close()
        need(rows==data['expected_rows'] and updates==len(data['train']),'PLACEMENT_COMPARISON_TRAIN_COVERAGE')
        if data['expected_windows']:need(windows==data['expected_windows'],'PLACEMENT_COMPARISON_WINDOW_COVERAGE')
        phases[name+'_training']=time.perf_counter()-tick; tick=time.perf_counter()
        records,payloads=measure.evaluate(model,target,labels,items,donors,bundle['statistics'],out/'final')
        held=next(i for i in items if i['side']=='query')
        controls=runtime.controls(model,optimizer,item,held,target,updates,policy)
        broken=deepcopy(donors); d=items[0]['document']; broken[str(d)][0]=d
        controls['same_case_donor']=runtime.reject(lambda:measure.validate_donors(metadata,broken))
        controls['checkpoint_cursor']=runtime.reject(lambda:restore(out/f'checkpoint-{updates}.pt',cps[str(updates)],
            model,optimizer,branch,updates+1,rows,chain,policy))
        results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),updates=updates,rows=rows,native_windows=windows,
            initial=initial_metric,final=measure.summary(records,target.roster),initial_rows=initial_state,final_rows=fingerprint(records),
            checkpoints=cps,ledger=chain,controls=controls)
        phases[name+'_final']=time.perf_counter()-tick; timings[name]=timing
        del model,optimizer,records,payloads,item;gc.collect()
    tick=time.perf_counter();data['close']()
    need(fingerprint(target.state_dict())==bundle['target_fingerprint'],'PLACEMENT_COMPARISON_TARGET_CHANGED')
    # Serialized checkpoint files have storage-dependent file bytes, but their tensor states are canonical.
    comparison={n:{k:v for k,v in r.items() if k!='checkpoints'} | {'checkpoint_states':{k:v['state'] for k,v in r['checkpoints'].items()}}
                for n,r in results.items()}
    result_pin=write_committed(a.output/'results.json',comparison)
    if previous:need(result_pin==previous['results_pin'],'PLACEMENT_COMPARISON_COLD_RESULT')
    for n,pin in pins.items():need(file_sha(Path(__file__).resolve().parents[1]/n)==pin,'PLACEMENT_COMPARISON_CODE_CHANGED')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'PLACEMENT_COMPARISON_PHASE_COVERAGE')
    need(device!='cpu' or not torch.cuda.is_initialized(),'PLACEMENT_COMPARISON_UNEXPECTED_GPU')
    report=dict(complete=True,binding=binding,previous_pin=a.previous_pin,results_pin=result_pin,results=results,files=files,
        phases=phases,timings=timings,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        acceptance='author-checked',number_status='measured',capacity_matched=False,model_promotion=False,
        scientific_admission=False,untouched_external_test=False)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=['small','qualify','full'],required=True)
    p.add_argument('--cpu-proof',action='store_true')
    for name in ('previous','runtime','runtime-cold','qualification','qualification-cold'):
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
    finally:release_lease()

"""Energy-batched continuation retaining complete branches and the original evidence format."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as original
from audit import mixed_placement_training as mixed
from audit import energy_placement_qualification as energy
from audit.surface_device_queue import identity
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model import placement_resume_publication as publication
from model.placement_resume_tail import Replay
from model.energy_placement_continuation import EnergyPlacementContinuation
from model import completed_placement_branch as completed
from model.native_population_training import fingerprint
from senses.contract import canonical
from text.constellation_corpus import file_sha,need


NEW = ('model/energy_placement_continuation.py', 'audit/energy_placement_training.py',
       'audit/energy_continuation_readback.py')
CPU = ('energy-placement-cpu-v1-20261025', '24f2a64cdce988871e9a6ebd872c48f4cc486a5a72945332b86118de9006f937')
CPU_COLD = ('energy-placement-cold-v1-20261025', 'd596b5f9b4702f24c6ceda79ccd224f02187dd17f56d056e90139e3323e31aff')


def code():
    root = Path(__file__).resolve().parents[1]
    result = dict(mixed.code())
    result.update({n: file_sha(root/n) for n in (*NEW, *energy.NEW)})
    return result


def energy_gate(a, device):
    root = Path(__file__).resolve().parents[1]
    report = original.pinned(a.root/original.SOURCE[0], original.SOURCE[1])
    pins = dict(original.code(report)); pins.update({n: file_sha(root/n) for n in energy.NEW})
    args = argparse.Namespace(cpu_source=a.root/CPU[0], cpu_pin=CPU[1],
        cpu_cold=a.root/CPU_COLD[0], cpu_cold_pin=CPU_COLD[1])
    energy.cpu_gate(args, pins)
    if device == 'cpu': return
    need(a.runtime and a.runtime_cold, 'ENERGY_DRIVER_CUDA_RUNTIME_REQUIRED')
    q = original.pinned(a.runtime, a.runtime_pin); c = original.pinned(a.runtime_cold, a.runtime_cold_pin)
    expected = dict(source=str(args.cpu_source), pin=args.cpu_pin, cold=str(args.cpu_cold), cold_pin=args.cpu_cold_pin)
    need(q['complete'] and c['complete'] and q['device'] == c['device'] == 'cuda'
         and q['cuda_producer_exact'] and c['gpu_qualified'] and q['all_updates_exact'] and c['all_updates_exact']
         and c['restarts_exact'] and c['reference_replay_exact'] and q['code'] == c['code'] == pins
         and c['source_pin'] == a.runtime_pin and q['cpu_qualification'] == expected
         and q['source_pin'] == original.SOURCE[1] and q['source_state'] == c['source_state'] == report['bundle_state'],
         'ENERGY_DRIVER_CUDA_RUNTIME')
    original.physical(a.runtime, q); original.physical(a.runtime_cold, c)


def gpu_gate(a, execution, manifest):
    need(a.cpu_qualification and a.cpu_qualification_pin, 'ENERGY_DRIVER_CPU_CONTINUATION_REQUIRED')
    cpu = original.pinned(a.cpu_qualification, a.cpu_qualification_pin)
    original.physical(a.cpu_qualification, cpu)
    need(cpu['complete'] and cpu['same_complete_comparison'] and cpu['device'] == 'cpu'
         and cpu['mixed_branch_roles'] == ['completed', 'partial', 'fresh'] and cpu['execution'] == execution
         and cpu['implementation'] == 'energy' and cpu['candidate_class_at_all_boundaries'],
         'ENERGY_DRIVER_CPU_CONTINUATION')
    if a.scope == 'full':
        need(manifest is not None and a.qualification and a.qualification_cold and a.match,
             'ENERGY_DRIVER_CUDA_CONTINUATION_REQUIRED')
        first = original.pinned(a.qualification, a.qualification_pin)
        cold = original.pinned(a.qualification_cold, a.qualification_cold_pin)
        match = original.pinned(a.match, a.match_pin)
        for root, report in ((a.qualification, first), (a.qualification_cold, cold), (a.match, match)):
            original.physical(root, report)
        need(first['complete'] and cold['complete'] and first['binding'] == cold['binding']
             and first['binding']['device'] == 'cuda' and first['binding']['scope'] == 'small'
             and first['execution']['code'] == cold['execution']['code'] == match['execution'] == execution
             and first['execution']['resume'] and cold['execution']['resume']
             and first['results_pin'] == cold['results_pin'] == match['results_pin']
             and cold['previous_pin'] == match['candidate_pin'] == a.qualification_pin
             and match['cold_pin'] == a.qualification_cold_pin and match['device'] == 'cuda'
             and match['same_complete_comparison'] and match['candidate_class_at_all_boundaries']
             and match['mixed_branch_roles'] == ['completed', 'partial', 'fresh']
             and first['execution']['implementation'] == cold['execution']['implementation'] == match['implementation'] == 'energy',
             'ENERGY_DRIVER_CUDA_CONTINUATION')
        mixed.terminal_owners(manifest['owners'])


load_census = mixed.load_census


def run(a):
    began=tick=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    need(not a.cpu_proof or a.scope=='small','PLACEMENT_DRIVER_CPU_SCOPE')
    need(a.boundary_step is None or (a.scope=='small' and 0<a.boundary_step<5),'PLACEMENT_DRIVER_BOUNDARY_SCOPE')
    need(a.scope!='full' or a.resume is not None,'PLACEMENT_DRIVER_FULL_REQUIRES_RESUME')
    device='cpu' if a.cpu_proof else 'cuda';execution=code()
    need(a.implementation == 'energy', 'ENERGY_DRIVER_IMPLEMENTATION')
    energy_gate(a, device)
    manifest=old_census=None
    if a.resume:manifest,old_census=publication.boundary(a.resume,a.resume_pin,execution=publication.code())
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    if device=='cuda':
        gpu_gate(a,execution,manifest)
        take_lease('29','resumed-placement-comparison')
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
             text=True).strip() and torch.cuda.is_available(),'PLACEMENT_DRIVER_GPU_BUSY')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source,bundle=original.source_bundle(a.root);pins=original.code(source)
    for name,pin in {**pins,**execution}.items():
        publication.copy_file(Path(__file__).resolve().parents[1]/name,a.output/'source'/name,pin)
    data=original.load_data(a,bundle);census=load_census(a,data);labels=data['labels'];items=data['items']
    target=deepcopy(labels.model).to(device).eval().requires_grad_(False);policy=dict(bundle['policy'],device=device)
    metadata=original.measure.metadata(items);donors=original.measure.donor_plan(metadata)
    original.measure.validate_donors(metadata,donors)
    donor_pin=write_committed(a.output/'donors.json',dict(metadata=metadata,donors=donors))
    original.measure.persist(a.output/'census.pt',census.value())
    match_authority=manifest['binding']['reader_match'] if manifest else None
    binding=dict(code=pins,policy=policy,source=data['binding'],target=bundle['target_ref'],scope=a.scope,
                 device=device,torch_version=str(torch.__version__),donors=donor_pin,reader_match=match_authority)
    if manifest:need(manifest['binding']==binding and old_census.pin==census.pin,'PLACEMENT_DRIVER_RESUME_BINDING')
    previous=original.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        original.physical(a.previous,previous)
        need(previous['complete'] and previous['binding']==binding and previous['execution']['code']==execution,
             'PLACEMENT_DRIVER_COLD_BINDING')
    phases=dict(authentication=time.perf_counter()-tick);results={};boundaries={};timings={};tail_replays={};reused=[]
    constructions=[]
    for name in original.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();out=a.output/name;out.mkdir();branch=dict(binding,architecture=name)
        engine=EnergyPlacementContinuation(bundle,target,branch,census)
        constructions.append(dict(architecture=name,kind='initial',step=engine.step,model=type(engine.model).__name__))
        inherited=manifest['branches'][name] if manifest else dict(fresh=True,step=0)
        if not inherited.get('fresh'):
            publication.import_branch(a.resume,a.output,manifest,name)
            initial,saved,_=publication.validate_checkpoints(out,inherited,branch,census)
            engine.restore(out/f"checkpoint-{inherited['step']}.pt",inherited['checkpoints'][str(inherited['step'])],
                out/'steps.jsonl',initial,cursor=inherited['step'],chain=saved['chain'])
            constructions.append(dict(architecture=name,kind='restore',step=engine.step,model=type(engine.model).__name__))
            records,_=publication.evaluation(out/'initial',items,name,target.roster)
            need(fingerprint(records)==inherited['initial_rows'] and original.measure.summary(records,target.roster)==inherited['initial'],
                 'PLACEMENT_DRIVER_INITIAL_REUSE')
            cps=deepcopy(inherited['checkpoints'])
            if 'completed' in inherited:
                completed.validate(a.resume,inherited,name,census,items,target.roster,manifest['files'])
                completed.import_final(a.resume,a.output,inherited)
        else:
            cps={'0':original.measure.persist(out/'checkpoint-0.pt',engine.snapshot())}
            records,_=original.measure.evaluate(engine.model,target,labels,items,donors,bundle['statistics'],out/'initial')
            (out/'steps.jsonl').touch(exist_ok=False)
        initial_metric=original.measure.summary(records,target.roster);initial_rows=fingerprint(records);del records
        phases[name+'_initial_or_restore']=time.perf_counter()-tick;tick=time.perf_counter()
        start_step=engine.step;end=a.boundary_step if a.boundary_step is not None else len(census.documents)
        need(start_step<=end,'PLACEMENT_DRIVER_BOUNDARY_BEFORE_CURSOR')
        timing=dict(native_read=0.,updates=0.,checkpoint=0.)
        witness=Replay(out/'retained-tail.jsonl',inherited['tail'],start_step) if 'tail' in inherited else nullcontext()
        with (out/'steps.jsonl').open('ab') as ledger,witness as replay:
            for cursor in range(start_step,end):
                at=time.perf_counter();item=data['read'](census.documents[cursor]);timing['native_read']+=time.perf_counter()-at
                if device=='cuda':torch.cuda.synchronize()
                at=time.perf_counter()
                if replay is not None:
                    record,gradients,noise=engine.commit(item,ledger,replay)
                else:
                    record,gradients,noise=engine.update(item)
                    raw=canonical(record)+b'\n'
                    if replay is not None and not replay.complete:replay.check(raw,cursor)
                    ledger.write(raw);ledger.flush();os.fsync(ledger.fileno())
                if device=='cuda':torch.cuda.synchronize()
                timing['updates']+=time.perf_counter()-at
                if device=='cpu':need(record['update']==bundle['cpu_reference'][name]['updates'][cursor]['record'],
                                      'PLACEMENT_DRIVER_PRIOR_CPU')
                if a.scope!='full' or engine.step==1 or engine.step%1024==0 or engine.step==end:
                    at=time.perf_counter();path=out/f'checkpoint-{engine.step}.pt'
                    cps[str(engine.step)]=original.measure.persist(path,engine.snapshot())
                    # Use the original exact checkpoint reopen, without rescanning
                    # the entire numerical prefix after every in-process save.
                    engine.reopen(path,cps[str(engine.step)])
                    constructions.append(dict(architecture=name,kind='reopen',step=engine.step,model=type(engine.model).__name__))
                    timing['checkpoint']+=time.perf_counter()-at
                del gradients,noise
                if engine.step%128==0 or a.scope=='small':print(json.dumps(dict(architecture=name,step=engine.step,
                    rows=census.counts(engine.step)['rows'],times=timing)),flush=True)
            if replay is not None:tail_replays[name]=replay.finish()
        phases[name+'_remaining_training']=time.perf_counter()-tick;tick=time.perf_counter()
        row=dict(step=engine.step,checkpoints=cps,initial=initial_metric,initial_rows=initial_rows)
        initial,saved,counts=publication.validate_checkpoints(out,row,branch,census);row['counts']=counts
        timings[name]=timing
        if a.boundary_step is not None:
            boundaries[name]=row;phases[name+'_boundary']=time.perf_counter()-tick
            del engine;gc.collect();continue
        need(engine.step==len(census.documents) and counts['rows']==data['expected_rows'],'PLACEMENT_DRIVER_COVERAGE')
        if data['expected_windows']:need(counts['windows']==data['expected_windows'],'PLACEMENT_DRIVER_WINDOWS')
        if 'completed' in inherited:
            completed.validate(a.output,inherited,name,census,items,target.roster,inherited['completed']['files'])
            records,_=publication.evaluation(out/'final',items,name,target.roster)
            engine.model.eval();reused.append(name)
        else:
            records,_=original.measure.evaluate(engine.model,target,labels,items,donors,bundle['statistics'],out/'final')
        item=data['read'](census.documents[-1]);held=next(i for i in items if i['side']=='query')
        controls=original.runtime.controls(engine.model,engine.optimizer,item,held,target,engine.step,policy)
        broken=deepcopy(donors);d=items[0]['document'];broken[str(d)][0]=d
        controls['same_case_donor']=original.runtime.reject(lambda:original.measure.validate_donors(metadata,broken))
        controls['checkpoint_cursor']=original.runtime.reject(lambda:original.restore(out/f'checkpoint-{engine.step}.pt',
            cps[str(engine.step)],engine.model,engine.optimizer,branch,engine.step+1,counts['rows'],engine.chain,policy))
        results[name]=dict(parameters=sum(p.numel() for p in engine.model.parameters()),updates=engine.step,rows=counts['rows'],
            native_windows=counts['windows'],initial=initial_metric,final=original.measure.summary(records,target.roster),
            initial_rows=initial_rows,final_rows=fingerprint(records),checkpoints=cps,ledger=engine.chain,controls=controls)
        phases[name+'_final']=time.perf_counter()-tick;del engine,records,item;gc.collect()
    tick=time.perf_counter();data['close']()
    need(fingerprint(target.state_dict())==bundle['target_fingerprint'],'PLACEMENT_DRIVER_FROZEN_TARGET')
    for n,pin in {**pins,**execution}.items():need(file_sha(Path(__file__).resolve().parents[1]/n)==pin,'PLACEMENT_DRIVER_CODE_CHANGED')
    if a.boundary_step is None:
        comparison={n:{k:v for k,v in r.items() if k!='checkpoints'} |
            {'checkpoint_states':{k:v['state'] for k,v in r['checkpoints'].items()}} for n,r in results.items()}
        result_pin=write_committed(a.output/'results.json',comparison)
        if previous:need(result_pin==previous['results_pin'],'PLACEMENT_DRIVER_COLD_RESULT')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['final_publication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'PLACEMENT_DRIVER_PHASE_COVERAGE')
    need(device!='cpu' or not torch.cuda.is_initialized(),'PLACEMENT_DRIVER_UNEXPECTED_GPU')
    need(all(v['model']=='EnergyBatchedForecaster' for v in constructions), 'ENERGY_DRIVER_CONSTRUCTIONS')
    common=dict(binding=binding,files=files,phases=phases,timings=timings,seconds=seconds,pid=os.getpid(),
                constructions=constructions,candidate_class_at_all_boundaries=True,
                retained_tail_replays=tail_replays,
                cpu_seconds=time.process_time()-cpu_start,acceptance='author-checked',number_status='measured',
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if a.boundary_step is not None:
        result=dict(**common,complete=True,kind='placement-resume-boundary-v1',training_complete=False,
            execution=execution,census=census.pin,branches=boundaries,owners=[],source_job=str(a.output))
        pin=write_committed(a.output/'boundary.json',result)
        print(json.dumps(dict(training_complete=False,boundary_pin=pin)),flush=True)
    else:
        result=dict(**common,complete=True,previous_pin=a.previous_pin,results=results,results_pin=result_pin,
            execution=dict(code=execution,resume=a.resume_pin,scope='complete_original_comparison',
                           implementation=a.implementation,completed_branches_reused=reused),
            capacity_matched=False,model_promotion=False,scientific_admission=False,untouched_external_test=False)
        print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=('small','full'),required=True)
    p.add_argument('--cpu-proof',action='store_true');p.add_argument('--boundary-step',type=int)
    p.add_argument('--implementation',choices=('energy',),required=True)
    for n in ('resume','previous','runtime','runtime-cold','qualification','qualification-cold','match','cpu-qualification'):
        p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
    finally:release_lease()

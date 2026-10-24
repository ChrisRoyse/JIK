"""Real-source frozen label publication, qualification, and complete cold replay."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_region_predictor_readback as qualification
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import native_target_cache as cache, native_target_baselines as baselines
from model.legal_native_training_stream import CertifiedNativeFrames
from model.native_population_training import fingerprint
from text.constellation_corpus import digest, file_sha, need

TARGET_COLD = ('legal-population-training-full-cold-v1-20261023',
               '5493534d6e60d067940783999c8d4fe2301eeaa47875e284a3ba31d65eea63bf')
POLICY = dict(device='cpu', threads=1, tile=128, shard=16384, native_memory_bytes=16<<20,
              code_dtype='uint8', count_dtype='uint8', native_encoder_calls=0, optimizer_updates=0,
              order='ascending_source_uid', cold='every_input_batched_and_canonical_singleton', scientific_admission=False,
              label_runtime='frozen_fp32_weights_expanded_exactly_to_fp64_v2', canonical='singleton_fp64_argmin_lowest_index')


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(qualification.code(), **{n:file_sha(root/n) for n in
        ['audit/legal_target_cache.py','model/native_target_cache.py','model/native_target_baselines.py']})


def prepare(root):
    _,bank,data,c,selected,held,_,_ = qualification.training.authenticate(root)
    path = root/qualification.initialization.STATS[0]
    r = qualification.training.pinned(path/'report.json',qualification.initialization.STATS[1])
    need(file_sha(path/'statistics.pt') == r['files']['statistics.pt'],'TARGET_CACHE_STATISTICS_PIN')
    stats = torch.load(path/'statistics.pt',map_location='cpu',weights_only=True)
    model,fit,ref = qualification.load_target(root,stats,{n:len(stats['centers'][n]) for n in data.roster})
    cold = qualification.training.pinned(root/TARGET_COLD[0]/'report.json',TARGET_COLD[1])
    need(cold['complete'] and cold['mode'] == 'cold' and cold['scope'] == 'full'
         and cold['previous_pin'] == qualification.FIT[1]
         and all(cold[k] == fit[k] for k in ['binding','rows','updates','ledger_head','native_windows','evaluations']),
         'TARGET_CACHE_FULL_TRAIN_COLD')
    labels = cache.FrozenLabels(model)
    reader = CertifiedNativeFrames(data,root/qualification.training.DEFAULTS['audit'][0],
                                   qualification.training.DEFAULTS['audit'][1])
    ids = sorted(set(selected.tolist()) | set(held['development']) | set(held['query']))
    binding = dict(code=code(), policy=POLICY, source=data.binding, population=bank.config['inputs'],
                   target_checkpoint=ref, target_fit=qualification.FIT[1], target_cold=TARGET_COLD[1],
                   model_state=labels.state, runtime_state=labels.runtime_state, sizes=list(labels.sizes), roster=list(labels.roster),
                   torch_version=str(torch.__version__), numpy_version=str(np.__version__))
    return data,c,reader,labels,np.asarray(ids,dtype='<i8'),binding


def seal(root,payload):
    pin = write_committed(root/'report.json',payload)
    with (root/'COMMITTED').open('x') as f:
        import os
        f.write(pin+'\n'); f.flush(); os.fsync(f.fileno())
    cache.sync_directory(root); cache.sync_directory(root.parent)
    return pin


def read_publication(root,pin,binding):
    r = qualification.training.pinned(root/'report.json',pin)
    need((root/'COMMITTED').is_file() and (root/'COMMITTED').read_text().strip() == pin,
         'TARGET_CACHE_PUBLICATION_COMMIT')
    need(r['complete'] and r['binding'] == binding,'TARGET_CACHE_PUBLICATION_BINDING')
    for name,p in r['files'].items(): need(file_sha(root/name) == p,'TARGET_CACHE_PUBLICATION_FILE')
    plan = qualification.training.pinned(root/'plan.json',r['plan_pin'])
    need(plan['scope'] == r['scope'] and plan['rows'] == r['rows'], 'TARGET_CACHE_PLAN_BINDING')
    if plan['scope'] == 'full': need(plan['rows'] == binding['population'],'TARGET_CACHE_FULL_PLAN')
    else:
        need(len(plan['uids']) == plan['rows'],'TARGET_CACHE_PROOF_PLAN')
        cache.validate_uids(np.asarray(plan['uids'],dtype='<i8'),binding['population'])
    return r,plan


def qualification_proof(root,reader,labels,binding,ids,shards):
    need(len(shards) == 1,'TARGET_CACHE_PROOF_SHARD_COUNT')
    _,(codes,counts) = cache.shard_data(root/shards[0]['path'],shards[0]['pin'],binding,ids)
    frames = []
    for lo in range(0,len(ids),128): frames.extend(reader.frames(ids[lo:lo+128].tolist()))
    before = fingerprint(tuple(frames)); singleton = []; singles_counts = []
    for f in frames:
        a,b = labels.encode((f,),reference=True); singleton.append(a); singles_counts.append(b)
    need(np.array_equal(np.concatenate(singleton),codes) and np.array_equal(np.concatenate(singles_counts),counts),
         'TARGET_CACHE_SINGLETON_CODES')
    rev = list(reversed(frames)); rc = []; rn = []
    for lo in range(0,len(rev),128):
        a,b = labels.encode(rev[lo:lo+128],reference=True); rc.append(a); rn.append(b)
    need(np.array_equal(np.concatenate(rc)[::-1],codes) and np.array_equal(np.concatenate(rn)[::-1],counts),
         'TARGET_CACHE_REVERSED_CODES')
    need(fingerprint(tuple(frames)) == before,'TARGET_CACHE_NATIVE_MUTATION')
    # Read count columns independently from authoritative shard offsets.
    for row,uid in enumerate(ids.tolist()):
        start = uid//16384*16384
        for lane,name in enumerate(reader.roster):
            d = reader.descriptor(name,start); a,b = d['offsets'][uid-start:uid-start+2]
            need(int(counts[row,lane]) == int(b-a),'TARGET_CACHE_SOURCE_COUNT')
    return dict(rows=len(ids), singleton_exact=True, reversed_exact=True, numpy_code_selection_exact=True,
                source_counts_exact=True, native_pin=before, native_windows=counts.astype(np.int64).sum(0).tolist())


def arithmetic_regression(reader,labels):
    ids=list(range(183424,183552));frames=reader.frames(ids)
    before=fingerprint(frames)
    batch=labels.encode(frames,reference=True)
    singleton=labels.encode(frames,reference=True,canonical=True)
    reverse=labels.encode(tuple(reversed(frames)),reference=True)
    need(all(np.array_equal(a,b) and np.array_equal(a,c[::-1]) for a,b,c in zip(batch,singleton,reverse)),
         'TARGET_CACHE_KNOWN_ARITHMETIC_REGRESSION')
    need(fingerprint(frames[28])=='474f38013b3b353041fe7e49dcf96f9e0c6c83b46fcaa394dea675ace3eb9b4c'
         and batch[0][28].tolist()==[0,2,18] and fingerprint(frames)==before,'TARGET_CACHE_KNOWN_NATIVE_WITNESS')
    return dict(uids=ids,native_pin=before,all_codes_counts_exact=True,
                witness_uid=183452,witness_codes=batch[0][28].tolist(),canonical='singleton_fp64')


def controls(root,reader,labels,binding,ids,shards):
    negative = root/'negative'; negative.mkdir()
    source = root/shards[0]['path']; pin = shards[0]['pin']
    _,(codes,counts) = cache.shard_data(source,pin,binding,ids)
    result = {}
    for key in ['source','target_checkpoint','code']:
        changed = dict(binding); changed[key] = {'changed':True}
        result['changed_'+key] = rejected(lambda:cache.shard_data(source,pin,changed,ids))
    result['missing_uid'] = rejected(lambda:cache.shard_data(source,pin,binding,ids[:-1]))
    duplicate = ids.copy(); duplicate[1] = duplicate[0]
    result['duplicate_uid'] = rejected(lambda:cache.shard_data(source,pin,binding,duplicate))
    result['reversed_uid'] = rejected(lambda:cache.shard_data(source,pin,binding,ids[::-1]))
    badcodes = codes.copy(); badcodes[0,0] = binding['sizes'][0]
    badcounts = counts.copy(); badcounts[0,0] = 0
    result['invalid_code'] = rejected(lambda:cache.validate_arrays(badcodes,counts,len(ids),labels.sizes,labels.roster))
    result['invalid_count'] = rejected(lambda:cache.validate_arrays(codes,badcounts,len(ids),labels.sizes,labels.roster))
    missing = negative/'missing-commit'; shutil.copytree(source,missing,ignore=shutil.ignore_patterns('COMMITTED'))
    result['missing_commit'] = rejected(lambda:cache.shard_data(missing,pin,binding,ids))
    corrupt = negative/'corrupted'; shutil.copytree(source,corrupt)
    with (corrupt/'codes.npy').open('r+b') as f: f.seek(-1,2); f.write(b'\xff')
    result['corrupted_bytes'] = rejected(lambda:cache.shard_data(corrupt,pin,binding,ids))
    before = fingerprint(labels.model.state_dict()); original = next(labels.model.parameters()).detach().clone()
    with torch.no_grad(): next(labels.model.parameters()).add_(1.)
    result['changed_model'] = rejected(labels.fresh)
    with torch.no_grad(): next(labels.model.parameters()).copy_(original)
    need(before == fingerprint(labels.model.state_dict()),'TARGET_CACHE_CONTROL_MODEL_RESTORE')
    return result


def cold_replay(root,pin,binding,reader,labels,output):
    r,plan = read_publication(root,pin,binding)
    exports = [np.load(root/'export'/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ['codes','counts']]
    need(len(r['shards']) == (plan['rows']+cache.SHARD-1)//cache.SHARD,'TARGET_CACHE_COLD_SHARDS')
    rows = 0; windows = np.zeros(len(labels.roster),dtype=np.int64); phases = dict(reading=0.,forward=0.,readback=0.)
    for ref,(lo,hi,ids) in zip(r['shards'],cache.ranges(plan)):
        tick=time.perf_counter()
        need(ref['start'] == lo and ref['stop'] == hi,'TARGET_CACHE_COLD_SHARD_ORDER')
        _,arrays = cache.shard_data(root/ref['path'],ref['pin'],binding,ids)
        need(all(np.array_equal(a,b[lo:hi]) for a,b in zip(arrays,exports)),'TARGET_CACHE_EXPORT_RECOUNT')
        phases['readback'] += time.perf_counter()-tick
        for j in range(0,len(ids),128):
            tick=time.perf_counter(); frames = reader.frames(ids[j:j+128].tolist())
            phases['reading'] += time.perf_counter()-tick; tick=time.perf_counter()
            actual = labels.encode(frames,reference=True)
            singleton = labels.encode(frames,reference=True,canonical=True)
            need(all(np.array_equal(a,b) for a,b in zip(actual,singleton)), 'TARGET_CACHE_COLD_CANONICAL_SINGLETON')
            need(all(np.array_equal(a,b[j:j+len(frames)]) for a,b in zip(actual,arrays)), 'TARGET_CACHE_COLD_LABEL')
            rows += len(frames); windows += actual[1].astype(np.int64).sum(0)
            phases['forward'] += time.perf_counter()-tick
        write_committed(output/f'shard-{lo:012d}.json',dict(start=lo,stop=hi,source_pin=ref['pin'],all_rows_exact=True))
        if len(r['shards']) > 1: print(json.dumps(dict(stage='cold',rows=rows,total=plan['rows'])),flush=True)
    need(rows == plan['rows'],'TARGET_CACHE_COLD_COVERAGE')
    baseline = None
    if plan['scope'] == 'full':
        expected = qualification.training.pinned(root/'baselines.json',r['files']['baselines.json'])
        baseline = baselines.recount(reader.data,*exports,labels.sizes,labels.roster)
        need(baseline == expected,'TARGET_CACHE_COLD_BASELINES')
        write_committed(output/'baselines.json',baseline)
    for a in exports: a._mmap.close()
    return dict(rows=rows,native_windows=windows.tolist(),all_rows_exact=True,all_exports_exact=True,
                all_rows_canonical_singleton_exact=True,
                baseline_rows=None if baseline is None else baseline['rows'],phases_seconds=phases)


def run(a):
    began=tick=time.perf_counter(); cpu=time.process_time()
    a.output.mkdir(parents=True,exist_ok=a.resume)
    lock = (Path('/home/croyse/mnlm/.lease')/'legal-native-target-cache').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(),'TARGET_CACHE_UNEXPECTED_CUDA')
    for name,pin in code().items():
        dest=a.output/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest)==pin,'TARGET_CACHE_SOURCE_SNAPSHOT')
    data,c,reader,labels,selected,binding = prepare(a.root)
    phases = dict(authentication=time.perf_counter()-tick); tick=time.perf_counter()
    if a.mode == 'cold':
        need(a.previous is not None and a.previous_pin is not None,'TARGET_CACHE_COLD_REQUIRED')
        previous,plan = read_publication(a.previous,a.previous_pin,binding)
        result = cold_replay(a.previous,a.previous_pin,binding,reader,labels,a.output)
        phases['complete_replay'] = time.perf_counter()-tick
        payload = dict(complete=True,binding=binding,mode='cold',scope=plan['scope'],previous_pin=a.previous_pin,result=result)
    else:
        if a.mode == 'full':
            previous,oldplan = read_publication(a.previous,a.previous_pin,binding)
            cold = qualification.training.pinned(a.cold_proof/'report.json',a.cold_pin)
            need(oldplan['scope']=='qualify' and previous['qualification']['rows']==len(selected)
                 and cold['complete'] and cold['mode']=='cold' and cold['binding']==binding
                 and cold['previous_pin']==a.previous_pin and cold['result']['all_rows_exact']
                 and cold['result']['all_rows_canonical_singleton_exact'], 'TARGET_CACHE_FULL_QUALIFICATION')
        plan = dict(scope=a.mode,rows=binding['population'] if a.mode=='full' else len(selected),
                    uids=None if a.mode=='full' else selected.tolist())
        if (a.output/'plan.json').exists():
            need(a.resume and json.loads((a.output/'plan.json').read_bytes())==plan,'TARGET_CACHE_RESUME_PLAN')
            plan_pin=file_sha(a.output/'plan.json')
        else: plan_pin=write_committed(a.output/'plan.json',plan)
        shards=[]
        for lo,hi,ids in cache.ranges(plan):
            path=a.output/'shards'/f'{lo:012d}-{hi:012d}'
            if path.exists():
                need(a.resume and (path/'COMMITTED').is_file(),'TARGET_CACHE_INCOMPLETE_SHARD')
                pin=(path/'COMMITTED').read_text().strip(); cache.shard_data(path,pin,binding,ids)
            else: pin=cache.publish(path,reader,labels,binding,ids)
            shards.append(dict(start=lo,stop=hi,path=str(path.relative_to(a.output)),pin=pin))
            print(json.dumps(dict(stage='publish',rows=hi,total=plan['rows'],shard_pin=pin)),flush=True)
        phases['shard_publication']=time.perf_counter()-tick; tick=time.perf_counter()
        exports=cache.export(a.output,shards,plan,binding)
        proof=negative=None
        if a.mode=='qualify':
            proof=qualification_proof(a.output,reader,labels,binding,selected,shards)
            proof['arithmetic_regression']=arithmetic_regression(reader,labels)
            negative=controls(a.output,reader,labels,binding,selected,shards)
        else:
            maps=[np.load(a.output/'export'/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ['codes','counts']]
            baseline=baselines.derive(data,*maps,labels.sizes,labels.roster)
            need(baseline==baselines.recount(data,*maps,labels.sizes,labels.roster),'TARGET_CACHE_BASELINE_RECOUNT')
            write_committed(a.output/'baselines.json',baseline)
            for m in maps:m._mmap.close()
        phases['export_qualification_baselines']=time.perf_counter()-tick
        payload=dict(complete=True,binding=binding,mode='publish',scope=plan['scope'],rows=plan['rows'],
                     plan_pin=plan_pin,shards=shards,qualification=proof,negative=negative,
                     previous_pin=a.previous_pin,qualification_cold=a.cold_pin)
    tick=time.perf_counter(); reader.close(); data.fresh(); c.fresh()
    need(fingerprint(labels.model.state_dict())==binding['model_state']
         and fingerprint(labels.runtime.state_dict())==binding['runtime_state'] and not torch.cuda.is_initialized(),
         'TARGET_CACHE_FINAL_MODEL_OR_DEVICE')
    payload['files']={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_authentication_hashes']=time.perf_counter()-tick
    seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'TARGET_CACHE_PHASE_ATTRIBUTION')
    payload.update(phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                   reader=reader.stats,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   number_status='measured',acceptance='author-checked',implicit_training_authority=False)
    pin=seal(a.output,payload)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['qualify','full','cold'],required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--cold-proof',type=Path);p.add_argument('--cold-pin');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists() and not (a.output/'failure.json').exists():
            write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise

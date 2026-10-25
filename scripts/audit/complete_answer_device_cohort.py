"""Declared-device, resident whole-answer evaluation with durable public cases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES')
if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.complete_answer_evaluation import artifact,read,reject,RUNTIME,CACHE,BASE
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model import complete_answer_cohort as cohort
from model.complete_answer_evaluation import ARCHITECTURES,METADATA,EvaluationCheckpoint
from model.legal_answer_public_pack import PublicPack
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import canonical,digest,file_sha,need

REQUIRED = ('audit/complete_answer_device_cohort.py','model/complete_answer_cohort.py',
    'model/complete_answer_evaluation.py','model/complete_answer_trace.py',
    'model/legal_answer_public_pack.py','model/native_open_window_answer.py',
    'model/native_window_end.py','model/task_answer_training.py')


def receipt(root, stem):
    i = 0
    while (root/f'{stem}-{i:06d}.json').exists():i += 1
    return root/f'{stem}-{i:06d}.json'


def gpu_ready(output):
    """No CUDA initialization or lease acquisition precedes this physical check."""
    need(not torch.cuda.is_initialized(), 'DEVICE_COHORT_EARLY_CUDA')
    snapshot = subprocess.check_output(['nvidia-smi'],text=True)
    resident = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                       '--format=csv,noheader'],text=True)
    write_committed(receipt(output,'gpu-before'),dict(snapshot=snapshot,resident=resident))
    need(not resident.strip(), 'DEVICE_COHORT_GPU_RESIDENT')


def validate(m, runtime, device):
    need(set(m['checkpoints']) == set(ARCHITECTURES), 'DEVICE_COHORT_ARCHITECTURES')
    need(set(runtime) == set(RUNTIME), 'DEVICE_COHORT_RUNTIME')
    need(device in ('cpu','cuda:0'), 'DEVICE_COHORT_DEVICE')
    target_pin = digest(canonical(dict(parent=runtime['target_ref'],
        runtime=file_sha(BASE/'model/native_open_window_answer.py'),
        physical_policy=file_sha(BASE/'model/native_window_end.py'),
        rule='preserve_TRAIN_position_prefix_then_TRAIN_mean')))
    need(fingerprint(runtime['target']) == runtime['target_fingerprint'], 'DEVICE_COHORT_TARGET')
    for name,row in m['checkpoints'].items():
        need(set(row) == {'reference','metadata'} and set(row['reference']) == {'path','pin','state'},
             'DEVICE_COHORT_CHECKPOINT_FIELDS')
        v = row['metadata']; need(set(v) == set(METADATA), 'DEVICE_COHORT_METADATA')
        b = v['binding']; policy = b['policy']
        need(v['schema'] == b['schema'] == 'legal-complete-answer-epoch-v1'
             and b['architecture'] == name, 'DEVICE_COHORT_CHECKPOINT_SCHEMA')
        need(b['device'] == policy['device'] == device and b['torch'] == str(torch.__version__)
             and b['cuda'] == (torch.version.cuda if device == 'cuda:0' else None), 'DEVICE_COHORT_DEVICE')
        need(policy == dict(runtime['policy'],device=device) and b['target'] == target_pin
             and b['target_state'] == runtime['target_fingerprint'], 'DEVICE_COHORT_POLICY')
        need(type(v['cursor']) is int and 0 <= v['cursor'] <= b['documents']
             and v['step'] == b['initial_step']+v['cursor']
             and v['noise_cursor'] == policy['noise_seed']+v['step']
             and v['counts']['documents'] == v['counts']['ends'] == v['cursor'], 'DEVICE_COHORT_CURSOR')
    first = m['checkpoints']['shared']['metadata']
    for row in m['checkpoints'].values():
        v = row['metadata']
        need(all(v[k] == first[k] for k in ('step','counts'))
             and all(v['binding'][k] == first['binding'][k] for k in ('source','schedule','full_population')),
             'DEVICE_COHORT_UNMATCHED')
    intervention = m['intervention']
    pressure = dict(kind=('CUDA' if device == 'cuda:0' else 'CPU')+'_memory_pressure_mechanics',cache_bytes=CACHE)
    need(intervention in (pressure,dict(kind='trained_runtime_policy')), 'DEVICE_COHORT_INTERVENTION')
    need(device == 'cuda:0' or intervention == pressure, 'DEVICE_COHORT_CPU_PROOF_ONLY')
    if m['ordinals'] is None:
        need(device == 'cuda:0' and intervention == dict(kind='trained_runtime_policy'), 'DEVICE_COHORT_FULL_SCOPE')
        for row in m['checkpoints'].values():
            v = row['metadata']; b = v['binding']; c = v['counts']
            need(b['full_population'] and b['documents'] == v['cursor'] == 91540
                 and c['atoms'] == 6547176 and c['answers'] == 6368511 and c['decisions'] == 6460051,
                 'DEVICE_COHORT_FULL_TRAINING')
    return target_pin,intervention == pressure


def access_guard(a, m):
    pack_root = Path(m['pack']['path']); manifest = read(pack_root/'report.json',m['pack']['pin'])
    allowed = {a.request.resolve(),Path(m['runtime']['path']).resolve(),(pack_root/'report.json').resolve()}
    allowed.update(Path(v['reference']['path']).resolve() for v in m['checkpoints'].values())
    for name in manifest['files']:
        need(Path(name).name == name, 'DEVICE_COHORT_PACK_FILE')
        allowed.add((pack_root/name).resolve())
    observed = set(); root = a.output.resolve()
    def access(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to('/zfs'):return
        need(path in allowed or path.is_relative_to(root), 'DEVICE_COHORT_UNDECLARED_ARTIFACT')
        if path in allowed:observed.add(str(path))
    sys.addaudithook(access)
    return allowed,observed


def worker(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=True)
    with (a.output/'worker.lock').open('a+') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('DEVICE_COHORT_OWNER_BUSY') from exc
        m = read(a.request,a.request_pin)
        need(set(m) == {'schema','runtime','checkpoints','code','authority','pack','ordinals','intervention'}
             and m['schema'] == 'public-pack-answer-cohort-v1', 'DEVICE_COHORT_SCHEMA')
        need(set(m['checkpoints']) == set(ARCHITECTURES), 'DEVICE_COHORT_ARCHITECTURES')
        need(set(m['pack']) == {'path','pin'} and set(REQUIRED) <= set(m['code']), 'DEVICE_COHORT_CODE_FIELDS')
        allowed,observed = access_guard(a,m)
        for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'DEVICE_COHORT_CODE')
        runtime = artifact(m['runtime']); device = 'cuda:0' if a.cuda else a.inspect_device if a.inspect else 'cpu'
        target_pin,pressure = validate(m,runtime,device)
        need(not a.cuda or (VISIBLE != '' and os.environ.get('CUDA_VISIBLE_DEVICES') == VISIBLE),
             'DEVICE_COHORT_GPU_HIDDEN')
        mode = 'inspect' if a.inspect else 'cuda' if a.cuda else 'cpu-proof'
        identity = dict(request=a.request_pin,device=device,mode=mode,intervention=m['intervention'])
        path = a.output/'worker-binding.json'
        if path.exists():need(json.loads(path.read_bytes()) == identity, 'DEVICE_COHORT_BINDING_CHANGED')
        else:write_committed(path,identity)
        binding_pin = file_sha(path); pack = None; leased = False
        try:
            pack = PublicPack(m['pack']['path'],m['pack']['pin']); ordinals = m['ordinals']
            if ordinals is None:cases = pack.cases
            else:
                need(isinstance(ordinals,list) and ordinals and ordinals == sorted(set(ordinals))
                     and all(type(x) is int and 0 <= x < len(pack.cases) for x in ordinals), 'DEVICE_COHORT_ORDINALS')
                cases = [pack.cases[i] for i in ordinals]
            phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
            if a.inspect:
                for row in m['checkpoints'].values():
                    saved = artifact(row['reference'])
                    need(set(saved) == {*METADATA,'model','optimizer'}
                         and all(saved[k] == row['metadata'][k] for k in METADATA), 'DEVICE_COHORT_SAVED_METADATA')
                    del saved
                result = dict(complete=True,cases=len(cases),checkpoint_loads=0,frozen_target_loads=0,
                    checkpoint_artifact_reads=3,private_reference_received=False,optimizer_updates=0,
                    capability_demonstrated=False,architecture_winner=False)
                phases['inspect_checkpoints'] = time.perf_counter()-tick
            else:
                if a.cuda:
                    gpu_ready(a.output); take_lease('29','complete-answer-evaluation'); leased = True
                    need(torch.cuda.is_available(), 'DEVICE_COHORT_CUDA_REQUIRED')
                    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
                target,actual_pin = extension(runtime); target.to(device)
                need(actual_pin == target_pin and fingerprint(target.state_dict()) == runtime['target_fingerprint'],
                     'DEVICE_COHORT_TARGET_RESTORED')
                loaded = {}
                for name in ARCHITECTURES:
                    row = m['checkpoints'][name]
                    v = EvaluationCheckpoint(row['reference']['path'],row['reference'],row['metadata'],
                                             runtime,target,target_pin,device=device)
                    need(v.architecture == name, 'DEVICE_COHORT_ARCHITECTURE')
                    if pressure:v.model.forecaster.predictor.cache_bytes = CACHE
                    need(v.model.forecaster.predictor.cache_bytes == (CACHE if pressure else runtime['policy']['cache_bytes']),
                         'DEVICE_COHORT_CACHE_POLICY')
                    loaded[name] = v
                phases['resident_load'] = time.perf_counter()-tick; tick = time.perf_counter()
                result = cohort.run(loaded,cases,pack.load,target=target,output=a.output/'run',authority=m['authority'])
                if a.cuda:torch.cuda.synchronize()
                phases['resident_cases'] = time.perf_counter()-tick
            tick = time.perf_counter(); pack.close(); pack = None
            for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'DEVICE_COHORT_CODE_CHANGED')
            need(file_sha(a.request) == a.request_pin and file_sha(path) == binding_pin, 'DEVICE_COHORT_FINAL')
            need(a.cuda or not torch.cuda.is_initialized(), 'DEVICE_COHORT_UNEXPECTED_CUDA')
            outside = Path(m['checkpoints']['shared']['reference']['path']).parent.parent/'report.json'
            refusal = reject(lambda:file_sha(outside),'DEVICE_COHORT_UNDECLARED_ARTIFACT')
            need(observed == {str(p) for p in allowed}, 'DEVICE_COHORT_INPUT_COVERAGE')
            phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
            need(.9 <= sum(phases.values())/seconds <= 1.001, 'DEVICE_COHORT_PHASES')
            result.update(request=a.request_pin,pack=m['pack'],worker_binding=binding_pin,mode=mode,device=device,
                memory_policy=dict(runtime=runtime['policy'],intervention=m['intervention']),
                observed_artifact_inputs=sorted(observed),undeclared_artifact_refused=refusal,
                access_guard='Python open audit hook over ZFS; not an OS sandbox',gpu_used=a.cuda,
                timings=phases,worker_seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.cuda else 0)
            dest = a.output/'report.json'
            if dest.exists():
                previous = json.loads(dest.read_bytes())
                need(previous['worker_binding'] == binding_pin and previous['request'] == a.request_pin,
                     'DEVICE_COHORT_PRIOR_REPORT')
                if not a.inspect:
                    stable = lambda rows:[{k:v for k,v in r.items() if k != 'action'} for r in rows]
                    need(stable(result['results']) == stable(previous['results']), 'DEVICE_COHORT_REOPEN_CHANGED')
                result['previous_report'] = file_sha(dest); dest = receipt(a.output,'reopen')
            pin = write_committed(dest,result)
            print(json.dumps(dict(complete=True,path=str(dest),pin=pin,cases=len(cases),seconds=seconds)),flush=True)
        finally:
            if pack is not None:pack.close()
            if leased:release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    for name in ('inspect','cpu-proof','cuda'):g.add_argument('--'+name,action='store_true')
    p.add_argument('--inspect-device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--request',type=Path,required=True); p.add_argument('--request-pin',required=True)
    p.add_argument('--output',type=Path,required=True); a = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:worker(a)
    except BaseException as exc:
        if a.output.exists():write_committed(receipt(a.output,'failure'),dict(error=type(exc).__name__,detail=str(exc)))
        raise

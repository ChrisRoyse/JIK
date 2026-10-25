"""Resident whole-answer evaluation from an immutable, public-only native pack."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.complete_answer_evaluation import artifact, read, reject, RUNTIME, CACHE, BASE
from claims.process_conditions import write_committed
from model import complete_answer_cohort as cohort
from model.complete_answer_evaluation import ARCHITECTURES, EvaluationCheckpoint
from model.legal_answer_public_pack import PublicPack
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need


def worker(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    m = read(a.request,a.request_pin)
    need(set(m) == {'schema','runtime','checkpoints','code','authority','pack','ordinals','intervention'}
         and m['schema'] == 'public-pack-answer-cohort-v1', 'PACKED_COHORT_SCHEMA')
    need(set(m['pack']) == {'path','pin'}, 'PACKED_COHORT_PACK_FIELDS')
    pack_root = Path(m['pack']['path']); manifest = read(pack_root/'report.json',m['pack']['pin'])
    allowed = {a.request.resolve(),Path(m['runtime']['path']).resolve(),(pack_root/'report.json').resolve()}
    allowed.update(Path(v['reference']['path']).resolve() for v in m['checkpoints'].values())
    for name in manifest['files']:
        need(Path(name).name == name, 'PACKED_COHORT_PACK_FILE')
        allowed.add((pack_root/name).resolve())
    observed = set(); root = a.output.resolve()
    def access(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to('/zfs'):return
        need(path in allowed or path.is_relative_to(root), 'PACKED_COHORT_UNDECLARED_ARTIFACT')
        if path in allowed:observed.add(str(path))
    sys.addaudithook(access)
    need(a.cpu_proof and m['intervention'] == dict(kind='CPU_memory_pressure_mechanics',cache_bytes=CACHE),
         'PACKED_COHORT_EXPLICIT_CPU_PROOF')
    for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'PACKED_COHORT_CODE')
    pack = PublicPack(pack_root,m['pack']['pin'])
    ordinals = m['ordinals']
    if ordinals is None:cases = pack.cases
    else:
        need(isinstance(ordinals,list) and ordinals and all(type(x) is int and 0 <= x < len(pack.cases) for x in ordinals)
             and ordinals == sorted(set(ordinals)), 'PACKED_COHORT_ORDINALS')
        cases = [pack.cases[i] for i in ordinals]
    runtime = artifact(m['runtime']); need(set(runtime) == set(RUNTIME), 'PACKED_COHORT_RUNTIME')
    target,target_pin = extension(runtime)
    need(fingerprint(target.state_dict()) == runtime['target_fingerprint'], 'PACKED_COHORT_TARGET')
    phases = dict(public_pack_and_runtime=time.perf_counter()-tick); tick = time.perf_counter()
    need(set(m['checkpoints']) == set(ARCHITECTURES), 'PACKED_COHORT_ARCHITECTURES')
    loaded = {}
    for name in ARCHITECTURES:
        row = m['checkpoints'][name]
        value = EvaluationCheckpoint(row['reference']['path'],row['reference'],row['metadata'],
                                     runtime,target,target_pin,device='cpu')
        need(value.architecture == name, 'PACKED_COHORT_ARCHITECTURE')
        value.model.forecaster.predictor.cache_bytes = CACHE; loaded[name] = value
    phases['checkpoint_load'] = time.perf_counter()-tick; tick = time.perf_counter()
    result = cohort.run(loaded,cases,pack.load,target=target,output=a.output/'run',authority=m['authority'])
    phases['resident_cases'] = time.perf_counter()-tick; tick = time.perf_counter()
    pack.close()
    for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'PACKED_COHORT_CODE_CHANGED')
    need(file_sha(a.request) == a.request_pin and not torch.cuda.is_initialized(), 'PACKED_COHORT_FINAL')
    outside = Path(m['checkpoints']['shared']['reference']['path']).parent.parent/'report.json'
    refusal = reject(lambda:file_sha(outside),'PACKED_COHORT_UNDECLARED_ARTIFACT')
    need(observed == {str(p) for p in allowed}, 'PACKED_COHORT_INPUT_COVERAGE')
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'PACKED_COHORT_PHASES')
    result.update(request=a.request_pin,pack=m['pack'],observed_artifact_inputs=sorted(observed),
        undeclared_artifact_refused=refusal,access_guard='Python open audit hook over ZFS; not an OS sandbox',
        timings=phases,worker_seconds=seconds,gpu_used=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    pin = write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,cases=len(cases),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True); p.add_argument('--request-pin',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--cpu-proof',action='store_true')
    a = p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    try:worker(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

"""Public-only resident evaluator. CPU flag is an explicit mechanics intervention."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.complete_answer_evaluation import artifact, read, reject, RUNTIME, CACHE, BASE
from claims.process_conditions import write_committed
from model import complete_answer_cohort as cohort
from model.complete_answer_evaluation import ARCHITECTURES, EvaluationCheckpoint
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need


def worker(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=True)
    need(re.fullmatch('[a-z0-9-]+', a.receipt) and not (a.output/(a.receipt+'.json')).exists(), 'COHORT_RECEIPT_NAME')
    m = read(a.request, a.request_pin)
    need(set(m) == {'schema', 'runtime', 'checkpoints', 'code', 'authority', 'cases', 'intervention'}
         and m['schema'] == 'public-only-answer-cohort-v1', 'COHORT_WORKER_SCHEMA')
    cohort.validate_cases(m['cases'])
    allowed = {a.request.resolve(), Path(m['runtime']['path']).resolve()}
    allowed.update(Path(v['reference']['path']).resolve() for v in m['checkpoints'].values())
    allowed.update(Path(v['public']['path']).resolve() for v in m['cases'])
    observed = set(); root = a.output.resolve()
    def access(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to('/zfs'):return
        need(path in allowed or path.is_relative_to(root), 'COHORT_UNDECLARED_ARTIFACT_ACCESS')
        if path in allowed:observed.add(str(path))
    sys.addaudithook(access)
    need(a.cpu_proof and m['intervention'] == dict(kind='CPU_memory_pressure_mechanics', cache_bytes=CACHE),
         'COHORT_EXPLICIT_CPU_PROOF')
    for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'COHORT_WORKER_CODE')
    runtime = artifact(m['runtime']); need(set(runtime) == set(RUNTIME), 'COHORT_RUNTIME_FIELDS')
    target, target_pin = extension(runtime)
    need(fingerprint(target.state_dict()) == runtime['target_fingerprint'], 'COHORT_TARGET_STATE')
    phases = dict(runtime=time.perf_counter()-tick); tick = time.perf_counter()
    need(set(m['checkpoints']) == set(ARCHITECTURES), 'COHORT_CHECKPOINT_ARCHITECTURES')
    loaded = {}
    for name in ARCHITECTURES:
        row = m['checkpoints'][name]
        value = EvaluationCheckpoint(row['reference']['path'], row['reference'], row['metadata'],
                                     runtime, target, target_pin, device='cpu')
        need(value.architecture == name, 'COHORT_CHECKPOINT_ARCHITECTURE')
        value.model.forecaster.predictor.cache_bytes = CACHE; loaded[name] = value
    phases['checkpoint_load'] = time.perf_counter()-tick; tick = time.perf_counter()
    # Faults exercise real durable boundaries in child processes, only in CPU proof.
    if a.fault == 'trace':
        from model import complete_answer_evaluation as evaluation
        original = evaluation.collect
        def stop_after_trace(*args, **kwargs):
            original(*args, **kwargs); os._exit(71)
        evaluation.collect = stop_after_trace
    elif a.fault == 'generation':
        original = cohort.generate
        def stop_after_generation(*args, **kwargs):
            original(*args, **kwargs); os._exit(72)
        cohort.generate = stop_after_generation
    result = cohort.run(loaded, m['cases'], artifact, target=target, output=a.output/'run', authority=m['authority'])
    phases['resident_cases'] = time.perf_counter()-tick; tick = time.perf_counter()
    for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'COHORT_WORKER_CODE_CHANGED')
    need(file_sha(a.request) == a.request_pin and not torch.cuda.is_initialized(), 'COHORT_WORKER_FINAL')
    outside = Path(m['checkpoints']['shared']['reference']['path']).parent.parent/'report.json'
    refusal = reject(lambda:file_sha(outside), 'COHORT_UNDECLARED_ARTIFACT_ACCESS')
    need(observed == {str(p) for p in allowed}, 'COHORT_INPUT_COVERAGE')
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'COHORT_WORKER_PHASES')
    result.update(request=a.request_pin, observed_artifact_inputs=sorted(observed), undeclared_artifact_refused=refusal,
        access_guard='Python open audit hook over ZFS artifacts; not an OS sandbox',
        timings=phases, worker_seconds=seconds, gpu_used=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    pin = write_committed(a.output/(a.receipt+'.json'), result)
    print(json.dumps(dict(pin=pin, actions=result['actions'], cases=len(result['results']), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True); p.add_argument('--request-pin', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--receipt', required=True)
    p.add_argument('--cpu-proof', action='store_true'); p.add_argument('--fault', choices=('trace', 'generation'))
    a = p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    try:worker(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/(a.receipt+'-failure.json'), dict(error=type(exc).__name__, detail=str(exc)))
        raise

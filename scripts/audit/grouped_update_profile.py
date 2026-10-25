"""Profile real grouped-placement CPU proof updates without changing their recipe."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/'lens')]
from audit import legal_placement_runtime_readback as proof
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need

SOURCE = ('legal-placement-runtime-source-v2-20261023',
          'a5976a125b4a41e42af703a2c12d10d835531250476fb460792aa900b07e2a97')


def rows(stats):
    return [dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0],
                 calls=value[1], self_seconds=value[2], cumulative_seconds=value[3])
            for key, value in sorted(stats.stats.items(), key=lambda kv: (-kv[1][2], kv[0]))]


def run(a):
    a.output.mkdir(exist_ok=False)
    original = proof.efficient.update
    observations = []
    seen = set()
    code = file_sha(__file__)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()

    def observe(model, optimizer, item, step, policy, *, target):
        key = (model.architecture, step)
        # The unchanged proof invokes the same update in its refusal controls.
        # Their work stays in that proof; each main update is profiled once.
        if key in seen or step >= 5:
            return original(model, optimizer, item, step, policy, target=target)
        need(item['side'] == 'train' and step == sum(x[0] == key[0] for x in seen),
             'GROUPED_PROFILE_CURSOR')
        seen.add(key)
        profiler = cProfile.Profile()
        path = a.output/f'{model.architecture}-{step+1}.pstats'
        tick = time.perf_counter()
        try:
            profiler.enable()
            value = original(model, optimizer, item, step, policy, target=target)
        finally:
            profiler.disable()
            elapsed = time.perf_counter()-tick
            profiler.dump_stats(path)
        stats = pstats.Stats(str(path))
        need(.9 <= stats.total_tt/elapsed <= 1.000001, 'GROUPED_PROFILE_COVERAGE')
        observations.append(dict(architecture=model.architecture, document=item['document'],
            step=step+1, targets=value[0]['targets'], seconds=elapsed,
            profiler_seconds=stats.total_tt, total_calls=stats.total_calls,
            primitive_calls=stats.prim_calls, profile=path.name, pin=file_sha(path)))
        return value

    proof.efficient.update = observe
    try:
        proof.run(argparse.Namespace(source=a.root/SOURCE[0], source_pin=SOURCE[1],
                                    output=a.output/'numerical', cpu_proof=True,
                                    previous=None, previous_pin=None))
    finally:
        intact = proof.efficient.update is observe
        proof.efficient.update = original
        need(intact, 'GROUPED_PROFILE_OWNERSHIP_CHANGED')
    report = json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and report['cpu_reference_exact']
         and not report['gpu_qualified'] and len(observations) == 15
         and sum(x['targets'] for x in observations) == 198, 'GROUPED_PROFILE_SCOPE')
    groups = {}
    for architecture in proof.original.ARCHITECTURES:
        selected = [x for x in observations if x['architecture'] == architecture]
        need([x['step'] for x in selected] == list(range(1, 6))
             and sum(x['targets'] for x in selected) == 66, 'GROUPED_PROFILE_GROUP')
        stats = pstats.Stats(*(str(a.output/x['profile']) for x in selected))
        groups[architecture] = dict(total_calls=stats.total_calls, primitive_calls=stats.prim_calls,
            profiler_seconds=stats.total_tt, functions=rows(stats))
    need(file_sha(__file__) == code and proof.efficient.update is original,
         'GROUPED_PROFILE_SOURCE_CHANGED')
    result = dict(complete=True, source=SOURCE, code=code,
        numerical_report=file_sha(a.output/'numerical/report.json'), observations=observations,
        groups=groups, seconds=time.perf_counter()-began, profiler_overhead_included=True,
        cumulative_times_overlap=True, gpu_used=False, speedup_claim=False,
        promoted=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(complete=True, report=pin, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists():
            write_committed(args.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise

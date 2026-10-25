"""Real matched existing-efficient/batched-validation updates, with cold arm reversal."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_placement_runtime_readback as proof
from audit.grouped_update_profile import SOURCE
from claims.process_conditions import write_committed
from model import batched_finite_scope as candidate
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need

NEW = ('model/batched_finite_scope.py', 'audit/grouped_finite_probe.py')


def controls(bundle):
    frame = next(i for i in bundle['items'] if i['side'] == 'train')['frames'][0]
    values = list(frame.values())
    source = values[0]
    before = fingerprint(values)
    result = {}

    def same(label, rows, expected, limit=candidate.JOIN_BYTES):
        checker = candidate.FiniteBatches(limit)
        need(candidate.ORIGINAL(rows) is expected and checker(iter(rows)) is expected,
             'FINITE_PROBE_PREDICATE:'+label)
        need(checker.counts['elements'] == sum(x.numel() for x in rows)
             and checker.counts['tensors'] == len(rows), 'FINITE_PROBE_COVERAGE:'+label)
        result[label] = checker.report()

    same('finite', values, True)
    same('empty', [], True)
    same('empty_tensor', [source[:0]], True)
    strided = source.reshape(-1).repeat(2).view(2, -1).T
    need(not strided.is_contiguous(), 'FINITE_PROBE_STRIDED_CONTROL')
    same('noncontiguous', [strided], True)
    same('mixed_dtype', [source, source.double(), source.to(torch.int64), source > 0], True)
    same('small_chunks', values, True, 64)
    for label, bad in (('nan', float('nan')), ('positive_inf', float('inf')), ('negative_inf', -float('inf'))):
        changed = source.clone()
        changed.view(-1)[-1] = bad
        same(label, [*values[1:], changed], False, 64)
        strided = changed.reshape(-1).repeat(2).view(2, -1).T
        need(not strided.is_contiguous(), 'FINITE_PROBE_STRIDED_CONTROL')
        same(label+'_strided', [strided], False)
    for label, limit in (('zero_allowance', 0), ('large_allowance', candidate.JOIN_BYTES+1), ('boolean_allowance', True)):
        result[label] = proof.reject(lambda n=limit: candidate.FiniteBatches(n))
        need('FINITE_BATCH_ALLOWANCE' in result[label], 'FINITE_PROBE_REFUSAL_CODE')
    def nested():
        with candidate.batched_finiteness():
            with candidate.batched_finiteness():
                pass
    def interrupted():
        with candidate.batched_finiteness():
            raise ValueError('FINITE_PROBE_INTERRUPTED')
    def replaced():
        with candidate.batched_finiteness():
            proof.efficient.finite_tensors = lambda _: True
    for label, fn, code in (('nested', nested, 'FINITE_BATCH_ALREADY_INSTALLED'),
                            ('exception', interrupted, 'FINITE_PROBE_INTERRUPTED'),
                            ('ownership', replaced, 'FINITE_BATCH_OWNERSHIP_CHANGED')):
        result[label] = proof.reject(fn)
        need(code in result[label], 'FINITE_PROBE_REFUSAL_CODE')
        need(not candidate.ACTIVE and proof.efficient.finite_tensors is candidate.ORIGINAL
             and candidate.transport.finite_tensors is candidate.ORIGINAL, 'FINITE_PROBE_RESTORE')
    with candidate.batched_finiteness() as checker:
        need(proof.efficient.finite_tensors(values), 'FINITE_PROBE_INSTALLED')
    result['closed'] = proof.reject(lambda: checker(values))
    need('FINITE_BATCH_CLOSED' in result['closed'] and fingerprint(values) == before, 'FINITE_PROBE_INPUT_CHANGED')
    return result


def run(a):
    a.output.mkdir(exist_ok=False)
    source = proof.pinned(a.root/SOURCE[0]/'report.json', SOURCE[1])
    need(file_sha(a.root/SOURCE[0]/'bundle.pt') == source['bundle_sha256'], 'FINITE_PROBE_SOURCE')
    bundle = torch.load(a.root/SOURCE[0]/'bundle.pt', map_location='cpu', weights_only=True)
    need(fingerprint(bundle) == source['bundle_state'], 'FINITE_PROBE_BUNDLE')
    code = {n: file_sha(Path(__file__).resolve().parents[1]/n) for n in NEW}
    for n, pin in code.items():
        path = a.output/'source'/n
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes())
        need(file_sha(path) == pin, 'FINITE_PROBE_SOURCE_COPY')
    checks = controls(bundle)
    write_committed(a.output/'controls.json', checks)
    previous = proof.pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['code'] == code and previous['controls'] == checks
             and not previous['reversed'], 'FINITE_PROBE_PRIOR')
    old_original, old_efficient = proof.original.update, proof.efficient.update
    observations = []

    def update(*args, **kwargs):
        with candidate.batched_finiteness() as checker:
            result = old_efficient(*args, **kwargs)
        model, _, item, step, _ = args
        observations.append(dict(architecture=model.architecture, document=item['document'], step=step+1,
                                 targets=result[0]['targets'], checks=checker.report()))
        return result

    roles = dict(original='batched' if previous else 'existing', efficient='existing' if previous else 'batched')
    proof.original.update, proof.efficient.update = (update, old_efficient) if previous else (old_efficient, update)
    try:
        proof.run(argparse.Namespace(source=a.root/SOURCE[0], source_pin=SOURCE[1],
            output=a.output/'numerical', cpu_proof=True,
            previous=a.previous/'numerical' if previous else None,
            previous_pin=previous['numerical_pin'] if previous else None))
    finally:
        intact = ((proof.original.update, proof.efficient.update) ==
                  ((update, old_efficient) if previous else (old_efficient, update)))
        proof.original.update, proof.efficient.update = old_original, old_efficient
        need(intact and not candidate.ACTIVE, 'FINITE_PROBE_UPDATE_RESTORE')
    numerical = json.loads((a.output/'numerical/report.json').read_bytes())
    need(numerical['complete'] and numerical['cpu_reference_exact'] and len(observations) == 15
         and sum(x['targets'] for x in observations) == 198, 'FINITE_PROBE_COMPLETE')
    need(code == {n: file_sha(Path(__file__).resolve().parents[1]/n) for n in NEW}, 'FINITE_PROBE_SOURCE_CHANGED')
    result = dict(complete=True, code=code, source=SOURCE, observations=observations, controls=checks,
        numerical_pin=file_sha(a.output/'numerical/report.json'), previous_pin=a.previous_pin,
        reversed=bool(previous), roles=roles, timings=numerical['timings'], paired_updates=30, paired_targets=396,
        gpu_used=False, promoted=False, capability_demonstrated=False, acceptance='author-checked')
    print(json.dumps(dict(complete=True, report=write_committed(a.output/'report.json', result))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--previous-pin')
    args = parser.parse_args()
    need(bool(args.previous) == bool(args.previous_pin), 'FINITE_PROBE_PRIOR_ARGUMENTS')
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists():
            write_committed(args.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise

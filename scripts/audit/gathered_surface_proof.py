"""Exact mixed-batch gather qualification, including actual AdamW and complete byte scores."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import mixed_surface_proof as previous
from audit import mixed_surface_readback as independent
from claims.process_conditions import write_committed
from model import gathered_surface_batches as gathered
from model import mixed_surface_batches as mixed
from model.native_population_training import cpu, fingerprint
from model.stream_surface import StreamSurfaceDecoder
from text.constellation_corpus import file_sha, need

SOURCE = ('legal-mixed-surface-cpu-v1-20261023', '792986624885936202b4a6f43e6dd307454dc8fa4e6f8bfb7b22a6a887f40754')
READBACK = ('legal-mixed-surface-cpu-readback-v2-20261023', '30d066f0cadc5f42ee111c3666d083ccac9ba0dcdb56faf805dbd9b958a5c138')


def code():
    root = Path(__file__).resolve().parents[1]
    names = set(previous.code()) | {'model/gathered_surface_batches.py',
             'audit/gathered_surface_proof.py', 'audit/gathered_surface_readback.py'}
    return {n: file_sha(root/n) for n in sorted(names)}


def load(root):
    p = root/SOURCE[0]; need(file_sha(p/'report.json') == SOURCE[1], 'GATHERED_PROOF_SOURCE_PIN')
    r = json.loads((p/'report.json').read_bytes())
    q = root/READBACK[0]/'report.json'; need(file_sha(q) == READBACK[1], 'GATHERED_PROOF_READBACK_PIN')
    v = json.loads(q.read_bytes())
    need(r['complete'] and v['complete'] and v['all_cold_arm_states_exact']
         and v['input_pin'] == SOURCE[1] and r['code'] == previous.code(), 'GATHERED_PROOF_AUTHORITY')
    for n, h in r['files'].items(): need(file_sha(p/n) == h, 'GATHERED_PROOF_PHYSICAL')
    return r, previous.load(root)


def batches(examples, config, specs, rows):
    for lo in range(0, len(examples), 128):
        for b in gathered.batches(examples[lo:lo+128], config, specs, rows=rows):
            yield replace(b, records=tuple(dict(r, atom=r['atom']+lo) for r in b.records), atoms=len(examples))


def evaluate(model, examples, rows, *, backward=False):
    if backward: need(all(e.side == 'train' for e in examples), 'GATHERED_PROOF_TRAIN_ONLY')
    result = [dict(losses=torch.full((len(e.raw)+1,), float('nan')),
                   argmax=torch.full((len(e.raw)+1,), -1, dtype=torch.int64)) for e in examples]
    calls = 0; objective = 0.; began = tick = time.perf_counter()
    phases = dict(packing=0., forward=0., scoring_backward=0., readout=0.)
    for b in batches(examples, model.config, model.specs, rows):
        phases['packing'] += time.perf_counter()-tick; tick = time.perf_counter()
        logits = model(b.prefix, b.native, offsets=b.offsets)['logits']
        phases['forward'] += time.perf_counter()-tick; tick = time.perf_counter()
        loss, token = mixed.loss(logits, b)
        if backward: loss.backward()
        phases['scoring_backward'] += time.perf_counter()-tick; tick = time.perf_counter()
        objective += float(loss.detach()); calls += 1
        for row, r in enumerate(b.records):
            i = r['atom']; a, end = r['scored']; left = a-r['offset']; right = end-r['offset']
            need(bool(torch.isnan(result[i]['losses'][a:end]).all()), 'GATHERED_PROOF_DUPLICATE_SCORE')
            result[i]['losses'][a:end] = token[row, left:right].detach()
            result[i]['argmax'][a:end] = logits[row, left:right].argmax(-1).detach()
        phases['readout'] += time.perf_counter()-tick; tick = time.perf_counter()
    phases['packing'] += time.perf_counter()-tick
    need(all(bool(v['losses'].isfinite().all()) and bool((v['argmax'] >= 0).all()) for v in result),
         'GATHERED_PROOF_MISSING_SCORE')
    seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.000001, 'GATHERED_PROOF_PHASE_COVERAGE')
    return result, dict(objective=objective, calls=calls, positions=sum(len(e.raw)+1 for e in examples),
                       seconds=seconds, phases=phases)


def execute(initial, config, specs, train, examples, shuffled, rows):
    model = StreamSurfaceDecoder(config, specs); model.load_state_dict(initial, strict=True); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False); model.zero_grad(set_to_none=True)
    fitting, fit = evaluate(model, train, rows, backward=True)
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    optimizer.step(); after = cpu(model.state_dict()); moments = cpu(optimizer.state_dict())
    with torch.no_grad():
        evaluation, score = evaluate(model, examples, rows); control, shuf = evaluate(model, shuffled, rows)
    artifact = dict(training=fitting, evaluation=evaluation, shuffled=control, gradients=gradients,
                    after=after, optimizer=moments)
    return artifact, dict(train=fit, evaluation=score, shuffled=shuf)


def packing_profile(examples, config, specs):
    results = []
    for rows in previous.COHORTS:
        for mode in ['reference', 'gathered', 'gathered', 'reference']:
            began = time.perf_counter(); count = positions = 0
            stream = previous.source_batches(examples, config, specs, rows) if mode == 'reference' else batches(examples, config, specs, rows)
            for b in stream:
                count += 1; positions += sum(y-x for r in b.records for x, y in [r['scored']])
            results.append(dict(mode=mode, rows=rows, calls=count, positions=positions, seconds=time.perf_counter()-began))
    return results


def run(a):
    began = tick = time.perf_counter(); process = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    pins = code(); prior, (_, config, specs, examples, metadata, initial) = load(a.root)
    saved = json.loads((a.root/SOURCE[0]/'results.json').read_bytes())
    train = [e for e in examples if e.side == 'train']
    selected = [next(e for e in train if e.occurrence == i) for i in saved['selected']]
    shuffled = [replace(e, native=examples[j].native) for e, j in zip(examples, saved['donors'])]
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    packing = {}
    for rows in previous.COHORTS:
        states = [previous.packed_state(b) for b in gathered.batches(selected, config, specs, rows=rows)]
        old = torch.load(a.root/SOURCE[0]/f'packed-{rows}.pt', map_location='cpu', weights_only=True)
        need(fingerprint(states) == fingerprint(old), 'GATHERED_PROOF_PACK_DIFFERENCE')
        torch.save(states, a.output/f'packed-{rows}.pt')
        manifest = json.loads((a.root/SOURCE[0]/f'schedule-{rows}.json').read_bytes())
        observed = []
        def full():
            for b in batches(examples, config, specs, rows):
                state = previous.packed_state(b); observed.append(fingerprint(state)); yield state
        counts = independent.check(full(), examples, config, specs)
        need(observed == [v['state'] for v in manifest], 'GATHERED_PROOF_FULL_PACK_DIFFERENCE')
        write_committed(a.output/f'schedule-{rows}.json', observed)
        packing[str(rows)] = dict(proof=independent.check(states, selected, config, specs), full=counts, all_exact=True)
    phases['packing_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    results = {}
    for rows in previous.COHORTS:
        started = time.perf_counter(); artifact, profile = execute(initial, config, specs, train, examples, shuffled, rows)
        baseline = torch.load(a.root/SOURCE[0]/f'rows-{rows}.pt', map_location='cpu', weights_only=True)
        need(fingerprint(artifact) == fingerprint(baseline) == saved['results'][f'rows-{rows}']['state'],
             'GATHERED_PROOF_MODEL_STATE_DIFFERENCE')
        torch.save(artifact, a.output/f'rows-{rows}.pt')
        results[str(rows)] = dict(state=fingerprint(artifact), profile=profile, seconds=time.perf_counter()-started,
                                 identical_to_phase80=True)
        print(json.dumps(dict(rows=rows, identical=True, seconds=results[str(rows)]['seconds'])), flush=True)
    phases['actual_model'] = time.perf_counter()-tick; tick = time.perf_counter()
    profiles = packing_profile(examples, config, specs)
    phases['packing_profile'] = time.perf_counter()-tick; tick = time.perf_counter()
    write_committed(a.output/'results.json', dict(packing=packing, results=results, profiles=profiles,
        donors=saved['donors'], metadata=metadata, selected=saved['selected']))
    need(code() == pins and not torch.cuda.is_initialized(), 'GATHERED_PROOF_ENVIRONMENT')
    for n, h in pins.items():
        p = a.output/'source'/n; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes()); need(file_sha(p) == h, 'GATHERED_PROOF_CODE_COPY')
    files = {str(p.relative_to(a.output)): file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.000001, 'GATHERED_PROOF_PHASE_COVERAGE')
    report = dict(complete=True, source=SOURCE, readback=READBACK, code=pins, files=files, results=results,
        profiles=profiles, phases=phases, seconds=seconds, cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        same_cohort_exact=True, original_per_atom_equivalence=False, promoted_to_training=False,
        scientific_admission=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', report); print(json.dumps(dict(complete=True, pin=pin)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(e)))
        raise

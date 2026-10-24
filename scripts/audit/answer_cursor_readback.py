"""Cold cursor restoration against independently generated reference events."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_cursor_source as source
from claims.process_conditions import write_committed
from model.native_answer_cursor import AnswerCursor
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.pin, 'CURSOR_COLD_REPORT')
    r = json.loads((a.input/'report.json').read_bytes()); need(r['complete'], 'CURSOR_COLD_COMPLETE')
    for n,h in r['files'].items(): need(file_sha(a.input/n) == h, 'CURSOR_COLD_BYTES')
    state = source.load(a.root); need(state['code'] == r['code'], 'CURSOR_COLD_SOURCE')
    need(state['cases'] == json.loads((a.input/'cases.json').read_bytes()), 'CURSOR_COLD_CASES')
    phases['authentication'] = time.perf_counter()-tick; results = []; checkpoints = events = 0
    for case in state['cases']:
        tick = time.perf_counter(); model = source.model(state, case); prefix = source.prefix(state, case)
        request = source.request(case); expected = state['traces'][case['id']]
        direct = model.rollout(prefix, request=request, target=state['target'], proposal=case['proposal'],
                               context=case['context'], seed=case['seed'])
        try: fresh = [cpu(next(direct)) for _ in expected]
        finally: direct.close()
        need(fingerprint(fresh) == fingerprint(expected), 'CURSOR_COLD_REFERENCE')
        observed = torch.load(a.input/case['id']/'events.pt', map_location='cpu', weights_only=True)
        need(fingerprint(observed) == fingerprint(fresh), 'CURSOR_COLD_RECORDED_EVENTS')
        index = json.loads((a.input/case['id']/'checkpoints.json').read_bytes())
        need([v['position'] for v in index] == list(range(len(fresh)+1)), 'CURSOR_COLD_COVERAGE')
        count = [0]
        def projected(module, args, output): count[0] += 1
        hooks = [m.register_forward_hook(projected) for m in model.forecaster.predictor.projectors.values()]
        for row in index:
            path = a.input/case['id']/row['file']
            need(file_sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'CURSOR_COLD_CHECKPOINT')
            saved = torch.load(path, map_location='cpu', weights_only=True); position = row['position']
            before = count[0]
            cursor = AnswerCursor.restore(saved, expected_pin=row['pin'], model=model, target=state['target'],
                request=request, proposal=case['proposal'], context=case['context'], prefix_pin=fingerprint(prefix))
            need(count[0] == before, 'CURSOR_COLD_REPROJECTED_PREFIX')
            need(cursor.offset == row['consumed_atoms'] == 2+min(position, sum(v['kind']=='constellation' for v in fresh))
                 and cursor.ended == row['ended'], 'CURSOR_COLD_POSITION')
            if position < len(fresh):
                event = cpu(next(iter(cursor))); need(fingerprint(event) == fingerprint(fresh[position]), 'CURSOR_COLD_NEXT')
                events += 1
            elif row['ended']:
                need(list(cursor) == [], 'CURSOR_COLD_TERMINAL')
            cursor.close(); need(cursor.state is None and cursor.hidden is None and cursor.stream is None, 'CURSOR_COLD_RELEASE')
            checkpoints += 1
        for hook in hooks: hook.remove()
        results.append(dict(case=case['id'], checkpoints=len(index), events=len(fresh),
                            prefix_projection_calls_on_restore=0, exact=True))
        phases[case['id']] = time.perf_counter()-tick
    tick = time.perf_counter(); state['fresh'](); need(not torch.cuda.is_initialized(), 'CURSOR_COLD_GPU')
    need(checkpoints == r['checkpoints'] and events == r['events'], 'CURSOR_COLD_TOTAL')
    phases['final_source_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'CURSOR_COLD_PHASES')
    value = dict(complete=True, source=dict(path=str(a.input), pin=a.pin), code=state['code'], results=results,
        checkpoints=checkpoints, events=events, all_events_exact=True, prefix_projection_calls_on_restore=0,
        phases=phases, seconds=seconds, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        acceptance='author-checked', capability_demonstrated=False, production_gpu_qualified=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', value), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

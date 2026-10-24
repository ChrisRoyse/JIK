"""Physical serialization and uninterrupted/resumed native answer comparison."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_cursor_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.native_answer_cursor import AnswerCursor
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need


def start(state, case, model):
    return AnswerCursor.start(model, source.prefix(state, case), target=state['target'], request=source.request(case),
        proposal=case['proposal'], context=case['context'], seed=case['seed'])


def restore(state, case, model, saved, **overrides):
    args = dict(expected_pin=saved['pin'], model=model, target=state['target'], request=source.request(case),
        proposal=case['proposal'], context=case['context'], prefix_pin=fingerprint(source.prefix(state, case)))
    args.update(overrides)
    return AnswerCursor.restore(saved, **args)


def controls(state):
    case = state['cases'][0]; model = source.model(state, case); cursor = start(state, case, model)
    first = cpu(next(cursor)); saved = cursor.snapshot(); controls = {}
    def bad(name, mutate, code, reseal=False):
        value = deepcopy(saved); mutate(value['payload'])
        if reseal: value['pin'] = fingerprint(value['payload'])
        controls[name] = rejected(lambda: restore(state, case, model, value), code)
    bad('changed_hidden', lambda v:v['hidden'].add_(1), 'ANSWER_CURSOR_CHECKPOINT_PIN')
    bad('changed_rng', lambda v:v['rng'].zero_(), 'ANSWER_CURSOR_CHECKPOINT_PIN')
    bad('changed_terminal', lambda v:v.update(ended=True), 'ANSWER_CURSOR_CHECKPOINT_PIN')
    controls['wrong_external_pin'] = rejected(lambda:restore(state, case, model, saved, expected_pin='0'*64),
                                              'ANSWER_CURSOR_CHECKPOINT_PIN')
    bad('position', lambda v:v.update(offset=v['offset']+1), 'ANSWER_CURSOR_RESTORE_POSITION', True)
    bad('cache_position', lambda v:next(iter(v['caches'].values())).update(next_atom=99),
        'ANSWER_CURSOR_RESTORE_CACHE_POSITION', True)
    bad('cache_bytes', lambda v:next(iter(v['caches'].values()))['layers'][0][0]['key'].add_(1),
        'PREFIX_STREAM_CHECKPOINT_PIN', True)
    bad('rng_shape', lambda v:v.update(rng=v['rng'][:2]), 'ANSWER_CURSOR_RESTORE_RNG', True)
    bad('hidden_shape', lambda v:v.update(hidden=v['hidden'][:,:,:1]), 'ANSWER_CURSOR_RESTORE_HIDDEN', True)
    controls['wrong_context'] = rejected(lambda:restore(state, case, model, saved, context='different'),
                                         'ANSWER_CURSOR_RESTORE_BINDING')
    controls['wrong_prefix'] = rejected(lambda:restore(state, case, model, saved, prefix_pin='0'*64),
                                        'ANSWER_CURSOR_RESTORE_BINDING')
    descriptor = legal(2); descriptor['operation'] = 'respond_to_source_question'
    controls['wrong_request'] = rejected(lambda:restore(state, case, model, saved, request=NativeRequest(descriptor)),
                                         'ANSWER_CURSOR_RESTORE_BINDING')
    wrong = source.model(state, case)
    with torch.no_grad(): next(wrong.parameters()).add_(1)
    controls['changed_weights'] = rejected(lambda:restore(state, case, wrong, saved), 'ANSWER_CURSOR_RESTORE_BINDING')
    altered = deepcopy(state['target'])
    with torch.no_grad(): next(altered.parameters()).add_(1)
    other_state = dict(state, target=altered); wrong = source.model(other_state, case)
    controls['changed_target'] = rejected(lambda:restore(other_state, case, wrong, saved), 'ANSWER_CURSOR_RESTORE_BINDING')
    whole = next(c for c in state['cases'] if c['architecture'] == 'whole_proposals')
    wm = source.model(state, whole); wc = start(state, whole, wm); ws = wc.snapshot()
    alternate = next(p for p in wm.proposals if p != whole['proposal'])
    controls['wrong_proposal'] = rejected(lambda:restore(state, whole, wm, ws, proposal=alternate),
                                         'ANSWER_CURSOR_RESTORE_BINDING'); wc.close()
    # Live mutation detection is separate from disk corruption.
    live = start(state, case, model)
    with torch.no_grad(): live.hidden.add_(1)
    controls['live_hidden'] = rejected(lambda:next(live), 'ANSWER_CURSOR_HIDDEN_CHANGED'); live.close()
    live = start(state, case, model)
    with torch.no_grad(): next(model.parameters()).add_(1)
    controls['live_weights'] = rejected(lambda:next(live), 'ANSWER_CURSOR_WEIGHTS_CHANGED'); live.close()
    model = source.model(state, case); live = start(state, case, model); first = cpu(next(live)); s = live.snapshot()
    s['payload']['hidden'].fill_(99)
    next(iter(s['payload']['caches'].values()))['layers'][0][0]['key'].fill_(99)
    first['slots'][model.roster[0]].fill_(99)
    need(fingerprint(cpu(next(live))) == fingerprint(state['traces'][case['id']][1]), 'CURSOR_CALLER_ALIAS')
    controls['caller_alias'] = 'snapshot_and_returned_event_changes_do_not_change_next_event'
    live.close(); controls['closed'] = rejected(lambda:next(live), 'ANSWER_CURSOR_CLOSED_OR_FAILED')
    cursor.close()
    controls['missing_request'] = rejected(lambda:AnswerCursor.start(model, source.prefix(state, case),
        target=state['target'], request=None, proposal=case['proposal'], context=case['context'], seed=case['seed']),
        'ANSWER_CURSOR_PREFIX')
    return controls


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    state = source.load(a.root); write_committed(a.output/'source.json', dict(code=state['code'], inputs=state['inputs']))
    write_committed(a.output/'cases.json', state['cases']); phases['source'] = time.perf_counter()-tick
    results = []; snapshot_count = total_events = total_bytes = 0
    for case in state['cases']:
        tick = time.perf_counter(); dest = a.output/case['id']; dest.mkdir(); model = source.model(state, case)
        original = model.rollout(source.prefix(state, case), target=state['target'], request=source.request(case),
            proposal=case['proposal'], context=case['context'], seed=case['seed'])
        expected = state['traces'][case['id']]; direct = []
        try:
            for _ in expected: direct.append(cpu(next(original)))
        finally: original.close()
        need(fingerprint(direct) == fingerprint(expected), 'CURSOR_UNCHANGED_GENERATOR')
        cursor = start(state, case, model); entries = []; events = []
        need(iter(cursor) is cursor, 'CURSOR_ITERATOR_PROTOCOL')
        for position in range(len(expected)+1):
            saved = cursor.snapshot(); path = dest/f'cursor-{position}.pt'; persist(path, saved)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            need(fingerprint(loaded) == fingerprint(saved), 'CURSOR_DISK_READBACK')
            entries.append(dict(position=position, pin=saved['pin'], file=path.name, sha256=file_sha(path),
                                bytes=path.stat().st_size, ended=cursor.ended, consumed_atoms=cursor.offset))
            snapshot_count += 1; total_bytes += path.stat().st_size
            if position == len(expected): break
            # Destroy the previous live state; the next event actually uses restored caches/RNG.
            cursor.close(); cursor = restore(state, case, model, loaded)
            event = cpu(next(iter(cursor))); events.append(event)
            need(fingerprint(event) == fingerprint(expected[position]), 'CURSOR_EVENT_DRIFT')
        if cursor.ended:
            try: next(cursor)
            except StopIteration: pass
            else: raise ValueError('CURSOR_END_NOT_TERMINAL')
        cursor.close(); total_events += len(events)
        persist(dest/'events.pt', events); write_committed(dest/'checkpoints.json', entries)
        results.append(dict(case=case['id'], events=len(events), checkpoints=len(entries),
            terminal=events[-1]['kind']=='end', censored=events[-1]['kind']!='end',
            seconds=time.perf_counter()-tick, all_events_exact=True))
        print(json.dumps(results[-1]), flush=True)
        phases[case['id']] = time.perf_counter()-tick
    tick = time.perf_counter(); negative = controls(state); state['fresh']()
    write_committed(a.output/'controls.json', negative)
    base = Path(__file__).resolve().parents[1]
    for n in source.NEW:
        dest = a.output/'source'/n; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((base/n).read_bytes())
    need(not torch.cuda.is_initialized(), 'CURSOR_CPU_PROOF_GPU')
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_and_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'CURSOR_PHASE_COVERAGE')
    report = dict(complete=True, acceptance='author-checked', code=state['code'], files=files, results=results,
        events=total_events, checkpoints=snapshot_count, checkpoint_bytes=total_bytes, controls=negative,
        seconds=seconds, phases=phases, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        capability_demonstrated=False, scientific_admission=False, production_gpu_qualified=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', report), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

"""Existing real, exposed answer traces; no new training or instrument loading."""
import json
from pathlib import Path

import torch

from audit import answer_history_source as history
from model.native_open_window_answer import answer_model
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need

ROOT = history.ROOT
BREADTH = ('answer-breadth-cpu-v4-20261024', '051f83dbd8cf3df36d8c4a76e3b45d7228d1904cdc86643c348d1c827b4e5f10')
COLD = ('answer-breadth-readback-v4-20261024', 'e5e97504b21b52a3a06b917c9be0444839fb9e6cd3ca08453101d3b6c568ed4b')
NEW = ('model/native_answer_cursor.py', 'audit/answer_cursor_source.py',
       'audit/answer_cursor.py', 'audit/answer_cursor_readback.py')


def load(root):
    state = history.load(root); inputs = {}; traces = {}; cases = []; checkpoints = {}
    def receipt(directory, pin):
        path = root/directory/'report.json'
        need(file_sha(path) == pin, 'CURSOR_SOURCE_RECEIPT')
        r = json.loads(path.read_bytes()); need(r['complete'], 'CURSOR_SOURCE_INCOMPLETE')
        inputs[str(path)] = pin; return r
    parent = receipt(*history.PARENT); prior = receipt(*BREADTH); cold = receipt(*COLD)
    need(cold['source']['pin'] == BREADTH[1], 'CURSOR_SOURCE_COLD')
    def payload(directory, relative, report):
        path = root/directory/relative; need(file_sha(path) == report['files'][relative], 'CURSOR_SOURCE_BYTES')
        inputs[str(path)] = report['files'][relative]
        return torch.load(path, map_location='cpu', weights_only=True)
    for architecture in history.ARCHITECTURES:
        key = 'parent/'+architecture
        checkpoints[key] = payload(history.PARENT[0], architecture+'/update-5.pt', parent)['model']
        records = payload(history.PARENT[0], architecture+'/evaluation.pt', parent)
        for row in records:
            if row['document'] == 22924:
                case = dict(checkpoint=key, architecture=architecture, document=22924,
                    mode=row['mode'], proposal=row['proposal'], context='count-answer-proof', seed=20261024+22924)
                case['id'] = str(len(cases)); cases.append(case); traces[case['id']] = row['events']
    key = 'breadth/whole_proposals'
    checkpoints[key] = payload(BREADTH[0], 'whole_proposals/learned_corpus/update-2.pt', prior)['model']
    records = payload(BREADTH[0], 'whole_proposals/learned_corpus/after.pt', prior)
    for row in records:
        if row['document'] == 58818 and row['mode'] == 'correct' and row['proposal'] in ('all-mpnet-base-v2','finbert-mean'):
            need(row['events'][-1]['kind'] == 'end', 'CURSOR_SOURCE_REAL_END')
            case = dict(checkpoint=key, architecture='whole_proposals', document=58818, mode='correct',
                proposal=row['proposal'], context='count-answer-proof', seed=20261024+58818, id=str(len(cases)))
            cases.append(case); traces[case['id']] = row['events']
    need(len(cases) == 18 and sum(v[-1]['kind'] == 'end' for v in traces.values()) == 2, 'CURSOR_SOURCE_CASES')
    base = Path(__file__).resolve().parents[1]
    code = dict(state['code'], **{n:file_sha(base/n) for n in NEW})
    old_fresh = state['fresh']; seal = fingerprint((checkpoints, traces, cases))
    def fresh():
        old_fresh(); need(fingerprint((checkpoints, traces, cases)) == seal, 'CURSOR_SOURCE_MUTATED')
        for path, h in inputs.items(): need(file_sha(path) == h, 'CURSOR_SOURCE_CHANGED')
        for path, h in code.items(): need(file_sha(base/path) == h, 'CURSOR_CODE_CHANGED')
    state.update(cases=cases, traces=traces, checkpoints=checkpoints, inputs=inputs, code=code, fresh=fresh)
    return state


def model(state, case):
    m = answer_model(state['target'], state['bundle'], case['architecture'], target_pin=state['pin'])
    m.load_state_dict(state['checkpoints'][case['checkpoint']], strict=True)
    return m.eval()


def request(case):
    desc = legal(2)
    if case['mode'] == 'wrong_operation': desc['operation'] = 'respond_to_source_question'
    return NativeRequest(desc)


def prefix(state, case): return tuple(state['items'][case['document']]['frames'][:2])

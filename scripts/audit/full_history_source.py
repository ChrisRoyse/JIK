"""Real, exposed legal sequences and controlled complete-answer comparisons."""
from pathlib import Path

from audit import answer_history_source as prior
from model.native_population_training import cpu
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need

ROOT = prior.ROOT
NEW = ('model/saved_gradient_storage.py', 'model/answer_full_history.py',
       'audit/full_history_source.py', 'audit/full_history_controls.py',
       'audit/full_history_training.py', 'audit/full_history_readback.py')


def load(root):
    s = prior.load(root); base = Path(__file__).resolve().parents[1]
    code = {n:file_sha(base/n) for n in NEW}; old_fresh = s['fresh']
    s['code'].update(code)
    def fresh():
        old_fresh()
        for n,h in code.items(): need(file_sha(base/n) == h, 'FULL_HISTORY_CODE')
    s['fresh'] = fresh
    s['train'] = sorted(k for k,d in s['items'].items() if d['side'] == 'train')
    need(len(s['train']) == 5, 'FULL_HISTORY_TRAIN_CENSUS')
    return s


def inputs(state, document, mode):
    d = state['items'][document]; prefix = tuple(d['frames'][:2]); desc = legal(2)
    need(mode in ('correct', 'reversed', 'train_donor', 'wrong_operation'), 'FULL_HISTORY_CONTROL_MODE')
    if mode == 'reversed': prefix = tuple(reversed(prefix))
    if mode == 'train_donor':
        donor = min(k for k in state['train'] if state['items'][k]['component'] != d['component'])
        prefix = tuple(state['items'][donor]['frames'][:2])
    if mode == 'wrong_operation': desc['operation'] = 'respond_to_source_question'
    return prefix, NativeRequest(desc)


def evaluate(state, model):
    rows = []; model.eval()
    for d in state['items'].values():
        if d['side'] == 'train': continue
        for proposal in model.proposals:
            for mode in ('correct', 'reversed', 'train_donor', 'wrong_operation'):
                prefix, request = inputs(state, d['document'], mode)
                result = prior.previous.observe(model, state['target'], prefix, proposal,
                    20261024+d['document'], len(d['frames'])-1, request)
                rows.append(dict(document=d['document'], side=d['side'], mode=mode, proposal=proposal,
                    **cpu(result), metrics=prior.previous.measured(result, d['frames'][2:], model.roster, d['frames'][1])))
    return rows

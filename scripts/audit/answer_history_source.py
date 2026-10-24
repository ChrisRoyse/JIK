"""Pinned, exposed legal source population for complete-answer history probes."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from audit.legal_grouped_placement_training import source_bundle
from audit import open_window_answer as previous
from model.native_conditioned_answer import document_item
from model.native_open_window_answer import answer_model
from model.native_placement_checked import optimizer_for, validate_optimizer
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
PARENT = ('open-window-answer-cpu-v2-20261024', 'a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339')
COLD = ('open-window-answer-readback-v2-20261024', 'abcd6121a90bc83118995db0877fc380f29d0581a18542d9d3b4f9a63b166242')
TRAIN = (27195, 106288)
ARCHITECTURES = ('shared', 'own_slot', 'whole_proposals')
NEW = ('model/answer_history_training.py', 'audit/answer_history_source.py',
       'audit/answer_history_training.py', 'audit/answer_history_readback.py')


def load(root):
    reports = []
    for directory, pin in (PARENT, COLD):
        path = root/directory/'report.json'
        need(file_sha(path) == pin, 'HISTORY_SOURCE_REPORT')
        record = json.loads(path.read_bytes()); need(record['complete'], 'HISTORY_SOURCE_INCOMPLETE')
        reports.append(record)
    parent, cold = reports
    need(cold['source']['pin'] == PARENT[1], 'HISTORY_SOURCE_COLD')
    for n, h in parent['files'].items(): need(file_sha(root/PARENT[0]/n) == h, 'HISTORY_SOURCE_BYTES')
    original, bundle = source_bundle(root); target, pin = extension(bundle)
    need(pin == parent['extension_pin'], 'HISTORY_SOURCE_REGION')
    base = Path(__file__).resolve().parents[1]; code = dict(original['code'], **parent['code'])
    for n, h in code.items(): need(file_sha(base/n) == h, 'HISTORY_SOURCE_PRIOR_CODE')
    for n in (*NEW, 'model/task_answer_training.py'): code[n] = file_sha(base/n)
    initial = {a: torch.load(root/PARENT[0]/a/'update-5.pt', map_location='cpu', weights_only=True)
               for a in ARCHITECTURES}
    items = {d['document']: d for d in bundle['items']}
    census = [dict(document=d['document'], side=d['side'], context_atoms=2,
                   answer_atoms=len(d['frames'])-2, native=d['native_pin'], labels=d['label_pin'])
              for d in items.values()]
    need(all(items[d]['side'] == 'train' for d in TRAIN), 'HISTORY_TRAIN_PARTITION')
    seal = fingerprint((initial, target.state_dict(), bundle['items']))
    def fresh():
        need(fingerprint((initial, target.state_dict(), bundle['items'])) == seal, 'HISTORY_SOURCE_MUTATION')
        for n, h in code.items(): need(file_sha(base/n) == h, 'HISTORY_SOURCE_CODE_CHANGED')
        for directory, h in (PARENT, COLD):
            need(file_sha(root/directory/'report.json') == h, 'HISTORY_SOURCE_CHANGED')
        for n, h in parent['files'].items(): need(file_sha(root/PARENT[0]/n) == h, 'HISTORY_SOURCE_CHANGED')
    return dict(root=root, bundle=bundle, target=target, pin=pin, code=code, initial=initial,
                items=items, census=census, fresh=fresh, policy=dict(bundle['policy'], device='cpu'))


def make(state, architecture, checkpoint=None, *, step=5):
    checkpoint = state['initial'][architecture] if checkpoint is None else checkpoint
    model = answer_model(state['target'], state['bundle'], architecture, target_pin=state['pin'])
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer = optimizer_for(model, state['policy'])
    optimizer.load_state_dict(deepcopy(checkpoint['optimizer']))
    validate_optimizer(model, optimizer, state['policy'], step)
    need(fingerprint((model.state_dict(), optimizer.state_dict())) ==
         fingerprint((checkpoint['model'], checkpoint['optimizer'])), 'HISTORY_CONTINUATION')
    return model, optimizer


def item(state, document):
    request = NativeRequest(legal(2)); value = document_item(state['items'][document], 2)
    value['provenance']['request'] = request.pin
    value['binding'] = fingerprint((value['document'], value['side'], value['provenance']))
    return value, request


def evaluate(state, model):
    rows = []; model.eval()
    for d in state['items'].values():
        if d['side'] == 'train': continue
        for proposal in model.proposals:
            for mode in ('correct', 'wrong_operation'):
                desc = legal(2)
                if mode == 'wrong_operation': desc['operation'] = 'respond_to_source_question'
                result = previous.observe(model, state['target'], tuple(d['frames'][:2]), proposal,
                    20261024+d['document'], len(d['frames'])-1, NativeRequest(desc))
                rows.append(dict(document=d['document'], side=d['side'], mode=mode, proposal=proposal,
                    **cpu(result), metrics=previous.measured(result, d['frames'][2:], model.roster, d['frames'][1])))
    return rows

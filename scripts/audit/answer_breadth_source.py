"""Completed full-TRAIN legal predictor states and real answer qualification data."""
import json
from pathlib import Path

import torch

from audit.legal_grouped_placement_training import source_bundle
from audit import answer_history_source as history
from model.native_open_window_answer import answer_model
from model.native_region_predictor import NativeRegionPredictor
from model.native_placement_checked import optimizer_for, validate_optimizer
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT = history.ROOT
SOURCE = ('legal-efficient-predictor-breadth-v1-20261023', '4921af55a18381dc978e25d593f946534a6e519d2d0b1a4623f640c0993c13f4')
COLD = ('legal-efficient-predictor-breadth-cold-v1-20261023', '9928a3993b4ea745ee50035a14210bb2d86fd34cdcdded10d947a43356b4450d')
NEW = ('model/answer_breadth_transfer.py', 'audit/answer_breadth_source.py',
       'audit/answer_breadth_training.py', 'audit/answer_breadth_readback.py')
ARMS = ('initial_corpus', 'learned_corpus')


def load(root):
    refs = {}; payloads = {}
    for name, pin in (SOURCE, COLD, history.PARENT):
        path = root/name/'report.json'; need(file_sha(path) == pin, 'BREADTH_SOURCE_REPORT')
        refs[name] = json.loads(path.read_bytes()); need(refs[name]['complete'], 'BREADTH_SOURCE_INCOMPLETE')
        payloads[path] = pin
    first, cold = refs[SOURCE[0]], refs[COLD[0]]
    need(first['scope'] == cold['scope'] == 'full' and first['arm'] == cold['arm'] == 'breadth'
         and first['updates'] == cold['updates'] == 91540 and first['target_presentations'] == cold['target_presentations'] == 6455636
         and cold['previous_pin'] == SOURCE[1] and first['binding'] == cold['binding'], 'BREADTH_SOURCE_COLD')
    need(first['ledger_head'] == cold['ledger_head'] and first['evaluations'] == cold['evaluations']
         and set(first['checkpoints']) == set(cold['checkpoints']) and
         all(v['state'] == cold['checkpoints'][k]['state'] for k, v in first['checkpoints'].items()), 'BREADTH_SOURCE_COLD_STATES')
    parent, bundle = source_bundle(root); target, pin = extension(bundle)
    need(pin == refs[history.PARENT[0]]['extension_pin'], 'BREADTH_SOURCE_EXTENSION')
    need(first['binding']['target'] == bundle['target_ref'], 'BREADTH_SOURCE_CODEBOOK')
    selection_path = root/SOURCE[0]/'selection.json'
    need(file_sha(selection_path) == first['binding']['selection'], 'BREADTH_SOURCE_SELECTION')
    selection = json.loads(selection_path.read_bytes()); payloads[selection_path] = file_sha(selection_path)
    train = set(selection['train_documents']); items = {v['document']: v for v in bundle['items']}
    need(len(train) == 91540 and all(v['document'] not in train for v in items.values() if v['side'] != 'train'),
         'BREADTH_SOURCE_HELDOUT_LEAK')
    need(all(d in train and items[d]['side'] == 'train' for d in history.TRAIN), 'BREADTH_SOURCE_TRAIN')
    checkpoints = {}
    for arm, step in zip(ARMS, (0, 91540)):
        path = root/SOURCE[0]/f'checkpoint-{step}.pt'; ref = first['checkpoints'][str(step)]
        need(file_sha(path) == ref['pin'], 'BREADTH_SOURCE_CHECKPOINT_BYTES'); payloads[path] = ref['pin']
        value = torch.load(path, map_location='cpu', weights_only=True)
        need(fingerprint(value) == ref['state'] and value['binding'] == first['binding']
             and value['step'] == step and value['rows'] == ref['rows'], 'BREADTH_SOURCE_CHECKPOINT')
        checkpoints[arm] = value
    cold_final = root/COLD[0]/'checkpoint-91540.pt'
    need(file_sha(cold_final) == first['checkpoints']['91540']['pin'], 'BREADTH_SOURCE_FINAL_COLD_BYTES')
    payloads[cold_final] = file_sha(cold_final)
    base = Path(__file__).resolve().parents[1]
    code = {}
    for manifest in (parent['code'], first['binding']['code'], refs[history.PARENT[0]]['code']):
        for n, h in manifest.items():
            need(n not in code or code[n] == h, 'BREADTH_SOURCE_CODE_CONFLICT')
            code[n] = h
    for n, h in code.items(): need(file_sha(base/n) == h, 'BREADTH_SOURCE_PRIOR_CODE')
    for n in (*NEW, 'audit/answer_history_source.py', 'audit/answer_history_readback.py',
              'model/task_answer_training.py'): code[n] = file_sha(base/n)
    seal = fingerprint((checkpoints, bundle, target.state_dict()))
    def fresh():
        need(fingerprint((checkpoints, bundle, target.state_dict())) == seal, 'BREADTH_SOURCE_MUTATED')
        for path, h in payloads.items(): need(file_sha(path) == h, 'BREADTH_SOURCE_CHANGED')
        for n, h in code.items(): need(file_sha(base/n) == h, 'BREADTH_SOURCE_CODE_CHANGED')
    fresh()
    return dict(root=root, bundle=bundle, target=target, pin=pin, code=code, items=items,
        checkpoints=checkpoints, fresh=fresh, policy=dict(bundle['policy'], device='cpu'),
        source_refs=dict(initial=first['checkpoints']['0'], learned=first['checkpoints']['91540']),
        census=[dict(document=v['document'], side=v['side'], context_atoms=2, answer_atoms=len(v['frames'])-2,
                     corpus_train=v['document'] in train) for v in items.values()])


def donor(state, arm):
    need(arm in ARMS, 'BREADTH_SOURCE_ARM')
    model = NativeRegionPredictor(state['target'], target_pin=state['bundle']['target_ref']['pin'],
                                  seed=state['bundle']['policy']['seed'])
    model.load_state_dict(state['checkpoints'][arm]['model'], strict=True)
    return model.eval().requires_grad_(False)


def make(state, architecture):
    model = answer_model(state['target'], state['bundle'], architecture, target_pin=state['pin'])
    optimizer = optimizer_for(model, state['policy']); validate_optimizer(model, optimizer, state['policy'], 0)
    return model, optimizer

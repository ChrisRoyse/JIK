"""Complete legal continuations with a source-bound, constant-size resume cursor.

The corpus and frozen target live across updates. A document is one optimizer
step; every answer atom and its true ending remain in that step's full graph.
"""
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import torch

from model import answer_full_history as learner
from model.guarded_transient_answer import guarded_transient_attention
from model.native_open_window_answer import answer_model
from model.native_placement_checked import optimizer_for, validate_optimizer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


def persist(path, value):
    path = Path(path)
    with path.open('xb') as f:
        torch.save(cpu(value), f); f.flush(); os.fsync(f.fileno())
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(saved) == fingerprint(value), 'ANSWER_EPOCH_PERSISTED_STATE')
    return dict(path=str(path), pin=file_sha(path), state=fingerprint(saved))


class AnswerEpoch:
    def __init__(self, corpus, bundle, target, target_pin, architecture, initial,
                 initial_step, policy, code, *, documents=None):
        need(architecture in ('shared', 'own_slot', 'whole_proposals'), 'ANSWER_EPOCH_ARCHITECTURE')
        need(type(initial_step) is int and initial_step >= 0, 'ANSWER_EPOCH_INITIAL_STEP')
        need(fingerprint(target.state_dict()) == bundle['target_fingerprint']
             and tuple(target.roster) == corpus.roster, 'ANSWER_EPOCH_FROZEN_SOURCE')
        self.corpus, self.bundle, self.target = corpus, bundle, target
        self.target_pin, self.architecture = target_pin, architecture
        self.policy = deepcopy(policy); self.device = torch.device(policy['device'])
        need(self.device.type in ('cpu', 'cuda') and next(target.parameters()).device == self.device,
             'ANSWER_EPOCH_TARGET_DEVICE')
        all_train = np.flatnonzero(corpus.rows['side'] == 0).astype(np.int64)
        selected = all_train if documents is None else np.asarray(documents, dtype=np.int64)
        need(selected.ndim == 1 and len(selected) > 0 and bool((selected >= 0).all())
             and bool((selected < len(corpus.rows)).all())
             and bool((np.diff(selected) > 0).all())
             and bool((corpus.rows['side'][selected] == 0).all()), 'ANSWER_EPOCH_TRAIN_ORDER')
        self.documents = tuple(map(int, selected)); rows = corpus.rows[selected]
        counts = dict(atoms=rows['stop']-rows['start'], answers=rows['stop']-rows['answer_start'],
                      decisions=rows['stop']-rows['answer_start']+1)
        self.prefix = {n:np.concatenate((np.zeros(1, dtype=np.int64), x.cumsum(dtype=np.int64)))
                       for n,x in counts.items()}
        self.window_prefix = np.concatenate((np.zeros((1,6), dtype=np.int64),
                                             rows['windows'].cumsum(0,dtype=np.int64)))
        self.initial_step = initial_step; self.cursor = 0
        self.binding = dict(schema='legal-complete-answer-epoch-v1', source=corpus.binding,
            target=target_pin, target_state=fingerprint(target.state_dict()), architecture=architecture,
            initial=fingerprint((initial['model'],initial['optimizer'])), initial_step=initial_step,
            policy=deepcopy(policy), code=deepcopy(code), torch=str(torch.__version__),
            device=str(self.device), cuda=torch.version.cuda if self.device.type == 'cuda' else None,
            schedule=fingerprint(self.documents), documents=len(selected), full_population=documents is None,
            history='all_prior_positions_with_gradients', endings='true_document_end_only')
        self.binding_pin = fingerprint(self.binding); self.chain = self.binding_pin
        self.target_versions = tuple(x._version for x in target.state_dict().values())
        self.model, self.optimizer = self.make(initial, initial_step)
        self.fresh()

    def fresh(self):
        self.corpus.fresh()
        need(fingerprint(self.binding) == self.binding_pin, 'ANSWER_EPOCH_BINDING_CHANGED')
        need(tuple(x._version for x in self.target.state_dict().values()) == self.target_versions,
             'ANSWER_EPOCH_TARGET_CHANGED')

    def make(self, value, step):
        model = answer_model(self.target,self.bundle,self.architecture,target_pin=self.target_pin).to(self.device)
        model.load_state_dict(value['model'],strict=True)
        optimizer = optimizer_for(model,self.policy); optimizer.load_state_dict(deepcopy(value['optimizer']))
        validate_optimizer(model,optimizer,self.policy,step)
        need(fingerprint((model.state_dict(),optimizer.state_dict())) ==
             fingerprint((value['model'],value['optimizer'])), 'ANSWER_EPOCH_RESTORED_STATE')
        model.train(); return model,optimizer

    def counts(self, cursor):
        need(type(cursor) is int and 0 <= cursor <= len(self.documents), 'ANSWER_EPOCH_CURSOR')
        return dict(documents=cursor, ends=cursor, **{n:int(v[cursor]) for n,v in self.prefix.items()},
                    windows={n:int(v) for n,v in zip(self.corpus.roster,self.window_prefix[cursor])})

    def snapshot(self):
        self.fresh(); step = self.initial_step+self.cursor
        validate_optimizer(self.model,self.optimizer,self.policy,step)
        return cpu(dict(schema=self.binding['schema'], binding=self.binding, cursor=self.cursor,
            step=step, noise_cursor=self.policy['noise_seed']+step, chain=self.chain,
            counts=self.counts(self.cursor), model=self.model.state_dict(), optimizer=self.optimizer.state_dict()))

    def restore(self, path, receipt, *, cursor, chain):
        self.fresh()
        need(file_sha(path) == receipt['pin'], 'ANSWER_EPOCH_CHECKPOINT_PIN')
        saved = torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(saved) == receipt['state'], 'ANSWER_EPOCH_CHECKPOINT_STATE')
        need(saved['schema'] == self.binding['schema'] and saved['binding'] == self.binding,
             'ANSWER_EPOCH_CHECKPOINT_BINDING')
        need(saved['cursor'] == cursor and saved['step'] == self.initial_step+cursor
             and saved['noise_cursor'] == self.policy['noise_seed']+self.initial_step+cursor
             and saved['chain'] == chain and saved['counts'] == self.counts(cursor), 'ANSWER_EPOCH_CHECKPOINT_CURSOR')
        model,optimizer = self.make(saved,saved['step'])
        self.model,self.optimizer,self.cursor,self.chain = model,optimizer,cursor,chain
        self.fresh()

    def update(self, document):
        self.fresh()
        need(type(document) is int and self.cursor < len(self.documents)
             and document == self.documents[self.cursor], 'ANSWER_EPOCH_NEXT_DOCUMENT')
        began = time.perf_counter()
        item,request,read_phases = self.corpus.example(document,fitting=True)
        read_seconds = time.perf_counter()-began
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        began = time.perf_counter()
        with guarded_transient_attention(self.model,heads=True):
            row,gradients,early = learner.update(self.model,self.optimizer,item,self.initial_step+self.cursor,
                self.policy,request=request,target=self.target,full=True)
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        update_seconds = time.perf_counter()-began
        before,after = self.counts(self.cursor),self.counts(self.cursor+1)
        need(row['decisions'] == after['decisions']-before['decisions']
             and len(item['answers']) == after['answers']-before['answers']
             and sum(r['end'] for r in row['records']) == 1, 'ANSWER_EPOCH_COVERAGE')
        self.corpus.validate(item,request,fitting=True)
        # Timing/storage observations stay out of the reproducible numerical chain.
        numerical = {n:v for n,v in row.items() if n not in ('phases','storage')}
        numerical['records'] = fingerprint(row['records'])
        entry = dict(parent=self.chain,cursor=self.cursor,document=document,update=numerical)
        self.chain = fingerprint(entry); self.cursor += 1; self.fresh()
        value = dict(record=row, gradients=gradients, early=early, model=cpu(self.model.state_dict()),
                     optimizer=cpu(self.optimizer.state_dict()))
        return value,dict(entry=entry,chain=self.chain,counts=after,read_phases=read_phases,
                          read_seconds=read_seconds,update_seconds=update_seconds)

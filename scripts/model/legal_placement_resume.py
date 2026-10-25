"""Resume the original placement objective at an authenticated document boundary.

The numerical checkpoint/ledger keep their original identities. New execution
code is recorded separately by the caller; this module never edits a live job.
"""
from copy import deepcopy
import hashlib
import json
import os

import numpy as np
import torch

from audit import legal_grouped_placement_training as original
from model.native_population_training import cpu, fingerprint
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need


class Census:
    def __init__(self, documents, targets, windows, roster):
        self.documents = tuple(documents); self.roster = tuple(roster)
        n = len(self.documents)
        need(n > 0 and all(type(d) is int and d >= 0 for d in self.documents)
             and len(set(self.documents)) == n, 'PLACEMENT_RESUME_DOCUMENTS')
        targets = np.asarray(targets); windows = np.asarray(windows)
        need(targets.dtype == windows.dtype == np.int64 and targets.shape == (n,)
             and windows.shape == (n,len(self.roster)) and len(set(self.roster)) == len(self.roster)
             and bool((targets > 0).all()) and bool((windows >= targets[:,None]+1).all()),
             'PLACEMENT_RESUME_CENSUS')
        self.targets = targets.copy(); self.windows = windows.copy()
        self.row_prefix = np.r_[np.int64(0),targets.cumsum(dtype=np.int64)]
        self.window_prefix = np.concatenate((np.zeros((1,len(self.roster)),dtype=np.int64),
                                             windows.cumsum(axis=0,dtype=np.int64)))
        need(bool((np.diff(self.row_prefix) > 0).all())
             and bool((np.diff(self.window_prefix,axis=0) > 0).all()), 'PLACEMENT_RESUME_OVERFLOW')
        for value in (self.targets,self.windows,self.row_prefix,self.window_prefix): value.flags.writeable = False
        self.pin = fingerprint(self.value())

    def value(self):
        return dict(documents=self.documents, roster=self.roster,
                    targets=torch.from_numpy(self.targets.copy()), windows=torch.from_numpy(self.windows.copy()))

    @classmethod
    def load(cls, value):
        return cls(value['documents'],value['targets'].numpy(),value['windows'].numpy(),value['roster'])

    def counts(self, step):
        need(type(step) is int and 0 <= step <= len(self.documents), 'PLACEMENT_RESUME_CURSOR')
        return dict(step=step,rows=int(self.row_prefix[step]),
                    windows=dict(zip(self.roster,map(int,self.window_prefix[step]))))


def recount(path, saved, initial, census, *, copy_to=None, allow_suffix=False):
    """Authenticate a fixed prefix with constant record memory, optionally sealing it."""
    step = saved['step']; counts = census.counts(step); binding = saved['binding']
    need(initial['binding'] == binding and initial['step'] == initial['rows'] == 0
         and initial['chain'] == digest(binding), 'PLACEMENT_RESUME_INITIAL')
    need(initial['noise_cursor'] == binding['policy']['noise_seed'], 'PLACEMENT_RESUME_INITIAL_NOISE')
    if binding['scope'] == 'full':
        need(tuple(binding['source']['plan']['train_documents']) == census.documents, 'PLACEMENT_RESUME_PLAN')
    need(saved['rows'] == counts['rows'] and saved['noise_cursor'] == binding['policy']['noise_seed']+step,
         'PLACEMENT_RESUME_CHECKPOINT_CURSOR')
    chain = digest(binding); model = fingerprint(initial['model']); optimizer = fingerprint(initial['optimizer'])
    sha = hashlib.sha256(); size = 0
    destination = copy_to.open('xb') if copy_to else None
    try:
        with path.open('rb') as source:
            for cursor in range(step):
                raw = source.readline(); need(raw.endswith(b'\n'), 'PLACEMENT_RESUME_TORN_PREFIX')
                row = json.loads(raw); body = {k:v for k,v in row.items() if k != 'hash'}
                need(set(row) == {'parent','cursor','update','hash'} and row['parent'] == chain
                     and row['cursor'] == cursor and row['hash'] == digest(body), 'PLACEMENT_RESUME_CHAIN')
                update = row['update']
                need(update['step'] == cursor+1 and update['document'] == census.documents[cursor]
                     and update['targets'] == int(census.targets[cursor]) and update['before'] == model
                     and update['noise_seed'] == binding['policy']['noise_seed']+cursor,
                     'PLACEMENT_RESUME_UPDATE')
                chain,model,optimizer = row['hash'],update['after'],update['optimizer']
                sha.update(raw); size += len(raw)
                if destination: destination.write(raw)
            need(allow_suffix or not source.read(1), 'PLACEMENT_RESUME_EXTRA_PREFIX')
        need(saved['chain'] == chain and fingerprint(saved['model']) == model
             and fingerprint(saved['optimizer']) == optimizer, 'PLACEMENT_RESUME_STATE')
        return dict(**counts,chain=chain,prefix_sha256=sha.hexdigest(),prefix_bytes=size,
                    model=model,optimizer=optimizer,census=census.pin,
                    next_document=census.documents[step] if step < len(census.documents) else None)
    finally:
        if destination: destination.flush(); os.fsync(destination.fileno()); destination.close()


class PlacementContinuation:
    def __init__(self, bundle, target, binding, census):
        self.binding = deepcopy(binding); self.binding_pin = digest(binding)
        self.census = census; self.bundle = bundle; self.target = target
        self.policy = deepcopy(binding['policy']); self.device = binding['device']
        need(self.device in ('cpu','cuda') and self.policy['device'] == self.device
             and str(next(target.parameters()).device).split(':')[0] == self.device,
             'PLACEMENT_RESUME_DEVICE')
        need(binding['torch_version'] == str(torch.__version__) and binding['target'] == bundle['target_ref']
             and fingerprint(target.state_dict()) == bundle['target_fingerprint']
             and tuple(target.roster) == census.roster, 'PLACEMENT_RESUME_TARGET')
        self.target_versions = tuple(t._version for t in target.state_dict().values())
        self.model,self.optimizer = original.runtime.make(bundle,target,binding['architecture'],self.device)
        self.step = 0; self.chain = digest(binding); self.poisoned = False

    def fresh(self, *, boundary=False):
        need(not self.poisoned, 'PLACEMENT_RESUME_POISONED')
        need(self.policy == self.binding['policy'], 'PLACEMENT_RESUME_BINDING_CHANGED')
        # The full binding contains the entire 91,540-document plan. Its bytes
        # are invariant during updates and authenticated at persistence/restore
        # boundaries, not reserialized once for every document (quadratic work).
        if boundary: need(digest(self.binding) == self.binding_pin, 'PLACEMENT_RESUME_BINDING_CHANGED')
        need(tuple(t._version for t in self.target.state_dict().values()) == self.target_versions,
             'PLACEMENT_RESUME_TARGET_CHANGED')

    def snapshot(self):
        self.fresh(boundary=True); original.efficient.validate_optimizer(self.model,self.optimizer,self.policy,self.step)
        return cpu(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),binding=self.binding,
                        step=self.step,rows=self.census.counts(self.step)['rows'],chain=self.chain,
                        noise_cursor=self.policy['noise_seed']+self.step))

    def restore(self, path, receipt, prefix, initial, *, cursor, chain):
        self.fresh(boundary=True); need(file_sha(path) == receipt['pin'], 'PLACEMENT_RESUME_CHECKPOINT_PIN')
        saved = torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(saved) == receipt['state'], 'PLACEMENT_RESUME_CHECKPOINT_STATE')
        need(saved['binding'] == self.binding and saved['step'] == cursor and saved['chain'] == chain,
             'PLACEMENT_RESUME_CHECKPOINT_BINDING')
        prefix_receipt = recount(prefix,saved,initial,self.census)
        # Restore into new objects: a refused checkpoint cannot partly mutate the current model.
        model,optimizer = original.runtime.make(self.bundle,self.target,self.binding['architecture'],self.device)
        original.restore(path,receipt,model,optimizer,self.binding,cursor,saved['rows'],chain,self.policy)
        self.model,self.optimizer,self.step,self.chain = model,optimizer,cursor,chain
        self.fresh(); return prefix_receipt

    def update(self, item):
        self.fresh(); need(self.step < len(self.census.documents), 'PLACEMENT_RESUME_FINISHED')
        need(type(item['document']) is int and item['document'] == self.census.documents[self.step],
             'PLACEMENT_RESUME_NEXT_DOCUMENT')
        actual = [sum(len(f[n]) for f in item['frames']) for n in self.census.roster]
        need(len(item['codes']) == int(self.census.targets[self.step])
             and actual == self.census.windows[self.step].tolist(), 'PLACEMENT_RESUME_ITEM_CENSUS')
        try:
            update,gradients,noise = original.efficient.update(self.model,self.optimizer,item,self.step,
                                                              self.policy,target=self.target)
            record = dict(parent=self.chain,cursor=self.step,update=update)
            self.chain = digest(record); record['hash'] = self.chain; self.step += 1
            self.fresh(); return record,gradients,noise
        except BaseException:
            # Once numerical execution began, recovery must use a persisted checkpoint.
            self.poisoned = True
            raise

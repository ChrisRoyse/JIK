"""Explicit energy-batched model construction at every continuation boundary.

The inherited update, commit and snapshot keep the qualified journal and tensor
formats. Restore installs fresh objects only after the original checks succeed.
"""
from copy import deepcopy

import torch

from audit import legal_grouped_placement_training as original
from model.batched_placement_continuation import BatchedPlacementContinuation
from model.legal_placement_resume import recount
from model.native_energy_forecaster import EnergyBatchedForecaster
from model.native_placement_training import optimizer_for
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha, need


def make(bundle, target, architecture, device):
    policy = bundle['policy']
    model = EnergyBatchedForecaster(target, target_pin=bundle['target_ref']['pin'],
        architecture=architecture, seed=policy['seed'], placement_seed=policy['placement_seed'],
        cache_bytes=policy['cache_bytes'], output_bytes=policy['output_bytes']).to(device)
    return model, optimizer_for(model, policy)


class EnergyPlacementContinuation(BatchedPlacementContinuation):
    def __init__(self, bundle, target, binding, census):
        self.binding = deepcopy(binding); self.binding_pin = digest(binding)
        self.census = census; self.bundle = bundle; self.target = target
        self.policy = deepcopy(binding['policy']); self.device = binding['device']
        need(self.device in ('cpu', 'cuda') and self.policy['device'] == self.device
             and str(next(target.parameters()).device).split(':')[0] == self.device,
             'PLACEMENT_RESUME_DEVICE')
        need(binding['torch_version'] == str(torch.__version__) and binding['target'] == bundle['target_ref']
             and fingerprint(target.state_dict()) == bundle['target_fingerprint']
             and tuple(target.roster) == census.roster, 'PLACEMENT_RESUME_TARGET')
        self.target_versions = tuple(t._version for t in target.state_dict().values())
        self.model, self.optimizer = make(bundle, target, binding['architecture'], self.device)
        self.step = 0; self.chain = digest(binding); self.poisoned = False
        self.fresh()

    def fresh(self, *, boundary=False):
        super().fresh(boundary=boundary)
        need(type(self.model) is EnergyBatchedForecaster, 'ENERGY_CONTINUATION_IMPLEMENTATION')

    def _load(self, path, receipt, cursor, rows, chain):
        model, optimizer = make(self.bundle, self.target, self.binding['architecture'], self.device)
        original.restore(path, receipt, model, optimizer, self.binding, cursor, rows, chain, self.policy)
        return model, optimizer

    def restore(self, path, receipt, prefix, initial, *, cursor, chain):
        self.fresh(boundary=True)
        need(file_sha(path) == receipt['pin'], 'PLACEMENT_RESUME_CHECKPOINT_PIN')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        need(fingerprint(saved) == receipt['state'], 'PLACEMENT_RESUME_CHECKPOINT_STATE')
        need(saved['binding'] == self.binding and saved['step'] == cursor and saved['chain'] == chain,
             'PLACEMENT_RESUME_CHECKPOINT_BINDING')
        prefix_receipt = recount(prefix, saved, initial, self.census)
        model, optimizer = self._load(path, receipt, cursor, saved['rows'], chain)
        self.model, self.optimizer, self.step, self.chain = model, optimizer, cursor, chain
        self.fresh(); return prefix_receipt

    def reopen(self, path, receipt):
        """Reopen a just-persisted state without rescanning an unchanged journal."""
        self.fresh(boundary=True)
        model, optimizer = self._load(path, receipt, self.step, self.census.counts(self.step)['rows'], self.chain)
        self.model, self.optimizer = model, optimizer
        self.fresh()

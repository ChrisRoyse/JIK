"""Versioned continuation using batched checks, without changing the old ledger.

The caller supplies a qualified runtime and authenticates its source pins. No
live module is replaced. Restore/snapshot retain the original checkpoint format.
"""
import os

from model.legal_placement_resume import PlacementContinuation
from model.native_placement_batched_checks import update
from senses.contract import canonical, digest
from text.constellation_corpus import need


class BatchedPlacementContinuation(PlacementContinuation):
    def update(self, item):
        self.fresh(); need(self.step < len(self.census.documents), 'PLACEMENT_RESUME_FINISHED')
        need(type(item['document']) is int and item['document'] == self.census.documents[self.step],
             'PLACEMENT_RESUME_NEXT_DOCUMENT')
        actual = [sum(len(f[n]) for f in item['frames']) for n in self.census.roster]
        need(len(item['codes']) == int(self.census.targets[self.step])
             and actual == self.census.windows[self.step].tolist(), 'PLACEMENT_RESUME_ITEM_CENSUS')
        try:
            record, gradients, noise = update(self.model, self.optimizer, item, self.step,
                                               self.policy, target=self.target)
            value = dict(parent=self.chain, cursor=self.step, update=record)
            self.chain = digest(value); value['hash'] = self.chain; self.step += 1
            self.fresh(); return value, gradients, noise
        except BaseException:
            self.poisoned = True
            raise

    def commit(self, item, ledger, replay):
        """Validate the retained witness before appending a newly executed update.

        The caller opens an authenticated prefix in append mode once per session.
        Any execution, witness or durable-write failure requires checkpoint restore;
        it must never leave a reusable in-memory state beyond the committed prefix.
        """
        try:
            cursor = self.step
            value, gradients, noise = self.update(item)
            raw = canonical(value) + b'\n'
            if not replay.complete: replay.check(raw, cursor)
            need(ledger.write(raw) == len(raw), 'PLACEMENT_RESUME_SHORT_WRITE')
            ledger.flush(); os.fsync(ledger.fileno())
            return value, gradients, noise
        except BaseException:
            self.poisoned = True
            raise

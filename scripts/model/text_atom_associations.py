"""Match atom-level relationship targets to atom-level observation representations."""
from pathlib import Path

import torch
from model.text_content_scale import ContentData
from senses.contract import digest
from text.constellation_corpus import file_sha, need


class AtomAssociationData(ContentData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config)
        need(all(len(self.native[n][i]) == 1 for n in self.native for i in self.nodes),
             "ATOM_ASSOCIATION_MEASURED_WORK")
        self.bind = {**self.bind, "association_representation": dict(
            unit="complete_current_constellation_before_context_identity_integration",
            native_windows_per_atom=6, typed_attention_tokens_per_atom=12,
            code=file_sha(Path(__file__)),
            driver=file_sha(Path(__file__).parents[1]/"audit/text_atom_learning.py"))}
        self.pin = digest(self.bind)

    def hidden(self, model, groups, examples=None):
        _, indices, succession = super().hidden(model, groups, examples)
        # The original context/identity path above supplies the succession loss.
        # Relation targets compare atoms, so their head reads the same atoms.
        atoms = []
        for index in indices:
            frame = self.cached[index]
            need(tuple((n, x._version) for n, x in frame.items()) == self.versions[id(frame)],
                 "ATOM_ASSOCIATION_NATIVE_CHANGED")
            atoms.append(model.core.sequence_transformer.encode((frame,), model.core.projectors)[0])
        return torch.stack(atoms), indices, succession

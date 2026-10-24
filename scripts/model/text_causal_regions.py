"""Source-causal region availability: prior documents and the observed local prefix."""
from pathlib import Path

from claims.codata_source import check
from model.text_dynamics import DynamicsData
from model.text_sequence_data import FLAGS
from senses.contract import digest
from text.constellation_corpus import file_sha


class CausalRegionData(DynamicsData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config)
        self.foreign = {}
        for group in self.groups:
            own = set(group["indices"])
            self.foreign[group["context"]] = tuple(i for i in self.refs if i not in own)
        self.bind = dict(schema="text-source-causal-regions-v1", base=self.bind,
            rule="other_document_shared_references_plus_observed_own_prefix",
            selection_inputs="current_source_lineage_and_cutoff_only; no_target_label",
            foreign={k: list(v) for k, v in self.foreign.items()}, code=file_sha(Path(__file__)), flags=FLAGS)
        self.pin = digest(self.bind)

    def region_indices(self, left, *, supplied=None):
        check(left in self.by_index, "TEXT_CONTEXT_SOURCE")
        group = self.by_index[left]
        prefix = group["indices"][:group["indices"].index(left)+1]
        check(supplied is None or supplied == prefix, "TEXT_CONTEXT_FUTURE_OR_OTHER")
        # The mask is defined before a target is consulted. Complete native
        # references from this document become available only when observed.
        return list(self.foreign[group["context"]])+prefix

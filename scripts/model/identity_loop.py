"""Intrinsic typed recurrent Identity Loop, from scratch (#10/#28).

Each context has its own recurrent state. The fixed reference bank participates
in every phase and every region/slot prediction. Heads predict; they never certify
conduct or truth. External checking and scientific instrument admission remain
separate. Native windows retain their instrument type throughout.
"""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Attention, Linear
from model.contracts import HEX, validate_bank
from model.native import NativeInput
from model.stream_attention import blocks, observe
from senses.contract import canonical, require

PHASES = ('standard', 'claim', 'act', 'echo', 'correction')
VERDICTS = ('supported', 'contradicted', 'unsupported_missing', 'evidence_mismatch')


@dataclass(frozen=True)
class ContextState:
    context: str
    standard_id: str
    sequence: int
    hidden: torch.Tensor  # K x H, separately typed instrument positions
    last_event: str | None = None


def reference_hash(specs, references, source_id):
    h = hashlib.sha256(canonical({'source_id': source_id, 'instruments': [s.identity for s in specs]}))
    for spec in specs:
        x = references[spec.name]
        h.update(canonical({'instrument': spec.name, 'rows': x.rows}))
        for t in (x.values, x.padding, x.absent, x.indices):
            if t is not None:
                a = t.detach().cpu().contiguous().numpy()
                h.update(canonical([list(a.shape), a.dtype.str]))
                h.update(a.tobytes())
    return h.hexdigest()


class IdentityLoopPredictor(nn.Module):
    """A bounded intrinsic predictor; not a trained/accepted identity controller.

    No token or external-prompt input. Frozen reference windows serve as the
    initial per-instrument identity-region codebooks. These are identity regions,
    not a claim that a universal world-meaning codebook has been learned.
    """
    def __init__(self, config, specs, references, *, source_id):
        super().__init__()
        config.validate()
        validate_bank(specs)
        require(HEX.fullmatch(source_id) is not None, 'MNLM_LOOP_SOURCE_ID', 'pin the source publication')
        require(set(references) == {s.name for s in specs}, 'MNLM_LOOP_REFERENCE_ROSTER', 'reference bank must match instruments')
        self.config, self.specs, self.source_id = config, tuple(specs), source_id
        self.reference_rows = {}
        sizes = []
        rows = None
        for spec in specs:
            require(spec.layout == 'multi', 'MNLM_LOOP_LAYOUT', 'v1 requires native full-window readings; do not pool')
            x = references[spec.name]
            x.check(spec, config)
            require(not bool(x.absent.any()) and (rows is None or x.rows == rows),
                    'MNLM_LOOP_REFERENCE_ALIGNMENT', 'every core component needs the same typed reference roster')
            rows = x.rows
            self.reference_rows[spec.name] = x.rows
            sizes.append(int((~x.padding).sum()))
        require(sum(x.values.numel() for x in references.values()) <= config.max_native_elements,
                'MNLM_LOOP_REFERENCE_BUDGET', 'reference bank exceeds fixed native budget')
        h, k = config.width, len(specs)
        # Exact count before allocation; checked against the actual module below.
        expected = 4*h*h + 11*h + k*h
        expected += sum(15*h*h + 20*h + 3*s.dims*h + 2*s.dims + h*r+r+4 for s,r in zip(specs,sizes))
        require(expected <= config.max_parameters, 'MNLM_LOOP_PARAMETER_BUDGET', 'reduce declared architecture before allocation')
        self.standard_id = reference_hash(specs, references, source_id)
        self.initial = nn.Parameter(torch.zeros(k, h))
        self.phase_embedding = nn.Embedding(len(PHASES), h)
        self.mix = Attention(config)
        self.norm = nn.LayerNorm(h)
        self.projectors = nn.ModuleDict()
        self.observe = nn.ModuleDict()
        self.anchor = nn.ModuleDict()
        self.compare = nn.ModuleDict()
        self.update = nn.ModuleDict()
        self.regions = nn.ModuleDict()
        self.place = nn.ModuleDict()
        self.expected_echo = nn.ModuleDict()
        self.verdict = nn.ModuleDict()
        for i, (spec, size) in enumerate(zip(specs, sizes)):
            name, x = spec.name, references[spec.name]
            self.register_buffer(f'ref_values_{i}', x.values.detach().clone())
            self.register_buffer(f'ref_padding_{i}', x.padding.detach().clone())
            self.register_buffer(f'ref_absent_{i}', x.absent.detach().clone())
            self.projectors[name] = Linear(spec.dims, h)
            self.observe[name], self.anchor[name] = Attention(config), Attention(config)
            self.compare[name] = Linear(h, h)
            self.update[name] = nn.GRUCell(h, h)
            self.regions[name] = Linear(h, size)
            self.place[name] = Linear(h, spec.dims)
            self.expected_echo[name] = Linear(h, spec.dims)
            self.verdict[name] = Linear(h, len(VERDICTS))
        actual = sum(p.numel() for p in self.parameters())
        require(actual == expected, 'MNLM_LOOP_COST_DRIFT', f'parameter accounting differs: {actual} != {expected}')

    def references(self):
        return {s.name: NativeInput(s.identity, self.reference_rows[s.name],
                    getattr(self, f'ref_values_{i}'), getattr(self, f'ref_padding_{i}'),
                    getattr(self, f'ref_absent_{i}')) for i,s in enumerate(self.specs)}

    def check_reference(self):
        # Publication/load boundary, not a host-copy/hash inside every train step.
        require(reference_hash(self.specs, self.references(), self.source_id) == self.standard_id,
                'MNLM_LOOP_STANDARD_DRIFT', 'restore the frozen reference; learning may not revise it')

    def start(self, context):
        require(isinstance(context, str) and bool(context), 'MNLM_LOOP_CONTEXT', 'explicit durable context identity required')
        return ContextState(context, self.standard_id, 0, self.initial.clone())

    def forward(self, native, states, *, phase, event_ids, evidence_origin,
                external_observation=None, advance_phase=True):
        """Consume one attested phase per context; return proposed next state.

        evidence_origin records source provenance, not a factual-support verdict.
        Echo phase accepts observations/receipts, never generated proposals. The
        caller must bind event_ids to its persisted source/evidence ledger before
        invocation. Publication must also check actual conduct and claim support.
        """
        b, h, k = len(states), self.config.width, len(self.specs)
        require(type(advance_phase) is bool and (advance_phase or
                all(x == 'source_observation' for x in evidence_origin)),
                'MNLM_LOOP_OBSERVATION_MODE', 'reading cannot fabricate an Act or Echo receipt')
        require(b > 0 and len({s.context for s in states}) == b, 'MNLM_LOOP_CONTEXT', 'contexts must be distinct')
        require(type(phase) is int and 0 <= phase < len(PHASES), 'MNLM_LOOP_PHASE', 'named phase required')
        require(all(s.standard_id == self.standard_id for s in states), 'MNLM_LOOP_STANDARD_ID', 'context belongs to another core standard')
        require(all(type(s.sequence) is int and s.sequence >= 0 and s.sequence % 5 == phase for s in states),
                'MNLM_LOOP_PHASE_ORDER', 'resume the pending phase; do not skip observed feedback')
        require(len(event_ids) == b and all(isinstance(e, str) and HEX.fullmatch(e) for e in event_ids)
                and all(e != s.last_event for e,s in zip(event_ids,states)),
                'MNLM_LOOP_EVENT', 'pin a new event for each context; replay is not new experience')
        require(len(evidence_origin) == b and all(x in ('source_observation', 'execution_receipt', 'generated_proposal') for x in evidence_origin),
                'MNLM_LOOP_PROVENANCE', 'distinguish source observations, executed receipts and generated proposals')
        require(phase != 3 or all(x != 'generated_proposal' for x in evidence_origin),
                'MNLM_LOOP_CAPTURED_ECHO', 'a predicted/generated echo cannot certify its own claim')
        device = self.initial.device
        if external_observation is not None:
            require(external_observation.shape == (b, k, h)
                    and external_observation.device == device
                    and external_observation.dtype == torch.float32
                    and bool(torch.isfinite(external_observation).all()),
                    'MNLM_LOOP_EXTERNAL_OBSERVATION', 'typed sensory coupling must match identity branches')
        require(all(s.hidden.shape == (k,h) and s.hidden.device == device and s.hidden.dtype == torch.float32
                    and bool(torch.isfinite(s.hidden).all()) for s in states),
                'MNLM_LOOP_STATE', 'aligned finite fp32 context state required')
        require(set(native) == {s.name for s in self.specs}, 'MNLM_LOOP_INPUT_ROSTER', 'missing typed observations must be explicit')
        rows = None
        present = torch.zeros(b, dtype=torch.int64, device=device)
        for spec in self.specs:
            x = native[spec.name]
            x.check(spec, self.config)
            require(len(x.rows) == b and (rows is None or x.rows == rows) and x.values.device == device,
                    'MNLM_LOOP_INPUT_ALIGNMENT', 'all instruments must measure the same event batch')
            present += (~x.absent).long()
            rows = x.rows
        require(sum(x.values.numel() for x in native.values()) <= self.config.max_native_elements,
                'MNLM_LOOP_INPUT_BUDGET', 'reduce active batch; never truncate native windows')
        require(external_observation is not None or bool((present > 0).all()),
                'MNLM_LOOP_NO_OBSERVATION', 'idle cycles are not experience')
        prior = torch.stack([s.hidden for s in states])
        phase_value = self.phase_embedding.weight[phase]
        updates = []
        for i, spec in enumerate(self.specs):
            name, x = spec.name, native[spec.name]
            q = (prior[:,i] + phase_value)[:,None,:]
            observed = torch.zeros((b,h), device=device)
            active = ~x.absent
            if bool(active.any()):
                values, padding = x.values[active], x.padding[active]
                counts = tuple((~padding).sum(1).tolist())
                observed[active] = observe(self.observe[name], self.projectors[name],
                    q[active], blocks(values, padding, self.config),
                    counts=counts, config=self.config)[:,0]
            if external_observation is not None:
                observed = observed + external_observation[:, i]
            values = getattr(self, f'ref_values_{i}')
            padding = getattr(self, f'ref_padding_{i}')
            reference = self.projectors[name](values[~padding])[None].expand(b,-1,-1)
            core = self.anchor[name](q, reference)[:,0]
            signal = observed + self.compare[name](core-observed) + phase_value
            updates.append(self.update[name](signal, prior[:,i]))
        typed = torch.stack(updates, 1)
        hidden = self.norm(typed + self.mix(typed, typed))
        regions, slots, echo, verdicts = {}, {}, {}, {}
        for i, spec in enumerate(self.specs):
            name, current = spec.name, hidden[:,i]
            book = getattr(self, f'ref_values_{i}')[~getattr(self, f'ref_padding_{i}')]
            regions[name] = self.regions[name](current)
            # Recognition of one reference region precedes within-region placement.
            mode = book[regions[name].argmax(-1)]
            slots[name] = F.normalize(mode + .1*self.place[name](current), dim=-1)
            echo[name] = F.normalize(self.expected_echo[name](current), dim=-1)
            verdicts[name] = self.verdict[name](current)
        require(bool(torch.isfinite(hidden).all()) and all(bool(torch.isfinite(x).all())
                for group in (regions,slots,echo,verdicts) for x in group.values()),
                'MNLM_LOOP_NONFINITE', 'reject state and restore the preceding committed version')
        return dict(states=tuple(ContextState(s.context,self.standard_id,s.sequence+int(advance_phase),hidden[j],event_ids[j])
                                 for j,s in enumerate(states)),
                    regions=regions, slots=slots, predicted_echo=echo, predicted_verdict=verdicts,
                    phase=PHASES[phase] if advance_phase else 'Observation', evidence_origin=tuple(evidence_origin),
                    meaning_asserted=False, behavioural_compliance_established=False,
                    truth_verified=False, scientific_admission=False)

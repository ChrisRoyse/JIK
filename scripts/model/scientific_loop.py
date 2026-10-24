"""Scientific evidence and intrinsic context transitions share one durable event chain."""
import base64
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.scientific_memory import ScientificMemory
from model.loop_session import output_digest
from model.world_checkpoint import load
from model.world_loop import WorldState


def code_pins():
    root = Path(__file__).parent
    return {n: sha((root/n).read_bytes()) for n in (
        "scientific_loop.py", "scientific_input.py", "world_loop.py", "world_checkpoint.py", "loop_session.py")}


class ScientificLoop(ScientificMemory):
    def __init__(self, path, source, inputs, checkpoint, *, model_pin, standard, world,
                 contexts, expected_head, expected_contract, device, attach=False,
                 expected_loop_contract=None, neural_replay=False):
        self.model, initial, _ = load(checkpoint, expected=model_pin, standard=standard, world=world, device=device)
        check(self.model.specs == inputs.specs and self.model.world_source_id == inputs.source_id, "MODEL_INPUT_JOIN")
        check(all(s.sequence == 0 for s in initial), "MODEL_NOT_INITIAL")
        check(tuple(sorted(contexts)) == tuple(sorted(s.context for s in initial)), "MODEL_CONTEXTS")
        self.model.eval()
        self.inputs, self.device = inputs, device
        self.states = {s.context: s for s in initial}
        self.support = {c: {} for c in contexts}
        self.context_edition = {c: None for c in contexts}
        self.dirty = set()
        self.initialized, self.neural_replay, self.replayed = False, neural_replay, 0
        self.loop_contract = dict(schema="scientific-intrinsic-loop-v1", checkpoint=model_pin,
                                  standard=standard, world=world, world_source=inputs.source_id,
                                  contexts=sorted(contexts), code=code_pins(), source=source.pin)
        self.loop_pin = digest(self.loop_contract)
        check(attach or self.loop_pin == expected_loop_contract, "LOOP_CONTRACT")
        self.allowed_key, self.allowed_cache = None, None
        self._replaying = True
        try:
            super().__init__(path, source, expected_head=expected_head, expected_contract=expected_contract)
        finally:
            self._replaying = False
        try:
            if attach:
                check(not self.initialized, "LOOP_ALREADY_ATTACHED")
                self._append(dict(kind="world_init", contract=self.loop_contract))
                self.initialized = True
            check(self.initialized, "LOOP_MISSING")
        except BaseException:
            self.close()
            raise

    def needs_review(self, context):
        check(context in self.states, "LOOP_CONTEXT")
        if context in self.dirty:
            return True
        if self.context_edition[context] not in (None, self.edition):
            return True
        for key, receipt in self.support[context].items():
            state = self.memory.state[key]
            if not state["active"] or state["grounded_scope"] is None or state["receipt"] != receipt:
                return True
        return False

    def _invalidate_consumers(self, record):
        affected = set(record["event"].get("affected", ()))
        self.dirty.update(c for c, refs in self.support.items() if affected.intersection(refs))

    def apply_memory(self, command):
        record = super().apply_memory(command)
        self._invalidate_consumers(record)
        return record

    def adopt(self, edition):
        body = super().adopt(edition)
        self.dirty.update(c for c, used in self.context_edition.items() if used not in (None, edition))
        return body

    def _request(self, context, quantity, receipt):
        self._live()
        check(self.initialized and context in self.states, "LOOP_CONTEXT")
        check(not self.needs_review(context), "LOOP_CONTEXT_NEEDS_REVIEW")
        phase = self.states[context].sequence % 5
        ep, checked = self._checked(self.edition, quantity)
        check(receipt == checked["id"] if phase == 3 else receipt is None, "LOOP_ECHO_RECEIPT")
        kind = "row" if phase in (3, 4) else "name"
        unit = self.inputs.unit(kind, self.edition if kind == "row" else "2022", quantity)
        return dict(context=context, quantity=quantity, edition=self.edition, unit=unit, kind=kind,
                    episode=ep["id"], checked_receipt=checked, phase=phase,
                    sequence=self.states[context].sequence, prior_head=self.head,
                    memory_head=self.memory.head, origin="execution_receipt" if phase == 3 else "source_observation")

    def _forward(self, request):
        if not self._replaying:
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            check(row is not None and row[0] == self.head, "STALE_LEDGER")
        key = (self.memory.head, self.edition)
        if key != self.allowed_key:
            self.allowed_cache = self.inputs.allowed(self, self.device, historical=self._replaying)
            self.allowed_key = key
        native = self.inputs.batch([request["unit"]], self.model.config, self.device)
        with torch.no_grad():
            return self.model(native, (self.states[request["context"]],), allowed=self.allowed_cache,
                              phase=request["phase"], event_ids=[digest(request)],
                              evidence_origin=[request["origin"]])

    def _support(self, request):
        self.support[request["context"]][request["episode"]] = request["checked_receipt"]
        self.context_edition[request["context"]] = request["edition"]

    def advance(self, context, quantity, *, receipt=None):
        request = self._request(context, quantity, receipt)
        output = self._forward(request)
        state = output["states"][0]
        raw = state.hidden.detach().cpu().contiguous().numpy().tobytes()
        body = dict(kind="world_advance", request=request, hidden=base64.b64encode(raw).decode(),
                    hidden_sha256=sha(raw), output_sha256=output_digest(output),
                    owner={n: v.detach().cpu().tolist() for n, v in output["world_owner"].items()})
        self._append(body)
        self.states[context] = state
        self._support(request)
        return output

    def reset(self, context, *, reason):
        self._live()
        check(self.needs_review(context) and isinstance(reason, str) and bool(reason.strip()), "LOOP_RESET_REVIEW")
        self._append(dict(kind="world_reset", context=context, reason=reason,
                          prior_sequence=self.states[context].sequence))
        self._reset(context)

    def _reset(self, context):
        self.states[context] = self.model.start(context)
        self.support[context] = {}
        self.context_edition[context] = None
        self.dirty.discard(context)

    def _replay(self, body):
        kind = body.get("kind")
        if kind == "world_init":
            check(not self.initialized and body == dict(kind="world_init", contract=self.loop_contract), "LOOP_INIT")
            self.initialized = True
        elif kind == "world_advance":
            check(set(body) == {"kind", "request", "hidden", "hidden_sha256", "output_sha256", "owner"}, "LOOP_EVENT")
            request = body["request"]
            receipt = request["checked_receipt"]["id"] if request["phase"] == 3 else None
            check(request == self._request(request["context"], request["quantity"], receipt), "LOOP_SOURCE_REPLAY")
            raw = base64.b64decode(body["hidden"], validate=True)
            check(sha(raw) == body["hidden_sha256"], "LOOP_HIDDEN_BYTES")
            array = np.frombuffer(raw, dtype=np.float32)
            shape = (len(self.model.specs), self.model.config.width)
            check(array.size == shape[0]*shape[1] and np.isfinite(array).all(), "LOOP_HIDDEN_SHAPE")
            if self.neural_replay:
                output = self._forward(request)
                check(output_digest(output) == body["output_sha256"]
                      and output["states"][0].hidden.detach().cpu().numpy().tobytes() == raw
                      and {n: v.detach().cpu().tolist() for n, v in output["world_owner"].items()} == body["owner"],
                      "LOOP_NEURAL_REPLAY")
                self.replayed += 1
            context = request["context"]
            self.states[context] = WorldState(context, self.model.standard_id, request["sequence"]+1,
                                               torch.from_numpy(array.copy().reshape(shape)).to(self.device),
                                               digest(request), self.model.world_id)
            self._support(request)
        elif kind == "world_reset":
            context = body["context"]
            check(set(body) == {"kind", "context", "reason", "prior_sequence"}
                  and self.needs_review(context) and body["prior_sequence"] == self.states[context].sequence
                  and isinstance(body["reason"], str) and bool(body["reason"].strip()), "LOOP_RESET_REPLAY")
            self._reset(context)
        else:
            super()._replay(body)
            if kind == "memory":
                self._invalidate_consumers(body["record"])
            elif kind == "adopt":
                self.dirty.update(c for c, used in self.context_edition.items() if used not in (None, self.edition))

    def snapshot(self):
        result = super().snapshot()
        return dict(**result, loop_contract=self.loop_pin, standard=self.model.standard_id, world=self.model.world_id,
                    contexts={c: dict(sequence=s.sequence, last_event=s.last_event, world=s.world_id,
                                      hidden_sha256=sha(s.hidden.detach().cpu().numpy().tobytes()),
                                      needs_review=self.needs_review(c), edition=self.context_edition[c],
                                      support=self.support[c]) for c, s in sorted(self.states.items())})

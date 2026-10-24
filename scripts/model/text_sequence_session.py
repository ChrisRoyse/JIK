"""Commit a typed next-passage forecast before scoring its actual source successor."""
from pathlib import Path

import torch

from claims.codata_source import check
from claims.event_journal import EventJournal
from model.constellation_learning import identity_pin
from model.native_world_checkpoint import load
from model.text_sequence_data import FLAGS, PURPOSE, state_pack
from senses.contract import digest


class TextSequenceSession:
    def __init__(self, path, model, data, *, checkpoint, checkpoint_pin, create, retained=None):
        check(type(create) is bool and (retained is None if create else isinstance(retained, dict)),
              "TEXT_SESSION_RETAINED")
        self.model, self.data, self.pending, self.broken = model, data, None, False
        states = load(checkpoint, model, expected=checkpoint_pin, binding=data.bind)
        check(all(s.sequence == 0 and not s.last_event for s in states), "TEXT_SESSION_INITIAL")
        self.states = {s.context: s for s in states}
        check(set(self.states) == {g["context"] for g in data.groups}, "TEXT_SESSION_CONTEXTS")
        self.cursors = {g["context"]: g["indices"][0] for g in data.groups}
        model.eval()
        self.stamps = self._stamps()
        contract = dict(schema="text-sequence-session-v1", checkpoint=checkpoint_pin,
                        data=data.bind, flags=FLAGS)
        self.journal = EventJournal(Path(path), contract, create=create,
                                    payload_bound=model.config.checkpoint_bytes)
        if not create:
            try:
                self.journal.restore(self._replay, **retained)
            except BaseException:
                self.broken = True
                raise

    def _stamps(self):
        return tuple((n, x._version) for n, x in list(self.model.named_parameters())+list(self.model.named_buffers()))

    def fresh(self):
        check(not self.broken, "TEXT_SESSION_RELOAD")
        self.journal.fresh()
        self.data.inputs.fresh()
        check(self.stamps == self._stamps(), "TEXT_SESSION_MODEL_CHANGED")

    def close(self):
        self.journal.close()
        self.broken = True

    def retained(self):
        self.fresh()
        return dict(head=self.journal.head, contract=self.journal.pin)

    def _predict(self, left):
        check(self.pending is None, "TEXT_TARGET_PENDING")
        check(left in self.data.successor, "TEXT_NO_SUCCESSOR")
        group = self.data.by_index[left]
        context, row = group["context"], self.data.sources[left][0]
        check(self.cursors[context] == left, "TEXT_PREFIX_ORDER")
        request = dict(context=context, left=left, source=row["atom_id"], through=row["char_end"],
                       prior_head=self.journal.head, phase="Observation", purpose=PURPOSE)
        event = digest(request)
        prior = self.states[context]
        with torch.no_grad():
            out = self.data.forward(self.model, prior, left, event)
        check(out["phase"] == "Observation" and out["states"][0].sequence == prior.sequence == 0,
              "TEXT_FABRICATED_ECHO")
        probabilities = torch.log_softmax(out["constellation_regions"][0], dim=-1).cpu().tolist()
        payload = dict(prior=state_pack(prior), after=state_pack(out["states"][0]),
            predictions=out["predicted_constellation"].pack(self.model.specs, self.model.config),
            region=out["predicted_constellation"].regions[0], probabilities=probabilities,
            identity=identity_pin(out), event=event)
        body = dict(kind="source_prediction", request=request, payload=digest(payload), flags=FLAGS)
        return body, payload, out["states"][0]

    def predict(self, left, *, expected_head, purpose):
        check(purpose == PURPOSE, "TEXT_SEQUENCE_ADMISSION_REQUIRED")
        self.fresh()
        check(expected_head == self.journal.head, "TEXT_WRITER_CONFLICT")
        body, payload, state = self._predict(left)
        try:
            self.journal.append(body, payload)
        except BaseException:
            self.broken = True
            raise
        self.pending = dict(left=left, prediction=self.journal.head, payload=body["payload"])
        self.states[state.context] = state
        self.journal.cache.clear()
        return self.pending.copy()

    def _reveal(self, right):
        check(self.pending is not None, "TEXT_PREDICTION_REQUIRED")
        left = self.pending["left"]
        self.data.edge(left, right)
        check(right == self.data.successor[left], "TEXT_NOT_IMMEDIATE_SUCCESSOR")
        payload = self.journal.payload(self.pending["payload"])
        outcome = self.data.reveal(left, right, payload["predictions"], payload["probabilities"])
        body = dict(kind="source_reveal", left=left, right=right,
            prediction=self.pending["prediction"], prior_head=self.journal.head,
            payload=digest(outcome), flags=FLAGS)
        return body, outcome

    def reveal(self, right, *, expected_head, purpose):
        check(purpose == PURPOSE, "TEXT_SEQUENCE_ADMISSION_REQUIRED")
        self.fresh()
        check(expected_head == self.journal.head, "TEXT_WRITER_CONFLICT")
        body, outcome = self._reveal(right)
        try:
            self.journal.append(body, outcome)
        except BaseException:
            self.broken = True
            raise
        context = self.data.by_index[body["left"]]["context"]
        self.cursors[context], self.pending = right, None
        self.journal.cache.clear()
        return outcome

    def _replay(self, body):
        check(body["flags"] == FLAGS and self.stamps == self._stamps(), "TEXT_REPLAY_CONTRACT")
        stored = self.journal.payload(body["payload"])
        if body["kind"] == "source_prediction":
            expected, payload, state = self._predict(body["request"]["left"])
            check(body == expected and stored == payload, "TEXT_PREDICTION_REPLAY")
            pin = digest(dict(seq=self.journal.count, previous=self.journal.head, body=body))
            self.pending = dict(left=body["request"]["left"], prediction=pin, payload=body["payload"])
            self.states[state.context] = state
        else:
            check(body["kind"] == "source_reveal", "TEXT_EVENT_KIND")
            expected, payload = self._reveal(body["right"])
            check(body == expected and stored == payload, "TEXT_TARGET_REPLAY")
            self.cursors[self.data.by_index[body["left"]]["context"]] = body["right"]
            self.pending = None
        self.journal.cache.clear()

    def records(self):
        """Independently read committed examples; no training from return values."""
        import json
        self.fresh()
        check(self.pending is None, "TEXT_TARGET_PENDING")
        pending, examples, outcomes = None, [], []
        for raw, pin in self.journal.db.execute("SELECT body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            payload = self.journal.payload(body["payload"])
            if body["kind"] == "source_prediction":
                check(pending is None, "TEXT_LEDGER_ORDER")
                pending = (body, payload, pin)
            else:
                check(pending is not None and body["prediction"] == pending[2], "TEXT_LEDGER_ORDER")
                prediction, prior, _ = pending
                left = body["left"]
                check(left == prediction["request"]["left"], "TEXT_LEDGER_ORDER")
                examples.append(dict(left=left, right=body["right"], prior=prior["prior"],
                    event=prior["event"], label=payload["label"], identity=prior["identity"],
                    evidence=pin, prediction=pending[2]))
                outcomes.append(dict(left=left, **payload))
                pending = None
            self.journal.cache.clear()
        check(pending is None, "TEXT_TARGET_PENDING")
        return examples, outcomes

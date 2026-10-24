"""Durable native episode processing under the unchanged admission boundary.

The checkpoint stays immutable. Each transaction publishes one context's two
recurrent states, and Act also retains every predicted native slot. Restart
replays the real model against the pinned source, not just the event hashes.
Recorded utterances are not consequences of model actions: Echo requires a
future, independently grounded feedback adapter and is deliberately unavailable.
"""
import base64
from pathlib import Path

import torch

from claims.codata_source import sha
from claims.event_journal import EventJournal
from model.loop_session import output_digest
from model.native_world_checkpoint import load
from senses.contract import canonical, digest, require


FLAGS = dict(scientific_admission=False, serving=False, learned_understanding=False,
             online_feedback_eligible=False, useful_capability_proven=False)


def check(ok, code):
    require(ok, "MNLM_"+code,
            "restore the retained source/checkpoint/session or supply the missing admitted evidence")


def tensor(value):
    a = value.detach().cpu().contiguous().numpy().astype("<f4")
    raw = a.tobytes()
    return dict(shape=list(a.shape), dtype="float32-le", sha256=sha(raw),
                base64=base64.b64encode(raw).decode())


class NativeEpisodeSession:
    def __init__(self, path, model, inputs, *, checkpoint, checkpoint_pin,
                 create, retained=None):
        check(type(create) is bool and (retained is None if create else
              isinstance(retained, dict) and set(retained) == {"head", "contract"}),
              "NATIVE_SESSION_RETAINED")
        inputs.fresh()
        self.inputs, self.model = inputs, model
        states = load(checkpoint, model, expected=checkpoint_pin, binding=inputs.binding)
        check(all(s.sequence == 0 and not s.last_event for s in states),
              "NATIVE_SESSION_INITIAL_CHECKPOINT")
        check(tuple(model.specs) == inputs.specs
              and model.core.temporal_id == digest(inputs.temporal_catalogue),
              "NATIVE_SESSION_INPUT_BINDING")
        self.model.eval()
        self.states = {s.context: s for s in states}
        self.initial = tuple(states)
        self.broken = False
        root = Path(__file__).resolve().parents[1]
        contract = dict(schema="native-episode-session-v1", checkpoint=checkpoint_pin,
            input_binding=inputs.binding, memory=inputs.memory.retained(),
            standard=model.standard_id, world=model.world_id,
            contexts=sorted(self.states), instruments=[s.identity for s in model.specs],
            scope="recorded_utterance_processing_not_action_feedback_or_forecast",
            code={n: sha((root/n).read_bytes()) for n in
                  ("model/native_episode_session.py", "model/loop_session.py",
                   "claims/event_journal.py")}, flags=FLAGS)
        self.journal = EventJournal(path, contract, create=create,
                                    payload_bound=model.config.checkpoint_bytes)
        if not create:
            try:
                self.journal.restore(self._replay, **retained)
                self.inputs.fresh()
            except BaseException:
                self.broken = True
                if not self.journal.broken:
                    self.journal.close()
                raise

    def close(self):
        # An aborted append poisons EventJournal without closing its connection.
        self.journal.close()
        self.broken = True

    def fresh(self):
        check(not self.broken, "NATIVE_SESSION_RELOAD")
        self.journal.fresh()
        self.inputs.fresh()

    def retained(self):
        self.fresh()
        return dict(head=self.journal.head, contract=self.journal.pin)

    def _request(self, context, index, expected_head, purpose):
        check(purpose == "engineering_boundary", "NATIVE_SESSION_ADMISSION_REQUIRED")
        # restore() already checks journal ordering under its read transaction.
        check(not self.broken, "NATIVE_SESSION_RELOAD")
        self.inputs.fresh()
        check(expected_head == self.journal.head, "NATIVE_SESSION_WRITER_CONFLICT")
        check(context in self.states, "NATIVE_SESSION_CONTEXT")
        check(type(index) is int and 0 <= index < len(self.inputs.moments),
              "NATIVE_SESSION_EPISODE")
        state, moment = self.states[context], self.inputs.moments[index]
        phase = state.sequence % 5
        check(phase < 3, "NATIVE_SESSION_ECHO_UNGROUNDED")
        return dict(context=context, index=index, episode=moment["id"],
            phase=phase, sequence=state.sequence, previous_event=state.last_event,
            prior_head=expected_head, input_binding=self.inputs.binding,
            source=moment["source"], event_time=moment["event_time"],
            purpose=purpose, origin="source_observation")

    def _execute(self, request):
        native = self.inputs.batch(request["index"], purpose=request["purpose"])
        rows = next(iter(self.model.reference_rows.values()))
        allowed = torch.ones(len(rows), dtype=torch.bool,
                             device=self.model.core.initial.device)
        with torch.no_grad():
            out = self.model(native, (self.states[request["context"]],),
                allowed=allowed, phase=request["phase"], event_ids=[digest(request)],
                evidence_origin=[request["origin"]])
        self.inputs.fresh()
        state = out["states"][0]
        payload = dict(world=tensor(state.hidden), identity=tensor(state.identity_hidden),
            sequence=state.sequence, last_event=state.last_event,
            prediction=(out["predicted_constellation"].pack(self.model.specs, self.model.config)
                        if request["phase"] == 2 else None),
            regions=list(out["predicted_constellation"].regions))
        body = dict(kind="native_episode_advance", request=request,
            payload=digest(payload), output=output_digest(out), flags=FLAGS)
        check(len(canonical(payload)) <= self.model.config.checkpoint_bytes,
              "NATIVE_SESSION_PAYLOAD_BUDGET")
        return body, payload, state, out

    def advance(self, context, index, *, expected_head, purpose="engineering_boundary"):
        self.fresh()
        request = self._request(context, index, expected_head, purpose)
        body, payload, state, output = self._execute(request)
        try:
            self.journal.append(body, payload)
        except BaseException:
            # Never expose uncommitted state after an uncertain append outcome.
            self.broken = True
            raise
        self.journal.cache.clear()  # PC-28: retained outputs live on disk, not in an ever-growing cache.
        self.states[context] = state
        return output

    def _replay(self, body):
        check(isinstance(body, dict) and set(body) ==
              {"kind", "request", "payload", "output", "flags"}
              and body["kind"] == "native_episode_advance", "NATIVE_SESSION_EVENT")
        r = body["request"]
        request = self._request(r["context"], r["index"], r["prior_head"], r["purpose"])
        check(r == request, "NATIVE_SESSION_REQUEST_REPLAY")
        stored = self.journal.payload(body["payload"])
        expected, payload, state, _ = self._execute(request)
        check(body == expected and stored == payload, "NATIVE_SESSION_NEURAL_REPLAY")
        self.journal.cache.clear()
        self.states[r["context"]] = state

    def snapshot(self):
        return dict(retained=self.retained(), flags=dict(FLAGS),
            contexts={key: dict(sequence=s.sequence, last_event=s.last_event,
                world=tensor(s.hidden)["sha256"], identity=tensor(s.identity_hidden)["sha256"])
                for key, s in sorted(self.states.items())})

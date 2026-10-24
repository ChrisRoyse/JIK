"""Durable, correctable native temporal memory using the existing evidence ledger."""
import json
from pathlib import Path

from claims.episodic_memory import Memory, canonical
from claims.event_journal import EventJournal
from claims.native_evidence import SCOPE, need, rational
from senses.perception_corpus import sha


class NativeMemory:
    def __init__(self, path, checker, *, create, retained=None):
        if not create:
            need(type(retained) == dict and set(retained) ==
                 {"head", "contract", "memory_head"}, "RETAINED_HEAD_REQUIRED")
        self.checker, self.memory = checker, Memory(checker)
        self.contract = dict(schema="native-temporal-memory-v1", evidence=checker.pin,
            code={n: sha(Path(__file__).with_name(n)) for n in
                  ("native_memory.py", "native_evidence.py", "episodic_memory.py", "event_journal.py")},
            scope=SCOPE, scientific_admission=False, training_ready=False)
        self.journal = EventJournal(path, self.contract, create=create, payload_bound=0)
        self.broken = False
        if not create:
            try:
                self.journal.restore(self._replay, head=retained["head"],
                                     contract=retained["contract"])
                need(self.memory.head == retained["memory_head"], "MEMORY_HEAD")
            except BaseException:
                self.broken = True
                if not self.journal.broken:
                    self.journal.close()
                raise

    def close(self):
        self.broken = True
        self.journal.close()

    def _fresh(self):
        need(not self.broken, "RELOAD_REQUIRED")
        self.journal.fresh()

    def retained(self):
        self._fresh()
        return dict(head=self.journal.head, contract=self.journal.pin, memory_head=self.memory.head)

    def _support(self, key):
        need(key in self.memory.state, "UNKNOWN_EPISODE")
        state = self.memory.state[key]
        need(state["active"] and state["grounded_scope"] == SCOPE
             and state["review"] == "source_checked" and state["receipt"] is not None,
             "SUPPORT_WITHDRAWN_OR_UNCHECKED")

    def _dependencies(self, command):
        if command.get("op") in ("check_source", "promote"):
            key = command.get("episode")
            need(key in self.memory.episodes, "UNKNOWN_EPISODE")
            for parent in self.memory.episodes[key]["evidence"]:
                self._support(parent)

    def _replay(self, body):
        need(type(body) == dict and set(body) == {"command", "record"}, "JOURNAL_EVENT")
        self._dependencies(body["command"])
        record = self.memory.apply(body["command"])
        need(record == body["record"], "MEMORY_REPLAY")

    def apply(self, command, *, expected_head):
        self._fresh()
        need(expected_head == self.journal.head, "WRITER_CONFLICT")
        self._dependencies(command)
        record = self.memory.apply(command)
        try:
            self.journal.append(dict(command=json.loads(canonical(command)), record=record))
        except BaseException:
            # RAM was advanced, but SQLite is the authority. This handle cannot publish again.
            self.broken = True
            raise
        return record

    def consume(self, key, *, clock_id, split_group, source_pin, through,
                required_media=("image", "audio"), purpose="engineering_boundary"):
        self._fresh()
        need(purpose == "engineering_boundary", "NATIVE_ADMISSION_REQUIRED")
        self._support(key)
        ep = self.memory.episodes[key]
        need(ep["claim_kind"] == "derived_relation", "MOMENT_REQUIRED")
        for parent in ep["evidence"]:
            self._support(parent)
        need(clock_id == ep["source"]["clock_id"], "CLOCK")
        need(split_group == ep["source"]["split_group"], "SPLIT")
        need(source_pin == ep["source"]["source"]["media"], "SOURCE")
        need(tuple(required_media) == ("image", "audio"), "MISSING_TIMED_TEXT_OR_MEDIUM")
        need(rational(through) >= rational(ep["event_time"]["available_after"]),
             "FUTURE_CONTEXT")
        # Return a snapshot labelled with its journal head. Every array has immutable bytes
        # backing it. A later request must re-check withdrawal and writer freshness.
        return dict(head=self.journal.head, moment=json.loads(canonical(ep)),
                    readings=self.checker.readings(ep), scientific_admission=False,
                    training_ready=False, physical_sync_calibrated=False)

    def snapshot(self):
        self._fresh()
        return json.loads(canonical(dict(retained=self.retained(), states=self.memory.state,
            grounded_recording_measurements=sorted(k for k, s in self.memory.state.items()
                if s["active"] and s["grounded_scope"] == SCOPE),
            grounded_world_claims=[], epistemic_distinction_learned=False)))

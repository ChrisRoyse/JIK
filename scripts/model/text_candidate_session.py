"""Durable selection and physical rollback of diagnostic recurrent learners."""
from pathlib import Path

from claims.codata_source import check
from claims.event_journal import EventJournal
from model.text_candidate_policy import POLICY, assess
from model.text_context_regions import PrefixSession
from model.text_sequence_data import FLAGS, PURPOSE
from senses.contract import digest
from text.constellation_corpus import file_sha


def code():
    root = Path(__file__).parent
    return {n: file_sha(root/n) for n in ("text_candidate_session.py", "text_candidate_policy.py", "text_candidate_evidence.py")}


class TextCandidateSession:
    def __init__(self, path, evidence, *, create, retained=None):
        check(type(create) is bool and (retained is None if create else isinstance(retained, dict)),
              "TEXT_SELECTION_RETAINED")
        self.evidence, self.broken = evidence, False
        self.selected, self.established, self.reviewed = "initial-training", [], []
        self.records = evidence.records()
        contract = dict(schema="text-candidate-session-v1", evidence=evidence.pin, policy=POLICY,
                        data=evidence.data.bind, code=code(), flags=FLAGS)
        self.journal = EventJournal(Path(path), contract, create=create,
                                    payload_bound=evidence.model.config.checkpoint_bytes)
        try:
            if create:
                payload = dict(selected=self.selected, record=self.records[self.selected], established=[])
                self.journal.append(dict(kind="initialize_diagnostic_learner", payload=digest(payload)), payload)
            else:
                self.journal.restore(self._replay, head=retained["head"], contract=retained["contract"])
                check(self.selected == retained["selected"] and self.established == retained["established"],
                      "TEXT_SELECTION_RETAINED_POINTER")
            evidence.activate(self.selected)
        except BaseException:
            self.broken = True
            if not self.journal.broken:
                self.journal.close()
            raise

    def fresh(self):
        check(not self.broken, "TEXT_SELECTION_RELOAD")
        self.journal.fresh()
        self.evidence.fresh()
        check(digest(self.journal.contract) == self.journal.pin and code() == self.journal.contract["code"]
              and self.records == self.evidence.records(), "TEXT_SELECTION_CONTRACT_CHANGED")
        check(self.evidence.fingerprint() == self.records[self.selected]["fingerprint"],
              "TEXT_SELECTION_LIVE_MODEL_DIVERGED")

    def retained(self):
        self.fresh()
        return dict(head=self.journal.head, contract=self.journal.pin,
                    selected=self.selected, established=list(self.established))

    def close(self):
        self.journal.close()
        self.broken = True

    def proposal(self, name):
        check(name in POLICY["stages"] and name not in self.reviewed, "TEXT_SELECTION_REVIEW_ORDER")
        stage = POLICY["stages"][name]
        parent, candidate = self.records[stage["parent"]], self.records[name]
        check(candidate["parent"] == parent["checkpoint"], "TEXT_SELECTION_PARENT")
        result = assess(name, parent["development"], candidate["development"], self.evidence.data.groups,
            [s.name for s in self.evidence.model.specs], retained=self.selected, established=self.established)
        selected = name if result["eligible_diagnostic_lineage"] else self.selected
        established = sorted(set(self.established+stage["acquire"])) if selected == name else list(self.established)
        return dict(candidate=name, before=self.selected, selected=selected, established=established,
            result=result, candidate_record=candidate, parent_record=parent,
            selected_fingerprint=self.records[selected]["fingerprint"])

    def review(self, name, *, expected_head, purpose):
        check(purpose == PURPOSE, "TEXT_SELECTION_ADMISSION_REQUIRED")
        self.fresh()
        check(expected_head == self.journal.head, "TEXT_SELECTION_WRITER_CONFLICT")
        payload = self.proposal(name)
        try:
            # Import the actual proposal, then enforce the selected version in
            # both weights and AdamW state before persisting the decision.
            self.evidence.activate(name)
            candidate_fingerprint = self.evidence.fingerprint()
            restored = self.evidence.activate(payload["selected"])
            check(candidate_fingerprint == payload["candidate_record"]["fingerprint"]
                  and restored == payload["selected_fingerprint"], "TEXT_SELECTION_PHYSICAL_RESTORE")
            self.journal.append(dict(kind="review_candidate", payload=digest(payload)), payload)
        except BaseException:
            self.broken = True
            raise
        self.selected, self.established = payload["selected"], payload["established"]
        self.reviewed.append(name)
        self.journal.cache.clear()
        return payload

    def recover(self, *, expected_head, purpose):
        check(purpose == PURPOSE, "TEXT_SELECTION_ADMISSION_REQUIRED")
        check(not self.broken, "TEXT_SELECTION_RELOAD")
        self.journal.fresh()
        check(expected_head == self.journal.head, "TEXT_SELECTION_WRITER_CONFLICT")
        self.evidence.activate(self.selected)
        self.fresh()

    def predictions(self, path, *, purpose):
        check(purpose == PURPOSE, "TEXT_SELECTION_ADMISSION_REQUIRED")
        self.fresh()
        return PrefixSession(path, self.evidence.model, self.evidence.data,
            checkpoint=self.evidence.root/self.selected/"model",
            checkpoint_pin=self.records[self.selected]["model"], create=True)

    def _replay(self, body):
        payload = self.journal.payload(body["payload"])
        if self.journal.count == 0:
            check(body["kind"] == "initialize_diagnostic_learner" and payload == dict(
                selected="initial-training", record=self.records["initial-training"], established=[]),
                "TEXT_SELECTION_INITIAL")
        else:
            check(body["kind"] == "review_candidate" and payload == self.proposal(payload["candidate"]),
                  "TEXT_SELECTION_DECISION_REPLAY")
            self.selected, self.established = payload["selected"], payload["established"]
            self.reviewed.append(payload["candidate"])
        self.journal.cache.clear()

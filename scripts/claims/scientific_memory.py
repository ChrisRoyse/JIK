"""Durable, evidence-bound adoption of published scientific reference editions.

Publication receipts feed the existing episodic Memory. Edition adoption is an
explicit policy event, not a learned assertion of truth. Withdrawing current
evidence refuses current queries; it never silently serves an older edition.
"""
import json
from pathlib import Path
import sqlite3

from claims.codata_source import DATES, SCOPE, check, sha
from claims.episodic_memory import Memory, ZERO, canonical, digest


def code_pins():
    root = Path(__file__).parent
    return {n: sha((root/n).read_bytes()) for n in (
        "scientific_memory.py", "codata_source.py", "episodic_memory.py")}


class ScientificMemory:
    def __init__(self, path, source, *, create=False, expected_head=None, expected_contract=None):
        path = Path(path)
        check(create or (expected_head is not None and expected_contract is not None), "RETAINED_PINS")
        check(not path.exists() if create else path.is_file(), "LEDGER_PATH")
        self.source, self.memory = source, Memory(source)
        self.edition, self.head, self.count, self.broken = None, ZERO, 0, False
        contract = dict(schema="scientific-reference-memory-v1", source=source.pin, code=code_pins(),
                        policy="latest-explicitly-adopted-pinned-codata-edition",
                        interpretation="published-recommendation-not-independent-replication",
                        serves_model_output=False)
        self.contract = digest(contract)
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb"):
                pass
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA synchronous=FULL")
        try:
            if create:
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.execute("CREATE TABLE metadata (contract BLOB NOT NULL)")
                    self.db.execute("INSERT INTO metadata VALUES (?)", (canonical(contract),))
                    self.db.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, previous TEXT NOT NULL,"
                                    "body BLOB NOT NULL, hash TEXT UNIQUE NOT NULL)")
                    self.db.execute("COMMIT")
            else:
                self.db.execute("BEGIN")
                actual = self.db.execute("SELECT contract FROM metadata").fetchall()
                check(actual == [(canonical(contract),)] and self.contract == expected_contract, "LEDGER_CONTRACT")
                for seq, previous, raw, pin in self.db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
                    check(len(raw) <= 65536, "EVENT_BOUND")
                    body = json.loads(raw)
                    check(raw == canonical(body) and seq == self.count and previous == self.head, "CHAIN_ORDER")
                    check(pin == digest(dict(seq=seq, previous=previous, body=body)), "CHAIN_HASH")
                    self._replay(body)
                    self.head, self.count = pin, seq+1
                check(self.head == expected_head, "HEAD_PIN")
                self.db.execute("COMMIT")
        except BaseException:
            self.db.close()
            self.broken = True
            raise

    def close(self):
        self.db.close()
        self.broken = True

    def _live(self):
        check(not self.broken, "RELOAD_REQUIRED")

    def _append(self, body):
        self._live()
        raw = canonical(body)
        check(len(raw) <= 65536, "EVENT_BOUND")
        pin = digest(dict(seq=self.count, previous=self.head, body=body))
        try:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            check((row[0] if row else ZERO) == self.head, "WRITER_CONFLICT")
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (self.count, self.head, raw, pin))
            self.db.execute("COMMIT")
        except BaseException:
            self.broken = True
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        self.head, self.count = pin, self.count+1

    def apply_memory(self, command):
        self._live()
        record = self.memory.apply(command)
        try:
            self._append(dict(kind="memory", record=record))
        except BaseException:
            self.broken = True
            raise
        return record

    def _checked(self, edition, quantity):
        ep = self.source.episode(edition, quantity)
        state = self.memory.state.get(ep["id"])
        check(state is not None and state["active"] and state["grounded_scope"] == SCOPE
              and state["review"] == "source_checked" and state["receipt"] == self.source.assess(ep),
              "CURRENT_EVIDENCE_UNAVAILABLE")
        return ep, state["receipt"]

    def _adoption(self, edition):
        check((self.edition, edition) in ((None, "2018"), ("2018", "2022")), "ADOPTION_ORDER")
        checked = [self._checked(edition, q)[1]["id"] for q in sorted(self.source.rows[edition])]
        return dict(kind="adopt", prior=self.edition, edition=edition, publication_date=DATES[edition],
                    receipt_set=digest(checked), quantity_count=len(checked),
                    history_sha256=self.source.manifest["files"]["history.html"]["sha256"],
                    delta=None if self.edition is None else self.source.changes(self.edition, edition))

    def adopt(self, edition):
        self._live()
        body = self._adoption(edition)
        self._append(body)
        self.edition = edition
        return body

    def _replay(self, body):
        if body.get("kind") == "memory":
            check(set(body) == {"kind", "record"}, "MEMORY_EVENT")
            command = {k: v for k, v in body["record"]["event"].items() if k not in ("result", "affected")}
            check(self.memory.apply(command) == body["record"], "MEMORY_REPLAY")
        else:
            check(body.get("kind") == "adopt" and body == self._adoption(body["edition"]), "ADOPTION_REPLAY")
            self.edition = body["edition"]

    def resolve(self, quantity, *, edition=None, require_complete_decimal=False, require_world_truth=False):
        self._live()
        row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        check((row[0] if row else ZERO) == self.head, "STALE_LEDGER")
        check(not require_world_truth, "NO_WORLD_CHECKER")
        selected = self.edition if edition is None else edition
        check(selected is not None and (edition is None or edition in DATES), "NO_ADOPTED_EDITION")
        ep, receipt = self._checked(selected, quantity)
        row = self.source.row(selected, quantity)
        check(not require_complete_decimal or not row["value"]["display_truncated"], "TRUNCATED_DISPLAY")
        value = dict(quantity=quantity, edition=selected, current_request=edition is None,
                     current_adopted_edition=self.edition, row=row, source=ep["source"],
                     publication_url=self.source.manifest["files"][selected+".txt"]["url"],
                     history_url=self.source.manifest["files"]["history.html"]["url"],
                     methodology_listing_url=self.source.manifest["files"]["methodology.html"]["url"]
                     if selected == "2022" else self.source.manifest["files"]["archive2018.html"]["url"],
                     receipt=receipt["id"], ledger_head=self.head, policy=self.contract,
                     who="NIST / CODATA", what="published recommended physical quantity",
                     where=None, where_status="no_experiment_location_in_table",
                     when=dict(publication=DATES[selected], adjustment=selected,
                               retrieved=self.source.manifest["files"][selected+".txt"]["retrieved_at"],
                               measurement=None),
                     why=None, why_status="adjustment_mechanism_not_reconstructed_from_table",
                     epistemic_status="published_exact_quantity" if row["exact_as_published"]
                     else "published_estimate_with_reported_uncertainty",
                     independent_replication=False, learned_understanding=False)
        return json.loads(canonical(value))

    def snapshot(self):
        self._live()
        return dict(contract=self.contract, head=self.head, events=self.count, adopted_edition=self.edition,
                    memory=self.memory.snapshot(), independently_established_world_claims=0,
                    learned_understanding=False, serving=False)

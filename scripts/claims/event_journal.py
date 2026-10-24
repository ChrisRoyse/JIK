"""Atomic evidence events and content-addressed payloads in one SQLite journal."""
import json
from pathlib import Path
import sqlite3

from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest


class EventJournal:
    def __init__(self, path, contract, *, create, payload_bound):
        path = Path(path)
        check(not path.exists() if create else path.is_file(), "JOURNAL_PATH")
        self.contract, self.pin = contract, digest(contract)
        self.head, self.count, self.broken = ZERO, 0, False
        self.payload_bound, self.cache = payload_bound, {}
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb"):
                pass
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA synchronous=FULL")
        try:
            if create:
                self.db.executescript("BEGIN IMMEDIATE;"
                    "CREATE TABLE metadata(contract BLOB NOT NULL);"
                    "CREATE TABLE events(seq INTEGER PRIMARY KEY, previous TEXT NOT NULL,"
                    "body BLOB NOT NULL, hash TEXT UNIQUE NOT NULL);"
                    "CREATE TABLE payloads(hash TEXT PRIMARY KEY, data BLOB NOT NULL);")
                self.db.execute("INSERT INTO metadata VALUES (?)", (canonical(contract),))
                self.db.execute("COMMIT")
            else:
                check(self.db.execute("SELECT contract FROM metadata").fetchall() ==
                      [(canonical(contract),)], "JOURNAL_CONTRACT")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.broken = True
        self.db.close()

    def fresh(self):
        check(not self.broken, "JOURNAL_RELOAD")
        row = self.db.execute("SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        check(row == (self.count-1, self.head) if row else self.count == 0 and self.head == ZERO,
              "JOURNAL_STALE")

    def payload(self, pin):
        if pin not in self.cache:
            row = self.db.execute("SELECT data FROM payloads WHERE hash=?", (pin,)).fetchone()
            check(row is not None and 0 < len(row[0]) <= self.payload_bound
                  and sha(row[0]) == pin, "JOURNAL_PAYLOAD")
            value = json.loads(row[0])
            check(canonical(value) == row[0], "JOURNAL_PAYLOAD_CANONICAL")
            self.cache[pin] = row[0]
        return json.loads(self.cache[pin])

    def append(self, body, payload=None):
        self.fresh()
        raw = canonical(body)
        check(len(raw) <= 65536, "JOURNAL_EVENT_BOUND")
        payload_raw = canonical(payload) if payload is not None else None
        check(payload_raw is None or (len(payload_raw) <= self.payload_bound
              and body["payload"] == sha(payload_raw)), "JOURNAL_PAYLOAD_BOUND")
        pin = digest(dict(seq=self.count, previous=self.head, body=body))
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.fresh()
            if payload_raw is not None:
                existing = self.db.execute("SELECT data FROM payloads WHERE hash=?",
                                           (body["payload"],)).fetchone()
                check(existing is None or existing == (payload_raw,), "JOURNAL_COLLISION")
                if existing is None:
                    self.db.execute("INSERT INTO payloads VALUES (?,?)", (body["payload"], payload_raw))
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (self.count, self.head, raw, pin))
            self.db.execute("COMMIT")
        except BaseException:
            self.broken = True
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        if payload_raw is not None:
            self.cache[body["payload"]] = payload_raw
        self.count, self.head = self.count+1, pin

    def restore(self, callback, *, head, contract):
        check(self.count == 0 and contract == self.pin, "JOURNAL_RETAINED_CONTRACT")
        try:
            self.db.execute("BEGIN")
            for seq, previous, raw, pin in self.db.execute(
                    "SELECT seq,previous,body,hash FROM events ORDER BY seq"):
                check(seq == self.count and previous == self.head and len(raw) <= 65536,
                      "JOURNAL_ORDER")
                body = json.loads(raw)
                check(canonical(body) == raw and pin == digest(dict(seq=seq, previous=previous, body=body)),
                      "JOURNAL_HASH")
                callback(body)
                self.count, self.head = seq+1, pin
            check(self.head == head, "JOURNAL_HEAD")
            self.db.execute("COMMIT")
        except BaseException:
            self.close()
            raise

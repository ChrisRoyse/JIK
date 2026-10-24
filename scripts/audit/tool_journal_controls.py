"""Corruption, stale-writer and frozen-target controls against the actual loop."""
import copy
import json
import sqlite3

from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from model.constellation import Constellation
from model.tool_learning import ToolLearningData
from model import constellation_training_checkpoint as checkpoint
from senses.contract import SenseError


def corruptions(path, open_at, *, head, contract, output):
    controls = []
    for name, code in (("head_pin", "JOURNAL_HEAD"), ("payload_bytes", "JOURNAL_PAYLOAD"),
                       ("event_hash", "JOURNAL_HASH"), ("rehashed_future", "TOOL_NEXT_REFERENCE")):
        clone = output/(name+".sqlite3")
        with sqlite3.connect(str(path)) as src, sqlite3.connect(str(clone)) as dst:
            src.backup(dst)
            expected_head = head
            if name == "head_pin":
                expected_head = "0"*64
            elif name == "payload_bytes":
                pin, raw = dst.execute("SELECT hash,data FROM payloads ORDER BY rowid LIMIT 1").fetchone()
                changed = bytes([raw[0] ^ 1])+raw[1:]
                dst.execute("UPDATE payloads SET data=? WHERE hash=?", (changed, pin))
            elif name == "event_hash":
                dst.execute("UPDATE events SET hash=? WHERE seq=0", ("f"*64,))
            else:
                entries = dst.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()
                altered, previous = False, ZERO
                for seq, raw in entries:
                    body = json.loads(raw)
                    if body["kind"] == "advance" and not altered:
                        body["request"]["reference"]["turn"] += 1
                        altered = True
                    pin = digest(dict(seq=seq, previous=previous, body=body))
                    dst.execute("UPDATE events SET previous=?,body=?,hash=? WHERE seq=?",
                                (previous, canonical(body), pin, seq))
                    previous = pin
                expected_head = previous
            dst.commit()
        try:
            session = open_at(clone, expected_head, contract)
        except (ValueError, SenseError) as exc:
            check(getattr(exc, "code", str(exc)) == "MNLM_CODATA_"+code, "TOOL_CORRUPTION_"+name)
        else:
            session.close()
            raise AssertionError(name+" did not refuse")
        controls.append(dict(name=name, code=code, source_copy=str(clone.name)))
    return controls


def runtime(session, refuse):
    import torch
    context = "withheld:0"
    task = session.contexts[context]
    refuse("duplicate_echo", "TOOL_FEEDBACK_RECEIPT",
           lambda: session.advance(context, receipt=session.source.receipt(task, 11)))
    refuse("unadmitted_candidate", "TOOL_CANDIDATE_NOT_SELECTED",
           lambda: session.require_selection("increment-2"))
    original = session.inputs.observed
    def missing(*args, **kwargs):
        value = original(*args, **kwargs)
        slots = dict(value.slots)
        slots.pop(next(iter(slots)))
        return Constellation(slots, "observed")
    session.inputs.observed = missing
    try:
        refuse("missing_instrument", "CONSTELLATION_ROSTER", lambda: session.advance(context))
    finally:
        session.inputs.observed = original
    keys = list(session.forecasts)
    train = [k for k in keys if session.forecasts[k]["record"]["request"]["context"] == "train:1"]
    evaluate = [k for k in keys if session.forecasts[k]["record"]["request"]["context"] == context]
    data = ToolLearningData(session, train=train, evaluate=evaluate)
    changed = copy.deepcopy(session.model)
    optimizer = checkpoint.optimizer_for(changed, data.plan)
    with torch.no_grad():
        changed.core.initial.view(-1)[0].add_(.01)
    refuse("frozen_identity_drift", "LEARNING_FROZEN_DRIFT",
           lambda: checkpoint.validate(changed, optimizer, data, 0))

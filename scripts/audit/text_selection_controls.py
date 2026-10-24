"""Negative controls and real selected-model prediction through the candidate controller."""
import copy
import shutil
import sqlite3

from claims.codata_source import check
from model.text_candidate_policy import assess
from model.text_candidate_session import TextCandidateSession
from model.text_sequence_data import PURPOSE
from senses.contract import digest


def selected_predictions(session, path):
    stream = session.predictions(path, purpose=PURPOSE)
    try:
        data = session.evidence.data
        for position in range(max(len(g["indices"])-1 for g in data.groups)):
            for group in data.groups:
                if position+1 < len(group["indices"]):
                    left, right = group["indices"][position:position+2]
                    stream.predict(left, expected_head=stream.journal.head, purpose=PURPOSE)
                    stream.reveal(right, expected_head=stream.journal.head, purpose=PURPOSE)
        _, outcomes = stream.records()
        check(len(outcomes) == 197, "TEXT_SELECTION_OUTPUT_COVERAGE")
        result = dict(retained=stream.retained(), outcomes=digest(outcomes), predictions=len(outcomes),
                      selected=session.selected, fingerprint=session.evidence.fingerprint())
    finally:
        stream.close()
    session.fresh()
    check(result["outcomes"] == session.records[session.selected]["outcomes"], "TEXT_SELECTION_OUTPUT_RESTORE")
    return result


def controls(session, root):
    rows = []
    def deny(name, function, code):
        try:
            function()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            check(actual == "MNLM_CODATA_"+code, "TEXT_SELECTION_CONTROL_CODE:"+name+":"+actual)
            rows.append(dict(name=name, code=actual))
        else:
            raise ValueError("MNLM_TEXT_SELECTION_CONTROL_ACCEPTED:"+name)
    before = session.retained()
    deny("serving", lambda: session.predictions(root/"forbidden.db", purpose="serving"), "TEXT_SELECTION_ADMISSION_REQUIRED")
    deny("stale_writer", lambda: session.review("cycle-1", expected_head="0"*64, purpose=PURPOSE), "TEXT_SELECTION_WRITER_CONFLICT")
    deny("duplicate_review", lambda: session.review("cycle-1", expected_head=before["head"], purpose=PURPOSE), "TEXT_SELECTION_REVIEW_ORDER")
    evidence = session.evidence
    data, records = evidence.data, evidence.records()
    old, new = records["initial-training"]["development"], records["cycle-1"]["development"]
    kwargs = dict(groups=data.groups, instruments=[s.name for s in evidence.model.specs],
                  retained="initial-training", established=[])
    missing = copy.deepcopy(new)
    del missing["general"]["instruments"][kwargs["instruments"][0]]
    deny("omitted_instrument", lambda: assess("cycle-1", old, missing, **kwargs), "TEXT_SELECTION_UNIT_COVERAGE")
    relabelled = copy.deepcopy(new)
    relabelled["general"]["count"] -= 1
    deny("omitted_source_unit", lambda: assess("cycle-1", old, relabelled, **kwargs), "TEXT_SELECTION_UNIT_COVERAGE")
    changed = copy.deepcopy(new)
    changed["general"]["instruments"][kwargs["instruments"][0]]["persistence"] += .1
    deny("changed_baseline", lambda: assess("cycle-1", old, changed, **kwargs), "TEXT_SELECTION_BASELINE_CHANGED")
    evidence.activate("cycle-2")
    deny("rejected_weights_in_memory", lambda: session.predictions(root/"rejected.db", purpose=PURPOSE), "TEXT_SELECTION_LIVE_MODEL_DIVERGED")
    session.recover(expected_head=before["head"], purpose=PURPOSE)
    import torch
    with torch.no_grad():
        evidence.model.core.identity.ref_values_0.view(-1)[0].add_(.125)
    deny("changed_identity_reference", session.fresh, "TEXT_SELECTION_LIVE_MODEL_DIVERGED")
    session.recover(expected_head=before["head"], purpose=PURPOSE)
    check(session.retained() == before, "TEXT_SELECTION_FAILED_CONTROL_MUTATION")
    check(not (root/"forbidden.db").exists() and not (root/"rejected.db").exists(), "TEXT_SELECTION_FAILED_OUTPUT")
    corrupt = root/"corrupt.db"
    shutil.copyfile(root/"selection.db", corrupt)
    with sqlite3.connect(corrupt) as db:
        db.execute("UPDATE payloads SET data=? WHERE hash=(SELECT hash FROM payloads LIMIT 1)", (b"{}",))
    deny("corrupted_decision", lambda: TextCandidateSession(corrupt, evidence, create=False, retained=before), "JOURNAL_PAYLOAD")
    deny("forged_retained_pointer", lambda: TextCandidateSession(root/"selection.db", evidence,
         create=False, retained={**before, "selected": "cycle-2"}), "TEXT_SELECTION_RETAINED_POINTER")
    check(session.retained() == before, "TEXT_SELECTION_FINAL_CONTROL_MUTATION")
    return rows

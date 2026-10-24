"""Physical replay/rollback checks for the native text sequence diagnostic."""
import json
import shutil
import sqlite3

import torch

from claims.codata_source import check
from model.constellation_learning import configure, frozen_pin, trainable
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_identity_world import NativeIdentityWorld, SensoryIdentityCore
from model.text_sequence_data import PLAN, PURPOSE, summarize
from model.text_sequence_session import TextSequenceSession
from senses.contract import digest


def build(data, a):
    identity = LoopInputs(a.identity_population, a.identity_native, a.identity_source)
    torch.manual_seed(20261020)
    core = IdentityLoopPredictor(data.config, identity.specs,
        identity.batch(identity.standard_units, data.config, "cpu"), source_id=identity.source_id)
    return NativeIdentityWorld(SensoryIdentityCore(core, data.inputs.specs, data.inputs.temporal_catalogue),
        data.inputs.references(data.refs, "cpu"), world_source_id=data.inputs.binding).cuda().eval()


def tape(path, model, data, checkpoint, pin, *, retained=None, session_class=TextSequenceSession):
    session = session_class(path, model, data, checkpoint=checkpoint, checkpoint_pin=pin,
                                  create=retained is None, retained=retained)
    try:
        if retained is None:
            # Interleave complete documents, while keeping each context's actual succession.
            for position in range(max(len(g["indices"])-1 for g in data.groups)):
                for group in data.groups:
                    if position+1 >= len(group["indices"]):
                        continue
                    left, right = group["indices"][position:position+2]
                    session.predict(left, expected_head=session.journal.head, purpose=PURPOSE)
                    session.reveal(right, expected_head=session.journal.head, purpose=PURPOSE)
        examples, outcomes = session.records()
        summary = {}
        for split in ("train", "development"):
            summary[split] = {}
            for domain in ("general", "legal", "science"):
                rows = [r for r in outcomes if data.by_index[r["left"]]["split"] == split
                        and data.by_index[r["left"]]["domain"] == domain]
                summary[split][domain] = summarize(rows)
        return dict(retained=session.retained(), examples=digest(examples), outcomes=digest(outcomes),
                    summary=summary, events=session.journal.count), examples
    finally:
        session.close()


def update(model, optimizer, data, example):
    optimizer.zero_grad(set_to_none=True)
    loss = data.loss(model, example)
    check(bool(torch.isfinite(loss)), "TEXT_UPDATE_NONFINITE")
    loss.backward()
    parameters = [p for _, p in trainable(model)]
    check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
          "TEXT_UPDATE_GRADIENTS")
    torch.nn.utils.clip_grad_norm_(parameters, 1.)
    optimizer.step()
    return float(loss.detach())


def train_cycle(model, optimizer, data, examples, domains, frozen):
    configure(model)
    chosen = [e for e in examples if data.by_index[e["left"]]["split"] == "train"
              and data.by_index[e["left"]]["domain"] in domains]
    expected = 86 if domains == PLAN["cycle_1"] else 104
    check(len(chosen) == expected and domains in (PLAN["cycle_1"], PLAN["cycle_2"]), "TEXT_TRAIN_POPULATION")
    result = []
    for epoch in range(PLAN["epochs"]):
        losses = [update(model, optimizer, data, e) for e in chosen]
        check(frozen_pin(model) == frozen, "TEXT_FROZEN_REGRESSION")
        result.append(dict(epoch=epoch, count=len(losses), mean=sum(losses)/len(losses)))
        print(json.dumps(dict(epoch=epoch, domains=domains, loss=result[-1]["mean"])), flush=True)
    return dict(epochs=result, steps=len(chosen)*PLAN["epochs"], evidence=digest(chosen))


def decisions(parent, candidate):
    result = {}
    for domain in parent["summary"]["development"]:
        old, new = parent["summary"]["development"][domain], candidate["summary"]["development"][domain]
        result[domain] = dict(region_bits_gain=old["region_nll_bits"]-new["region_nll_bits"],
            metric="next_source_region_logscore_gain_not_conditional_lens_information",
            lanes={name: dict(parent=old["instruments"][name]["prediction"], **r,
                better_than_parent=r["prediction"] < old["instruments"][name]["prediction"],
                better_than_persistence=r["prediction"] < r["persistence"])
                for name, r in new["instruments"].items()})
    passed = all(r["better_than_parent"] and r["better_than_persistence"]
                 for d in result.values() for r in d["lanes"].values())
    return dict(domains=result, per_lane_development_gate=passed, selected=False,
                reasons=["CANDIDATE_INSTRUMENTS_NOT_ADMITTED", "EXPOSED_SOURCE_REPLAY_NOT_CAPABILITY_PROOF"]
                + ([] if passed else ["PER_LANE_DEVELOPMENT_GATE_FAILED"]))


def controls(root, model, data, checkpoint, pin):
    results = []
    def deny(name, function, code):
        try:
            function()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            prefix = "MNLM_" if code == "TEXT_STREAM_ADMISSION_REQUIRED" else "MNLM_CODATA_"
            check(actual == prefix+code, "TEXT_CONTROL_CODE:"+name+":"+actual)
            results.append(dict(name=name, code=actual))
        else:
            raise ValueError("MNLM_TEXT_CONTROL_ACCEPTED:"+name)
    path = root/"controls.db"
    session = TextSequenceSession(path, model, data, checkpoint=checkpoint, checkpoint_pin=pin, create=True)
    first = data.groups[0]["indices"][0]
    head = session.journal.head
    def predict(index, purpose=PURPOSE):
        return session.predict(index, expected_head=session.journal.head, purpose=purpose)
    def reveal(index):
        return session.reveal(index, expected_head=session.journal.head, purpose=PURPOSE)
    deny("reveal_without_prediction", lambda: reveal(first+1), "TEXT_PREDICTION_REQUIRED")
    deny("serving", lambda: predict(first, "serving"), "TEXT_SEQUENCE_ADMISSION_REQUIRED")
    deny("skipped_prefix", lambda: predict(first+1), "TEXT_PREFIX_ORDER")
    predict(first)
    prediction = session.retained()
    deny("skipped_target", lambda: reveal(first+2), "TEXT_NOT_IMMEDIATE_SUCCESSOR")
    deny("cross_document_target", lambda: reveal(data.groups[1]["indices"][0]), "TEXT_NOT_IMMEDIATE_SUCCESSOR")
    deny("stale_writer", lambda: session.reveal(first+1, expected_head=head, purpose=PURPOSE), "TEXT_WRITER_CONFLICT")
    deny("target_still_pending", lambda: predict(first+1), "TEXT_TARGET_PENDING")
    check(session.retained() == prediction, "TEXT_FAILED_ACTION_MUTATION")
    session.close()
    # A process can stop after commit and before target exposure. Restore this exact boundary.
    session = TextSequenceSession(path, model, data, checkpoint=checkpoint, checkpoint_pin=pin,
                                  create=False, retained=prediction)
    reveal(first+1)
    retained = session.retained()
    session.close()
    session = TextSequenceSession(path, model, data, checkpoint=checkpoint, checkpoint_pin=pin,
                                  create=False, retained=retained)
    check(len(session.records()[0]) == 1, "TEXT_CONTROL_SOURCE_READBACK")
    session.close()
    bad = root/"corrupt.db"
    shutil.copyfile(path, bad)
    with sqlite3.connect(bad) as db:
        db.execute("UPDATE payloads SET data=? WHERE hash=(SELECT hash FROM payloads LIMIT 1)", (b"{}",))
    deny("corrupted_payload", lambda: TextSequenceSession(bad, model, data, checkpoint=checkpoint,
        checkpoint_pin=pin, create=False, retained=retained), "JOURNAL_PAYLOAD")
    deny("unadmitted_training", lambda: data.inputs.batch(first, through=data.sources[first][0]["char_end"],
         purpose="training"), "TEXT_STREAM_ADMISSION_REQUIRED")
    return dict(refusals=results, pending_prediction_restarted=True, committed_source_read_back=True)

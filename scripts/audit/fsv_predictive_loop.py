"""Real publication replay through committed forecasts and delayed intrinsic echoes."""
import argparse
import base64
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_scientific_loop import setup, prefix_copy, write, CONTEXTS, QUANTITIES
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest, ZERO
from lens.lenscommon import run_cli
from model.predictive_loop import PredictiveLoop
from senses.contract import SenseError


def independent(path, inputs, source, expected):
    """Raw chain readback and scalar arithmetic, separate from the forecast scorer."""
    import struct
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    head, count, forecasts, resolved, contexts = ZERO, 0, {}, {}, {}
    try:
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            assert seq == count and previous == head and canonical(body) == raw
            assert pin == digest(dict(seq=seq, previous=previous, body=body))
            kind = body["kind"]
            if kind == "world_init":
                contexts = {c: dict(sequence=0) for c in body["contract"]["contexts"]}
            elif kind == "world_advance":
                request = body["request"]
                c = request["context"]
                assert request["sequence"] == contexts[c]["sequence"]
                contexts[c] = dict(sequence=request["sequence"]+1, last_event=digest(request),
                                   hidden_sha256=body["hidden_sha256"])
            elif kind == "prediction_commit":
                r = body["record"]
                assert body["id"] == digest(r) and r["prior_head"] == head
                assert r["mode"] == "retrospective_temporal_replay" and r["issued_at"][:10] >= r["target_published"]
                assert r["context_sequence"] == contexts[r["context"]]["sequence"]
                forecasts[body["id"]] = (seq, r)
            elif kind == "prediction_observe":
                key, request = body["forecast"], body["request"]
                committed, record = forecasts[key]
                assert key not in resolved and committed < seq
                assert request["quantity"] == record["quantity"] and request["context"] == record["context"]
                ep = source.episode(record["target_edition"], record["quantity"])
                assert request["checked_receipt"] == source.assess(ep)
                assert request["prior_sequence"] == record["context_sequence"]
                for name, value in body["assessment"]["instruments"].items():
                    encoded = record["predictions"][name]
                    raw_prediction = base64.b64decode(encoded["base64"], validate=True)
                    assert sha(raw_prediction) == encoded["sha256"]
                    pred = list(struct.unpack("<"+"f"*encoded["size"], raw_prediction))
                    arrays, offsets = inputs.arrays[name]
                    old = record["baseline_unit"]
                    future = inputs.unit("row", record["target_edition"], record["quantity"])
                    a = [list(map(float, v)) for v in arrays[offsets[old]:offsets[old+1]]]
                    b = [list(map(float, v)) for v in arrays[offsets[future]:offsets[future+1]]]
                    def cosine(x, y):
                        return max(-1., min(1., math.fsum(i*j for i, j in zip(x, y)) /
                            math.sqrt(math.fsum(i*i for i in x)*math.fsum(j*j for j in y))))
                    pred_cos = [cosine(pred, v) for v in b]
                    baseline_cos = [max(cosine(u, v) for u in a) for v in b]
                    assert all(abs(x-y) < 1e-12 for x, y in zip(pred_cos, value["prediction_cosines"]))
                    assert all(abs(x-y) < 1e-12 for x, y in zip(baseline_cos, value["persistence_cosines"]))
                    assert abs(value["prediction_loss"]-math.fsum(1-x for x in pred_cos)/len(b)) < 1e-12
                    assert abs(value["persistence_loss"]-math.fsum(1-x for x in baseline_cos)/len(b)) < 1e-12
                    assert abs(value["delta"]-(value["prediction_loss"]-value["persistence_loss"])) < 1e-12
                resolved[key] = body["assessment"]
                contexts[request["context"]] = dict(sequence=request["prior_sequence"]+1,
                    last_event=digest(request), hidden_sha256=body["hidden_sha256"])
            head, count = pin, count+1
        assert head == expected["head"] and count == expected["events"]
        assert set(forecasts) == set(resolved) == set(expected["forecasts"])
        for key, result in resolved.items():
            assert expected["forecasts"][key]["result"] == result
            assert expected["forecasts"][key]["grounding"]["establishes_empirical_truth"] is False
        for c, state in contexts.items():
            assert all(expected["contexts"][c][k] == v for k, v in state.items())
        return dict(events=count, forecasts=len(forecasts), scored=len(resolved),
                    instrument_scores=sum(len(x["instruments"]) for x in resolved.values()),
                    scalar_score_readback=True, head=head)
    finally:
        db.close()


def main(*, loop_type=PredictiveLoop, readback=independent, boundary=None):
    # Reuse this real protocol for separately versioned prediction representations.
    PredictiveLoop, independent = loop_type, readback
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("source", "population", "native", "parent", "science", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("source-pin", "population-pin", "native-pin", "parent-pin", "science-run-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--run-pin")
    args = p.parse_args()
    started = time.perf_counter()
    source, inputs, resident = setup(args)
    phases = dict(source_native_readback=time.perf_counter()-started)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "PARENT_RUN_PIN")
    parent = json.loads((args.parent/"run.json").read_bytes())
    if args.cold:
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["parent"] == args.parent_pin and run["world_source"] == inputs.source_id, "RUN_SOURCE")
        session = PredictiveLoop(args.output/"science.sqlite3", source, inputs, args.parent/"model",
                                 **run["open"], device="cuda")
        expected = json.loads((args.output/"snapshot.json").read_bytes())
        check(session.snapshot() == expected, "COLD_SNAPSHOT")
        result = independent(args.output/"science.sqlite3", inputs, source, expected)
        check(session.replayed+session.prediction_replayed == 20, "COLD_NEURAL_COUNT")
        session.close()
        write(args.output/"cold-readback.json", dict(complete=True, acceptance="author-checked",
              independent=result, neural_transitions=20, wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(cold=True, complete=True, readback=result)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"gpu-before.txt").write_text(resident, encoding="utf-8")
    tick = time.perf_counter()
    check(sha((args.science/"run.json").read_bytes()) == args.science_run_pin, "SCIENCE_RUN_PIN")
    retained = json.loads((args.science/"run.json").read_bytes())
    lineage = prefix_copy(args.science/"science.sqlite3", args.output/"science.sqlite3", source, retained)
    kw = dict(parent["open"], expected_head=lineage["prefix_head"])
    kw.pop("expected_loop_contract")
    session = PredictiveLoop(args.output/"science.sqlite3", source, inputs, args.parent/"model",
                             **kw, device="cuda", attach=True)
    extension = boundary(session, args.output) if boundary is not None else {}
    phases["load_and_attach"] = time.perf_counter()-tick
    tick = time.perf_counter()
    controls, boundaries = [], []
    def refuse(label, code, fn, *, subject=None, same=True):
        target = subject or session
        before = digest(target.snapshot()) if same else None
        try:
            fn()
        except (ValueError, SenseError) as exc:
            check(getattr(exc, "code", str(exc)) == code, "WRONG_REFUSAL_"+label)
        else:
            raise AssertionError(label+" did not refuse")
        if same:
            check(digest(target.snapshot()) == before, "REFUSAL_MUTATED")
        controls.append(dict(name=label, code=code, unchanged=same))
    before = digest(session.snapshot())
    # Real past/current source exposure, then a new act which predicts its next echo.
    for phase in range(5):
        for c, q in zip(CONTEXTS, QUANTITIES):
            session.advance(c, q, receipt=session.resolve(q)["receipt"] if phase == 3 else None)
    for phase in range(3):
        for c, q in zip(CONTEXTS, QUANTITIES):
            session.advance(c, q)
    refuse("target_not_later", "MNLM_CODATA_PREDICTION_TARGET_ORDER",
           lambda: session.commit_forecast(CONTEXTS[0], target_edition="2018"))
    forecasts = {c: session.commit_forecast(c, target_edition="2022") for c in CONTEXTS}
    original = {c: dict(session.forecasts[f]["record"]["predictions"]) for c, f in forecasts.items()}
    refuse("duplicate_commitment", "MNLM_CODATA_PREDICTION_DUPLICATE",
           lambda: session.commit_forecast(CONTEXTS[0], target_edition="2022"))
    refuse("outcome_not_yet_adopted", "MNLM_CODATA_PREDICTION_OUTCOME_NOT_ADOPTED",
           lambda: session.observe_forecast(forecasts[CONTEXTS[0]], receipt="future"))
    refuse("unknown_forecast", "MNLM_CODATA_PREDICTION_UNKNOWN",
           lambda: session.observe_forecast("0"*64, receipt="future"))
    boundaries.append(dict(name="commit-before-feedback", before=before, after=digest(session.snapshot()),
                           forecasts=forecasts, mode="retrospective_temporal_replay"))
    kw.update(expected_head=session.head, expected_loop_contract=session.loop_pin,
              expected_prediction_contract=digest(session.prediction_contract()))
    pending = session.snapshot()
    write(args.output/"pending.json", pending)
    session.close()
    session = PredictiveLoop(args.output/"science.sqlite3", source, inputs, args.parent/"model", **kw, device="cuda")
    check(session.snapshot() == pending, "PENDING_RESTART")
    stale = PredictiveLoop(args.output/"science.sqlite3", source, inputs, args.parent/"model", **kw, device="cuda")
    session.adopt("2022")
    refuse("stale_outcome_writer", "MNLM_CODATA_STALE_LEDGER",
           lambda: stale.observe_forecast(forecasts[CONTEXTS[0]], receipt="future"), subject=stale)
    stale.close()
    refuse("late_commit_after_edition_change", "MNLM_CODATA_PREDICTION_CONTEXT",
           lambda: session.commit_forecast(CONTEXTS[0], target_edition="2022"))
    # Withdrawal on a separate real ledger copy must survive restoration of the same receipt.
    shadow_path = args.output/"withdrawn-pending.sqlite3"
    with sqlite3.connect(shadow_path) as db:
        session.db.backup(db)
    shadow = PredictiveLoop(shadow_path, source, inputs, args.parent/"model",
                            **dict(kw, expected_head=session.head), device="cuda")
    old_ep = source.episode("2018", QUANTITIES[0])
    shadow.apply_memory(dict(op="invalidate", episode=old_ep["id"], reason="Withdraw forecast input evidence."))
    target_receipt = session.resolve(QUANTITIES[0])["receipt"]
    refuse("withdrawn_pending", "MNLM_CODATA_PREDICTION_UNAVAILABLE",
           lambda: shadow.observe_forecast(forecasts[CONTEXTS[0]], receipt=target_receipt), subject=shadow)
    shadow.apply_memory(dict(op="check_source", episode=old_ep["id"]))
    shadow.apply_memory(dict(op="promote", episode=old_ep["id"], receipt=source.assess(old_ep)["id"]))
    refuse("rechecking_cannot_restore_forecast", "MNLM_CODATA_PREDICTION_UNAVAILABLE",
           lambda: shadow.observe_forecast(forecasts[CONTEXTS[0]], receipt=target_receipt), subject=shadow)
    shadow.close()
    for c, q in zip(CONTEXTS, QUANTITIES):
        other = next(x for x in CONTEXTS if x != c)
        untouched = session.snapshot()["contexts"][other]
        before = digest(session.snapshot())
        refuse("generated_echo_"+c, "MNLM_CODATA_PREDICTION_OUTCOME_RECEIPT",
               lambda: session.observe_forecast(forecasts[c], receipt="model-confidence"))
        other_receipt = session.resolve(next(x for x in QUANTITIES if x != q))["receipt"]
        refuse("other_observable_"+c, "MNLM_CODATA_PREDICTION_OUTCOME_RECEIPT",
               lambda: session.observe_forecast(forecasts[c], receipt=other_receipt))
        result = session.observe_forecast(forecasts[c], receipt=session.resolve(q)["receipt"])
        check(session.snapshot()["contexts"][other] == untouched and not session.needs_review(c), "ECHO_CONTEXT_ISOLATION")
        check(session.forecasts[forecasts[c]]["record"]["predictions"] == original[c], "PREDICTION_REWRITTEN")
        session.advance(c, q)  # the real correction phase consumes the observed source
        refuse("reuse_observed_forecast_"+c, "MNLM_CODATA_PREDICTION_UNAVAILABLE",
               lambda: session.observe_forecast(forecasts[c], receipt=session.resolve(q)["receipt"]))
        refuse("cannot_promote_temporal_replay_"+c, "MNLM_CODATA_PREDICTION_GROUNDING_INSUFFICIENT",
               lambda: session.require_empirical_grounding(forecasts[c]))
        boundaries.append(dict(name="observed-echo-and-correction-"+c, before=before,
                               after=digest(session.snapshot()), score=result, predictions_unchanged=True))
    target = source.episode("2022", QUANTITIES[0])
    session.apply_memory(dict(op="invalidate", episode=target["id"], reason="Withdraw scored outcome evidence."))
    check(not session.grounding_status(forecasts[CONTEXTS[0]])["prediction_usable_as_learning_evidence"]
          and session.grounding_status(forecasts[CONTEXTS[1]])["prediction_usable_as_learning_evidence"],
          "SCORED_OUTCOME_WITHDRAWAL")
    session.model.check_reference()
    final = session.snapshot()
    kw["expected_head"] = session.head
    session.close()
    phases["forecast_commit_echo_correction"] = time.perf_counter()-tick
    tick = time.perf_counter()
    # A forged, correctly rehashed chain must still fail neural-forecast readback.
    for label in ("forged-prediction", "mixed-observable", "truncated-chain"):
        path = args.output/(label+".sqlite3")
        shutil.copyfile(args.output/"science.sqlite3", path)
        with sqlite3.connect(path) as db:
            if label == "truncated-chain":
                db.execute("DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)")
                head = kw["expected_head"]
                error = "MNLM_CODATA_HEAD_PIN"
            else:
                rows = db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()
                head, changed = ZERO, False
                for seq, raw in rows:
                    body = json.loads(raw)
                    if not changed and body["kind"] == "prediction_commit":
                        r = body["record"]
                        if label == "mixed-observable":
                            r["quantity"] = QUANTITIES[1]
                        else:
                            name = next(iter(r["predictions"]))
                            prediction = r["predictions"][name]
                            data = bytearray(base64.b64decode(prediction["base64"]))
                            data[0] ^= 1
                            prediction.update(base64=base64.b64encode(data).decode(), sha256=sha(data))
                        body["id"] = digest(r)
                        changed = True
                    pin = digest(dict(seq=seq, previous=head, body=body))
                    db.execute("UPDATE events SET previous=?,body=?,hash=? WHERE seq=?", (head, canonical(body), pin, seq))
                    head = pin
                error = "MNLM_CODATA_PREDICTION_COMMIT_REPLAY"
        refuse(label, error, lambda: PredictiveLoop(path, source, inputs, args.parent/"model",
                                                   **dict(kw, expected_head=head), device="cuda"), same=False)
    phases["semantic_corruption_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    restored = PredictiveLoop(args.output/"science.sqlite3", source, inputs, args.parent/"model", **kw, device="cuda")
    check(restored.snapshot() == final and restored.replayed+restored.prediction_replayed == 20, "FINAL_RESTART")
    result = independent(args.output/"science.sqlite3", inputs, source, final)
    boundaries.append(dict(name="prediction-score-state-readback", before=digest(final),
                           after=digest(restored.snapshot()), raw_sql=result, neural_transitions=20))
    restored.close()
    phases["independent_readback"] = time.perf_counter()-tick
    write(args.output/"snapshot.json", final)
    write(args.output/"boundaries.json", boundaries)
    write(args.output/"run.json", dict(parent=args.parent_pin, source=source.pin, world_source=inputs.source_id,
                                       lineage=lineage, open=kw))
    report = dict(complete=True, acceptance="author-checked", measured=result, controls=controls,
                  extension=extension,
                  phase_seconds=phases, wall_seconds=time.perf_counter()-started,
                  useful_capability_gain_demonstrated=False, prospective_forecast_validation=False,
                  independently_grounded_world_knowledge=False, scientific_admission=False,
                  files={n: sha((args.output/n).read_bytes()) for n in
                         ("snapshot.json", "pending.json", "run.json", "boundaries.json", "science.sqlite3")})
    write(args.output/"report.json", report)
    print(json.dumps(dict(complete=True, measured=result, controls=len(controls), wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    sys.exit(run_cli(main))

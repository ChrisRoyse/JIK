"""Two real, fixed-recipe outcome-learning trials and exact restart/rejection."""
import argparse
import copy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_scientific_loop import setup, prefix_copy, write, CONTEXTS, QUANTITIES
from audit.constellation_learning_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import digest, canonical, ZERO
from lens.lenscommon import run_cli
from model.constellation_learning import LearningData, configure, frozen_pin, trainable
from model.constellation_learning_session import LearningConstellationLoop as Session
from model import constellation_training_checkpoint as checkpoint
from model.loop_training_checkpoint import fingerprint
from senses.contract import SenseError


def main():
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
    import torch
    phases = dict(source_native=time.perf_counter()-started)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "PARENT_PIN")
    parent = json.loads((args.parent/"run.json").read_bytes())
    candidates = args.output/"candidates"
    def open_session(kw, **extra):
        return Session(args.output/"science.sqlite3", source, inputs, args.parent/"model",
                       **kw, candidate_root=candidates, device="cuda", **extra)
    if args.cold:
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["parent"] == args.parent_pin and run["source"] == source.pin, "RUN_SOURCE")
        session = open_session(run["open"])
        expected = json.loads((args.output/"snapshot.json").read_bytes())
        check(session.snapshot() == expected, "COLD_SNAPSHOT")
        result = readback(args.output, inputs, expected)
        session.close()
        write(args.output/"cold-readback.json", dict(result=result, complete=True,
              wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(cold=True, complete=True, result=result)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    candidates.mkdir()
    (args.output/"gpu-before.txt").write_text(resident, encoding="utf-8")
    tick = time.perf_counter()
    check(sha((args.science/"run.json").read_bytes()) == args.science_run_pin, "SCIENCE_PIN")
    lineage = prefix_copy(args.science/"science.sqlite3", args.output/"science.sqlite3",
                          source, json.loads((args.science/"run.json").read_bytes()))
    kw = dict(parent["open"], expected_head=lineage["prefix_head"])
    kw.pop("expected_loop_contract")
    session = open_session(kw, attach=True)
    controls, boundaries = [], []
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as e:
            check(getattr(e, "code", str(e)) == "MNLM_CODATA_"+code, "CONTROL_"+name)
        else:
            raise AssertionError(name+" did not refuse")
        check(digest(session.snapshot()) == before, "CONTROL_MUTATED_"+name)
        controls.append(dict(name=name, code=code, state_unchanged=True))
    for phase in range(5):
        for c, q in zip(CONTEXTS, QUANTITIES):
            session.advance(c, q, receipt=session.resolve(q)["receipt"] if phase == 3 else None)
    for phase in range(3):
        for c, q in zip(CONTEXTS, QUANTITIES):
            session.advance(c, q)
    forecasts = [session.commit_forecast(c, target_edition="2022") for c in CONTEXTS]
    refuse("pending_feedback", "LEARNING_EVIDENCE_INELIGIBLE",
           lambda: LearningData(session, train=forecasts[:1], evaluate=forecasts[1:]))
    before = digest(session.snapshot())
    session.adopt("2022")
    for c, q, f in zip(CONTEXTS, QUANTITIES, forecasts):
        refuse("self_authored_"+c, "PREDICTION_OUTCOME_RECEIPT",
               lambda: session.observe_forecast(f, receipt="generated-prediction"))
        session.observe_forecast(f, receipt=session.resolve(q)["receipt"])
        session.advance(c, q)
    boundaries.append(dict(name="pending_to_eligible_external_feedback", before=before,
                           after=digest(session.snapshot()), forecasts=forecasts))
    phases["connected_feedback"] = time.perf_counter()-tick
    for cycle, (train, evaluate) in enumerate(((forecasts[1], forecasts[0]), (forecasts[0], forecasts[1])), 1):
        tick = time.perf_counter()
        name = "cycle-"+str(cycle)
        data = LearningData(session, train=[train], evaluate=[evaluate])
        refuse("overlap_"+name, "LEARNING_SPLIT",
               lambda: LearningData(session, train=[train], evaluate=[train]))
        example = data.examples[train]
        forbidden = (~example["allowed"]).nonzero()[0, 0]
        example["allowed"][forbidden] = True
        refuse("future_region_"+name, "LEARNING_CUTOFF", lambda: data.forward(session.model, train))
        example["allowed"][forbidden] = False
        baseline = data.evaluate(session.model)
        model = configure(copy.deepcopy(session.model))
        optimizer = checkpoint.optimizer_for(model, data.plan)
        initial = fingerprint(model, optimizer)
        state_before = digest(session.snapshot())
        initial_pin = checkpoint.save(candidates/(name+"-initial"), model, optimizer, data,
                                      step=0, parent=kw["model_pin"])
        def update(model, optimizer, step):
            optimizer.zero_grad(set_to_none=True)
            loss, detail = data.loss(model)
            check(bool(torch.isfinite(loss)), "LEARNING_NONFINITE")
            loss.backward()
            check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in trainable(model)),
                  "LEARNING_GRADIENT")
            check(all(p.grad is None for p in model.core.parameters()), "LEARNING_FROZEN_GRADIENT")
            optimizer.step()
            return dict(step=step, loss=float(loss.detach()), lanes=detail)
        trace = []
        for step in range(1, 9):
            trace.append(update(model, optimizer, step))
            if step == 4:
                mid_pin = checkpoint.save(candidates/(name+"-step4"), model, optimizer, data,
                                          step=step, parent=kw["model_pin"])
        final_pin = checkpoint.save(candidates/name, model, optimizer, data, step=8, parent=kw["model_pin"])
        final_fp = fingerprint(model, optimizer)
        check(initial["model"] != final_fp["model"] and frozen_pin(model) == data.frozen, "LEARNING_NO_UPDATE")
        assessment = data.evaluate(model)
        resumed, opt, m = checkpoint.load(candidates/(name+"-step4"), data, expected=mid_pin)
        resumed_trace = [update(resumed, opt, step) for step in range(m["step"]+1, 9)]
        resumed_fp = fingerprint(resumed, opt)
        check(resumed_trace == trace[4:] and resumed_fp == final_fp, "LEARNING_RECOVERY")
        refuse("cursor_"+name, "LEARNING_CURSOR", lambda: checkpoint.validate(model, optimizer, data, 7))
        with torch.no_grad():
            original = model.core.initial.clone()
            model.core.initial.add_(1)
        refuse("core_drift_"+name, "LEARNING_FROZEN_DRIFT",
               lambda: checkpoint.validate(model, optimizer, data, 8))
        with torch.no_grad():
            model.core.initial.copy_(original)
        if cycle == 1:
            corrupt = candidates/"corrupt-optimizer"
            shutil.copytree(candidates/name, corrupt)
            raw = bytearray((corrupt/"training.pt").read_bytes())
            raw[-1] ^= 1
            (corrupt/"training.pt").write_bytes(raw)
            refuse("optimizer_payload_bytes", "LEARNING_OPTIMIZER_PIN",
                   lambda: checkpoint.load(corrupt, data, expected=final_pin))
            # A self-consistent file hash cannot launder an inconsistent step.
            raw = (candidates/name/"training.pt").read_bytes()
            payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
            next(iter(payload["optimizer"]["state"].values()))["step"].fill_(7)
            buf = io.BytesIO()
            torch.save(payload, buf)
            raw = buf.getvalue()
            (corrupt/"training.pt").write_bytes(raw)
            manifest = json.loads((corrupt/"manifest.json").read_bytes())
            manifest.update(bytes=len(raw), sha256=sha(raw))
            raw = canonical(manifest)
            (corrupt/"manifest.json").write_bytes(raw)
            (corrupt/"COMMITTED").write_text(sha(raw), encoding="ascii")
            refuse("rehashed_optimizer_cursor", "LEARNING_CURSOR",
                   lambda: checkpoint.load(corrupt, data, expected=sha(raw)))
        result = session.consider(name, pin=final_pin, train=[train], evaluate=[evaluate])
        check(result["result"]["selected"] is False, "UNEXPECTED_SELECTION")
        check(digest(baseline) == result["baseline"] and digest(assessment) == result["assessment"], "CANDIDATE_READBACK")
        refuse("failed_selection_"+name, "LEARNING_CANDIDATE_NOT_SELECTED", lambda: session.require_selection(name))
        refuse("stale_learning_"+name, "LEARNING_STALE_BINDING", data.fresh)
        write(args.output/(name+".json"), dict(initial_pin=initial_pin, mid_pin=mid_pin, final_pin=final_pin,
            initial_fingerprint=initial, final_fingerprint=final_fp, resumed_fingerprint=resumed_fp,
            trace=trace, resumed_trace=resumed_trace, baseline=baseline, assessment=assessment, decision=result))
        boundaries.append(dict(name=name, before=state_before, after=digest(session.snapshot()),
                               candidate_changed=True, active_unchanged=True, exact_optimizer_recovery=True))
        phases[name] = time.perf_counter()-tick
    tick = time.perf_counter()
    before = digest(session.snapshot())
    session.apply_memory(dict(op="invalidate", episode=source.episode("2022", QUANTITIES[0])["id"],
                              reason="Withdraw learning outcome evidence after candidate review."))
    refuse("withdrawn_learning", "LEARNING_EVIDENCE_INELIGIBLE",
           lambda: LearningData(session, train=forecasts[:1], evaluate=forecasts[1:]))
    boundaries.append(dict(name="withdraw_feedback", before=before, after=digest(session.snapshot())))
    final = session.snapshot()
    kw.update(expected_head=session.head, expected_loop_contract=session.loop_pin,
              expected_prediction_contract=digest(session.prediction_contract()))
    session.close()
    session = open_session(kw)
    check(session.snapshot() == final, "RESTART")
    independent = readback(args.output, inputs, final)
    session.close()
    forged = args.output/"forged-decision.sqlite3"
    shutil.copyfile(args.output/"science.sqlite3", forged)
    with sqlite3.connect(forged) as db:
        head, changed = ZERO, False
        for seq, raw in db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall():
            body = json.loads(raw)
            if body["kind"] == "constellation_learning_decision" and not changed:
                body["result"]["selected"], changed = True, True
            pin = digest(dict(seq=seq, previous=head, body=body))
            db.execute("UPDATE events SET previous=?,body=?,hash=? WHERE seq=?",
                       (head, canonical(body), pin, seq))
            head = pin
    try:
        Session(forged, source, inputs, args.parent/"model", **dict(kw, expected_head=head),
                candidate_root=candidates, device="cuda")
    except (ValueError, SenseError) as e:
        check(getattr(e, "code", str(e)) == "MNLM_CODATA_LEARNING_DECISION_REPLAY", "FORGED_DECISION")
    else:
        raise AssertionError("forged selection was accepted")
    controls.append(dict(name="forged_selection_rehashed_chain", code="LEARNING_DECISION_REPLAY"))
    phases["withdraw_restart_readback"] = time.perf_counter()-tick
    write(args.output/"snapshot.json", final)
    write(args.output/"run.json", dict(source=source.pin, parent=args.parent_pin, open=kw))
    wall = time.perf_counter()-started
    check(sum(phases.values())/wall >= .9, "PHASE_COVERAGE")
    report = dict(complete=True, acceptance="author-checked", controls=controls, boundaries=boundaries,
        readback=independent, phases_seconds=phases, wall_seconds=wall, serving=False,
        useful_capability_established=False, files={n: sha((args.output/n).read_bytes())
            for n in ("run.json", "snapshot.json", "cycle-1.json", "cycle-2.json")})
    write(args.output/"report.json", report)
    print(json.dumps(dict(complete=True, controls=len(controls), readback=independent,
                          wall_seconds=wall, report_sha256=sha((args.output/"report.json").read_bytes()))))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

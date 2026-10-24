"""Bounded physical audit of the developer changes (#12, PC-04/07/13/28).

Input is a persisted excerpt of real canonical triples, real calibration rows and
real Multi slots. Synthetic inputs occur only in named boundary controls. Separate
JSON reads establish expected claims; disk corruption tests operate on owned copies.
The output is author-checked evidence, never an acceptance certificate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
import contextlib
import io

import blake3
import numpy as np
import zstandard

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / p) for p in ("claims", "lens", "census")]
from assertion import ClaimError, parse_assertion, parse_verdict
from check_wikidata import SHARDS, SOURCE, build_index, check
from lenscommon import Paths, read_family, write_json, sealed_registry
import admission_state as state
import slotio
import vinfo

RESULTS = []


def record(name, ok, **detail):
    RESULTS.append({"boundary": name, "ok": bool(ok), **detail})
    print(name, bool(ok), flush=True)


def refusal(fn, code):
    try:
        fn()
    except Exception as exc:
        return code in str(exc)
    return False


def load_old(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def claims(proof, run, before):
    paths = Paths(run / "archive", run / "source", "proof-v1")
    paths.canon_version = "proof-v1"
    canon = paths.canon(SOURCE)
    canon.mkdir(parents=True)
    for f in proof.glob("wikidata5m*"):
        shutil.copyfile(f, canon / f.name)
    shutil.copyfile(proof / "manifest.jsonl", canon / "manifest.jsonl")
    records = []
    for sh in SHARDS:
        data = zstandard.ZstdDecompressor().decompress((canon / f"{sh}.jsonl.zst").read_bytes())
        records.extend((sh, json.loads(line)) for line in data.splitlines())
    initial = {f.name: state.digest(f) for f in canon.iterdir()}
    idx = build_index(paths, run / "cache")
    cold = []
    for sh, r in records:
        f = r["f"]
        a = {"subject": f["s"], "relation": f["p"], "object": f["o"], "polarity": "asserted",
             "epistemic_status": "claimed", "evidence": [{"source": SOURCE, "shard": sh, "row": r["row"]}]}
        verdict = check(parse_assertion(a), idx).to_json()
        cold.append(verdict)
        record(f"claims.truth.{sh}.{r['row']}", verdict["outcome"] == "supported")
        neg = check(parse_assertion({**a, "polarity": "negated"}), idx)
        wrong = check(parse_assertion({**a, "object": "Q999999999999"}), idx)
        qualified = check(parse_assertion({**a, "qualifiers": {"P580": "2020-01-01"}}), idx)
        record(f"claims.boundaries.{sh}.{r['row']}", neg.outcome == "contradicted" and wrong.outcome != "supported" and qualified.outcome == "unsupported_missing")
    write_json(run / "claims-cold.json", cold)
    warm = build_index(paths, run / "cache")
    replay = [check(parse_assertion(v["assertion"]), warm).to_json() for v in cold]
    write_json(run / "claims-warm.json", replay)
    record("claims.cold_warm_exact", cold == replay, real_records=len(records))
    a = cold[0]["assertion"]
    broken = {**a, "polarity": "negated", "evidence": [{"source": SOURCE, "shard": "missing", "row": 0}]}
    record("claims.broken_citation_precedes_truth", check(parse_assertion(broken), idx).outcome == "evidence_mismatch")
    # Corrupt a real cached row while leaving the source and cache manifest intact.
    p = run / "cache" / f"shard-{SHARDS[0]}-o.npy"
    orig = p.read_bytes()
    obj = np.load(p); obj[0] = 999999999; np.save(p, obj)
    record("cache.corruption_refused", refusal(lambda: build_index(paths, run / "cache"), "MNLM_CLAIM_CACHE_CORRUPT"))
    p.write_bytes(orig)
    if before:
        old = load_old(before / "before_check_wikidata.py", "old_claim_checker")
        old.build_index(paths, run / "before-cache")
        p = run / "before-cache" / f"shard-{SHARDS[0]}-o.npy"
        obj = np.load(p); obj[0] = 999999999; np.save(p, obj)
        oldidx = old.build_index(paths, run / "before-cache")
        fake = {**a, "object": "Q999999999"}
        false_verdict = old.check(parse_assertion(fake), oldidx).to_json()
        write_json(run / "before-cache-false-support.json", false_verdict)
        record("before.cache_false_support_reproduced", false_verdict["outcome"] == "supported")
    src = canon / f"{SHARDS[0]}.jsonl.zst"
    orig = src.read_bytes(); changed = bytearray(orig); changed[-2] ^= 1; src.write_bytes(changed)
    record("source.changed_bytes_refused", refusal(lambda: build_index(paths, run / "cache"), "MNLM_CLAIM_SOURCE_UNSEALED"))
    src.write_bytes(orig)
    record("source.before_after_unchanged", initial == {f.name: state.digest(f) for f in canon.iterdir()})
    bad = [({**a, "subject": "Q1\n"}, "MNLM_CLAIM_BAD_SUBJECT"),
           ({**a, "polarity": []}, "MNLM_CLAIM_BAD_TYPE"),
           ({**a, "epistemic_status": {}}, "MNLM_CLAIM_BAD_TYPE"),
           ({**a, "time": {"start": "2020-02-31", "end": None}}, "MNLM_CLAIM_BAD_TIME"),
           ({**a, "time": {"start": "2021-01-01", "end": "2020-01-01"}}, "MNLM_CLAIM_BAD_TIME"),
           ({**a, "evidence": [{"source": SOURCE, "shard": SHARDS[0], "row": -1}]}, "MNLM_CLAIM_BAD_TYPE")]
    record("schema.invalid_assertions_refused", all(refusal(lambda d=d: parse_assertion(d), code) for d, code in bad))
    record("schema.invalid_verdict_refused", refusal(lambda: parse_verdict({**cold[0], "cited": "fake"}), "MNLM_CLAIM_BAD_TYPE"))
    # Execute the actual CLI writer with a valid row followed by an invalid row.
    # The last accepted output must survive a mid-stream refusal byte for byte.
    import check_wikidata as cli
    batch, result = run / "batch.jsonl", run / "published-verdicts.jsonl"
    batch.write_text(json.dumps(a) + "\n" + json.dumps({**a, "subject": "Q1\n"}) + "\n", encoding="utf-8")
    result.write_bytes((json.dumps(cold[0]) + "\n").encode())
    old_output = result.read_bytes()
    argv = ["check", "--assertions", str(batch), "--out", str(result), "--cache", str(run / "cache")]
    with patch.object(cli, "default_paths", return_value=paths), patch.object(sys, "argv", argv):
        record("claims.partial_cli_publish_refused", refusal(cli.main, "MNLM_CLAIM_BAD_SUBJECT") and result.read_bytes() == old_output)
    record("claims.partial_output_cleaned", not list(run.glob("published-verdicts.jsonl.tmp.*")))
    for r in records:
        if r[1]["f"]["p"] in {"P19", "P21", "P569", "P570"}:
            f = r[1]["f"]
            alt = {"subject": f["s"], "relation": f["p"], "object": "Q999999999", "polarity": "asserted", "epistemic_status": "hypothesis"}
            record("claims.alternate_object_open_world", check(parse_assertion(alt), idx).outcome == "unsupported_missing")
            break


def multi(proof, run):
    path = proof.parent / "multi.npz"
    z = np.load(path)
    tokens, offsets, absent = z["tokens"], z["offsets"], z["absent"]
    before = state.digest(path)
    mapped = slotio.npz_member(path, "tokens")
    record("multi.native_member_exact", np.array_equal(tokens, mapped), atoms=len(absent), token_rows=len(tokens))
    x, lm = slotio.maxsim_landmarks(path, absent, "a2-wikidata-p54", device="cpu")
    ref = np.empty_like(x)
    for i in range(len(absent)):
        for j, row in enumerate(lm):
            s = tokens[offsets[i]:offsets[i + 1]].astype(np.float64) @ tokens[offsets[row]:offsets[row + 1]].astype(np.float64).T
            ref[i, j] = 0.5 * (s.max(1).mean() + s.max(0).mean())
    np.savez(run / "multi-reference.npz", actual=x, independent=ref, landmarks=lm)
    record("multi.native_maxsim_matches_independent", np.max(np.abs(x - ref)) < 1e-6, max_abs=float(np.max(np.abs(x - ref))), runtime="explicit CPU diagnostic; GPU acceptance not claimed")
    from multi_view import maxsim_pair, validate_multi
    # Tiny tiles force both axes through multiple iterations on the real tokens.
    got = maxsim_pair(tokens[:31], tokens[31:62], device="cpu", chunk=7)
    s = tokens[:31].astype(float) @ tokens[31:62].astype(float).T
    record("multi.both_axes_tiled", abs(got - 0.5 * (s.max(1).mean() + s.max(0).mean())) < 1e-6)
    record("multi.bad_offsets_refused", refusal(lambda: validate_multi(tokens, offsets + 1, absent), "MNLM_SLOT_LAYOUT"))
    record("multi.source_unchanged", before == state.digest(path))


def probes(proof, run):
    n = 600
    x = np.random.default_rng(42).uniform(-1, 1, (n, 1))
    y = (np.sin(80 * x[:, 0]) > 0).astype(int)
    blocks = {"panel": vinfo.block(x), "computed": y[:, None].astype(float)}
    rows = {k: np.arange(n) for k in blocks}
    conf = vinfo.confirm(blocks, rows, y, ["panel"], "computed", np.arange(n))
    write_json(run / "computed-control.json", {k: v for k, v in conf.items() if k not in ("common", "diffs")})
    record("probe.computation_is_not_certification", not vinfo.SHANNON_CONDITIONAL_CERTIFIED,
           probe_pass=vinfo.admissible(conf, 0.05, False), true_conditional_shannon_bits=0)
    prior, ce = vinfo.cv_cross_entropy(blocks["panel"], y, np.arange(n))
    subset = np.arange(n)[np.arange(n) % 7 != 0]
    blocks["partial"], rows["partial"] = blocks["computed"][subset], subset
    screen = vinfo.conditional_gain(blocks, rows, y, ["panel"], "partial", (np.arange(n), ce), np.arange(n))
    confirm = vinfo.confirm(blocks, rows, y, ["panel"], "partial", np.arange(n))
    record("probe.partial_coverage_paired", abs(screen["gain"] - confirm["seeds"][0]["gain"]) < 1e-12)
    record("probe.seed_identity", not np.array_equal(vinfo.sign_matrix(8, 4, 1), vinfo.sign_matrix(8, 4, 2)))
    record("probe.missing_seed_refused", not vinfo.admissible({"seeds": conf["seeds"][:-1]}, 0.05, False))
    invalid = [{**s, "lo": float("nan")} for s in conf["seeds"]]
    record("probe.nonfinite_confirmation_refused", not vinfo.admissible({"seeds": invalid}, 0.05, False))
    yy = np.repeat([0, 1, 2], 20)
    record("probe.missing_training_class_refused", refusal(lambda: vinfo.cv_cross_entropy(np.arange(60)[:, None], yy, yy, folds=3), "MNLM_PROBE_CLASS_SUPPORT"))
    failure = vinfo.gains_parallel({"a": np.arange(60)[:, None]}, {"a": np.arange(60)}, yy, [], ["a"], None, 1, yy)
    record("probe.failure_diagnostic_retained", failure["a"].get("error") == "MNLM_PROBE_REFUSED" and "MNLM_PROBE_INSUFFICIENT_SUPPORT" in failure["a"]["message"])
    cal = [json.loads(l) for l in (proof / "a2-wikidata-p54.jsonl").read_text(encoding="utf-8").splitlines()]
    z = np.load(proof / "record-vector.npz")
    keep = np.flatnonzero(~z["absent"])
    real_y = np.array([r["y"] for r in cal]); groups = np.unique([r["cluster"] for r in cal], return_inverse=True)[1]
    a, b = vinfo.cv_cross_entropy(vinfo.block(z["X"][keep]), real_y[keep], groups[keep])
    a2, b2 = vinfo.cv_cross_entropy(vinfo.block(z["X"][keep]), real_y[keep], groups[keep])
    np.savez(run / "real-probe.npz", prior=a, probe=b, rows=keep)
    record("probe.real_rows_replay_exact", np.array_equal(a, a2) and np.array_equal(b, b2), rows=len(keep), usable_bits=float((a - b).mean()))


def publication(proof, run):
    root = run / "state"
    (root / "cal").mkdir(parents=True)
    q = "a2-wikidata-p54"
    for suffix in (".jsonl", ".summary.json"):
        shutil.copyfile(proof / (q + suffix), root / "cal" / (q + suffix))
    reg = {"lenses": {}, "families": [q], "runtime": {}}
    body = (json.dumps(reg, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    from ledger import Ledger
    ledger = Ledger(run / "ledger.jsonl", "developer-audit")
    dg = blake3.blake3(body).hexdigest()
    entry = ledger.append("lenses.measure", "boundary-control", {"registry_blake3": dg})
    reg.update(ledger_seq=entry["seq"], ledger_head=entry["hash"], registry_blake3_before_ledger=dg)
    write_json(root / "registry.json", reg)
    cal, summary = read_family(root / "cal", q)
    rep = {"inputs": state.bindings(root, q, reg), "conditional_estimator": {"instrument": vinfo.INSTRUMENT},
           "admission": {"panel": [], "eligible": []}, "served": False, "shannon_conditional_certified": False,
           "sufficiency": {"sufficient": False}, "n": len(cal), "n_eff": None, "H_Y_bits": summary["H_Y_bits"]}
    (root / "assay").mkdir()
    write_json(root / "assay" / f"{q}.json", rep)
    admission = {"registry_blake3": dg, "digests": {q: state.digest(root / "assay" / f"{q}.json")}}
    e = ledger.append("lenses.admission", "boundary-control", admission)
    admission.update(ledger_seq=e["seq"], ledger_head=e["hash"])
    state.validate_ledger(run / "ledger.jsonl", reg, admission)
    record("publication.bound_generation_readback", state.load_reports(root, sealed_registry(root), admission)[q] == rep)
    p = root / "assay" / f"{q}.json"; before = p.read_bytes()
    p.write_text(json.dumps({**rep, "served": True}))
    record("publication.changed_report_refused", refusal(lambda: state.load_reports(root, reg, admission), "MNLM_ASSAY_RECORD_TORN"))
    p.write_bytes(before)
    bad = {**rep, "inputs": {**rep["inputs"], "instrument": "vinfo-v2"}}
    record("publication.old_instrument_refused", refusal(lambda: state.validate_report(root, q, bad, reg), "MNLM_ASSAY_STALE"))
    record("publication.probe_cannot_certify", refusal(lambda: state.validate_report(root, q, {**rep, "served": True}, reg), "MNLM_ASSAY_UNCERTIFIED"))
    record("publication.forged_ledger_refused", refusal(lambda: state.validate_ledger(run / "ledger.jsonl", reg, {**admission, "digests": {}}), "MNLM_ASSAY_UNSEALED"))
    p = root / "cal" / f"{q}.jsonl"; before = p.read_bytes(); p.write_bytes(before.replace(b'"y": ', b'"y": 9', 1))
    record("publication.changed_calibration_refused", refusal(lambda: read_family(root / "cal", q), "MNLM_LENS_CAL_TORN"))
    p.write_bytes(before)
    write_json(root / "assay" / "admission.json", admission)
    import panel as publish
    argv = ["panel", "--root", str(run), "--out", str(run), "--repo", str(run / "repo"), "--supersede"]
    old_panel = b'{"panel_blake3": "previous-generation"}\n'
    (root / "panel.json").write_bytes(old_panel)
    with patch.object(sys, "argv", argv), patch.object(publish, "lens_root", return_value=root), patch.object(publish, "validate_ledger", side_effect=lambda *a: state.validate_ledger(run / "ledger.jsonl", reg, admission)):
        with patch.object(publish, "ledger_append", side_effect=OSError("injected ledger write failure")):
            record("publication.failed_ledger_preserves_panel", refusal(publish.main, "injected ledger write failure") and (root / "panel.json").read_bytes() == old_panel)
        with patch.object(publish, "ledger_append", side_effect=lambda _p, _a, kind, subject, data: ledger.append(kind, subject, data)), contextlib.redirect_stdout(io.StringIO()):
            publish.main()
    new = state.load_panel(root, reg, admission, run / "ledger.jsonl")
    record("publication.atomic_panel_readback", not new["gates"]["families_served_ge3"] and new == json.loads((run / "repo" / "config" / "panel-p0-v1.json").read_text()))
    write_json(root / "panel.json", {**new, "host": "corrupted"})
    record("publication.panel_corruption_refused", refusal(lambda: state.load_panel(root, reg, admission, run / "ledger.jsonl"), "MNLM_PANEL_STALE"))
    from catalogue import CATALOGUE
    senses = {}
    for lens in CATALOGUE:
        if not lens.get("retired"):
            senses.setdefault(lens["perspective"], lens["name"])
    names = list(senses.values())[:5]
    qs = ["a2-wikidata-p31", "a2-wikidata-p17", "a2-wikidata-p106", "a2-wikidata-p19", "a2-wikidata-p54"]
    reps = {q: {"admission": {"panel": [name]}, "correlation": {"order": names, "matrix": np.eye(5).tolist()}} for q, name in zip(qs, names)}
    union = publish.union_panel(reps)["A2"]
    record("publication.question_union_not_served_meaning", union["candidate_count_gate"] and not union["served"])
    reps[qs[0]]["correlation"]["matrix"][0][1] = float("nan")
    record("publication.unreadable_correlation_refused", publish.max_corr(reps, qs, names[0], names[1]) is None)
    write_json(root / "registry-stale.json", reg)
    write_json(root / "registry.json", {**reg, "lenses": {"stale": {}}})
    from census_senses import load_registry
    import lenscommon
    record("census.no_snapshot_fallback", refusal(lambda: load_registry(root, lenscommon), "MNLM_REGISTRY_UNSEALED"))


def cache_and_lease(proof, run):
    import slot_binding as binding
    import lenscommon as lc
    root = run / "reuse"
    q = "a2-wikidata-p54"
    (root / "cal").mkdir(parents=True)
    shutil.copyfile(proof / (q + ".jsonl"), root / "cal" / (q + ".jsonl"))
    slot = root / "slots" / q / "record-vector.npz"
    slot.parent.mkdir(parents=True)
    shutil.copyfile(proof / "record-vector.npz", slot)
    contract = {"name": "boundary-test", "inputs": "fields", "weights_sha256": "fixed-test-identity"}
    record("cache.legacy_slot_remeasured", binding.reuse(slot, contract) is None)
    meta = slotio.slot_digest(slot)
    binding.seal(slot, contract, meta)
    record("cache.bound_slot_reused", binding.reuse(slot, contract)["digest"] == meta["digest"])
    record("cache.changed_lens_remeasured", binding.reuse(slot, {**contract, "weights_sha256": "changed"}) is None)
    record("cache.context_dependencies_not_assumed", binding.reuse(slot, {**contract, "inputs": "context"}) is None)
    cal = root / "cal" / (q + ".jsonl"); original = cal.read_bytes(); cal.write_bytes(original + b"\n")
    record("cache.changed_calibration_remeasured", binding.reuse(slot, contract) is None)
    cal.write_bytes(original)
    with np.load(slot) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["X"][0, 0] += 1
    np.savez(slot, **arrays)
    record("cache.changed_slot_refused", refusal(lambda: binding.reuse(slot, contract), "MNLM_SLOT_STALE"))
    # No GPU work: a separate, owned lease path tests process exclusion/cleanup.
    with patch.object(lc, "LEASE", run / "lease" / "gpu"):
        lc.write_json(lc.LEASE, {"pid": os.getpid()})
        original = lc.LEASE.read_bytes()
        record("lease.live_pid_refused", refusal(lambda: lc.take_lease("12", "boundary"), "MNLM_GPU_LEASE_HELD") and lc.LEASE.read_bytes() == original)
        lc.LEASE.unlink()
        def job():
            lc.take_lease("12", "boundary-no-gpu")
            code = "import sys;from pathlib import Path;sys.path[:0]=sys.argv[2:];import lenscommon as lc;lc.LEASE=Path(sys.argv[1]);sys.exit(lc.run_cli(lambda:lc.take_lease('12','boundary-no-gpu')))"
            child = subprocess.run([sys.executable, "-I", "-c", code, str(lc.LEASE), str(REPO / "scripts" / "lens"), str(Path(blake3.__file__).resolve().parents[1])], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record("lease.concurrent_process_refused", child.returncode == 2 and "MNLM_GPU_LEASE_HELD" in child.stderr)
            raise lc.LensError("MNLM_AUDIT_INJECTED_FAILURE", "owned boundary control")
        with contextlib.redirect_stderr(io.StringIO()):
            result = lc.run_cli(job)
        record("lease.failed_job_released", result == 2 and not lc.LEASE.exists() and lc._LEASE_LOCK is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--before", type=Path)
    args = ap.parse_args()
    run = args.out / time.strftime("readback-%Y%m%dT%H%M%SZ", time.gmtime())
    run.mkdir(parents=True)
    try:
        claims(args.proof, run, args.before)
        multi(args.proof, run)
        probes(args.proof, run)
        publication(args.proof, run)
        cache_and_lease(args.proof, run)
    except BaseException as exc:
        record("unexpected_failure", False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sources = [p for d in ("claims", "lens", "audit") for p in (REPO / "scripts" / d).glob("*.py")]
        report = {"acceptance": "author-checked only", "checks": RESULTS, "passed": sum(r["ok"] for r in RESULTS), "total": len(RESULTS),
                  "code_blake3": {str(p.relative_to(REPO)): state.digest(p) for p in sources},
                  "inputs_blake3": {p.name: state.digest(p) for p in args.proof.iterdir() if p.is_file()},
                  "multi_blake3": state.digest(args.proof.parent / "multi.npz"), "runtime": sys.version}
        write_json(run / "report.json", report)
        write_json(run / "manifest.json", {str(p.relative_to(run)): state.digest(p) for p in run.rglob("*") if p.is_file() and p != run / "manifest.json"})
        print(json.dumps({"report": str(run / "report.json"), "passed": report["passed"], "total": report["total"]}))
    return 0 if all(r["ok"] for r in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

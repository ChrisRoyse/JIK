"""Real retained text -> exact tokenizer windows -> native publication (#16).

Five real observations, four cached instrument kinds, no downloads/admission.
Independent tokenizer/native readback plus altered-provenance/cache boundaries.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical, digest, CHANNELS
from production.input_plan import sha, code_hashes, environment, POLICY_SHA256
from production.stream import run, paths
from production.stream_slots import audit_checkpoint
from production.session import Session
from learned import snapshot_dir, weights_sha256
from learned_ucr import visible_batch
from catalogue import by_name
from universal import ChannelReader
from lenscommon import LensError, take_lease, release_lease, LEASE
from slotio import slot_digest
import slot_binding
import text_windows

NAMES = ("ucr-all-minilm-l6-v2", "ucr-opensearch-sparse-doc-v2", "ucr-answerai-colbert-small", "ucr-potion-base-8m")


def write(path, obj):
    with path.open("xb") as f:
        f.write(canonical(obj) + b"\n")


def prepare(store, out):
    db = sqlite3.connect(Path(store).resolve().as_uri() + "?mode=ro", uri=True)
    longs, short, absent = [], None, None
    try:
        generation = db.execute("SELECT hash FROM generations ORDER BY seq DESC LIMIT 1").fetchone()[0]
        for raw, in db.execute("SELECT body FROM records ORDER BY atom"):
            r = json.loads(raw)
            items = r["channels"]["surface"]["items"]
            t = items[0]["text"] if items else None
            if r["locator"]["source"] == "swe-agent-trajectories":
                if 4000 <= len(t) <= 8000 and len(longs) < 3:
                    longs.append(r)
                elif short is None and 10 < len(t) < 400:
                    short = r
            if absent is None and r["locator"]["source"] == "web-audio-mpj" and t is None:
                absent = r
    finally:
        db.close()
    assert len(longs) == 3 and short is not None and absent is not None
    selected = longs + [short, absent]
    write(out / "source-records.json", selected)
    rows = [{**r["locator"], "atom": r["atom_id"], "i": i,
             "text": "\n".join(s["text"] for s in r["channels"]["surface"]["items"])} for i, r in enumerate(selected)]
    row_root = out / "rows"
    row_root.mkdir()
    policy = {"schema": "ucr-view-v1", "question": "window-boundary", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "Tokenizer/native-window boundary only, no anchor fitting or admission."}
    policy_path = out / "policies.json"
    write(policy_path, {"window-boundary": policy})
    plan = {"schema": "tier-input-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "generation": generation, "policies_sha256": sha(policy_path),
            "code_sha256": code_hashes(), "environment": environment(), "batch_rows": 2,
            "slot_budget_bytes": 16 << 20, "metadata_budget_bytes": 1 << 20,
            "blob_budget_bytes": 256 << 20, "work_budget_bytes": 1 << 30, "instruments": {}, "shards": {}}
    for i, name in enumerate(NAMES):
        spec = by_name(name)
        snap = snapshot_dir(spec["repo"])
        tier = "multi" if spec["kind"] == "multi" else "panel" if spec["kind"] == "sparse" else "cheap-semantic"
        plan["instruments"][name] = {"spec": spec, "weights_sha256": weights_sha256(snap)[0],
                                     "revision": snap.name, "tier": tier,
                                     "slot_budget_bytes": (2 if spec["kind"] == "multi" else 1) << 20}
        key = f"window-{i}"
        rp = row_root / (key + ".jsonl")
        rp.write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
        plan["shards"][key] = {"tier": tier, "family": "window-boundary", "n": len(rows),
                                "instruments": [name], "rows_sha256": sha(rp), "rows_bytes": rp.stat().st_size}
    write(out / "plan.json", plan)
    return plan, policy_path, row_root, rows


def reference_tokens(lens, trace, text):
    """Separate tokenization call, no text_windows.build or validate."""
    assert trace["surface_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    prefixed = lens.spec.get("prefix", "") + text
    if lens.kind == "static":
        e = lens.tok.encode(prefixed, add_special_tokens=False)
        reference = [(e.ids, e.offsets, e.special_tokens_mask)]
    else:
        e = lens.tok(prefixed, truncation=True, max_length=lens.spec["max_len"],
                     return_overflowing_tokens=True, return_offsets_mapping=True,
                     return_special_tokens_mask=True, padding=False)
        reference = list(zip(e["input_ids"], e["offset_mapping"], e["special_tokens_mask"]))
    assert len(reference) == len(trace["windows"])
    for window, (ids, offsets, specials) in zip(trace["windows"], reference):
        use = [i for i, mask in enumerate(window["attention_mask"]) if mask]
        assert [window["input_ids"][i] for i in use] == ids
        assert [window["offset_mapping"][i] for i in use] == [list(p) for p in offsets]
        assert [window["special_tokens_mask"][i] for i in use] == specials
    return len(reference), sum(len(r[0]) for r in reference)


def native_equal(stored, arrays, start, stop):
    for key in ("absent", "windows"):
        assert np.array_equal(stored[key][start:stop], arrays[key])
    if "X" in stored:
        assert np.array_equal(stored["X"][start:stop], arrays["X"])
    else:
        off = "offsets" if "tokens" in stored else "indptr"
        lo, hi = int(stored[off][start]), int(stored[off][stop])
        assert np.array_equal(stored[off][start:stop + 1] - lo, arrays[off])
        for key in (("tokens",) if off == "offsets" else ("data", "indices")):
            assert np.array_equal(stored[key][lo:hi], arrays[key])


def replay_native(plan, store, policies, rows, out, check):
    reader = ChannelReader(store, policies, generation=plan["generation"])
    take_lease("16", "text-window-independent-readback")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    totals = []
    try:
        for i, name in enumerate(NAMES):
            session = Session(plan["instruments"][name], reader, plan["code_sha256"])
            try:
                slot, _ = paths(out / "job", f"window-{i}", name)
                provenance = json.loads(slot.with_suffix(".views.json").read_bytes())
                with np.load(slot, allow_pickle=False) as z:
                    stored = {k: z[k].copy() for k in z.files}
                policy = reader.policy("window-boundary")
                for start in range(0, len(rows), 2):
                    chosen = rows[start:start + 2]
                    texts, _ = visible_batch(reader, policy, chosen)
                    arrays, bindings, details = session.batch(chosen, policy)
                    native_equal(stored, arrays, start, start + len(chosen))
                    assert details == provenance["details"][start:start + len(chosen)]
                    for j, text in enumerate(texts):
                        trace = provenance["details"][start + j]["text_windows"]
                        if text is None:
                            assert trace["windows"] == [] and trace["surface_sha256"] is None
                        else:
                            windows, tokens = reference_tokens(session.lens, trace, text)
                            totals.append({"instrument": name, "atom": chosen[j]["atom"], "chars": len(text),
                                           "windows": windows, "tokens": tokens})
                check(name + "_all_rows_native_and_tokenizer_replay", True)
                if session.kind != "static":
                    assert all(r["windows"] > 1 for r in totals[-4:-1])
                # Prefix/Unicode are a tokenizer boundary, not new source records.
                if i == 0:
                    prefix, text = "query: ", "café Ω e\u0301\n  alpha [MASK] 🙂"
                    e = session.lens.tok([prefix + text], return_offsets_mapping=True,
                                         return_special_tokens_mask=True, return_tensors="pt", padding=True)
                    trace = text_windows.build([text], prefix, e, [0], "dense")[0]
                    write(out / "unicode-prefix-boundary.json", {"text": text, "trace": trace, "artificial_boundary_only": True})
                    text_windows.validate(trace, "dense", 1, text=text)
                    check("prefix_unicode_normalization_gaps_retained", bool(trace["unmapped_source_spans"]) and
                          trace["windows"][0]["source_char_span"][1] <= len(text))
                    try:
                        session.lens._tokenize(["window " * (session.spec["max_len"] * 8 + 1)])
                    except LensError as exc:
                        check("window_budget_still_refuses_before_forward", exc.code == "MNLM_LENS_WINDOW_BUDGET")
                    else:
                        check("window_budget_still_refuses_before_forward", False)
            finally:
                session.close()
    finally:
        reader.close()
        release_lease()
    write(out / "tokenizer-readback.json", totals)


def negative_boundaries(plan, out, rows, check):
    name = NAMES[0]
    slot, cp = paths(out / "job", "window-0", name)
    provenance = json.loads(slot.with_suffix(".views.json").read_bytes())
    trace = provenance["details"][0]["text_windows"]
    count = len(trace["windows"])
    def refuse(label, action, code="MNLM_LENS_WINDOW_PROVENANCE"):
        before = sha(slot), sha(cp), sha(slot.with_suffix(".views.json"))
        try:
            action()
        except LensError as exc:
            check(label, exc.code == code, code=exc.code)
        else:
            check(label, False, code="unexpected success")
        assert before == (sha(slot), sha(cp), sha(slot.with_suffix(".views.json")))
    refuse("count_only_record_refuses", lambda: text_windows.validate({}, "dense", count))
    bad = copy.deepcopy(trace)
    bad["windows"][0]["offset_mapping"][1][1] = bad["surface_chars"] + 100
    refuse("out_of_source_offset_refuses", lambda: text_windows.validate(bad, "dense", count))
    refuse("missing_window_refuses", lambda: text_windows.validate(trace, "dense", count + 1))
    refuse("different_surface_refuses", lambda: text_windows.validate(trace, "dense", count, text="different source"))
    bad = copy.deepcopy(trace)
    bad["unmapped_source_spans"] = []
    refuse("hidden_normalization_gaps_refuse", lambda: text_windows.validate(bad, "dense", count))
    bad = copy.deepcopy(trace)
    bad["windows"][0]["input_ids"].pop()
    refuse("missing_token_refuses", lambda: text_windows.validate(bad, "dense", count))
    multi, _ = paths(out / "job", "window-2", NAMES[2])
    mt = json.loads(multi.with_suffix(".views.json").read_bytes())["details"][0]["text_windows"]
    mt["windows"][0]["native_token_span"][1] += 1
    refuse("multi_alignment_refuses", lambda: text_windows.validate(mt, "multi", len(mt["windows"])))
    # Exercise the actual calibration-cache consumer on real retained rows/slots.
    cache = out / "calibration"
    dest = cache / "slots" / "window-boundary" / "lens.npz"
    dest.parent.mkdir(parents=True)
    (cache / "cal").mkdir()
    raw = b"".join(canonical(r) + b"\n" for r in rows)
    (cache / "cal" / "window-boundary.jsonl").write_bytes(raw)
    shutil.copyfile(slot, dest)
    shutil.copyfile(slot.with_suffix(".views.json"), dest.with_suffix(".views.json"))
    contract = json.loads(cp.read_bytes())["slot"]["contract"]
    record = slot_digest(dest)
    slot_binding.seal(dest, contract, record)
    check("bound_window_cache_reuses", slot_binding.reuse(dest, contract)["reused"])
    mutated = copy.deepcopy(provenance)
    mutated["details"][0]["text_windows"]["windows"][0]["input_ids"][1] += 1
    dest.with_suffix(".views.json").write_bytes(canonical(mutated))
    refuse("changed_window_cache_refuses", lambda: slot_binding.reuse(dest, contract), "MNLM_SLOT_STALE")
    write(out / "negative-boundaries.json", {"source_slot_sha256": sha(slot), "source_views_sha256": sha(slot.with_suffix(".views.json"))})


def calibration_paths(plan, store, policies, rows, out, check):
    """Both real calibration writers must persist the same native/window state."""
    from learned_ucr import measure_learned_ucr
    from measure import measure_learned
    take_lease("16", "text-window-calibration-readback")
    try:
        for mode in ("ucr", "legacy"):
            cache = out / (mode + "-measurement")
            sdir = cache / "slots"
            (sdir / "window-boundary").mkdir(parents=True)
            (cache / "cal").mkdir()
            # Legacy diagnostics have no absence contract; only observed texts.
            selected = rows if mode == "ucr" else rows[:4]
            (cache / "cal" / "window-boundary.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in selected))
            for i, name in enumerate(NAMES):
                spec = by_name(name if mode == "ucr" else name.removeprefix("ucr-"))
                pins = {spec["name"]: plan["instruments"][name]["weights_sha256"]}
                if mode == "ucr":
                    entry = measure_learned_ucr(spec, {"window-boundary": selected}, sdir, pins, store, policies, 2)
                else:
                    entry = measure_learned(spec, {"window-boundary": selected}, sdir, pins, False, 2)
                write(cache / (spec["name"] + ".json"), entry)
                assert entry["probe"]["byte_identical"] and entry["probe"]["window_provenance_byte_identical"]
                assert entry["probe"]["batch_invariance_byte_identical"] and entry["probe"]["batch_invariance_rows"] < len(selected)
                dest = Path(entry["families"]["window-boundary"]["slot"])
                original, _ = paths(out / "job", f"window-{i}", name)
                with np.load(original, allow_pickle=False) as source, np.load(dest, allow_pickle=False) as produced:
                    native_equal(source, produced, 0, len(selected))
                details = json.loads(dest.with_suffix(".views.json").read_bytes())["details"]
                reference = json.loads(original.with_suffix(".views.json").read_bytes())["details"][:len(selected)]
                assert [d["text_windows"] for d in details] == [d["text_windows"] for d in reference]
                assert slot_binding.reuse(dest, entry["contract"])["reused"]
                check(mode + "_calibration_" + spec["name"], True, rows=len(selected))
        # Corrupt only a retained probe trace, leaving native values untouched.
        slot, _ = paths(out / "job", "window-0", NAMES[0])
        traces = [d["text_windows"] for d in json.loads(slot.with_suffix(".views.json").read_bytes())["details"]]
        windows = [len(t["windows"]) for t in traces]
        before = text_windows.probe_digest(windows, traces)
        for field in ("windows", "tokens"):
            altered_windows, altered_traces = copy.deepcopy(windows), copy.deepcopy(traces)
            if field == "windows":
                altered_windows[0] += 1
            else:
                altered_traces[0]["windows"][0]["input_ids"][1] += 1
            after = text_windows.probe_digest(altered_windows, altered_traces)
            try:
                text_windows.check_probe(before, after, NAMES[0])
            except LensError as exc:
                check("probe_" + field + "_drift_refuses", exc.code == "MNLM_LENS_NONDETERMINISTIC",
                      before=before, after=after, native_slot_sha256=sha(slot), code=exc.code)
            else:
                check("probe_" + field + "_drift_refuses", False)
    finally:
        release_lease()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    before, start = sha(a.store), time.monotonic()
    phases, checks = {}, []
    def check(name, ok, **facts):
        checks.append({"name": name, "pass": bool(ok), **facts})
        with (a.out / "checks.jsonl").open("ab") as f:
            f.write(canonical(checks[-1]) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(checks[-1]), flush=True)
        assert ok, name
    plan, policies, root, rows = prepare(a.store, a.out)
    phases["prepare"] = time.monotonic() - start
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    write(a.out / "resident-before.json", {"nvidia_smi": resident})
    at = time.monotonic()
    receipt = run(plan, a.store, policies, root, a.out / "job")
    write(a.out / "receipt.json", receipt)
    check("real_four_instrument_publication", receipt["checkpoints_computed"] == 4 and receipt["atom_instrument_pairs"] == 20)
    phases["measurement"] = time.monotonic() - at
    at = time.monotonic()
    replay_native(plan, a.store, policies, rows, a.out, check)
    phases["independent_replay"] = time.monotonic() - at
    at = time.monotonic()
    calibration_paths(plan, a.store, policies, rows, a.out, check)
    phases["calibration_writers"] = time.monotonic() - at
    at = time.monotonic()
    negative_boundaries(plan, a.out, rows, check)
    cached = run(plan, a.store, policies, root, a.out / "job")
    write(a.out / "cached.receipt.json", cached)
    check("cached_checkpoint_no_model_or_mutation", cached["checkpoints_reused"] == 4 and cached["loaded_instruments"] == []
          and receipt["after"] == cached["after"])
    check("source_unchanged_and_lease_released", sha(a.store) == before and not LEASE.exists())
    phases["negative_controls_and_cached_read"] = time.monotonic() - at
    write(a.out / "report.json", {"acceptance": "author-checked only", "stage4_accepted": False, "checks": checks,
                                 "source_before": before, "source_after": sha(a.store), "phases_seconds": phases,
                                 "wall_seconds": time.monotonic() - start})
    files = sorted(p for p in a.out.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm")))
    write(a.out / "SHA256.json", {str(p.relative_to(a.out)): sha(p) for p in files})
    print(json.dumps({"passed": len(checks), "files": len(files), "report_sha256": sha(a.out / "report.json")}))


if __name__ == "__main__":
    main()

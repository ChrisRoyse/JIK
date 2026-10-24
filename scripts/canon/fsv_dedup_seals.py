"""Physical dedup acceptance proof (#21), real retained identity sections.

Linux builds with the actual suffix executable. Other hosts can cold-read that
accepted tree and exercise the same consumer/refusal boundaries (--accepted).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Paths, iter_canon_docs, sealed_shards
from dedup import build_population, exact_layer, near_layer, unique_layer, load_population
from artifact_seals import Audit
from sa_dedup import build_corpus, build_table, self_similar, map_spans, read_ranges
from atomise import plan_jobs, atomise_shard
from minhash import accept_signature


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def run(args):
    start = time.perf_counter()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    p = Paths(root / "root", root / "published", "seal-proof")
    checks, phases = [], {}
    def check(name, ok, **detail):
        checks.append({"boundary": name, "pass": bool(ok), **detail})
    if args.accepted:
        shutil.copytree(args.accepted, p.out)
    else:
        shutil.copytree(args.canonical, p.canon("identity-v1"))
        rows = sealed_shards(p.canon("identity-v1"))
        meta = build_population(p, ["identity-v1"])
        cmd = [sys.executable, "-I", str(Path(__file__).with_name("minhash.py")),
               "--root", str(p.root), "--out", str(p.out), "--version", p.version,
               "--source", "identity-v1", "--workers", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (root / "minhash.log").write_text(proc.stdout + proc.stderr)
        check("actual_minhash_cli", proc.returncode == 0, docs=meta["docs"], shards=len(rows))
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        exact_layer(p)
        near_layer(p)
        unique_layer(p)
        build_corpus(p)
        build_table(p, args.tool, 1000000, 1, 1)
        self_similar(p, args.tool, 200, 1)
        map_spans(p, 200)
    phases["real_build_or_transfer"] = time.perf_counter() - start
    t = time.perf_counter()
    before = inventory(p.out)
    Audit(p).require("spans")
    pop, meta = load_population(p)
    text = [r["text"].encode() for r in iter_canon_docs(p.canon("identity-v1"), "profile")]
    check("all_population_bytes_rehashed", len(text) == meta["docs"] and all(
        len(b) == int(pop["n"][i]) and blake3.blake3(b).digest(length=16) == bytes(pop["b3"][i])
        for i, b in enumerate(text)))
    unique = np.load(p.dedup / "unique.npy")
    exact = np.load(p.dedup / "exact.removed.npy")
    near = np.load(p.dedup / "near.removed.npy")
    check("independent_survivor_recount", unique.tolist() == sorted(set(range(len(text))) - set(exact) - set(near)))
    corpus = (p.sa / "corpus.bin").read_bytes()
    docs, offsets = np.load(p.sa / "docs.npy"), np.load(p.sa / "offsets.npy")
    check("every_corpus_byte_and_offset", corpus == b"".join(text[d] + b"\0" for d in docs)
          and offsets.tolist() == [0] + list(np.cumsum([len(text[d]) + 1 for d in docs])))
    spans = np.load(p.dedup / "spans.npy")
    check("every_cut_equals_retained_copy", all(text[d][s:e] == text[k][ks:ks+e-s] and e-s >= 200
          for d, s, e, k, ks in spans), spans=len(spans))
    build_corpus(p)
    if args.tool:
        build_table(p, args.tool, 1000000, 1, 1)
    check("sealed_cache_replay_read_only", inventory(p.out) == before)
    jobs, stats = plan_jobs(p, None)
    results = [atomise_shard(job) for job in jobs]
    check("actual_stage2_consumer", len(results) == 1 and sum(r["docs"] for r in results) == len(unique),
          atoms=sum(r["atoms"] for r in results))
    phases["physical_readback_and_consumer"] = time.perf_counter() - t
    t = time.perf_counter()

    def refusal(name, file, action, code, missing=False, transform=None):
        raw = file.read_bytes()
        if missing:
            file.unlink()
        else:
            file.write_bytes(transform(raw) if transform else raw[:-1] + bytes([raw[-1] ^ 1]))
        before_call = inventory(p.out)
        error = None
        try:
            action()
        except (Exception, SystemExit) as exc:
            error = str(exc)
        check(name, error is not None and code in error and inventory(p.out) == before_call, error=error)
        file.write_bytes(raw)

    refusal("missing_population_seal", p.dedup / "population.seal.json", lambda: load_population(p),
            "MNLM_DEDUP_UNSEALED", missing=True)
    refusal("same_size_population_corruption", p.dedup / "population.npz", lambda: exact_layer(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("population_metadata_corruption", p.dedup / "population.json", lambda: load_population(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("stale_exact_map_refuses_near", p.dedup / "exact.removed.npy", lambda: near_layer(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("stale_near_map_refuses_unique", p.dedup / "near.kept_of.npy", lambda: unique_layer(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("stale_unique_map_refuses_corpus", p.dedup / "unique.npy", lambda: build_corpus(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("same_size_corpus_cache_corruption", p.sa / "corpus.bin", lambda: build_corpus(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("same_size_offsets_corruption", p.sa / "offsets.npy", lambda: build_corpus(p), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("table_bytes_corruption", p.sa / "corpus.bin.table.bin", lambda: Audit(p).require("table"), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("range_bytes_corruption", p.sa / "ranges.txt", lambda: map_spans(p, 200), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("span_corruption_refuses_stage2", p.dedup / "spans.npy", lambda: plan_jobs(p, None), "MNLM_DEDUP_BYTES_MISMATCH")
    refusal("incomplete_spans_refuse_stage2", p.dedup / "spans.seal.json", lambda: plan_jobs(p, None), "MNLM_DEDUP_UNSEALED", missing=True)
    sh = meta["shards"][0]
    refusal("signature_corruption", p.sig("identity-v1") / "profile.sig.npy",
            lambda: accept_signature(p.sig("identity-v1"), "profile", sh["canon_blake3"]), "MNLM_DEDUP_BYTES_MISMATCH")
    # A valid new parent with the same row count still invalidates its descendants.
    original_meta = (p.dedup / "population.json").read_bytes()
    original_seal = (p.dedup / "population.seal.json").read_bytes()
    changed = json.loads(original_meta); changed["ts"] = "different generation, same rows"
    (p.dedup / "population.json").write_text(json.dumps(changed))
    Audit(p).finish("population", {})
    error = None
    try:
        Audit(p).require("exact")
    except SystemExit as exc:
        error = str(exc)
    check("same_count_new_parent_rejects_old_maps", error and "MNLM_DEDUP_PARENT_MISMATCH" in error, error=error)
    (p.dedup / "population.json").write_bytes(original_meta)
    (p.dedup / "population.seal.json").write_bytes(original_seal)
    error = None
    try:
        map_spans(p, 201)
    except SystemExit as exc:
        error = str(exc)
    check("different_threshold_refuses", error and "MNLM_SA_THRESHOLD_MISMATCH" in error, error=error)
    malformed = root / "malformed-ranges.txt"; malformed.write_text("garbage\n")
    error = None
    try:
        read_ranges(malformed)
    except SystemExit as exc:
        error = str(exc)
    check("missing_collect_marker_refuses", error and "MNLM_SA_RANGES_INVALID" in error, error=error)
    Audit(p).require("spans")
    check("source_canonical_unchanged", all(sha(p.out / rel) == h for rel, h in before.items() if rel.startswith("canon/")))
    phases["negative_controls_and_restoration"] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    report = {"acceptance": "author-checked only", "full_scope": False, "scientific_admission": False,
              "checks": checks, "passed": sum(bool(c["pass"]) for c in checks),
              "failed": sum(not c["pass"] for c in checks), "seconds": elapsed, "phases": phases,
              "phase_coverage": sum(phases.values()) / elapsed, "docs": meta["docs"],
              "canonical_bytes": meta["canonical_bytes"], "source": str(args.canonical or args.accepted)}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "SHA256.json").write_text(json.dumps(inventory(root), sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "seconds", "phase_coverage")}))
    return bool(report["failed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", type=Path)
    ap.add_argument("--accepted", type=Path)
    ap.add_argument("--tool", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    sys.exit(run(ap.parse_args()))

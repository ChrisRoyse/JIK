"""Physical Stage 1 integrity proof using the retained real identity profile (#19)."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unicodedata

import blake3
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonicalise import process_plan
from common import sealed_shards


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(raw, out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    source, version = "identity-v1", "integrity-proof"
    sandbox = out / "real-copy"
    sandbox.mkdir()
    for name in ("profile.md", "receipts.jsonl"):
        shutil.copyfile(raw / name, sandbox / name)
    receipts = {r["path"]: r for r in map(json.loads, (raw / "receipts.jsonl").read_text().splitlines())}
    expected = receipts["profile.md"]["sha256"]
    assert sha(sandbox / "profile.md") == expected
    checks, phases = [], {}

    def check(name, ok, **detail):
        checks.append({"boundary": name, "pass": bool(ok), **detail})

    def snapshot(root):
        return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}

    def cli(label):
        cmd = [sys.executable, "-I", str(Path(__file__).with_name("canonicalise.py")),
               "--root", str(out / "root"), "--out", str(out / "published"), "--version", version,
               "--source", source, "--raw-dir", str(sandbox), "--workers", "1", "--shards", "profile"]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (out / (label + ".log")).write_text(p.stdout + p.stderr, encoding="utf-8")
        return p

    result = cli("build")
    check("real_cli_build", result.returncode == 0, returncode=result.returncode)
    cdir = out / "published" / "canon" / version / source
    manifest = [json.loads(x) for x in (cdir / "manifest.jsonl").read_text().splitlines()][-1]
    with (cdir / "profile.jsonl.zst").open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z:
        canonical_rows = [json.loads(line) for line in io.TextIOWrapper(z, encoding="utf-8")]
    text = (sandbox / "profile.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    # Independent section/normalization restatement against the persisted raw.
    import re
    sections, lines = [], []
    for line in text.split("\n"):
        if re.match(r"^#{1,3} ", line) and lines:
            sections.append("\n".join(lines))
            lines = []
        lines.append(line)
    if lines:
        sections.append("\n".join(lines))
    normalized = ["\n".join(line.rstrip() for line in unicodedata.normalize("NFC", s).split("\n")).strip("\n")
                  for s in sections]
    check("raw_to_every_canonical_section", [r["text"] for r in canonical_rows] == normalized,
          raw_bytes=(sandbox / "profile.md").stat().st_size, sections=len(canonical_rows))
    hashes = np.fromfile(cdir / "profile.b3.bin", dtype=np.uint8).reshape(-1, 16)
    rows = np.load(cdir / "profile.rows.npy", allow_pickle=False)
    sizes = np.load(cdir / "profile.n.npy", allow_pickle=False)
    check("all_physical_sidecars_match", len(rows) == len(canonical_rows) == manifest["docs"]
          and all(int(rows[i]) == r["row"] and int(sizes[i]) == len(r["text"].encode())
                  and bytes(hashes[i]) == blake3.blake3(r["text"].encode()).digest(length=16)
                  for i, r in enumerate(canonical_rows)))
    before = snapshot(cdir)
    replay = cli("reuse")
    check("cached_cli_reuse_preserves_canonical_bytes", replay.returncode == 0 and before == snapshot(cdir))
    from common import Paths
    from dedup import build_population
    paths = Paths(out / "root", out / "published", version)
    population = build_population(paths, [source])
    with np.load(paths.dedup / "population.npz", allow_pickle=False) as p:
        check("real_population_consumer_counts_all_rows", len(p["row"]) == len(canonical_rows)
              and p["row"].tolist() == [r["row"] for r in canonical_rows])
    phases["build_replay_and_independent_readback"] = time.perf_counter() - start
    t = time.perf_counter()
    plan = {"shard": "profile", "files": [sandbox / "profile.md"], "sha256": [expected], "kind": "prose"}
    excluded_dest = out / "excluded-output"
    result = process_plan(("cuyahoga-eighth-district", plan, excluded_dest, sandbox))
    check("direct_extraction_excluded_source_refuses", result[0].get("error", {}).get("code") == "MNLM_SOURCE_EXCLUDED"
          and not excluded_dest.exists())
    for name, changed in (("missing_hash", {**plan, "sha256": []}),
                          ("extra_hash", {**plan, "sha256": [expected, expected]})):
        dest = out / name
        results = process_plan((source, changed, dest, sandbox))
        code = next((r.get("error", {}).get("code") for r in results if r.get("error")), None)
        check(name, code == "MNLM_CANON_PLAN" and not dest.exists(), code=code)
    original = (sandbox / "profile.md").read_bytes()
    flipped = bytearray(original)
    pos = next(i for i, b in enumerate(flipped) if 65 <= b <= 89)
    flipped[pos] += 1
    (sandbox / "profile.md").write_bytes(flipped)
    dest = out / "raw-corrupt-output"
    results = process_plan((source, plan, dest, sandbox))
    code = next((r.get("error", {}).get("code") for r in results if r.get("error")), None)
    check("same_size_raw_corruption_refused_before_extraction", code == "MNLM_CANON_RAW_DIGEST_MISMATCH"
          and not dest.exists(), code=code)
    before = snapshot(cdir)
    retry = cli("corrupt-raw-reuse")
    check("cached_reuse_checks_current_raw_receipt", retry.returncode != 0
          and "MNLM_CANON_RAW_DIGEST_MISMATCH" in retry.stdout + retry.stderr
          and before == snapshot(cdir), returncode=retry.returncode)
    (sandbox / "profile.md").write_bytes(original)
    phases["raw_refusal_controls"] = time.perf_counter() - t
    t = time.perf_counter()
    for name in ("canon_flip", "hash_sidecar_flip", "row_sidecar_flip", "size_sidecar_flip",
                 "missing_canon", "latest_error", "legacy_valid", "legacy_sidecar_flip"):
        target = out / name
        shutil.copytree(cdir, target)
        m = copy.deepcopy(manifest)
        if name.startswith("legacy"):
            m.pop("sidecars", None)
            m.pop("plan_seal_required", None)
            (target / "manifest.jsonl").write_text(json.dumps(m) + "\n", encoding="utf-8")
        filename = {"canon_flip": "profile.jsonl.zst", "hash_sidecar_flip": "profile.b3.bin",
                    "row_sidecar_flip": "profile.rows.npy", "size_sidecar_flip": "profile.n.npy",
                    "legacy_sidecar_flip": "profile.b3.bin"}.get(name)
        if filename:
            p = target / filename
            body = bytearray(p.read_bytes())
            body[-1] ^= 1
            p.write_bytes(body)
        elif name == "missing_canon":
            (target / "profile.jsonl.zst").unlink()
        elif name == "latest_error":
            with (target / "manifest.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**m, "error": {"code": "MNLM_CONTROL_FAILED"}}) + "\n")
        before, code, found = snapshot(target), None, []
        try:
            if name.startswith("legacy"):
                from attestation import attest_shard
                found = [attest_shard(target, m)]
            else:
                found = sealed_shards(target)
        except Exception as exc:
            code = getattr(exc, "code", str(exc).split()[0])
        expected_ok = len(found) == 1 and code is None if name == "legacy_valid" else bool(code and code.startswith("MNLM_CANON_"))
        check(name, expected_ok and before == snapshot(target), code=code, accepted=len(found))
        if name == "legacy_valid":
            code = None
            try:
                sealed_shards(target)
            except Exception as exc:
                code = getattr(exc, "code", str(exc).split()[0])
            check("legacy_cold_shard_is_not_plan_completion", code == "MNLM_CANON_PLAN_UNSEALED", code=code)
    # The existing consumer must propagate corruption rather than build a
    # plausible smaller population. No substitute test-only reader here.
    data_file = cdir / "profile.b3.bin"
    good = data_file.read_bytes()
    damaged = bytearray(good)
    damaged[-1] ^= 1
    data_file.write_bytes(damaged)
    before, code = snapshot(paths.dedup), None
    try:
        build_population(paths, [source])
    except Exception as exc:
        code = getattr(exc, "code", str(exc).split()[0])
    check("real_population_consumer_refuses_corruption", code == "MNLM_CANON_SIDECAR_MISMATCH"
          and before == snapshot(paths.dedup), code=code)
    data_file.write_bytes(good)
    for filename in ("profile.jsonl.zst", "profile.n.npy"):
        p = cdir / filename
        good = p.read_bytes()
        corrupt = bytearray(good)
        corrupt[-1] ^= 1
        p.write_bytes(corrupt)
        before = snapshot(cdir)
        result = cli("cached-corrupt-" + filename)
        check("cached_cli_refuses_" + filename, result.returncode != 0
              and "MNLM_CANON_" in result.stdout + result.stderr and before == snapshot(cdir))
        p.write_bytes(good)
    # Legacy corpus content remains readable, but its old size-only status is
    # recertified by the real driver before downstream source acceptance.
    shutil.copyfile(cdir / "plans.jsonl", out / "previous-plans.jsonl")
    (cdir / "plans.jsonl").unlink()
    before = {p.name: sha(p) for p in cdir.iterdir() if p.suffix not in (".jsonl",)}
    rebuilt = cli("legacy-plan-recertification")
    after = {p.name: sha(p) for p in cdir.iterdir() if p.suffix not in (".jsonl",)}
    check("legacy_plan_recertified_by_real_driver", rebuilt.returncode == 0
          and "REBUILD MNLM_CANON_PLAN_UNSEALED" in rebuilt.stdout and before == after
          and len(sealed_shards(cdir)) == 1)
    phases["cached_shard_refusal_controls"] = time.perf_counter() - t
    t = time.perf_counter()
    from fsv_plan_seals import controls
    controls(out / "known-truth-archive", check)
    phases["archive_commit_controls"] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    report = {"acceptance": "author-checked", "scientific_admission": False, "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "seconds": elapsed, "phases": phases, "phase_coverage": sum(phases.values()) / elapsed,
              "raw_sha256": expected, "source_unchanged": sha(raw / "profile.md") == expected,
              "code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("canonicalise.py"),
                                                     Path(__file__).with_name("common.py"),
                                                     Path(__file__).with_name("attestation.py"),
                                                     Path(__file__).with_name("plan_seals.py"),
                                                     Path(__file__).with_name("fsv_plan_seals.py")]}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256.json").write_text(json.dumps(snapshot(out), sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "seconds", "phase_coverage")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.exit(run(args.raw, args.output))

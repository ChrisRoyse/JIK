"""Shared helpers for Stage 3 (issue #12, spec §19): paths under <out>/lenses/<version>/, deterministic ordering by BLAKE3,
record fetching by (shard, rows) from the canonical shards, and the family table. No policy lives here."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "canon"))
from common import HOST, Paths, iter_canon_docs, ledger_append, now_iso, read_jsonl, run_stamp, write_json  # noqa: E402,F401

FAMILIES = ("a1-oci-tests", "a2-wikidata-p31", "a3-gdelt-dyad-7d", "a4-scripture-genre", "a5-gdelt-sources", "a6-cuyahoga-cited",
            "a1-oci-outcome3", "a2-wikidata-p17", "a2-wikidata-p106", "a2-wikidata-p31-rare", "a3-gdelt-next-quad", "a6-cuyahoga-cite-tier",
            "a2-wikidata-p27", "a2-wikidata-p19", "a2-wikidata-p69", "a2-wikidata-p641", "a2-wikidata-p136", "a2-wikidata-p495",
            "a2-wikidata-p54", "a2-wikidata-p166", "a2-wikidata-p463", "a2-wikidata-p108", "a2-wikidata-p1412", "a2-wikidata-p131",
            "a1-oci-ran-tests", "a4-scripture-genre-keyfree", "a6-cuyahoga-cited-cohort")
FAMILY_SPEC = {  # anchor family -> spec Table 2 family, the oracle and the population version it reads
    "a1-oci-tests": {"family": "A1", "oracle": "unit tests executed by this stage in a bubblewrap sandbox", "version": "p0-v1"},
    "a2-wikidata-p31": {"family": "A2", "oracle": "the P31 statement held by the structured source (wikidata5m transductive triples)", "version": "p0-v1"},
    "a3-gdelt-dyad-7d": {"family": "A3", "oracle": "a later GDELT coding of the same (Actor1, Actor2, QuadClass) within 7 days", "version": "p0-v1"},
    "a4-scripture-genre": {"family": "A4", "oracle": ">= 2 independent editions assert the verse (Stage 2 alignment keys); genre by config/book_classes.json", "version": "scripture-v1"},
    "a5-gdelt-sources": {"family": "A5", "oracle": "the same (Actor1, Actor2, EventCode) coded again on a later day from a different source domain (re-observation under new evidence)", "version": "p0-v1"},
    "a6-cuyahoga-cited": {"family": "A6", "oracle": "an in-corpus citation by another opinion (Stage 2 edges/cites.npz)", "version": "p0-v1"},
    # second questions of the same anchor families (spec §19.1: lenses are admitted per anchor family, the panel is the union over its questions)
    "a1-oci-outcome3": {"family": "A1", "oracle": "the same sandboxed test run read as all pass / some fail / any error or timeout", "version": "p0-v1"},
    "a2-wikidata-p17": {"family": "A2", "oracle": "the P17 (country) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p106": {"family": "A2", "oracle": "the P106 (occupation) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p31-rare": {"family": "A2", "oracle": "the P31 statement, classes ranked 13-24 by frequency", "version": "p0-v1"},
    "a3-gdelt-next-quad": {"family": "A3", "oracle": "the QuadClass of the next GDELT coding of the same (Actor1, Actor2) within 7 days, or none", "version": "p0-v1"},
    "a6-cuyahoga-cite-tier": {"family": "A6", "oracle": "in-corpus citations received: none, one, two or more (Stage 2 edges/cites.npz)", "version": "p0-v1"},
    # third round (Johari 2, solution 3): more A2 statement questions, each the top-12 values of one property
    "a2-wikidata-p27": {"family": "A2", "oracle": "the P27 (country of citizenship) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p19": {"family": "A2", "oracle": "the P19 (place of birth) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p69": {"family": "A2", "oracle": "the P69 (educated at) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p641": {"family": "A2", "oracle": "the P641 (sport) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p136": {"family": "A2", "oracle": "the P136 (genre) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p495": {"family": "A2", "oracle": "the P495 (country of origin) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p54": {"family": "A2", "oracle": "the P54 (member of sports team) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p166": {"family": "A2", "oracle": "the P166 (award received) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p463": {"family": "A2", "oracle": "the P463 (member of) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p108": {"family": "A2", "oracle": "the P108 (employer) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p1412": {"family": "A2", "oracle": "the P1412 (languages spoken, written or signed) statement held by the structured source", "version": "p0-v1"},
    "a2-wikidata-p131": {"family": "A2", "oracle": "the P131 (located in the administrative territorial entity) statement held by the structured source", "version": "p0-v1"},
    # round 4 (agent decisions D-3/D-4 of 2026-10-16, #12): the A4 question without the fields that identify the book, and an A6 question with the clock removed
    "a4-scripture-genre-keyfree": {"family": "A4", "oracle": "as a4-scripture-genre (same keys, same editions); chapter, verse and editions_of_key are not lens-readable fields", "version": "scripture-v1"},
    "a6-cuyahoga-cited-cohort": {"family": "A6", "oracle": "an in-corpus citation by another opinion, sampled balanced within each filing-year cohort so the year carries no bits; year and era withheld", "version": "p0-v1"},
    "a1-oci-ran-tests": {"family": "A1", "oracle": "the same sandboxed test run on the solutions the oracle recorded as loaded (no code error, no timeout): all tests pass or not", "version": "p0-v1"},
}
# Questions retired from the panel union (agent decisions of 2026-10-16, #12): still built and assayed for the record, never counted toward a family.
RETIRED_QUESTIONS = {
    "a6-cuyahoga-cited-cohort": "Operator excluded Cuyahoga from training; a new A6 substrate must be independently selected",
    "a4-scripture-genre": "D-3: chapter, verse and editions_of_key identify the book and the genre is a function of the book; replaced by a4-scripture-genre-keyfree",
    "a6-cuyahoga-cited": "D-4: largely a publication-date clock (readback F2); replaced by a6-cuyahoga-cited-cohort",
    "a6-cuyahoga-cite-tier": "D-4: largely a publication-date clock (readback F2); replaced by a6-cuyahoga-cited-cohort",
}
# Declared strata for the stratified override of Def. 13.3 (assay.py): the values of these categorical fields, 50 <= rows <= n/2.
STRATA_FIELDS = {"A1": ["domain", "generation_algorithm"], "A2": [], "A3": ["QuadClass", "EventRootCode", "ActionGeo_CountryCode", "IsRootEvent"], "A4": ["edition"],
                 "A5": ["QuadClass", "EventRootCode", "ActionGeo_CountryCode", "IsRootEvent"], "A6": ["era", "optype", "precedential", "court"]}


def sealed_registry(root: Path) -> dict:
    """The registry as measure sealed it: its content without the ledger fields must hash to the digest measure ledgered. A
    registry rewritten by a measure that did not finish (chain 11: killed by the OOM killer, the panel then read a half-updated
    registry) refuses here instead of being assayed or frozen."""
    reg = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in reg.items() if k not in ("ledger_seq", "ledger_head", "registry_blake3_before_ledger")}
    own = blake3.blake3((json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str) + chr(10)).encode("utf-8")).hexdigest()
    if not reg.get("ledger_seq") or own != reg.get("registry_blake3_before_ledger"):
        raise LensError("MNLM_REGISTRY_UNSEALED", f"{root / 'registry.json'}: content digest {own[:16]} != sealed {str(reg.get('registry_blake3_before_ledger'))[:16]} (a measure did not finish); re-run measure")
    return reg


def spec_family(fam: str) -> str:
    return FAMILY_SPEC[fam]["family"]


def families_present(root) -> list[str]:
    """The declared families that have a calibration file under <lens_root>/cal (a declared question is measured only once built)."""
    from training_scope import POLICY
    return [f for f in FAMILIES if f not in POLICY["excluded_questions"] and (Path(root) / "cal" / f"{f}.jsonl").exists()]


class LensError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


LEASE = Path.home() / "mnlm" / ".lease" / "gpu"
_LEASE_LOCK = None


def _pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied is not proof of absence
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def take_lease(issue: str, job: str) -> None:
    """One MNLM GPU job per host: refuse MNLM_GPU_LEASE_HELD while another live pid holds the lease file."""
    global _LEASE_LOCK
    LEASE.parent.mkdir(parents=True, exist_ok=True)
    lock = LEASE.with_suffix(".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, 2) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if LEASE.exists():
            old = json.loads(LEASE.read_text(encoding="utf-8"))
            if _pid_alive(old.get("pid")):
                raise LensError("MNLM_GPU_LEASE_HELD", f"{LEASE} held by pid {old.get('pid')}; wait for its owner")
        write_json(LEASE, {"pid": os.getpid(), "start": now_iso(), "vram_budget_mib": 30000, "issue": issue, "job": job})
    except (OSError, ValueError) as exc:
        lock.close()
        raise LensError("MNLM_GPU_LEASE_HELD", f"{LEASE}: lease is busy or malformed; inspect its owner") from exc
    except BaseException:
        lock.close()
        raise
    _LEASE_LOCK = lock


def run_cli(main):
    """Release only this process's lease, including on a fail-closed exception."""
    try:
        return main()
    except LensError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    finally:
        release_lease()


def release_lease():
    """Release a successfully acquired lease for in-process streaming callers."""
    global _LEASE_LOCK
    if _LEASE_LOCK is not None:
        try:
            if LEASE.exists() and json.loads(LEASE.read_text(encoding="utf-8")).get("pid") == os.getpid():
                LEASE.unlink()
        finally:
            _LEASE_LOCK.close()
            _LEASE_LOCK = None


def require_gpu(device: str) -> None:
    """The assay and the FSV run the estimator on the GPU; a CPU run is a different runtime and refuses MNLM_ASSAY_NO_GPU."""
    if device != "cuda":
        raise LensError("MNLM_ASSAY_NO_GPU", f"estimator device is {device!r}; the Stage 3 assay runs on CUDA only (no silent CPU fallback)")


def lens_root(paths: Paths) -> Path:
    return paths.out / "lenses" / paths.version


def hash_order(seed: int, *parts) -> int:
    """Deterministic 64-bit ordinal of an identity under a seed (spec §23: BLAKE3 of seed, ordinal and identity)."""
    h = blake3.blake3()
    h.update(str(seed).encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


def text_b3(text: str) -> str:
    return blake3.blake3(text.encode("utf-8")).hexdigest()


def fetch_records(canon_dir: Path, shard: str, rows: set[int]) -> dict[int, dict]:
    """The canonical records of one shard whose row is in `rows` (one streaming pass, stops when all are found)."""
    out: dict[int, dict] = {}
    if not rows:
        return out
    for rec in iter_canon_docs(canon_dir, shard):
        if rec["row"] in rows:
            out[rec["row"]] = rec
            if len(out) == len(rows):
                break
    return out


def atoms_by_row(adir: Path, source: str, shard: str) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """The atom arrays of one shard and, per record row, the indices of its atoms in start order."""
    a = np.load(adir / source / f"{shard}.atoms.npz")
    arrays = {k: a[k] for k in ("row", "start", "end", "role", "unit")}
    order = np.lexsort((arrays["start"], arrays["row"]))
    rows = arrays["row"][order]
    bounds = np.flatnonzero(np.diff(rows)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(rows)]])
    per_row = {int(rows[i]): order[i:j] for i, j in zip(starts, ends)} if len(rows) else {}
    return arrays, per_row


def population_shards(paths: Paths, version: str) -> list[dict]:
    meta = json.loads((paths.out / "dedup" / version / "population.json").read_text(encoding="utf-8"))
    return meta["shards"]


def entropy_bits(labels: list[int]) -> float:
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def write_family(cal_dir: Path, family: str, rows: list[dict], summary: dict) -> dict:
    """Writes <family>.jsonl (one calibration atom per line, `i` ascending) and <family>.summary.json; returns the summary
    with the file digest, class histogram and H(Y)."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    path = cal_dir / f"{family}.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    h = blake3.blake3()
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows):
            r["i"] = i
            r["family"] = family
            r["text_b3"] = text_b3(r["text"])
            line = json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
            fh.write(line)
            h.update(line.encode("utf-8"))
    tmp.replace(path)
    labels = [r["y"] for r in rows]
    names = {}
    for r in rows:
        names.setdefault(str(r["y"]), r.get("y_name", str(r["y"])))
    summary = dict(summary)
    summary.update({"family": family, "n": len(rows), "classes": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}, "class_names": names,
                    "H_Y_bits": entropy_bits(labels) if rows else 0.0, "file_blake3": h.hexdigest(), "host": HOST, "ts": now_iso(), **FAMILY_SPEC[family]})
    write_json(cal_dir / f"{family}.summary.json", summary)
    return summary


def read_family(cal_dir: Path, family: str) -> tuple[list[dict], dict]:
    from training_scope import require_training_question, require_training_source
    require_training_question(family)
    rows = read_jsonl(cal_dir / f"{family}.jsonl")
    for source in {r.get("source") for r in rows} - {None}:
        require_training_source(source)
    summary = json.loads((cal_dir / f"{family}.summary.json").read_text(encoding="utf-8"))
    if len(rows) != summary["n"]:
        raise LensError("MNLM_LENS_CAL_TORN", f"{family}: {len(rows)} rows, summary says {summary['n']}")
    actual = blake3.blake3((cal_dir / f"{family}.jsonl").read_bytes()).hexdigest()
    if actual != summary.get("file_blake3") or any(r.get("i") != i for i, r in enumerate(rows)):
        raise LensError("MNLM_LENS_CAL_TORN", f"{family}: digest/row order differs from calibration summary")
    hist = {str(int(k)): int(v) for k, v in zip(*np.unique([r["y"] for r in rows], return_counts=True))}
    if hist != summary.get("classes") or abs(entropy_bits([r["y"] for r in rows]) - summary["H_Y_bits"]) > 1e-9:
        raise LensError("MNLM_LENS_CAL_TORN", f"{family}: labels/entropy differ from calibration summary")
    return rows, summary

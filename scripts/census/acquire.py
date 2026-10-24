"""Stage 0â†’1 â€” acquire the raw bytes listed by the census, verify each file against the provider's digest, and write
receipts (spec Â§16/Â§17: nothing is deleted from the raw store; every byte is traceable to a sealed manifest entry).

    python -I scripts/census/acquire.py --root /zfs/archive/mnlm --version p0-v1 [--only <source-id>] [--workers 4]

Reads <root>/census/<version>/census.json and each source's manifest.jsonl, builds the download URL per access type,
streams each file to <root>/raw/<source-id>/<path> while hashing (sha256 and md5), and appends one receipt per file to
<root>/raw/<source-id>/receipts.jsonl: {path, bytes, secs, rate_Bps, sha256, md5, expected, expected_kind, verified}.
A file whose digest does not match is kept under <path>.REJECTED and recorded verified=false; it is never used.
Idempotent: a file with a verified receipt is skipped. Cost: O(bytes) once; workers bound concurrency; no payload is
parsed here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "MNLM-acquire/0.1 (+https://github.com/ChrisRoyse/MNLM)"
CHUNK = 1 << 20


def build_url(access: dict, info: dict, entry: dict) -> str:
    t = access["type"]
    if t == "hf-tree":
        rev = info.get("revision_sha") or access["revision"]
        return f"https://huggingface.co/datasets/{access['repo']}/resolve/{rev}/{entry['path']}"
    if t == "cc-pathlist":
        return access["base"] + entry["path"]
    if t in ("gdelt-masterfilelist", "http-head", "html-links"):
        return entry["path"] if entry["path"].startswith("http") else entry.get("final_url", entry["path"])
    if t == "git-tree":
        return info["raw_base"] + urllib.parse.quote(entry["path"])
    if t == "gcs-prefix":
        return info["base_url"] + urllib.parse.quote(entry["path"])
    raise ValueError(f"no download rule for access type {t}")


NAME_MAX = 240  # 255 bytes per path component on ext4/ZFS, minus room for the .part/.REJECTED suffixes


def local_name(access: dict, entry: dict) -> str:
    p = entry["path"]
    if access["type"] == "cc-pathlist":
        p = p.rsplit("/", 1)[-1]
    elif p.startswith("http"):
        p = p.rsplit("/", 1)[-1].split("?")[0]
    head, _, base = p.rpartition("/")
    if len(base.encode("utf-8")) > NAME_MAX:  # the receipt keeps the full path; the file carries a hash of it
        stem, dot, ext = base.rpartition(".")
        tag = "~" + hashlib.blake2b(base.encode("utf-8"), digest_size=8).hexdigest()
        keep = (stem or ext).encode("utf-8")[: NAME_MAX - len(tag) - len(ext) - 8].decode("utf-8", "ignore")
        base = keep + tag + (dot + ext if stem else "")
        p = f"{head}/{base}" if head else base
    return p


def accepted(r: dict) -> bool:
    """A receipt is accepted when its digest verified, or when the provider offers no digest and the byte count equals
    the declared Content-Length (attestation "content-length"; the receipt's own sha256 is then the pinned digest)."""
    return r.get("verified") is True or (r.get("verified") is None and r.get("attestation") == "content-length" and r.get("size_match") is True)


def current(r: dict, e: dict) -> bool:
    """Accepted AND for this manifest entry: the digest the receipt verified against (any kind) and the byte count must
    be the entry's. A provider re-publication (new md5, new size) or a rebuilt authored file (new sha256) is otherwise
    carried for ever on the older receipt."""
    if not accepted(r):
        return False
    if e.get("digest") is not None and (r.get("expected") != e["digest"] or r.get("expected_kind") != e.get("digest_kind")):
        return False
    return not (isinstance(e.get("bytes"), int) and e["bytes"] >= 0 and r.get("bytes") != e["bytes"])


def load_receipts(path: Path) -> dict:
    out = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["path"]] = r
    return out


RETRIES = 4
LOCAL_TYPES = ("aiwonder-local", "local-manifest-pending", "manifest-copy", "poly-sealed-store", "not-on-this-host")
NET_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, TimeoutError, OSError, http.client.HTTPException)


class AcquisitionError(RuntimeError):
    """A contract refusal is terminal, never a retryable network error."""


@contextmanager
def source_lease(directory):
    lock = directory / ".acquire.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError as exc:
        raise AcquisitionError("MNLM_ACQUIRE_BUSY: inspect the source writer lease before retrying") from exc
    try:
        yield
    finally:
        lock.unlink()


def destination(directory, access, entry):
    root = directory.resolve()
    path = (root / local_name(access, entry)).resolve()
    if path == root or not path.is_relative_to(root):
        raise AcquisitionError("MNLM_ACQUIRE_PATH: manifest destination escapes the named source directory")
    return path


def _rehash(path: Path, sha, md5) -> int:
    n = 0
    with path.open("rb") as f:
        while b := f.read(CHUNK):
            sha.update(b)
            md5.update(b)
            n += len(b)
    return n


def fetch(url: str, dest: Path, expected: str | None, kind: str | None, hf_token: str | None, expected_bytes: int | None = None, max_bytes: int | None = None) -> dict:
    """Stream to <dest>.part while hashing. A broken stream is resumed with an HTTP Range request from the bytes already on
    disk (re-hashed from disk first), up to RETRIES attempts; a server that ignores the range restarts the file. A previous
    `.part` may resume only under the same pinned metadata. Rejected bytes never resume."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0 or
                                  expected_bytes is None or expected_bytes > max_bytes):
        raise AcquisitionError("MNLM_ACQUIRE_BUDGET: declared payload exceeds the approved file cap")
    limit = expected_bytes if max_bytes is None else min(expected_bytes, max_bytes)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    meta = tmp.with_suffix(tmp.suffix + ".meta")
    identity = {"url": url, "expected": expected, "kind": kind, "bytes": expected_bytes}
    if tmp.exists() and (not meta.exists() or json.loads(meta.read_text()) != identity):
        raise AcquisitionError("MNLM_ACQUIRE_RESUME_IDENTITY: preserve the old partial; use a separate acquisition path")
    meta.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    # a .REJECTED file is never a resume base: bytes that failed their digest are not a prefix of anything
    headers = {"User-Agent": UA}
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    if kind == "etag" and expected:
        headers["If-Match"] = '"' + expected.strip('"') + '"'
    sha, md5, t0 = hashlib.sha256(), hashlib.md5(), time.time()
    n = _rehash(tmp, sha, md5) if tmp.exists() else 0
    if expected_bytes is not None and n > expected_bytes:
        sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
        tmp.unlink()
    attempts, status, last_err, resumed = 0, None, None, 0
    tmp.touch()  # a zero-byte object never enters the loop; the file must still exist to be verified and renamed
    while expected_bytes is None or n < expected_bytes:
        attempts += 1
        if attempts > RETRIES:  # RETRIES counts consecutive attempts without progress; a stream that advances resets it
            raise OSError(f"MNLM_ACQUIRE_RETRIES_EXHAUSTED after {RETRIES} attempts without progress at byte {n}: {last_err}")
        n_before = n
        h = dict(headers)
        if n:
            h["Range"] = f"bytes={n}-"
            resumed += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r, tmp.open("ab") as f:
                status = r.status
                if kind == "etag" and expected and r.headers.get("ETag", "").strip('"') != expected.strip('"'):
                    raise AcquisitionError("MNLM_ACQUIRE_PROVIDER_CHANGED: response ETag differs from the sealed manifest")
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != n or (expected_bytes is not None and int(match[3]) != expected_bytes):
                        raise AcquisitionError("MNLM_ACQUIRE_RANGE: response does not continue the retained prefix")
                if n and status != 206:
                    f.seek(0)
                    f.truncate()
                    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
                while b := r.read(CHUNK):
                    if limit is not None and n + len(b) > limit:
                        raise AcquisitionError("MNLM_ACQUIRE_BUDGET: response exceeds declared payload; partial remains unaccepted")
                    f.write(b)
                    sha.update(b)
                    md5.update(b)
                    n += len(b)
                f.flush()
                os.fsync(f.fileno())
            if expected_bytes is None:
                break
            if n > n_before:
                attempts, last_err = 0, "stream ended early"
        except NET_ERRORS as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if n > n_before:
                attempts = 0
            time.sleep(min(60, 2 ** attempts))
    secs = time.time() - t0
    got = {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}
    size_match = (n == expected_bytes) if expected_bytes is not None else None
    if kind == "sha256":
        verified, attestation = got["sha256"] == expected, "sha256"
    elif kind in ("git-blob", "git-blob-sha1"):  # provider git identity
        blob = hashlib.sha1(f"blob {n}\0".encode())
        with tmp.open("rb") as f:
            while b := f.read(CHUNK):
                blob.update(b)
        got["git_blob"] = blob.hexdigest()
        verified, attestation = got["git_blob"] == expected, "git-blob"
    elif kind in ("md5", "s3-etag", "etag") and expected and re.fullmatch(r"[0-9a-fA-F]{32}", expected.strip('"')):
        verified, attestation = got["md5"].lower() == expected.strip('"').lower(), "md5"
    else:  # the provider gave no content digest (Dropbox-style opaque ETag): the declared Content-Length is the only attestation
        verified, attestation = None, "content-length"
        if size_match is False:
            verified = False
    final = dest if verified is not False else dest.with_name(dest.name + ".REJECTED")
    os.replace(tmp, final)
    meta.unlink()
    return {"bytes": n, "secs": round(secs, 3), "rate_Bps": int(n / secs) if secs > 0 else None, "status": status,
            **got, "expected": expected, "expected_kind": kind, "verified": verified, "attestation": attestation, "size_match": size_match, "stored": str(final), "attempts": attempts, "resumed": resumed}


def verify_local(src_dir: Path, entries: list[dict], todo: list[dict], rpath: Path) -> dict:
    """Authored files (identity-v1): re-hash each manifest entry in place; a mismatch is recorded verified=false and the
    file is never planned. The receipt's `stored` is the file itself (nothing is copied)."""
    stats = {"files": len(entries), "fetched": 0, "verified": 0, "rejected": 0, "unverifiable": 0, "errors": 0, "bytes": 0, "secs": 0.0}
    with rpath.open("a", encoding="utf-8") as out:
        for e in todo:
            p = src_dir / e["path"]
            t0, sha, md5 = time.time(), hashlib.sha256(), hashlib.md5()
            try:
                n = _rehash(p, sha, md5)
                ok = sha.hexdigest() == e["digest"] and n == e["bytes"]
                r = {"bytes": n, "secs": round(time.time() - t0, 3), "sha256": sha.hexdigest(), "md5": md5.hexdigest(), "expected": e["digest"], "expected_kind": "sha256",
                     "verified": ok, "attestation": "sha256", "size_match": n == e["bytes"], "stored": str(p)}
            except OSError as exc:
                r = {"error": f"{type(exc).__name__}: {exc}", "verified": False, "bytes": 0, "secs": 0.0}
            out.write(json.dumps({"path": e["path"], "url": p.as_uri(), "host": socket.gethostname(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r}, sort_keys=True) + "\n")
            stats["fetched"] += 1
            stats["bytes"] += r.get("bytes", 0)
            stats["errors" if "error" in r else ("verified" if r["verified"] else "rejected")] += 1
            print(f"  {e['path']}: {r.get('bytes')} B verified={r.get('verified')} {r.get('error', '')}", flush=True)
    print(f"local-authored {src_dir}: {json.dumps(stats)}", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selection", type=Path, help="Explicit bounded selection; sequential acquisition only")
    ap.add_argument("--selection-sha256")
    args = ap.parse_args()
    if args.selection:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bounded import run
        return run(Path(args.root), args.version, args.selection, args.selection_sha256)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
    from training_scope import allowed, require_training_source
    if args.only:
        require_training_source(args.only)
    root = Path(args.root)
    census = json.loads((root / "census" / args.version / "census.json").read_text(encoding="utf-8"))
    if args.only and args.only not in census["sources"]:
        raise AcquisitionError("MNLM_ACQUIRE_SOURCE: requested source is not in the sealed census")
    hf_token = os.environ.get("HF_TOKEN") or None
    summary = {}
    for sid, row in census["sources"].items():
        if not allowed(sid):
            continue
        if args.only and sid != args.only:
            continue
        access, info = row["access"], row.get("info") or {}
        if access["type"] == "by-host":
            access = access.get(socket.gethostname()) or {"type": "not-on-this-host"}
        if not row["discovery_complete"] or access["type"] in LOCAL_TYPES:
            continue
        mpath = root / "census" / args.version / sid / "manifest.jsonl"
        entries = [json.loads(l) for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        rdir = root / "raw" / sid
        rdir.mkdir(parents=True, exist_ok=True)
        with source_lease(rdir):
            rpath = rdir / "receipts.jsonl"
            done = load_receipts(rpath)
            todo = [e for e in entries if not current(done.get(e["path"], {}), e)]
            if access["type"] == "local-authored":  # nothing to fetch: the receipt is the in-place sha256 against the sealed manifest
                summary[sid] = verify_local(Path(access["path"]), entries, todo, rpath)
                continue
            print(f"{sid}: {len(entries)} files, {len(entries) - len(todo)} already verified, {len(todo)} to fetch", flush=True)
            stats = {"files": len(entries), "fetched": 0, "verified": 0, "rejected": 0, "unverifiable": 0, "errors": 0, "bytes": 0, "secs": 0.0}

            def job(e):
                url = build_url(access, info, e)
                dest = destination(rdir, access, e)
                try:
                    r = fetch(url, dest, e.get("digest"), e.get("digest_kind"), hf_token, e.get("bytes"))
                except NET_ERRORS as exc:
                    r = {"error": f"{type(exc).__name__}: {exc}", "verified": False, "bytes": 0, "secs": 0.0}
                return e["path"], url, r

            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex, rpath.open("a", encoding="utf-8") as out:
                for path, url, r in ex.map(job, todo):
                    rec = {"path": path, "url": url, "host": socket.gethostname(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r}
                    out.write(json.dumps(rec, sort_keys=True) + "\n")
                    out.flush()
                    stats["fetched"] += 1
                    stats["bytes"] += r.get("bytes", 0)
                    stats["secs"] += r.get("secs", 0.0)
                    if "error" in r:
                        stats["errors"] += 1
                    elif r["verified"] is True:
                        stats["verified"] += 1
                    elif r["verified"] is False:
                        stats["rejected"] += 1
                    elif not accepted(r):
                        stats["unverifiable"] += 1
                    print(f"  {path}: {r.get('bytes')} B in {r.get('secs')} s verified={r.get('verified')} {r.get('error', '')}", flush=True)
            summary[sid] = stats
            print(f"{sid}: {json.dumps(stats)}", flush=True)
    (root / "raw" / f"acquire-summary-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary and all(s["errors"] == 0 and s["rejected"] == 0 and s["unverifiable"] == 0 for s in summary.values()) else 3


if __name__ == "__main__":
    sys.exit(main())

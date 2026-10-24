"""Manifest-pinned bounded acquisition. O(manifest bytes + selected payload bytes).

One writer, three explicit boundaries: select -> acquire -> cold receipt readback.
Never infer approval from a census entry. The caller pins an adjudicated selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import blake3
from acquire import (AcquisitionError, accepted, build_url, current, fetch,
                     load_receipts, destination, source_lease, NET_ERRORS)


def require(ok, message):
    if not ok:
        raise AcquisitionError(message)


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt_bytes(receipt, dest, entry):
    require(current(receipt, entry), "MNLM_ACQUIRE_RECEIPT: no accepted current receipt")
    require(Path(receipt["stored"]).resolve() == dest.resolve() and dest.is_file(),
            "MNLM_ACQUIRE_RETAINED: receipt does not locate a retained file")
    require(dest.stat().st_size == receipt["bytes"] and sha_file(dest) == receipt["sha256"],
            "MNLM_ACQUIRE_RETAINED: bytes changed; preserve evidence and repair before reuse")


def plan(root, version, selection_path, selection_sha256):
    raw = selection_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == selection_sha256,
            "MNLM_ACQUIRE_SELECTION: pin the exact approved selection SHA256")
    selection = json.loads(raw)
    require(selection.get("contract") == "mnlm-bounded-acquisition-v1" and
            selection.get("approval") and selection.get("version") == version,
            "MNLM_ACQUIRE_SELECTION: selection contract, approval and census version required")
    cdir = root / "census" / version
    require(sha_file(cdir / "census.json") == selection["census_sha256"],
            "MNLM_ACQUIRE_CENSUS: census differs from approved selection")
    census = json.loads((cdir / "census.json").read_bytes())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
    from training_scope import require_training_source
    jobs, seen, destinations, total = [], set(), set(), 0
    for sid, scope in selection["sources"].items():
        require_training_source(sid)
        row = census["sources"][sid]
        require(row["discovery_complete"], "MNLM_ACQUIRE_CENSUS: source census incomplete")
        manifest = (cdir / sid / "manifest.jsonl").read_bytes()
        require(blake3.blake3(manifest).hexdigest() == row["manifest_blake3"],
                "MNLM_ACQUIRE_MANIFEST: manifest changed after census")
        entries = [json.loads(line) for line in manifest.splitlines() if line.strip()]
        bypath = {e["path"]: e for e in entries}
        require(len(bypath) == len(entries), "MNLM_ACQUIRE_MANIFEST: duplicate manifest path")
        subtotal = 0
        require(scope["paths"] and type(scope["max_bytes"]) is int and scope["max_bytes"] > 0,
                "MNLM_ACQUIRE_SELECTION: empty or invalid source budget")
        for path in scope["paths"]:
            require((sid, path) not in seen and path in bypath,
                    "MNLM_ACQUIRE_SELECTION: duplicate or unlisted path")
            seen.add((sid, path))
            entry = bypath[path]
            require(type(entry.get("bytes")) is int and entry["bytes"] > 0,
                    "MNLM_ACQUIRE_BUDGET: unknown or empty payload size")
            rdir = (root / "raw" / sid).resolve()
            dest = destination(rdir, row["access"], entry)
            require(dest.is_relative_to(rdir) and dest != rdir and dest not in destinations,
                    "MNLM_ACQUIRE_PATH: escaping or colliding destination")
            destinations.add(dest)
            subtotal += entry["bytes"]
            jobs.append((sid, row, entry, dest))
        require(subtotal <= scope["max_bytes"], "MNLM_ACQUIRE_BUDGET: source exceeds approved cap")
        total += subtotal
    require(jobs and type(selection["max_bytes"]) is int and total <= selection["max_bytes"],
            "MNLM_ACQUIRE_BUDGET: selection exceeds approved aggregate cap")
    return selection, jobs, total


def run(root, version, selection_path, selection_sha256):
    selection, jobs, total = plan(root, version, selection_path, selection_sha256)
    # Lock before any raw mutation; a different source-wide job must not run concurrently.
    leases = ExitStack()
    try:
        for sid in sorted(selection["sources"]):
            rdir = root / "raw" / sid
            rdir.mkdir(parents=True, exist_ok=True)
            leases.enter_context(source_lease(rdir))
        report = {"selection_sha256": selection_sha256, "approval": selection["approval"],
                  "declared_bytes": total, "readback_bytes": 0, "files": []}
        done_by_source = {sid: load_receipts(root / "raw" / sid / "receipts.jsonl")
                          for sid in selection["sources"]}
        for sid, row, entry, dest in jobs:
            before = done_by_source[sid].get(entry["path"])
            if dest.exists():
                require(before is not None, "MNLM_ACQUIRE_RETAINED: file has no receipt; investigate before overwrite")
                receipt_bytes(before, dest, entry)
                rec = before
            else:
                url = build_url(row["access"], row.get("info") or {}, entry)
                result = fetch(url, dest, entry.get("digest"), entry.get("digest_kind"),
                               os.environ.get("HF_TOKEN"), entry["bytes"], entry["bytes"])
                rec = {"path": entry["path"], "url": url, "host": socket.gethostname(),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "selection_sha256": selection_sha256, **result}
                with (root / "raw" / sid / "receipts.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                require(accepted(rec), "MNLM_ACQUIRE_CONTENT: provider digest/length verification failed")
                # Reopen the receipt and bytes; a successful fetch return is not acceptance.
                rec = load_receipts(root / "raw" / sid / "receipts.jsonl")[entry["path"]]
                receipt_bytes(rec, dest, entry)
            report["readback_bytes"] += rec["bytes"]
            report["files"].append({"source": sid, "path": entry["path"], "sha256": rec["sha256"],
                                    "bytes": rec["bytes"], "stored": str(dest), "reused": dest.exists() and rec is before})
            print(json.dumps(report["files"][-1]), flush=True)
        require(report["readback_bytes"] == total, "MNLM_ACQUIRE_READBACK: aggregate size differs")
        data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        report_hash = hashlib.sha256(data).hexdigest()
        output = root / "raw" / ("bounded-" + selection_sha256 + "-" + report_hash + ".json")
        if output.exists():
            require(sha_file(output) == report_hash, "MNLM_ACQUIRE_REPORT: content-addressed report changed")
        else:
            with output.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        print(json.dumps({"author_checked": True, "report": str(output), "sha256": sha_file(output)}))
        return 0
    except (AcquisitionError, *NET_ERRORS) as exc:
        print(json.dumps({"error": str(exc), "remediation": "Resolve the named boundary; do not enlarge the cap."}), file=sys.stderr)
        return 3
    finally:
        leases.close()

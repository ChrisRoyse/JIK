"""Acquire the delegated, bounded BridgeData observation pilot (#3).

O(metadata + selected bytes), one network stream, no provider code execution.
Uses bounded.py for locks, resume, size/ETag refusal and physical receipt reuse.
The census covers the named selection, not the provider's full dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded import require, run, sha_file
from ledger import Ledger, digest, canonical, verify_chain

PROPOSAL = "ce43d85b9167146d2829a2566fb776fe96b503cd8a1284d5e675902018528057"
MANIFEST = "f2c8c1150d30bba3b85c8d92d0fdddc1ae60ae469c5310dc7ed835e491965341"
VERSION = "bridge-observations-v1"
APPROVAL = "https://github.com/ChrisRoyse/MNLM/issues/3#issuecomment-6232430707"


def immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "MNLM_BRIDGE_DRIFT: retained metadata differs")
        return
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def prepare(root, metadata):
    require(sha_file(metadata / "proposal.json") == PROPOSAL and
            sha_file(metadata / "manifest.json") == MANIFEST,
            "MNLM_BRIDGE_PROPOSAL: frozen proposal or metadata inventory differs")
    inventory = json.loads((metadata / "manifest.json").read_bytes())
    for item in inventory["files"]:
        p = (metadata / item["path"]).resolve()
        require(p.is_relative_to(metadata.resolve()) and p.stat().st_size == item["bytes"]
                and sha_file(p) == item["sha256"],
                "MNLM_BRIDGE_METADATA: retained evidence changed")
    proposal = json.loads((metadata / "proposal.json").read_bytes())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
    from training_scope import POLICY
    scope = POLICY["delegated_decisions"]
    files = proposal["selected_files"]
    require(scope["record"] == APPROVAL and scope["bridge_proposal_sha256"] == PROPOSAL
            and len(files) == scope["bridge_shards"] == 12
            and sum(f["content_length"] for f in files) <= scope["bridge_max_bytes"],
            "MNLM_BRIDGE_SCOPE: proposal is outside the recorded delegated decision")
    # Recheck all remote identities before writing a census or acquiring payloads.
    for item in files:
        request = urllib.request.Request(item["url"], method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            require(response.status == 200
                    and int(response.headers["Content-Length"]) == item["content_length"]
                    and response.headers["ETag"] == item["etag"],
                    "MNLM_BRIDGE_PROVIDER_CHANGED: length or version changed; inspect before selecting again")
    entries = [{"path": f["url"], "bytes": f["content_length"], "digest": f["etag"],
                "digest_kind": "etag", "episodes_declared": f["episodes_declared"]} for f in files]
    manifest = b"".join(canonical(e) + b"\n" for e in entries)
    sid = proposal["source_id"]
    cdir = root / "census" / VERSION
    immutable(cdir / sid / "manifest.jsonl", manifest)
    # License/schema/source observations stay beside the census as read-only evidence.
    for name in ("proposal.json", "bridge-info.raw", "bridge-features.raw", "bridge-license-page.raw"):
        immutable(cdir / sid / name, (metadata / name).read_bytes())
    census = {"version": VERSION, "scope": "exact selected observation pilot; not full provider discovery",
              "approval": APPROVAL, "discovery_complete": True,
              "sources": {sid: {"access": {"type": "http-head"}, "discovery_complete": True,
                  "manifest_blake3": digest(manifest), "entries": len(entries),
                  "bytes_total": sum(e["bytes"] for e in entries), "licence": proposal["license"],
                  "clock": "ordinal sequence only; physical timing/alignment uncommissioned",
                  "anchor_families": [], "admission": "provisional observations"}}}
    raw = canonical(census) + b"\n"
    immutable(cdir / "census.json", raw)
    ledger = Ledger(root / "ledger" / "ledger.jsonl", actor="mnlm-bridge-acquisition")
    seal_path = cdir / "SEAL.json"
    if not seal_path.exists():
        row = ledger.append("census.artifact", VERSION, {"census_blake3": digest(raw),
                            "census_bytes": len(raw), "discovery_complete": True})
        immutable(seal_path, canonical({"ledger_seq": row["seq"], "ledger_head": row["hash"],
                                       "census_blake3": digest(raw)}) + b"\n")
    verify_chain(root / "ledger" / "ledger.jsonl")
    seal = json.loads(seal_path.read_bytes())
    with (root / "ledger" / "ledger.jsonl").open(encoding="utf-8") as f:
        matches = [r for line in f if (r := json.loads(line))["seq"] == seal["ledger_seq"]]
    require(len(matches) == 1 and matches[0]["hash"] == seal["ledger_head"]
            and matches[0]["payload_digests"]["census_blake3"] == seal["census_blake3"] == digest(raw),
            "MNLM_BRIDGE_LEDGER: census seal does not bind the current bytes")
    selection = {"contract": "mnlm-bounded-acquisition-v1", "version": VERSION,
                 "approval": APPROVAL, "census_sha256": hashlib.sha256(raw).hexdigest(),
                 "sources": {sid: {"paths": [e["path"] for e in entries],
                                   "max_bytes": scope["bridge_max_bytes"]}},
                 "max_bytes": scope["bridge_max_bytes"]}
    path = cdir / "selection.json"
    immutable(path, canonical(selection) + b"\n")
    return path, sha_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    selection, sha = prepare(args.root, args.metadata)
    print(json.dumps({"selection": str(selection), "sha256": sha}), flush=True)
    return 0 if args.prepare_only else run(args.root, VERSION, selection, sha)


if __name__ == "__main__":
    raise SystemExit(main())

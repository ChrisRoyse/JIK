"""Cross-host seal (issue #8): append a `stage.peer` ledger entry on this host naming the other host's ledger head and
evidence digest for the same stage and version, so that a claim spanning both hosts is chained on both ledgers.

    python -I scripts/canon/peer_seal.py --root R --version V --stage 1 --peer-host H --peer-head X --peer-seq N \
        --peer-evidence-blake3 Y --peer-evidence-path P
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, ledger_append  # noqa: E402

# Where each host keeps MNLM evidence. A peer path outside its host's roots is refused: a Git Bash shell rewrites a
# leading `/zfs/...` argument to `C:/Program Files/Git/zfs/...` (MSYS path conversion), which the workstation ledger
# recorded in every peer seal from seq 26 to 69 on 2026-10-15; the digests were right, the paths were not.
PEER_ROOTS = {"aiwonder": ("/zfs/archive/mnlm/", "/zfs/hot/mnlm/"), "cabdru": ("D:/MNLMData/", "D:\\MNLMData\\")}
HEX64 = frozenset("0123456789abcdef")


def refuse(code: str, **detail) -> int:
    print(json.dumps({"ok": False, "code": code, **detail}), file=sys.stderr)
    return 2


def check_args(args) -> int:
    roots = PEER_ROOTS.get(args.peer_host)
    if roots is None:
        return refuse("MNLM_PEER_HOST_UNKNOWN", peer_host=args.peer_host, known=sorted(PEER_ROOTS))
    if not args.peer_evidence_path.startswith(roots):
        return refuse("MNLM_PEER_EVIDENCE_PATH_FOREIGN", peer_host=args.peer_host, path=args.peer_evidence_path,
                      expected_prefixes=list(roots), hint="run with MSYS_NO_PATHCONV=1 or from PowerShell")
    for name in ("peer_head", "peer_evidence_blake3"):
        v = getattr(args, name)
        if len(v) != 64 or set(v) - HEX64:
            return refuse("MNLM_PEER_DIGEST_MALFORMED", field=name, value=v)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--peer-host", required=True)
    ap.add_argument("--peer-head", required=True)
    ap.add_argument("--peer-seq", type=int, required=True)
    ap.add_argument("--peer-evidence-blake3", required=True)
    ap.add_argument("--peer-evidence-path", required=True)
    args = ap.parse_args()
    rc = check_args(args)
    if rc:
        return rc
    paths = Paths(args.root, args.root, args.version)
    e = ledger_append(paths, f"peer_seal@{HOST}", "stage.peer", f"stage{args.stage}/{args.version}",
                      {"peer_host": args.peer_host, "peer_ledger_head": args.peer_head, "peer_ledger_seq": args.peer_seq,
                       "peer_evidence_blake3": args.peer_evidence_blake3, "peer_evidence_path": args.peer_evidence_path})
    print(json.dumps({"seq": e["seq"], "head": e["hash"], "subject": e["subject"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

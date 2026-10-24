"""Stage 0 FSV: read the persisted census and ledger back through a separate path and audit three boundaries
(spec §16) with negative controls. Writes the report under <root>/evidence/fsv/<issue>/<run>/stage0-fsv.json.

    python -I scripts/census/fsv_stage0.py --root D:/MNLMData --version p0-v1 --issue 7

B1 manifest bytes -> census rows: recount every manifest.jsonl (entries, byte sum, BLAKE3) and compare with census.json
   and with the ledger's census.source payload.
B2 census -> sealed ledger position: verify the whole chain; census.json's BLAKE3 and byte count must equal the
   exact census.artifact named by SEAL.json. Later entries cannot substitute a different artifact or source attestation.
B3 tamper: a copy of the ledger with one payload byte altered must refuse MNLM_LEDGER_CHAIN_BROKEN; a copy of a manifest
   with one byte altered must not reproduce the sealed digest.
Negative control: every source with discovery_complete=false has bytes_total null (never 0) and a typed error code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import LedgerError, digest_file, read_entries, verify_chain  # noqa: E402


def recount(mpath: Path) -> tuple[int, int | None, str]:
    n, total, any_sized = 0, 0, False
    with mpath.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            n += 1
            if isinstance(e.get("bytes"), int) and e["bytes"] >= 0:
                total += e["bytes"]
                any_sized = True
    return n, (total if any_sized else None), digest_file(mpath)[0]


def source_key(subject, payload):
    # The same source can have several snapshots, including inherited --only
    # rows. Bind its exact manifest attestation, not its latest global status.
    return json.dumps([subject, payload.get("manifest_blake3"), payload.get("entries"), payload.get("bytes_total")], sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--issue", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    cdir = root / "census" / args.version
    lpath = root / "ledger" / "ledger.jsonl"
    census = json.loads((cdir / "census.json").read_text(encoding="utf-8"))
    seal = json.loads((cdir / "SEAL.json").read_text(encoding="utf-8"))
    entries = read_entries(lpath)
    by_source = {source_key(e["subject"], e["payload_digests"]): e for e in entries
                 if e["kind"] == "census.source" and e["seq"] <= seal["ledger_seq"]}
    checks: list[dict] = []

    def check(name: str, passed: bool, **detail):
        detail.pop("ok", None)  # payloads from verify_chain / LedgerError carry their own ok flag
        checks.append({"check": name, "ok": bool(passed), **detail})

    # B1
    for sid, row in census["sources"].items():
        if not row["discovery_complete"]:
            continue
        n, total, dig = recount(cdir / sid / "manifest.jsonl")
        led = by_source.get(source_key(sid, row), {}).get("payload_digests", {})
        check(f"B1.{sid}.entries", n == row["entries"] == led.get("entries"), recount=n, census=row["entries"], ledger=led.get("entries"))
        check(f"B1.{sid}.bytes", total == row["bytes_total"] == led.get("bytes_total"), recount=total, census=row["bytes_total"], ledger=led.get("bytes_total"))
        check(f"B1.{sid}.digest", dig == row["manifest_blake3"] == led.get("manifest_blake3"), recount=dig, census=row["manifest_blake3"], ledger=led.get("manifest_blake3"))

    # B2
    try:
        v = verify_chain(lpath)
        check("B2.chain", True, **{k: x for k, x in v.items() if k != "ok"})
    except LedgerError as exc:
        check("B2.chain", False, **{k: x for k, x in exc.as_dict().items() if k != "ok"})
        v = {"head": None}
    # the seal names a POSITION in the chain, not the current head: later lawful entries must not fail the census
    at = next((e for e in entries if e["seq"] == seal["ledger_seq"]), None)
    check("B2.seal_entry_in_chain", at is not None and at["hash"] == seal["ledger_head"] and at["kind"] == "census.artifact" and at["subject"] == args.version and v.get("ok", False),
          seal_seq=seal["ledger_seq"], seal_head=seal["ledger_head"][:16], entry_kind=(at or {}).get("kind"), chain_head=(v.get("head") or "")[:16])
    cdig, cbytes = digest_file(cdir / "census.json")
    artifact = (at or {}).get("payload_digests", {})
    check("B2.census_digest_sealed", artifact.get("census_blake3") == cdig == seal["census_blake3"],
          recomputed=cdig, sealed=artifact.get("census_blake3"), seal_seq=seal["ledger_seq"])
    check("B2.census_bytes_sealed", artifact.get("census_bytes") == cbytes,
          actual=cbytes, sealed=artifact.get("census_bytes"))

    # B3 tamper controls on copies
    with tempfile.TemporaryDirectory() as td:
        tl = Path(td) / "ledger.jsonl"
        shutil.copy(lpath, tl)
        data = bytearray(tl.read_bytes())
        i = data.find(b'"manifest_blake3":"')
        if i < 0:
            i = data.find(b'"payload_digests"')
        j = i + 25
        data[j] = (data[j] + 1) & 0x7F if chr(data[j]).isalnum() else ord("x")
        tl.write_bytes(bytes(data))
        try:
            verify_chain(tl)
            check("B3.tampered_ledger_refused", False, note="tampered chain verified as ok")
        except LedgerError as exc:
            check("B3.tampered_ledger_refused", exc.code == "MNLM_LEDGER_CHAIN_BROKEN", **exc.as_dict())
        complete = [s for s, r in census["sources"].items() if r["discovery_complete"]]
        if complete:
            sid = complete[0]
            tm = Path(td) / "manifest.jsonl"
            shutil.copy(cdir / sid / "manifest.jsonl", tm)
            mb = bytearray(tm.read_bytes())
            mb[0] = ord("X") if mb[0] != ord("X") else ord("Y")
            tm.write_bytes(bytes(mb))
            check("B3.tampered_manifest_digest_differs", digest_file(tm)[0] != census["sources"][sid]["manifest_blake3"], source=sid)

    # negative control: refused sources are null, not zero
    for sid, row in census["sources"].items():
        if not row["discovery_complete"]:
            check(f"NC.{sid}.refused_is_null", row["bytes_total"] is None and row["entries"] is None and bool((row.get("error") or {}).get("code")),
                  error=row.get("error"))

    ok = all(c["ok"] for c in checks)
    run = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    edir = root / "evidence" / "fsv" / str(args.issue) / f"stage0-{run}"
    edir.mkdir(parents=True, exist_ok=True)
    report = {"ok": ok, "issue": args.issue, "version": args.version, "root": str(root), "run_utc": run,
              "census_blake3": cdig, "ledger_head": v.get("head"), "ledger_entries": len(entries), "checks": checks,
              "acceptance": "author-checked only"}
    rpath = edir / "stage0-fsv.json"
    rpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (edir / "stage0-fsv.json.blake3").write_text(f"{digest_file(rpath)[0]}  stage0-fsv.json\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "checks": len(checks), "failed": [c["check"] for c in checks if not c["ok"]], "report": str(rpath)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

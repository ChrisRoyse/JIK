"""Stage 0 writer/readback/crash boundaries on isolated copies of a real ledger.

The 1,000-event chain is a known-truth checkpoint control, never source/training
evidence. O(K*B) existing full-prefix validation; no concurrent performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import CHECKPOINT_EVERY, GENESIS, Ledger, LedgerError, entry_hash, verify_chain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_hash(value):
    return blake3.blake3(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def cold(path):
    """Separate raw-byte path; no Ledger verifier, entry_hash or merkle helper."""
    data = Path(path).read_bytes()
    assert not data or data.endswith(b"\n")
    previous, hashes, checkpoint_sizes, entries = GENESIS, [], [], []
    for i, raw in enumerate(data.splitlines(), 1):
        e = json.loads(raw)
        assert e["seq"] == i and e["prev_hash"] == previous
        assert e["hash"] == independent_hash({k: v for k, v in e.items() if k != "hash"})
        if e["kind"] == "ledger.checkpoint":
            leaves = [bytes.fromhex(h) for h in hashes]
            assert e["payload_digests"]["entries"] == len(leaves) and leaves
            while len(leaves) > 1:
                leaves += leaves[-1:] if len(leaves) % 2 else []
                leaves = [blake3.blake3(a + b).digest() for a, b in zip(leaves[::2], leaves[1::2])]
            assert leaves[0].hex() == e["payload_digests"]["merkle_root"]
            checkpoint_sizes.append(len(hashes))
            hashes = []
        else:
            hashes.append(e["hash"])
        previous = e["hash"]
        entries.append(e)
    return {"entries": len(entries), "head": previous, "checkpoint_sizes": checkpoint_sizes,
            "pending": len(hashes), "sha256": sha(path)}, entries


def write_control_prefix(path, source_sha):
    previous = GENESIS
    with path.open("xb") as f:
        for n in range(1, CHECKPOINT_EVERY):
            e = {"seq": n, "prev_hash": previous, "kind": "fsv.checkpoint-control",
                 "subject": f"boundary-only-{n}", "actor": "fsv-not-training",
                 "payload_digests": {"real_ledger_sha256": source_sha}, "ts": "2026-10-17T00:00:00+00:00"}
            e["hash"] = independent_hash(e)
            f.write(json.dumps(e, sort_keys=True).encode() + b"\n")
            previous = e["hash"]
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--contend", type=Path)
    ap.add_argument("--crash", type=Path)
    args = ap.parse_args()
    if args.contend:
        try:
            Ledger(args.contend, "fsv-competing-writer").append("fsv.contender", "concurrent-boundary", {})
        except LedgerError as exc:
            print(json.dumps(exc.as_dict()))
            return 2
        return 0
    if args.crash:
        class CrashAfterEvent(Ledger):
            def _write(self, kind, subject, payload):
                entry = super()._write(kind, subject, payload)
                if kind == "fsv.crash":
                    os._exit(91)  # after the real fsync, while the real lock is held
                return entry
        CrashAfterEvent(args.crash, "fsv-crash").append("fsv.crash", "durable-event-before-checkpoint", {})
        return 1
    if not args.source or not args.output:
        ap.error("--source and --output are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started, checks, phases = time.perf_counter(), [], {}

    def check(name, passed, **evidence):
        checks.append({"boundary": name, "pass": bool(passed), **evidence})
        if not passed:
            (args.output / "failed-checks.json").write_text(json.dumps(checks, indent=2))
            raise AssertionError(name)

    def refuses(name, call, code, path):
        before = sha(path)
        try:
            call()
        except LedgerError as exc:
            check(name, exc.code == code and sha(path) == before, before=before, after=sha(path), refusal=exc.as_dict())
        else:
            check(name, False, expected=code)

    at = time.perf_counter()
    real = args.source.read_bytes()
    source_sha = hashlib.sha256(real).hexdigest()
    initial, _ = cold(args.source)
    check("real_chain_compatible", verify_chain(args.source)["head"] == initial["head"], source=initial)
    copy = args.output / "actual-prefix.jsonl"
    copy.write_bytes(real)
    writer = Ledger(copy, "fsv-boundary")
    before = writer.seq
    event = writer.append("fsv.source-readback", "retained-real-ledger", {"source_sha256": source_sha})
    after, rows = cold(copy)
    check("actual_prefix_unchanged", copy.read_bytes().startswith(real) and event["seq"] > before, before_entries=before, after=after)
    check("payload_cold_readback", next(e for e in rows if e["hash"] == event["hash"])["payload_digests"] == {"source_sha256": source_sha})
    for name, kind, subject, payload in (("nonfinite", "fsv.invalid", "nan", {"value": float("nan")}),
                                          ("bad-payload", "fsv.invalid", "list", []),
                                          ("bad-text", "fsv.invalid", "\ud800", {}),
                                          ("reserved-checkpoint", "ledger.checkpoint", "manual", {})):
        refuses(name, lambda: writer.append(kind, subject, payload),
                "MNLM_LEDGER_RESERVED_KIND" if kind == "ledger.checkpoint" else "MNLM_LEDGER_ENTRY_INVALID", copy)
    broken = args.output / "late-tamper.jsonl"
    broken.write_bytes(real)
    stale = Ledger(broken, "fsv-stale")
    lines = broken.read_bytes().splitlines(keepends=True)
    changed = json.loads(lines[0]); changed["subject"] += "-tampered"
    lines[0] = json.dumps(changed).encode() + b"\n"
    broken.write_bytes(b"".join(lines))
    refuses("tamper_after_writer_open", lambda: stale.append("fsv.probe", "refuse", {}), "MNLM_LEDGER_CHAIN_BROKEN", broken)
    refuses("tamper_before_writer_open", lambda: Ledger(broken, "fsv-cold"), "MNLM_LEDGER_CHAIN_BROKEN", broken)
    for name, body in (("missing-newline", real.rstrip(b"\r\n")), ("partial-tail", real + b'{"seq":'),
                       ("malformed-entry", real + b'[]\n'), ("duplicate-key", real + b'{"seq":1,"seq":2}\n')):
        p = args.output / (name + ".jsonl")
        p.write_bytes(body)
        refuses(name, lambda: Ledger(p, "fsv-cold"), "MNLM_LEDGER_CHAIN_BROKEN", p)
    phases["prefix_and_negative_controls"] = time.perf_counter() - at

    at = time.perf_counter()
    boundary = args.output / "checkpoint.jsonl"
    write_control_prefix(boundary, source_sha)
    initial_bytes = boundary.read_bytes()
    attempts = []
    class ContendedWriter(Ledger):
        def _write(self, kind, subject, payload):
            entry = super()._write(kind, subject, payload)
            # Try both moments: after boundary event and after checkpoint bytes.
            child = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--contend", str(self.path)], capture_output=True)
            attempts.append({"kind": kind, "returncode": child.returncode,
                             "result": json.loads(child.stdout), "stderr": child.stderr.decode()})
            return entry
    w = ContendedWriter(boundary, "fsv-primary")
    w.append("fsv.boundary", "cross-real-checkpoint-interval", {})
    report, rows = cold(boundary)
    check("checkpoint_and_event_one_lock", len(attempts) == 2 and all(a["returncode"] == 2 and a["result"]["code"] == "MNLM_LEDGER_BUSY" for a in attempts), attempts=attempts)
    check("checkpoint_exact_frontier", report["checkpoint_sizes"] == [CHECKPOINT_EVERY] and report["entries"] == CHECKPOINT_EVERY + 1 and report["pending"] == 0, readback=report)
    check("checkpoint_prefix_unchanged", boundary.read_bytes().startswith(initial_bytes))
    Ledger(boundary, "fsv-next-writer").append("fsv.next", "after-lock-release", {})
    check("lock_released_after_checkpoint", cold(boundary)[0]["entries"] == CHECKPOINT_EVERY + 2)
    for name, change in (("count", {"entries": CHECKPOINT_EVERY - 1}), ("root", {"merkle_root": GENESIS})):
        p = args.output / ("wrong-checkpoint-" + name + ".jsonl")
        altered = json.loads(json.dumps(rows))
        altered[-1]["payload_digests"].update(change)
        altered[-1]["hash"] = entry_hash(altered[-1])  # valid outer hash, invalid checkpoint proof
        p.write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in altered))
        refuses("checkpoint_" + name + "_refused", lambda: Ledger(p, "fsv-bad-checkpoint"), "MNLM_LEDGER_CHAIN_BROKEN", p)
    phases["checkpoint_and_contention"] = time.perf_counter() - at

    at = time.perf_counter()
    crash = args.output / "crash.jsonl"
    crash.write_bytes(initial_bytes)
    child = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--crash", str(crash)], capture_output=True)
    (args.output / "crash.stderr.txt").write_bytes(child.stderr)
    crashed = crash.read_bytes()
    interrupted, _ = cold(crash)
    check("crash_after_durable_event", child.returncode == 91 and interrupted["entries"] == CHECKPOINT_EVERY and interrupted["pending"] == CHECKPOINT_EVERY, readback=interrupted, returncode=child.returncode)
    restarted = Ledger(crash, "fsv-restart")
    check("restart_readonly_until_append", crash.read_bytes() == crashed)
    event = restarted.append("fsv.resume", "after-pending-checkpoint", {})
    recovered, recovered_rows = cold(crash)
    check("pending_checkpoint_completed_before_next_event", recovered_rows[-2]["kind"] == "ledger.checkpoint" and event["seq"] == CHECKPOINT_EVERY + 2 and recovered["pending"] == 1, readback=recovered)
    check("crash_recovery_preserves_prefix", crash.read_bytes().startswith(crashed))
    check("recovered_production_verifier", verify_chain(crash)["head"] == recovered["head"])
    phases["crash_and_recovery"] = time.perf_counter() - at
    check("live_ledger_unchanged", args.source.read_bytes() == real, source_sha256=source_sha)
    wall = time.perf_counter() - started
    report = {"acceptance": "author-checked", "source": str(args.source), "source_readback": initial,
              "source_bytes": len(real), "checks": checks, "passed": len(checks), "failed": 0,
              "wall_seconds": wall, "phase_seconds": phases, "timed_fraction": sum(phases.values()) / wall,
              "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("ledger.py"))},
              "limits": "Stage 0 ledger boundary repair only; no revised base or scientific admission completion"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("acceptance", "passed", "failed", "wall_seconds", "timed_fraction")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

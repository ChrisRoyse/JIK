"""Persisted acquisition boundary controls; never a training corpus or speed claim."""
import argparse
import hashlib
import http.server
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire import AcquisitionError, fetch, destination, source_lease
from bounded import plan, receipt_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--selection-sha256", required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = args.selection.read_bytes()  # retained approved evidence, no invented training data

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"actual-identity"')
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/evidence"
    checks = []

    def control(name, call, code, nofinal=None):
        error = None
        try:
            call()
        except AcquisitionError as exc:
            error = str(exc)
        checks.append({"boundary": name, "pass": error is not None and error.startswith(code)
                       and (nofinal is None or not nofinal.exists()), "refusal": error})

    try:
        selection, jobs, total = plan(args.root, "candidates-v1", args.selection, args.selection_sha256)
        checks.append({"boundary": "manifest selection", "pass": len(jobs) == 3 and total == 89682439,
                       "files": len(jobs), "bytes": total})
        dest = args.output / "accepted.json"
        before = dest.exists()
        result = fetch(url, dest, hashlib.sha256(payload).hexdigest(), "sha256", None, len(payload), len(payload))
        entry = {"bytes": len(payload), "digest": hashlib.sha256(payload).hexdigest(), "digest_kind": "sha256"}
        (args.output / "receipt.json").write_text(json.dumps(result))
        retained = json.loads((args.output / "receipt.json").read_text())
        receipt_bytes(retained, dest, entry)
        checks.append({"boundary": "download and independent cold receipt", "pass": not before and dest.read_bytes() == payload,
                       "before_exists": before, "after_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
        oversize = args.output / "oversize.json"
        control("stream bound", lambda: fetch(url, oversize, None, None, None, 8, 8), "MNLM_ACQUIRE_BUDGET", oversize)
        changed = args.output / "changed.json"
        control("provider identity", lambda: fetch(url, changed, "old-identity", "etag", None, len(payload), len(payload)),
                "MNLM_ACQUIRE_PROVIDER_CHANGED", changed)
        old = hashlib.sha256(dest.read_bytes()).hexdigest()
        dest.write_bytes(b"!" + payload[1:])
        control("corrupt retained file", lambda: receipt_bytes(retained, dest, entry), "MNLM_ACQUIRE_RETAINED")
        checks[-1].update(before_sha256=old, after_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        control("wrong selection identity", lambda: plan(args.root, "candidates-v1", args.selection, "0" * 64),
                "MNLM_ACQUIRE_SELECTION")
        limited = args.output / "insufficient-budget.json"
        selection["max_bytes"] = total - 1
        limited.write_text(json.dumps(selection))
        control("aggregate budget", lambda: plan(args.root, "candidates-v1", limited,
                                                 hashlib.sha256(limited.read_bytes()).hexdigest()), "MNLM_ACQUIRE_BUDGET")
        partial = args.output / "resume.json"
        partial.with_suffix(".json.part").write_bytes(payload[:10])
        control("unbound resume", lambda: fetch(url, partial, None, None, None, len(payload), len(payload)),
                "MNLM_ACQUIRE_RESUME_IDENTITY", partial)
        control("source traversal", lambda: destination(args.output, {"type": "hf-tree"}, {"path": "../escape"}),
                "MNLM_ACQUIRE_PATH")
        with source_lease(args.output):
            def competing_writer():
                with source_lease(args.output):
                    raise AssertionError("another source writer entered")
            control("shared writer exclusion", competing_writer, "MNLM_ACQUIRE_BUSY")
        check = not (args.output / ".acquire.lock").exists()
        checks.append({"boundary": "source lease released", "pass": check})
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())

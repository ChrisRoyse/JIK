"""Runs inside the bubblewrap sandbox only (scripts/lens/sandbox.py): reads {"code": str, "tests": [str]} from stdin,
executes the solution once, then each test in the same namespace, and prints one JSON line
{"statuses": ["pass"|"fail"|"error"], "code_error": str|null}. Nothing is written to disk; stdout and stderr of the
solution are discarded. The caller enforces the wall-clock, CPU and memory limits and the absence of network and home."""
import io
import json
import sys


def main() -> int:
    job = json.loads(sys.stdin.read())
    sys.stdin = io.StringIO("")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    ns: dict = {"__name__": "__main__"}
    code_error = None
    try:
        exec(compile(job["code"], "<solution>", "exec"), ns)
    except BaseException as exc:  # noqa: BLE001 - every failure of the solution is a measured outcome, never a crash of the harness
        code_error = f"{type(exc).__name__}: {exc}"[:200]
    statuses = []
    for t in job["tests"]:
        if code_error is not None:
            statuses.append("error")
            continue
        try:
            exec(compile(t, "<test>", "exec"), ns)
            statuses.append("pass")
        except AssertionError:
            statuses.append("fail")
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, SystemExit):
                statuses.append("error")
            else:
                statuses.append("error")
    sys.stdout, sys.stderr = real_out, real_err
    print(json.dumps({"statuses": statuses, "code_error": code_error}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

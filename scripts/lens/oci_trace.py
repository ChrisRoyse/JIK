"""Runs inside the bubblewrap sandbox only (scripts/lens/encoders_ctx3.py, lens code-exec-trace): reads {"code": str} from
stdin, executes the solution once with no tests and prints one JSON line describing the execution trace: whether it ran,
the exception class if not, stdout and stderr sizes, wall time, how many callables and classes the module defined, and
whether it tried to read stdin. The tests are never run here (their outcome is the A1 label); this is the "execution
trace" input of spec Table 3. Nothing is written to disk; the caller enforces the limits and the absence of network."""
import io
import json
import sys
import time


def main() -> int:
    job = json.loads(sys.stdin.read())
    sys.stdin = io.StringIO("")
    real_out, real_err = sys.stdout, sys.stderr
    out, err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out, err
    ns: dict = {"__name__": "__main__"}
    exc, stdin_read = None, False
    t0 = time.perf_counter()
    try:
        exec(compile(job["code"], "<solution>", "exec"), ns)
    except BaseException as e:  # noqa: BLE001 - every failure of the solution is a measured outcome
        exc = type(e).__name__
        stdin_read = isinstance(e, EOFError)
    secs = time.perf_counter() - t0
    sys.stdout, sys.stderr = real_out, real_err
    funcs = sum(1 for v in ns.values() if callable(v) and not isinstance(v, type) and getattr(v, "__module__", None) in (None, "__main__", "builtins") and not getattr(v, "__name__", "").startswith("__"))
    classes = sum(1 for v in ns.values() if isinstance(v, type) and getattr(v, "__module__", None) == "__main__")
    print(json.dumps({"ran": exc is None, "exception": exc, "stdout_chars": len(out.getvalue()), "stderr_chars": len(err.getvalue()), "wall_ms": round(1000 * secs, 2),
                      "functions": funcs, "classes": classes, "stdin_read": stdin_read, "names": len(ns) - 1}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

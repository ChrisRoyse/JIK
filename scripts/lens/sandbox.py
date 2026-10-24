"""A1 execution oracle (issue #12, spec Table 2): runs an OpenCodeInstruct solution and its unit tests under bubblewrap on
aiwonder with no network, no home directory, a private tmpfs, a CPU-time limit and an address-space limit; one short-lived
child per record; a record that exceeds its wall clock is `timeout` for every test. The harness is oci_harness.py.
Refuses with MNLM_SANDBOX_UNAVAILABLE when bwrap is missing or the probe (a trivial program) does not run, and with
MNLM_SANDBOX_ESCAPE when the probe can reach the network or the home directory (the sandbox must fail closed)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lenscommon import LensError

HARNESS = Path(__file__).resolve().parent / "oci_harness.py"
WALL_SECS = 10
CPU_SECS = 8
MEM_KB = 2_000_000


def _bwrap_cmd(wall: int = WALL_SECS) -> list[str]:
    binds = []
    for d in ("/usr", "/lib", "/lib64", "/bin", "/etc/alternatives"):
        if Path(d).exists():
            binds += ["--ro-bind", d, d]
    return ["timeout", "-s", "KILL", str(wall), "bwrap", *binds, "--ro-bind", str(HARNESS), "/sandbox/harness.py", "--tmpfs", "/tmp",
            "--unshare-all", "--die-with-parent", "--new-session", "--", "/bin/sh", "-c",
            f"ulimit -v {MEM_KB} -t {CPU_SECS}; exec /usr/bin/python3 -I /sandbox/harness.py"]


def run_one(code: str, tests: list[str]) -> dict:
    """{"statuses": [...], "code_error": ..., "rc": int, "timeout": bool}."""
    try:
        p = subprocess.run(_bwrap_cmd(), input=json.dumps({"code": code, "tests": tests}).encode("utf-8"), capture_output=True, timeout=WALL_SECS + 5)
    except subprocess.TimeoutExpired:
        return {"statuses": ["timeout"] * len(tests), "code_error": "wall clock", "rc": -1, "timeout": True}
    if p.returncode == 137 or p.returncode == -9:
        return {"statuses": ["timeout"] * len(tests), "code_error": "wall clock", "rc": p.returncode, "timeout": True}
    line = p.stdout.decode("utf-8", "replace").strip().splitlines()
    if not line:
        return {"statuses": ["error"] * len(tests), "code_error": f"no output rc={p.returncode} {p.stderr[-200:]!r}", "rc": p.returncode, "timeout": False}
    try:
        out = json.loads(line[-1])
    except json.JSONDecodeError:
        return {"statuses": ["error"] * len(tests), "code_error": f"bad output rc={p.returncode}", "rc": p.returncode, "timeout": False}
    out.update({"rc": p.returncode, "timeout": False})
    return out


def probe() -> dict:
    """Fail-closed probe: bwrap present; a trivial solution passes its test; a network connect and a home read both fail."""
    if shutil.which("bwrap") is None or shutil.which("timeout") is None:
        raise LensError("MNLM_SANDBOX_UNAVAILABLE", "bwrap or timeout not on PATH")
    ok = run_one("def f(x):\n    return x + 1\n", ["assert f(1) == 2", "assert f(1) == 3"])
    if ok["statuses"] != ["pass", "fail"]:
        raise LensError("MNLM_SANDBOX_UNAVAILABLE", f"probe statuses {ok}")
    net = run_one("import socket\ns = socket.create_connection(('1.1.1.1', 53), timeout=2)\n", ["assert True"])
    home = run_one(f"import os\nassert not os.path.exists({os.path.expanduser('~')!r})\nassert os.listdir('/') and not os.path.exists('/zfs')\n", ["assert True"])
    if net["statuses"] != ["error"] or home["statuses"] != ["pass"]:
        raise LensError("MNLM_SANDBOX_ESCAPE", f"network {net} home {home}")
    return {"bwrap": shutil.which("bwrap"), "python": "/usr/bin/python3", "wall_secs": WALL_SECS, "cpu_secs": CPU_SECS, "mem_kb": MEM_KB,
            "probe": {"trivial": ok["statuses"], "network": net["statuses"], "home_hidden": home["statuses"]}}


def run_many(jobs: list[tuple[str, list[str]]], workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: run_one(*j), jobs))

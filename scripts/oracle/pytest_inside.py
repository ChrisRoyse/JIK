"""Runs only inside the scoped container. It never sees other conditions or secrets."""
import json
import os
from pathlib import Path
import socket
import sys
import time

PREFIX = "MNLM_EXECUTION_RESULT="


def probe():
    results = {"uid": os.getuid(), "hidden": not any(Path(p).exists() for p in
                ("/home/croyse", "/zfs", "/var/run/docker.sock"))}
    for name, path in (("payload_readonly", "/work/__mnlm_probe__"), ("root_readonly", "/__mnlm_probe__")):
        try:
            Path(path).write_text("probe")
            results[name] = False
        except OSError:
            results[name] = True
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        results["network_blocked"] = False
    except OSError:
        results["network_blocked"] = True
    caps = [s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("CapEff:")]
    results["no_capabilities"] = caps == ["0000000000000000"]
    print(PREFIX + json.dumps({"probe": results, "python": sys.version}))


def build_sdist():
    """Build one pure-Python sdist (read-only /work) with the pinned /runtime/site backend.

    Network is absent and the root is read-only; the build happens in /tmp and the
    wheel bytes leave only through the single bounded result envelope.
    """
    import base64
    import shutil
    import subprocess
    src, out = Path("/tmp/src"), Path("/tmp/out")
    shutil.copytree("/work", src)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": "/runtime/site",
           "SOURCE_DATE_EPOCH": "1700000000", "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    p = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
                        "--no-cache-dir", "-w", str(out), str(src)], env=env, capture_output=True, timeout=25)
    wheels = sorted(out.glob("*.whl")) if out.exists() else []
    print(PREFIX + json.dumps({"exitcode": p.returncode, "python": sys.version,
                               "log_tail": (p.stdout + p.stderr).decode("utf-8", "replace")[-3000:],
                               "wheels": [{"name": w.name, "b64": base64.b64encode(w.read_bytes()).decode()}
                                          for w in wheels]}))


def main():
    mode = sys.argv[1]
    if mode == "probe":
        probe()
        return
    if mode == "timeout-control":
        time.sleep(60)
        return
    if mode == "output-control":
        while True:
            print("CONTROL" * 4096, flush=True)
    if mode == "build-sdist":
        build_sdist()
        return
    layout = Path("/runtime/mnlm-runtime.json")
    extra, root = [], "/work"
    if layout.exists():  # site-v1: a venv-equivalent runtime plus the repository's declared source roots
        import shutil
        import site
        spec = json.loads(layout.read_bytes())
        # writable copy, as a real checkout is; the host-side payload stays read-only
        root = "/work-rw/work" if Path("/work-rw").is_dir() else "/tmp/work"
        if not Path(root).exists():
            try:
                shutil.copytree("/work", root, symlinks=True)
            except OSError as exc:
                print(PREFIX + json.dumps({"exitcode": 3, "collected": [], "phases": [], "junit": None,
                                           "workdir_copy": f"{type(exc).__name__}: {exc}"[:500], "python": sys.version}))
                raise SystemExit(3)
        os.chdir(root)
        sys.path[:0] = [root + "/" + r for r in spec["source_roots"]] + [root]
        site.addsitedir("/runtime/" + spec["site"])  # after the stdlib, .pth processed: venv semantics
        for module in spec["plugins"]:
            extra += ["-p", module]  # explicit equivalent of autoload, restricted to the pinned runtime
    else:
        sys.path[:0] = sorted(str(p) for p in Path("/runtime").glob("*.whl")) + ["/work"]
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["HOME"] = "/tmp"
    import pytest

    class Recorder:
        def __init__(self):
            self.collected, self.phases = [], []

        def pytest_collection_finish(self, session):
            self.collected = [item.nodeid for item in session.items]

        def pytest_runtest_logreport(self, report):
            self.phases.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})

    base_args = ["-c", "/dev/null", "--rootdir=" + root, "--import-mode=prepend", "-p", "no:cacheprovider", *extra]
    if mode == "collect-only":  # child of the site-v1 run: which required modules import at all
        class Collect:
            def __init__(self):
                self.failed, self.items = {}, []

            def pytest_collectreport(self, report):
                if report.failed:
                    self.failed[report.nodeid] = str(report.longrepr)[-2000:]

            def pytest_collection_finish(self, session):
                self.items = [item.nodeid for item in session.items]
        c = Collect()
        code = pytest.main([*base_args, "--collect-only", "-q", "--continue-on-collection-errors", "--", *sys.argv[2:]],
                           plugins=[c])
        print(PREFIX + json.dumps({"exitcode": int(code), "failed_modules": c.failed, "collected": c.items}))
        return
    args = sys.argv[2:]
    uncollectable, collect_exit = {}, None
    if layout.exists():
        import subprocess
        modules = sorted({a.split("::", 1)[0] for a in args})
        child = subprocess.run([sys.executable, "-I", "-B", __file__, "collect-only", *modules],
                               capture_output=True, timeout=20, cwd=root)
        lines = [l for l in child.stdout.decode("utf-8", "replace").splitlines() if l.startswith(PREFIX)]
        if len(lines) != 1:
            print(PREFIX + json.dumps({"exitcode": 3, "collected": [], "phases": [], "junit": None,
                                       "collect_probe": "no envelope", "python": sys.version,
                                       "collect_stderr": child.stderr.decode("utf-8", "replace")[-3000:]}))
            raise SystemExit(3)
        probe_result = json.loads(lines[0][len(PREFIX):])
        collect_exit = probe_result["exitcode"]
        uncollectable = {m: e for m, e in probe_result["failed_modules"].items() if m in modules}
        args = [a for a in args if a.split("::", 1)[0] not in uncollectable]
    recorder = Recorder()
    junit = Path("/tmp/result.xml")
    # No project config, setup scripts or inferred test command; plugins only from the pinned runtime.
    code = None
    if args:
        code = int(pytest.main([*base_args, "-o", "junit_family=xunit2", "--junitxml=" + str(junit), "--tb=short",
                                "-rA", "--", *args], plugins=[recorder]))
    # A required module that cannot import fails its tests; any other abnormal exit stays abnormal.
    final = 1 if (uncollectable and code in (None, 0)) else (code if code is not None else 1)
    envelope = {"exitcode": final, "collected": recorder.collected, "phases": recorder.phases,
                "junit": junit.read_text() if junit.exists() else None, "python": sys.version,
                "pytest": pytest.__version__}
    if layout.exists():
        envelope.update(main_exitcode=code, collect_exitcode=collect_exit, uncollectable=uncollectable)
    print(PREFIX + json.dumps(envelope))
    raise SystemExit(final)


if __name__ == "__main__":
    main()

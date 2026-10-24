"""Resolve and fetch a hash-pinned site-v1 runtime from a snapshot's declared dependencies.

Ladder (decision #12 6238665963 R2), first success wins and is recorded:
  (a) declared specifiers, distributions uploaded no later than the task's created_at
  (b) declared specifiers, no date limit
  (c) names, extras and markers only
Binary wheels only (`--only-binary :all:`): no sdist is built or executed.
pytest is pinned to the harness version. Runs on the Linux host.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from oracle.runtime_site import rank
from senses.contract import SenseError, require

CODE = "MNLM_RUNTIME_RESOLUTION"
PYTEST = "pytest==8.3.3"
PYTEST_37 = "pytest==7.4.4"  # pytest 8 needs CPython >= 3.8; the 3.7 era harness pins the last 7.x


def pytest_pin(interpreter=None):
    return PYTEST_37 if interpreter and interpreter["python"].startswith("3.7.") else PYTEST
def target(interpreter=None):
    it = interpreter or {"python": "3.12.13", "glibc": "2.41"}
    libc = min(int(it["glibc"].split(".")[1]), 40)  # uv's newest manylinux target is 2_40
    return ["--python-version", it["python"], "--python-platform", f"x86_64-manylinux_2_{libc}"]
LADDER = (("declared_before_created_at", False, True), ("declared_any_date", False, False), ("names_only", True, False))
META_CAP, WHEEL_CAP = 4 << 20, 256 << 20
HARNESS_CLOSURE = ("pytest", "pluggy", "iniconfig", "packaging")
HARNESS_DATE = "2026-10-17T00:00:00Z"  # fixed decision date, not "now"


# Declared, recorded substitutions (decision #12 roadblocks). psycopg2-binary is the psycopg project's own
# wheel build of the same code and import name; pytest-capturelog was merged into pytest core (its logging
# plugin), which the harness pytest already provides.
SUBSTITUTE = {"psycopg2": "psycopg2-binary"}
PROVIDED_BY_HARNESS = {"pytest-capturelog"}


SETUPTOOLS_DEFAULT = "setuptools<82"  # 81.0.0 ships pkg_resources, 82.0.1 does not (measured wheels)


def inputs(declared, relaxed, extra=()):
    lines = set()
    for r in declared["runtime"] + declared["test"]:
        q = Requirement(r["requirement"])
        name = canonicalize_name(q.name)
        if name in PROVIDED_BY_HARNESS:
            continue
        if name in SUBSTITUTE:
            q.name = SUBSTITUTE[name]
        if relaxed:
            q.specifier, q.url = SpecifierSet(), None
        if q.url is None:
            lines.add(str(q))
    # pytest is supplied by the harness; setuptools is the environment default every venv of these
    # projects' era shipped with (pkg_resources, the distutils shim). Both are recorded in the lock.
    return sorted(lines | {"pytest", SETUPTOOLS_DEFAULT} | set(extra))


def rfc3339(created_at):
    return datetime.fromisoformat(created_at).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_lock(text):
    pins, current = [], None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s;\\]+)", line)
        if m:
            current = {"name": canonicalize_name(m[1]), "version": m[3], "hashes": []}
            pins.append(current)
        for h in re.findall(r"--hash=sha256:([0-9a-f]{64})", line):
            require(current is not None, CODE, "hash before requirement in lock")
            current["hashes"].append(h)
    require(all(p["hashes"] for p in pins), CODE, "lock entry without hashes")
    return pins


WHEEL_LESS = (r"versions of (\S+) have no usable wheels", r"Wheels are required for `([^`]+)`",
              r"Because (\S+)==\S+ has no usable wheels")
EXEMPT_ROUNDS = 10


def wheel_less(stderr):
    """Package names uv reports as lacking a usable wheel in the attempted range."""
    text = stderr.replace(chr(10), " ")
    return sorted({canonicalize_name(n.strip("`,.")) for pat in WHEEL_LESS for n in re.findall(pat, text)})


def resolve(declared, created_at, work, cache, find_links=None, rungs=("a", "b", "c"), interpreter=None, extra=()):
    """Rung (a) keeps the task date for every package and lifts it only for packages that uv names as
    having no usable cp312 wheel in range (iterated). Rungs (b) and (c) are the recorded fallbacks."""
    work.mkdir(parents=True, exist_ok=False)
    (work / "constraints.txt").write_text(pytest_pin(interpreter) + chr(10))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update(UV_CACHE_DIR=str(cache), UV_NO_CONFIG="1", UV_PYTHON_DOWNLOADS="never")
    attempts = []

    def attempt(n, rule, relaxed, dated, exempt):
        req = work / f"in-{n}.txt"
        req.write_text(chr(10).join(inputs(declared, relaxed, extra)) + chr(10))
        lock = work / f"lock-{n}.txt"
        argv = ["uv", "pip", "compile", str(req), "-c", str(work / "constraints.txt"), *target(interpreter),
                "--only-binary", ":all:", "--generate-hashes", "--no-header", "--no-annotate",
                "--index-url", "https://pypi.org/simple", "-o", str(lock), "--quiet"]
        if find_links is not None:  # sandbox-built wheels of sdist-only dependencies
            argv += ["--find-links", str(find_links)]
        if dated:  # the harness closure (and named wheel-less packages) are exempt from the task's date
            argv += ["--exclude-newer", rfc3339(created_at)]
            for name in sorted(set(HARNESS_CLOSURE) | exempt):
                argv += ["--exclude-newer-package", f"{name}={HARNESS_DATE}"]
        else:  # reproducible: never resolve against an unbounded "now"
            argv += ["--exclude-newer", HARNESS_DATE]
        started = time.perf_counter()
        try:
            p = subprocess.run(argv, env=env, capture_output=True, timeout=300)
            rc, err = p.returncode, p.stderr.decode("utf-8", "replace")[-4000:]
        except subprocess.TimeoutExpired:
            rc, err = None, "timeout 300 s"
        attempts.append({"rule": rule, "date_exempt": sorted(exempt), "returncode": rc, "stderr_tail": err,
                         "seconds": time.perf_counter() - started})
        return lock if rc == 0 else None

    n, exempt, plan = 0, set(), [("declared_before_created_at", False, True)] * EXEMPT_ROUNDS
    found = None
    for rule, relaxed, dated in plan if "a" in rungs else []:
        found = attempt(n, rule, relaxed, dated, exempt); n += 1
        if found:
            break
        more = set(wheel_less(attempts[-1]["stderr_tail"])) - exempt
        if not more:
            break
        exempt |= more
    if not found:
        for key, (rule, relaxed, dated) in zip("bc", LADDER[1:]):
            if key not in rungs:
                continue
            found = attempt(n, rule, relaxed, dated, set()); n += 1
            if found:
                break
    if found:
        pins = parse_lock(found.read_text())
        names = [p["name"] for p in pins]
        pinned = pytest_pin(interpreter).split("==")[1]
        require(len(names) == len(set(names)) and any(p["name"] == "pytest" and p["version"] == pinned for p in pins),
                CODE, "lock has duplicates or lacks the harness pytest")
        last = attempts[-1]
        names = {canonicalize_name(Requirement(r["requirement"]).name) for r in declared["runtime"] + declared["test"]}
        return {"rule": last["rule"], "attempt": n - 1, "date_exempt": last["date_exempt"], "pins": pins,
                "attempts": attempts, "substituted": sorted(names & set(SUBSTITUTE)),
                "provided_by_harness": sorted(names & PROVIDED_BY_HARNESS)}
    it = interpreter or {"python": "3.12.13", "glibc": "2.41"}
    exc = SenseError(CODE, f"no ladder rung resolved with binary wheels for CPython {it['python']}/glibc {it['glibc']}")
    exc.attempts = attempts  # retained in the case record; the case stays a runtime gap
    raise exc


class Fetcher:
    """Content-addressed PyPI metadata and wheel downloads with running byte totals."""

    def __init__(self, root, raw, meta_budget, wheel_budget):
        self.root, self.raw = root, raw
        self.budget = {"metadata": meta_budget, "wheels": wheel_budget}
        self.used = {"metadata": 0, "wheels": 0}
        (raw / "pypi").mkdir(parents=True, exist_ok=True)
        (raw / "wheels").mkdir(parents=True, exist_ok=True)

    def get(self, url, kind, cap):
        limit = min(cap, self.budget[kind] - self.used[kind])
        require(limit > 0, CODE, f"{kind} budget exhausted")
        req = urllib.request.Request(url, headers={"User-Agent": "MNLM-bounded-runtime-audit"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read(limit + 1)
        self.used[kind] += min(len(data), limit)
        require(len(data) <= limit, CODE, f"response above cap at {url}")
        return data

    def binding(self, path):
        data = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    def wheel(self, pin, built=None, interpreter=None):
        """Locked pin -> (wheel binding, metadata binding). `hashes=None` is only for the pinned build
        backend, which is checked against the PyPI digest alone. Built wheels come from `built`."""
        key = (pin["name"], pin["version"])
        if built and key in built:
            wheel, receipt = built[key]
            require(pin["hashes"] is not None and wheel["sha256"] in pin["hashes"], CODE,
                    "locally built wheel differs from the lock")
            return wheel, receipt
        meta_path = self.raw / "pypi" / f"{pin['name']}-{pin['version']}.json"
        if not meta_path.exists():
            data = self.get(f"https://pypi.org/pypi/{pin['name']}/{pin['version']}/json", "metadata", META_CAP)
            with meta_path.open("xb") as f:
                f.write(data)
        meta = json.loads(meta_path.read_bytes())
        candidates = sorted((rank(u["filename"], interpreter), u["filename"], u) for u in meta["urls"]
                            if u["packagetype"] == "bdist_wheel" and not u.get("yanked")
                            and rank(u["filename"], interpreter) is not None and (pin["hashes"] is None or u["digests"]["sha256"] in pin["hashes"]))
        require(bool(candidates), CODE, f"no compatible locked wheel for {pin['name']}=={pin['version']}")
        u = candidates[0][2]
        path = self.raw / "wheels" / u["filename"]
        if not path.exists():
            require(u["size"] <= WHEEL_CAP, CODE, f"wheel {u['filename']} above per-wheel cap")
            data = self.get(u["url"], "wheels", WHEEL_CAP)
            require(hashlib.sha256(data).hexdigest() == u["digests"]["sha256"], CODE, "wheel digest differs from PyPI")
            with path.open("xb") as f:
                f.write(data)
        b = self.binding(path)
        require(b["sha256"] == u["digests"]["sha256"], CODE, "stored wheel differs from PyPI digest")
        return b, self.binding(meta_path)

"""FSV for runtime layout site-v1 on one real prepared case.

Boundaries: (1) plan wheels -> unpacked site (every file traced to exactly one
pinned wheel, bytes equal); (2) snapshot -> declared record/source roots
(re-derived independently of the plan); (3) prepared payload -> read-time
verification. Broken-state controls mutate copies under the output only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import declared_deps
from oracle.prepare_swe import bytes_bound, prepare, source_files
from oracle.replay_swe import verify_prepared
from oracle.runtime_site import plugins, rank, unpack
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_RUNTIME_SITE_FSV"


def run(root, plan_path, plan_sha, prepared, prepared_sha, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, fn, expect=None):
        try:
            detail = fn()
            ok, rec = expect is None, {"detail": detail}
        except SenseError as exc:
            ok = expect is not None and exc.code == expect[0] and expect[1] in exc.message
            rec = {"refused": exc.envelope()}
        checks.append({"name": name, "passed": ok, **rec})
        print(name, "PASS" if ok else "FAIL", flush=True)

    plan = read_pinned(plan_path, plan_sha)
    spec, _ = verify_prepared(prepared, prepared_sha)
    site = prepared / "runtime" / "site"

    def trace():
        # Independent re-unpack straight from the zip bytes; no shared helper state.
        expected = {}
        for b in plan["wheels"]:
            with zipfile.ZipFile(io.BytesIO(bytes_bound(root, b))) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    p = PurePosixPath(info.filename)
                    if p.parts[0].endswith(".data"):
                        if len(p.parts) > 2 and p.parts[1] in ("purelib", "platlib"):
                            p = PurePosixPath(*p.parts[2:])
                        else:
                            continue
                    expected.setdefault(p.as_posix(), hashlib.sha256(z.read(info)).hexdigest())
        actual = {p.relative_to(site).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in site.rglob("*") if p.is_file()}
        require(actual == expected, CODE, "site differs from the pinned wheels")
        require(all(rank(Path(b["path"]).name, plan.get("interpreter")) is not None for b in plan["wheels"]),
                CODE, "incompatible tag")
        return {"files": len(actual), "wheels": len(plan["wheels"])}
    check("wheels-to-site-exact", trace)

    def declared():
        files = source_files(root, plan)
        record = declared_deps.read(files)
        layout = json.loads((prepared / "runtime" / "mnlm-runtime.json").read_bytes())
        require(record == plan["declared"] and layout["source_roots"] == [r["path"] for r in record["source_roots"]]
                and layout["plugins"] == plugins(site) == spec["plugins"], CODE, "declared/runtime record differs")
        return {"source_roots": layout["source_roots"], "plugins": layout["plugins"]}
    check("snapshot-to-declared-record", declared)

    work = output / "controls"
    work.mkdir()

    def wheel_bytes(entries):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)
        return buf.getvalue()

    def unpack_control(label, filename, entries, owners=None, expect=None):
        dst = work / label
        dst.mkdir()
        own = dict(owners or {})
        if owners:
            for rel in owners:
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                (dst / rel).write_bytes(b"first wheel bytes")
        check(label, lambda: unpack(wheel_bytes(entries), filename, dst, own, []), expect)

    unpack_control("incompatible-tag", "x-1.0-cp27-cp27mu-manylinux1_x86_64.whl", {"x.py": b""},
                   expect=("MNLM_EXECUTION_RUNTIME", "not compatible"))
    unpack_control("macos-only-tag", "x-1.0-cp312-cp312-macosx_11_0_arm64.whl", {"x.py": b""},
                   expect=("MNLM_EXECUTION_RUNTIME", "not compatible"))
    unpack_control("entry-escapes", "x-1.0-py3-none-any.whl", {"../evil.py": b"x"},
                   expect=("MNLM_EXECUTION_RUNTIME", "escapes"))
    unpack_control("conflicting-bytes", "x-1.0-py3-none-any.whl", {"shared.py": b"second"},
                   owners={"shared.py": "a-1.0-py3-none-any.whl"}, expect=("MNLM_EXECUTION_RUNTIME", "different bytes"))

    def prepare_control(label, change, expect):
        p = copy.deepcopy(plan)
        change(p)
        path = work / (label + ".json")
        sha = publish(path, p)
        check(label, lambda: prepare(root, path, sha, work / (label + "-out")), expect)
    prepare_control("declared-record-forged", lambda p: p["declared"]["test"].append(
        {"requirement": "forged==1", "source": "forged"}), ("MNLM_EXECUTION_PREPARATION", "declared dependencies differ"))
    prepare_control("wheel-binding-mutated", lambda p: p["wheels"][0].update(sha256="0" * 64),
                    ("MNLM_UCR_INPUT_DIGEST", "digest changed"))
    prepare_control("unknown-layout", lambda p: p.update(runtime_layout="site-v0"),
                    ("MNLM_EXECUTION_PREPARATION", "unknown runtime layout"))

    copied = work / "prepared"
    shutil.copytree(prepared, copied)
    victim = next(p for p in sorted((copied / "runtime" / "site").rglob("*.py")))
    victim.write_bytes(victim.read_bytes() + b"\n# mutated\n")
    check("site-file-mutated-at-read", lambda: verify_prepared(copied, prepared_sha),
          ("MNLM_EXECUTION_EVIDENCE", "prepared bytes changed"))
    shutil.rmtree(work)
    passed = sum(c["passed"] for c in checks)
    report = {"schema": "runtime-site-fsv-v1", "task": plan["task"], "checks": checks, "passed": passed,
              "failed": len(checks) - passed, "admitted": False, "author_checked": True}
    sha = publish(output / "report.json", report)
    print(json.dumps({"passed": passed, "failed": len(checks) - passed, "sha256": sha}))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "plan", "prepared", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("plan-sha256", "prepared-sha256"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    r = run(a.root, a.plan, a.plan_sha256, a.prepared, a.prepared_sha256, a.output)
    sys.exit(0 if r["failed"] == 0 else 1)

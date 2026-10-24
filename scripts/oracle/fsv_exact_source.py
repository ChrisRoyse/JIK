"""FSV for exact Git sources: export-subst blobs and recorded executable modes.

Real fsspec/parsel plans are read from the data root. Broken-state controls
mutate copies under the output directory only, never the retained raw bytes.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.prepare_swe import check_modes, export_subst, git_sha1, source_files
from oracle.replay_swe import verify_prepared
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_EXACT_SOURCE_FSV"


def rebind(root, binding, data):
    path = root / binding["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {**binding, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def copy_plan(src_root, dst_root, plan):
    for key in ("tree", "commit", "archive"):
        rebind(dst_root, plan[key], (src_root / plan[key]["path"]).read_bytes())
    for b in plan.get("exact_blobs", []):
        rebind(dst_root, b, (src_root / b["path"]).read_bytes())
    return copy.deepcopy(plan)


def run(root, fsspec_plan, fsspec_sha, parsel_plan, parsel_sha, prepared, prepared_sha, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, fn, expect=None):
        try:
            detail = fn()
            ok = expect is None
            outcome = {"name": name, "passed": ok, "detail": detail}
        except SenseError as exc:
            ok = expect is not None and exc.code == expect[0] and expect[1] in exc.message
            outcome = {"name": name, "passed": ok, "refused": exc.envelope()}
        checks.append(outcome)
        print(name, "PASS" if ok else "FAIL", flush=True)

    plan = read_pinned(fsspec_plan, fsspec_sha)
    tree = json.loads((root / plan["tree"]["path"]).read_bytes())
    entry = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    name = plan["exact_blobs"][0]["name"]
    raw_before = {k: hashlib.sha256((root / plan[k]["path"]).read_bytes()).hexdigest() for k in ("tree", "commit", "archive")}

    # Boundary 1: archive -> exact source. Before: the transformed archive is refused.
    bare = {k: v for k, v in plan.items() if k != "exact_blobs"}
    check("before-archive-only-refused", lambda: source_files(root, bare),
          ("MNLM_EXECUTION_PREPARATION", "source differs from pinned Git blob"))

    def after():
        files = source_files(root, plan)
        require(git_sha1(files[name]) == entry[name]["sha"] and files.keys() == entry.keys()
                and all(git_sha1(d) == entry[n]["sha"] for n, d in files.items()), CODE, "exact snapshot")
        return {"files": len(files), "bytes": sum(map(len, files.values())), "replaced": [name],
                "replaced_bytes": len(files[name])}
    check("after-exact-blob-accepted", after)

    work = output / "controls"
    def mutated(label, change, expect):
        dst = work / label
        p = copy_plan(root, dst, plan)
        change(dst, p)
        check(label, lambda: source_files(dst, p), expect)

    def blob_bytes(dst, p, edit):
        b = p["exact_blobs"][0]
        j = json.loads((dst / b["path"]).read_bytes())
        edit(j)
        p["exact_blobs"][0] = {**rebind(dst, b, json.dumps(j).encode()), "name": b["name"]}

    def wrong_content(j):
        data = base64.b64decode(j["content"])
        j["content"] = base64.b64encode(data[:-1] + b"#").decode()
    mutated("blob-content-mutated", lambda d, p: blob_bytes(d, p, wrong_content),
            ("MNLM_EXECUTION_PREPARATION", "exact blob bytes differ"))
    mutated("blob-sha-field-mutated", lambda d, p: blob_bytes(d, p, lambda j: j.update(sha="0" * 40)),
            ("MNLM_EXECUTION_PREPARATION", "not the pinned tree entry"))
    mutated("blob-duplicated", lambda d, p: p["exact_blobs"].append(dict(p["exact_blobs"][0])),
            ("MNLM_EXECUTION_PREPARATION", "not the pinned tree entry"))

    def superfluous(dst, p):
        with tarfile.open(dst / p["archive"]["path"]) as tar:
            member = next(m for m in tar if m.isfile() and m.name.endswith("/setup.py"))
            data = tar.extractfile(member).read()
        target = "setup.py"
        response = {"sha": entry[target]["sha"], "size": len(data), "encoding": "base64",
                    "content": base64.b64encode(data).decode()}
        b = {"name": target, "path": p["exact_blobs"][0]["path"].replace(".json", "-setup.json")}
        p["exact_blobs"].append({**rebind(dst, b, json.dumps(response).encode()), "name": target})
    mutated("blob-superfluous-unchanged-member", superfluous,
            ("MNLM_EXECUTION_PREPARATION", "unchanged member"))

    def archive_member(dst, p):
        src = dst / p["archive"]["path"]
        out = io.BytesIO()
        with tarfile.open(src) as tin, tarfile.open(fileobj=out, mode="w:gz") as tout:
            for m in tin:
                data = tin.extractfile(m).read() if m.isfile() else None
                if m.isfile() and m.name.endswith("/setup.py"):
                    data = data.replace(b"setup(", b"setup (", 1)
                    m.size = len(data)
                tout.addfile(m, io.BytesIO(data) if data is not None else None)
        p["archive"] = rebind(dst, p["archive"], out.getvalue())
    mutated("archive-other-member-mutated", archive_member,
            ("MNLM_EXECUTION_PREPARATION", "source differs from pinned Git blob"))

    # .gitattributes rule: the real file declares exactly this path; other spellings must not match.
    real = b"fsspec/_version.py export-subst\n"
    cases = {"real-declares": (real, True), "other-path": (b"fsspec/other.py export-subst\n", False),
             "attribute-absent": (b"fsspec/_version.py text\n", False),
             "negated-comment": (b"# fsspec/_version.py export-subst\n", False),
             "basename-pattern": (b"_version.py export-subst\n", True),
             "wrong-depth": (b"_version.py/x export-subst\n", False)}
    check("gitattributes-rule", lambda: require(all(export_subst(a, name) == e for a, e in cases.values()),
                                                CODE, "export-subst matcher") or sorted(cases))

    # Boundary 2: Git tree mode -> archive exec bit -> prepared filesystem.
    pplan = read_pinned(parsel_plan, parsel_sha)
    ptree = json.loads((root / pplan["tree"]["path"]).read_bytes())
    exe = sorted(r["path"] for r in ptree["tree"] if r.get("mode") == "100755")
    modes = {}
    check("modes-recorded", lambda: (source_files(root, pplan, modes),
          require(sorted(n for n, m in modes.items() if m == "100755") == exe, CODE, "modes")) and exe)

    def tree_mode(new_mode, label, expect):
        dst = work / label
        p = copy_plan(root, dst, pplan)
        t = json.loads((dst / p["tree"]["path"]).read_bytes())
        next(r for r in t["tree"] if r["path"] == exe[0])["mode"] = new_mode
        # The commit binds the tree sha, not its JSON bytes; the mode check is what must refuse.
        p["tree"] = rebind(dst, p["tree"], json.dumps(t).encode())
        check(label, lambda: source_files(dst, p), expect)
    tree_mode("100644", "tree-mode-cleared", ("MNLM_EXECUTION_PREPARATION", "executable bit differs"))
    tree_mode("120000", "tree-mode-symlink", ("MNLM_EXECUTION_PREPARATION", "unsupported source entry"))

    spec, _ = verify_prepared(prepared, prepared_sha)
    check("prepared-modes", lambda: [check_modes(prepared / c, spec.get("executables", [])) for c in spec["conditions"]])
    if os.name == "posix":
        regular = next(n for n in sorted(spec["conditions"]["base"]) if n not in exe)
        for label, target, mode in (("exec-bit-cleared", exe[0], 0o644), ("exec-bit-added", regular, 0o755)):
            dst = work / label
            shutil.copytree(prepared, dst)
            (dst / "base" / target).chmod(mode)
            check(label, lambda: verify_prepared(dst, prepared_sha),
                  ("MNLM_EXECUTION_PREPARATION", "executable bits differ"))

    raw_after = {k: hashlib.sha256((root / plan[k]["path"]).read_bytes()).hexdigest() for k in ("tree", "commit", "archive")}
    check("raw-unchanged", lambda: require(raw_before == raw_after, CODE, "retained raw changed") or raw_after)
    shutil.rmtree(work)
    passed = sum(c["passed"] for c in checks)
    report = {"schema": "exact-source-fsv-v1", "host": os.name, "checks": checks, "passed": passed,
              "failed": len(checks) - passed, "admitted": False, "author_checked": True}
    sha = publish(output / "report.json", report)
    print(json.dumps({"passed": passed, "failed": len(checks) - passed, "sha256": sha}))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "fsspec-plan", "parsel-plan", "prepared", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("fsspec-sha256", "parsel-sha256", "prepared-sha256"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    r = run(a.root, a.fsspec_plan, a.fsspec_sha256, a.parsel_plan, a.parsel_sha256, a.prepared, a.prepared_sha256, a.output)
    sys.exit(0 if r["failed"] == 0 else 1)

"""One task from pinned source to a witnessed outcome or a named gap, with every recorded fix (#12).

Predeclared runtime candidates, in order: CPython 3.12 dated, 3.12 names-only, era CPython dated,
era names-only (era = newest of 3.7..3.11 released by the task date). Within a candidate, up to six
measured-closure rounds add a third-party distribution only after the reference run reported
`No module named X`. Sdist-only dependencies are built in the sandbox (compiler image when needed).
Plans declare SWE-bench evaluation semantics (agent patch fallback, test files reset to base),
setuptools_scm version files and truncated test-ID completion where the snapshot calls for them.
The agent condition never runs before the environment stage passes, so it never selects a runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from oracle import declared_deps
from oracle.build_sdist import build as build_sdist
from oracle.complete_ids import RULE as COMPLETION_RULE, complete, truncated
from oracle.fsv_reexecution import run as fsv
from oracle.prepare_swe import patch_paths, prepare, source_files
from oracle.replay_swe import ENVIRONMENT, run as execute
from oracle.resolve_runtime import Fetcher, resolve, wheel_less
from oracle.runtime_site import LAYOUT
from oracle.scm_version import generated as scm_generated
from senses.contract import SenseError
from senses.trajectory_population import publish

RUNTIME = {  # measured 2026-10-17 on aiwonder: (slim runtime image, interpreter, compiler image)
    "3.7": ("sha256:a255ffcb469f2ec40f2958a76beb0c2bbebfe92ce9af67a9b48d84b4cb695ac8", {"python": "3.7.17", "glibc": "2.34"},
            "sha256:16d93ae3411be3db255b6b52fdfc155a0dff0f697c2e4e3d862caf8d978830b2"),
    "3.8": ("sha256:b5f62925bd0f63f48cc8acd5e87d0c3a07e2f229cd2fb0a9586e68ed17f45ee3", {"python": "3.8.20", "glibc": "2.36"},
            "sha256:3ea6eaad4f175bd42f39dae10098b1820ee522628ff04670b3ac8b89fe00c1c8"),
    "3.9": ("sha256:085da638e1b8a449514c3fda83ff50a3bffae4418b050cfacd87e5722071f497", {"python": "3.9.25", "glibc": "2.41"},
            "sha256:bcd3da5974912584a81ed86fd944ab5fba9093ff1c9a0b0ed18349f9a69e4762"),
    "3.10": ("sha256:0f61d91517f6602b9d99096696845f54ddfe01a52029b4f55fdddb277ebe034a", {"python": "3.10.22", "glibc": "2.41"},
             "sha256:44d5bf756e00d5a2653ec9ce52d1939acbc8b90568babfae8f353a4739c7b1e4"),
    "3.11": ("sha256:541096ae9b05d3b850341b78790e2ed31e94d5aa0ea086933c6ff0c09ffa2e63", {"python": "3.11.17", "glibc": "2.41"},
             "sha256:855c886888c435587b6c0884eef92acbd5e28fd78c265dcdc0122af214cc0821"),
    "3.12": ("sha256:25c5b8011a3425a140bf5fa73be0feabd3c0d5b323eecb19dc02437a368ae075", None,
             "sha256:720762d97dcd513c0e9e578cbc6cd97bb633f8f9e268ef75b73fd4f4621ea200"),
}
RELEASES = (("3.8", "2019-10-14"), ("3.9", "2020-10-05"), ("3.10", "2021-10-04"), ("3.11", "2022-10-24"), ("3.12", "2023-10-02"))
MODULE_TO_DIST = {"yaml": "PyYAML", "dateutil": "python-dateutil", "bs4": "beautifulsoup4", "PIL": "Pillow",
                  "attr": "attrs", "sklearn": "scikit-learn", "cv2": "opencv-python-headless", "jwt": "PyJWT",
                  "OpenSSL": "pyOpenSSL", "Crypto": "pycryptodome", "magic": "python-magic", "dotenv": "python-dotenv",
                  "serial": "pyserial", "usb": "pyusb", "git": "GitPython", "jose": "python-jose", "zmq": "pyzmq",
                  "slugify": "python-slugify", "Levenshtein": "python-Levenshtein", "pkg_resources": "setuptools",
                  "ruamel": "ruamel.yaml", "nacl": "PyNaCl", "docx": "python-docx", "tomli_w": "tomli-w",
                  "google": "protobuf", "skimage": "scikit-image", "Bio": "biopython", "github": "PyGithub"}
STDLIB_REMOVED = {"imp", "distutils", "asynchat", "asyncore", "smtpd"}
ROUNDS = 6
DECISION = "https://github.com/ChrisRoyse/MNLM/issues/12#issuecomment-6241921740"


def era(created_at):
    day, chosen = created_at[:10], "3.7"
    for version, released in RELEASES:
        if released <= day:
            chosen = version
    return chosen


def candidates(created_at):
    e = era(created_at)
    out = [("3.12", "a"), ("3.12", "c")]
    if e != "3.12":
        out += [(e, "a"), (e, "c")]
    return out


def missing_module(exe, files, pins):
    for cond in ("reference", "base"):
        d = exe / cond
        if not (d / "stdout.bin").exists():
            continue
        text = (d / "stdout.bin").read_text(errors="replace") + (d / "stderr.bin").read_text(errors="replace")
        for m in re.findall(r"No module named '([\w.]+)'", text):
            top = m.split(".")[0]
            first_party = any(PurePosixPath(n).parts[0] == top or PurePosixPath(n).parts[:2] == ("src", top)
                              or n in (top + ".py", "src/" + top + ".py") for n in files)
            if first_party or top in sys.stdlib_module_names or top in STDLIB_REMOVED:
                return None, f"missing module {m!r} is first-party or stdlib"
            dist = MODULE_TO_DIST.get(top, top)
            if canonicalize_name(dist) in pins:
                return None, f"{m!r} missing although {dist} is installed"
            return dist, f"reference output reported missing module {m!r} -> {dist}"
    return None, "no missing-module signature"


class Builds:
    """Per-interpreter sandbox-built wheels and recorded failures, reused across tasks."""

    def __init__(self, root, ev, fetcher):
        self.root, self.ev, self.fetcher, self.built, self.failed = root, ev, fetcher, {}, {}

    def load(self, v):
        if v in self.built:
            return
        self.built[v], self.failed[v] = {}, {}
        for receipt in sorted((self.ev / f"sdist-builds-py{v}").rglob("receipt.json")):
            rec = json.loads(receipt.read_bytes())
            data = receipt.read_bytes()
            self.built[v][(rec["package"], rec["version"])] = (rec["wheel"], {"path": receipt.relative_to(self.root).as_posix(),
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def links(self, v):
        d = self.root / f"raw/swe-runtime-v2/built-py{v}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def make(self, v, pkg, specifier, log):
        if pkg in self.failed[v]:
            raise SenseError("MNLM_RUNTIME_RESOLUTION", f"sdist-only dependency {pkg} did not build: {self.failed[v][pkg][:200]}")
        image, it, build_image = RUNTIME[v]
        try:
            try:  # pure first in the runtime image; the compiler image only if that fails
                wb, rb = build_sdist(self.fetcher, image, pkg, specifier, self.ev / f"sdist-builds-py{v}",
                                     interpreter=it, built_dir=f"built-py{v}")
            except SenseError:
                wb, rb = build_sdist(self.fetcher, image, pkg, specifier, self.ev / f"sdist-builds-py{v}",
                                     interpreter=it, built_dir=f"built-py{v}", build_image=build_image)
            self.built[v][(pkg, Path(wb["path"]).name.split("-")[1])] = (wb, rb)
            log.append({"package": pkg, "ok": True})
        except (SenseError, FileExistsError) as exc:
            self.failed[v][pkg] = str(exc); log.append({"package": pkg, "ok": False, "error": str(exc)[:300]})
            raise SenseError("MNLM_RUNTIME_RESOLUTION", f"sdist-only dependency {pkg} did not build: {str(exc)[:200]}")


def run_case(root, ev, case, rows, fetcher, builds, cache, out):
    """case: {index, task, src, benchmark, attempt, test_modules}; rows: BoundRows covering both bindings."""
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    i, task = case["index"], case["task"]
    files = source_files(root, case["src"])
    declared = declared_deps.read(files)
    m = rows.read(case["benchmark"], ["instance_id", "repo", "base_commit", "environment_setup_commit",
                                      "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "test_patch", "created_at"])
    a = rows.read(case["attempt"], ["instance_id", "generated_patch", "target"])
    if m["instance_id"] != task or a["instance_id"] != task:
        raise SenseError("MNLM_PIPELINE", "benchmark/attempt rows are another task")
    empty = not (a["generated_patch"] or "").strip()
    allowed = {k: sorted(patch_paths(t)) for k, t in (("reference", m["patch"]), ("test", m["test_patch"]))}
    allowed["agent"] = [] if empty else sorted(patch_paths(a["generated_patch"]))
    required = list(m["FAIL_TO_PASS"]) + list(m["PASS_TO_PASS"])
    needs_completion = any(truncated(n) for n in required)
    specs = {}
    for r in declared["runtime"] + declared["test"]:
        q = Requirement(r["requirement"]); specs.setdefault(canonicalize_name(q.name), []).append(str(q.specifier))
    gen = [{"path": g["path"], "rule": g["rule"], "sha256": hashlib.sha256(g["content"]).hexdigest()} for g in scm_generated(files)]
    runs = []
    for v, rung in candidates(m["created_at"]):
        image, it, _ = RUNTIME[v]
        builds.load(v)
        extra = []
        for rnd in range(ROUNDS):
            tag = f"{i}-{v}-{rung}-{rnd}"; rec = {"tag": tag, "python": v, "rung": rung, "extra": list(extra)}
            t0, res = time.perf_counter(), None
            try:
                for attempt_build in range(4):  # sandbox builds for sdist-only dependencies, then re-resolve
                    try:
                        res = resolve(declared, m["created_at"], out / "resolution" / f"{tag}-b{attempt_build}", cache,
                                      find_links=builds.links(v), rungs=(rung,), interpreter=it, extra=extra)
                        break
                    except SenseError as exc:
                        want = [p for p in wheel_less(exc.attempts[-1]["stderr_tail"]) if not any(k[0] == p for k in builds.built[v])]
                        if not want or attempt_build == 3:
                            raise
                        rec.setdefault("builds", [])
                        for pkg in want:
                            builds.make(v, pkg, ",".join(x for x in specs.get(pkg, []) if x), rec["builds"])
                fetched = [fetcher.wheel(p, builds.built[v], it) for p in res["pins"]]
                plan = {**case["src"], "schema": "swe-execution-plan-v1", "image": image, "test_modules": case["test_modules"],
                        "decision": DECISION, "benchmark": case["benchmark"], "attempt": case["attempt"], "allowed_paths": allowed,
                        "runtime_layout": LAYOUT, "declared": declared, "wheels": [w for w, _ in fetched],
                        "wheel_metadata": [x for _, x in fetched], "created_at": m["created_at"],
                        "agent_patch_fallback": True, "reset_test_files": True,
                        "resolution": {"rule": res["rule"], "attempt": res["attempt"], "python": v, "round": rnd,
                                       "extra": list(extra), "pins": [{"name": p["name"], "version": p["version"]} for p in res["pins"]],
                                       "attempts": res["attempts"], "substituted": res.get("substituted", []),
                                       "provided_by_harness": res.get("provided_by_harness", []), "builds": rec.get("builds", [])}}
                if it:
                    plan["interpreter"] = it
                if empty:
                    plan["agent_patch_empty"] = True
                if gen:
                    plan["generated_files"] = gen
                if needs_completion:
                    mapping, evidence = complete(root, plan, required, out / "ids" / tag)
                    plan["test_id_completion"] = {"map": mapping, "evidence": evidence, "rule": COMPLETION_RULE}
                pb = out / "plans" / f"{tag}.json"; psha = publish(pb, plan)
                prep = out / "prepared" / tag; exe = out / "execution" / tag
                prepare(root, pb, psha, prep, rows=rows)
                result = execute(prep, sha(prep / "prepared.json"), exe)
                proof = fsv(prep, sha(prep / "prepared.json"), exe, sha(exe / "report.json"), out / "fsv" / tag)
                report = json.loads((exe / "report.json").read_bytes())
                rec.update(status="observed", plan={"path": pb.relative_to(root).as_posix(), "sha256": psha},
                           prepared=str(prep), execution=str(exe), conditions=result["conditions"],
                           agent_outcome=report["conditions"]["agent"]["outcome"], fsv_report_sha256=proof)
            except SenseError as exc:
                rec.update(status="gap", error=exc.envelope())
                if exc.code == ENVIRONMENT and res is not None:
                    dist, why = missing_module(out / "execution" / tag, files, {canonicalize_name(p["name"]) for p in res["pins"]})
                    rec["closure"] = why
                    if dist and dist not in extra:
                        extra.append(dist); rec["seconds"] = time.perf_counter() - t0; runs.append(rec)
                        continue
            except Exception as exc:  # recorded per case, never silently dropped
                rec.update(status="gap", error={"code": "MNLM_PIPELINE", "message": str(exc)[:400],
                                                 "trace": traceback.format_exc()[-1500:]})
            rec["seconds"] = time.perf_counter() - t0
            runs.append(rec)
            break
        last = runs[-1]
        if last["status"] == "observed":
            break
        if last["error"]["code"] not in (ENVIRONMENT, "MNLM_RUNTIME_RESOLUTION", "MNLM_TEST_ID_COMPLETION"):
            break  # an agent-stage, patch or evidence refusal is terminal: never try another runtime after it
    last = runs[-1]
    return {"index": i, "task": task, "status": last["status"], "runs": runs,
            **({"error": last["error"]} if last["status"] == "gap" else
               {"agent_outcome": last["agent_outcome"], "plan": last["plan"], "prepared": last["prepared"],
                "execution": last["execution"]})}

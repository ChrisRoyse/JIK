"""Prepare hash-bound SWE execution evidence without executing project code."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import zipfile

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, require
from senses.trajectory_population import open_bound, publish, read_pinned
from oracle import declared_deps
from oracle.runtime_site import LAYOUT, plugins, unpack
from oracle.test_contract import selectors

CODE = "MNLM_EXECUTION_PREPARATION"
BENCHMARK_COLUMNS = ["instance_id", "repo", "base_commit", "environment_setup_commit",
                     "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "test_patch"]
ATTEMPT_COLUMNS = ["instance_id", "generated_patch", "target"]


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            CODE, "portable relative path required")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name,
            CODE, "path escapes execution payload")
    return name


def bytes_bound(root, binding):
    relative(binding["path"])
    with open_bound(root, binding) as stream:
        return stream.read()


def row_bound(root, binding, columns):
    with open_bound(root, binding) as stream:
        parquet = pq.ParquetFile(stream)
        index = binding["row"]
        require(type(index) is int and 0 <= index < parquet.metadata.num_rows, CODE, "row outside source")
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if index < count:
                return parquet.read_row_group(group, columns=columns).slice(index, 1).to_pylist()[0]
            index -= count
    raise SenseError(CODE, "source row missing")


def inventory(root):
    result = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), CODE, "symlinks are not execution payloads")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def patch_paths(text):
    """Every path a Git patch touches. Edits, new regular files, deletions and exact renames
    (`rename from/to` matching the diff header) are supported; mode changes, copies and binary
    patches refuse."""
    require(isinstance(text, str) and text.strip(), CODE, "nonempty patch required")
    paths, current, renames = set(), None, {}
    for line in text.splitlines():
        if line.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            require(match is not None, CODE, "quoted or unparseable patch paths unsupported")
            current = (relative(match[1]), relative(match[2]))
            paths.update(current)
            if current[0] != current[1]:
                renames[current] = set()
        elif line.startswith(("rename from ", "rename to ")):
            side = 0 if line.startswith("rename from ") else 1
            require(current is not None and current[0] != current[1] and line.split(" ", 2)[2] == current[side],
                    CODE, "rename header differs from the diff paths")
            renames[current].add(side)
        elif line.startswith(("--- ", "+++ ")):
            name = line[4:]
            if name != "/dev/null":
                require(name.startswith(("a/", "b/")) and relative(name[2:]) in paths,
                        CODE, "patch header differs from declared path")
        elif line.startswith(("old mode ", "new mode ", "copy from ", "copy to ", "GIT binary patch")):
            raise SenseError(CODE, "mode changes, copies and binary patches unsupported")
        elif line.startswith("deleted file mode "):
            require(line in ("deleted file mode 100644", "deleted file mode 100755"), CODE, "deleted entry is not a regular file")
        elif line.startswith("new file mode "):
            require(line in ("new file mode 100644", "new file mode 100755"), CODE, "new files must be regular files")
    require(all(sides == {0, 1} for sides in renames.values()), CODE, "a changed path without rename headers")
    require(bool(paths), CODE, "patch has no supported file headers")
    return paths


def new_executables(text):
    """Paths a patch creates with mode 100755 (git apply sets the bit on POSIX)."""
    found, current = set(), None
    for line in (text or "").splitlines():
        if line.startswith("diff --git "):
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            current = m[2] if m else None
        elif line == "new file mode 100755" and current:
            found.add(current)
    return found


FUZZ = ["patch", "--batch", "--fuzz=5", "-p1", "--no-backup-if-mismatch", "--reject-file=-"]


def apply_patch(tree, patch, text, allowed, fallback=False):
    """`git apply` exactly; with fallback (agent submissions only) the SWE-bench harness's own second
    step, `patch --batch --fuzz=5 -p1`. Returns the method used. Changed files must equal the patch's
    declared paths either way."""
    paths = patch_paths(text)
    require(paths == set(allowed), CODE, "patch paths differ from the pinned allowlist")
    before = inventory(tree)
    patch.write_text(text, encoding="utf-8", newline="")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_CEILING_DIRECTORIES=str(tree.parent.resolve()))
    method = "git-apply"
    check = subprocess.run(["git", "apply", "--whitespace=nowarn", "--check", str(patch.resolve())],
                           cwd=tree, env=env, capture_output=True, timeout=10)
    if check.returncode == 0:
        result = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch.resolve())],
                                cwd=tree, env=env, capture_output=True, timeout=10)
    elif fallback:
        method = "patch-fuzz5"
        dry = subprocess.run([*FUZZ, "--dry-run", "-i", str(patch.resolve())], cwd=tree, capture_output=True, timeout=10)
        require(dry.returncode == 0, "MNLM_EXECUTION_PATCH_REJECTED",
                "git apply: " + check.stderr.decode("utf-8", "replace")[:500]
                + " | patch --fuzz=5: " + dry.stdout.decode("utf-8", "replace")[:500])
        result = subprocess.run([*FUZZ, "-i", str(patch.resolve())], cwd=tree, capture_output=True, timeout=10)
    else:
        result = check
    require(result.returncode == 0, "MNLM_EXECUTION_PATCH_REJECTED",
            (result.stderr or result.stdout).decode("utf-8", "replace")[:1000])
    after = inventory(tree)
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    require(changed == paths, CODE, "patch changed an undeclared file or failed to change a declared file")
    return method


def reset_test_files(tree, files, test_paths):
    """SWE-bench evaluation step `git checkout <base> -- <test files>` before the test patch: files the
    test patch touches return to their pinned base bytes (or are removed if new). Returns reset paths."""
    reset = []
    for path in test_paths:
        target = tree / relative(path)
        if path in files:
            if not target.exists() or target.read_bytes() != files[path]:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files[path]); reset.append(path)
        elif target.exists():
            target.unlink(); reset.append(path)
    return sorted(reset)


MODES = {"100644": False, "100755": True}  # Git blob mode -> executable


def git_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def export_subst(attributes, name):
    """Whether the snapshot's root .gitattributes declares export-subst for name."""
    for line in attributes.decode("utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or "export-subst" not in fields[1:]:
            continue
        pattern = fields[0].lstrip("/")
        subject = name if "/" in fields[0].rstrip("/") else PurePosixPath(name).name
        if PurePosixPath(subject).match(pattern) and subject.count("/") == pattern.count("/"):
            return True
    return False


def exact_blobs(root, plan, blobs):
    """Git-blob API responses that replace export-subst archive members, SHA-1 checked."""
    result = {}
    for binding in plan.get("exact_blobs", []):
        name = relative(binding["name"])
        response = json.loads(bytes_bound(root, binding))
        require(name in blobs and name not in result and response.get("encoding") == "base64"
                and response.get("sha") == blobs[name]["sha"], CODE, "exact blob is not the pinned tree entry")
        data = base64.b64decode(response["content"], validate=False)
        require(len(data) == blobs[name]["size"] == response.get("size") and git_sha1(data) == blobs[name]["sha"],
                CODE, "exact blob bytes differ from the pinned Git blob")
        result[name] = data
    return result


def source_files(root, plan, modes=None):
    """One pinned Git snapshot; no benchmark, trajectory or evaluation reads.

    Archive bytes must equal the pinned blobs, except members whose snapshot
    .gitattributes declares export-subst: those come from a bound exact blob,
    and only when the archive member actually differs.
    """
    tree = json.loads(bytes_bound(root, plan["tree"]))
    commit = json.loads(bytes_bound(root, plan["commit"]))
    require(commit["sha"] == plan["base_commit"] and tree["sha"] == commit["tree"]["sha"]
            and tree["truncated"] is False, CODE, "Git tree is not the pinned commit's complete tree")
    blobs = {relative(r["path"]): r for r in tree["tree"] if r["type"] == "blob"}
    require(all(r["type"] in ("blob", "tree") for r in tree["tree"])
            and all(r["mode"] in MODES for r in blobs.values()), CODE,
            "unsupported source entry (symlink, gitlink or special mode)")
    replace = exact_blobs(root, plan, blobs)
    files, transformed = {}, set()
    archive = bytes_bound(root, plan["archive"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isdir():
                continue
            require(member.isfile(), CODE, "archive links/devices refused")
            prefix, sep, name = member.name.partition("/")
            require(sep and prefix.endswith("-" + plan["base_commit"]), CODE, "archive commit prefix differs")
            relative(name)
            require(name in blobs and name not in files, CODE, "archive inventory differs from Git tree")
            require(bool(member.mode & 0o100) == MODES[blobs[name]["mode"]], CODE,
                    "archive executable bit differs from Git tree mode")
            data = tar.extractfile(member).read()
            if name in replace:
                require(git_sha1(data) != blobs[name]["sha"], CODE, "exact blob supplied for an unchanged member")
                transformed.add(name)
                data = replace[name]
            require(len(data) == blobs[name]["size"] and git_sha1(data) == blobs[name]["sha"],
                    CODE, "source differs from pinned Git blob")
            files[name] = data
    require(files.keys() == blobs.keys(), CODE, "archive is incomplete")
    require(transformed == replace.keys(), CODE, "exact blob has no archive member")
    if transformed:
        attributes = files.get(".gitattributes", b"")
        require(all(export_subst(attributes, n) for n in transformed), CODE,
                "exact blob replaces a member that .gitattributes does not declare export-subst")
    if modes is not None:
        modes.update((n, blobs[n]["mode"]) for n in files)
    return files


def executables(tree, names):
    """Apply recorded Git executable modes where the filesystem can carry them."""
    if os.name == "posix":
        for name in names:
            path = tree / name
            path.chmod(path.stat().st_mode | 0o111)


def check_modes(tree, names):
    """On POSIX every executable bit must be exactly the recorded set."""
    if os.name != "posix":
        return "not_representable_on_" + os.name
    actual = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*")
                    if p.is_file() and p.stat().st_mode & 0o111)
    require(actual == sorted(names), CODE, "executable bits differ from the recorded Git modes")
    return "checked"


def prepare(root, plan_path, plan_sha256, output, rows=None):
    started = time.perf_counter()
    plan = read_pinned(plan_path, plan_sha256)
    require(plan["schema"] == "swe-execution-plan-v1" and not output.exists(), CODE, "plan/output contract")
    modes = {}
    files = source_files(root, plan, modes)
    execs = sorted(n for n, m in modes.items() if MODES[m])
    source_done = time.perf_counter()
    if rows is not None:
        require(rows.root == Path(root).resolve(), CODE, "batch source root differs")
    read = (lambda b, c: row_bound(root, b, c)) if rows is None else rows.read
    metadata = read(plan["benchmark"], BENCHMARK_COLUMNS)
    attempt = read(plan["attempt"], ATTEMPT_COLUMNS)
    require(metadata["instance_id"] == attempt["instance_id"] == plan["task"]
            and metadata["base_commit"] == metadata["environment_setup_commit"] == plan["base_commit"],
            CODE, "task/base/environment binding differs")
    roles = {}
    completion = plan.get("test_id_completion")
    if completion is not None:  # whitespace-truncated IDs completed from the reference tree's collection
        from oracle.complete_ids import check
        evidence = json.loads(bytes_bound(root, completion["evidence"]))
        check(evidence["result"]["collected"], completion["map"])
    for role in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        values = metadata[role]
        values = json.loads(values) if isinstance(values, str) else values
        if completion is not None:
            values = [completion["map"].get(v, v) for v in values]
        require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
                and len(values) == len(set(values)) and not (set(values) & roles.keys()), CODE, "invalid required tests")
        roles.update((v, role) for v in values)
    require(any(r == "FAIL_TO_PASS" for r in roles.values()), CODE, "empty fail-to-pass contract")
    exact = selectors(roles)
    require(plan["test_modules"] == sorted({n.split("::", 1)[0] for n in exact}),
            CODE, "declared modules differ from required test inventory")
    case_done = time.perf_counter()
    output.mkdir(parents=True)
    wheels = output / "runtime"
    wheels.mkdir()
    layout = plan.get("runtime_layout")
    require(layout in (None, LAYOUT), CODE, "unknown runtime layout")
    if layout is None:  # earlier plans: pure wheels on sys.path as zips
        for binding in plan["wheels"]:
            data = bytes_bound(root, binding)
            with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                require(all(not n.endswith((".so", ".pyd", ".pth")) for n in wheel.namelist()),
                        CODE, "only pure Python wheels without startup hooks")
            (wheels / Path(binding["path"]).name).write_bytes(data)
    else:
        declared = declared_deps.read(files)
        require(plan["declared"] == declared, CODE, "plan's declared dependencies differ from the snapshot's own")
        roots = [r["path"] for r in declared["source_roots"]]
        require(all(relative(r) and any(n.startswith(r + "/") for n in files) for r in roots),
                CODE, "declared source root absent from the snapshot")
        site, owners, skipped = wheels / "site", {}, []
        site.mkdir()
        names = [Path(b["path"]).name for b in plan["wheels"]]
        require(len(names) == len(set(names)), CODE, "duplicate wheel in runtime")
        for binding in plan["wheels"]:
            unpack(bytes_bound(root, binding), Path(binding["path"]).name, site, owners, skipped,
                   plan.get("interpreter"))
        runtime_spec = {"schema": "swe-runtime-layout-v1", "layout": LAYOUT, "site": "site",
                        "source_roots": roots, "plugins": plugins(site), "skipped_data_entries": sorted(skipped)}
        publish(wheels / "mnlm-runtime.json", runtime_spec)
        if os.name == "posix":  # publish() creates 0600; the sandbox reads as uid 65534
            (wheels / "mnlm-runtime.json").chmod(0o644)
    generated = []
    if plan.get("generated_files") is not None:  # install-time files the repository itself declares
        from oracle.scm_version import generated as scm_generated
        generated = scm_generated(files)
        require(plan["generated_files"] == [{"path": g["path"], "rule": g["rule"],
                                             "sha256": hashlib.sha256(g["content"]).hexdigest()} for g in generated]
                and generated, CODE, "declared generated files differ from the snapshot's own declaration")
    conditions, agent_method, agent_reset, cond_execs = {}, None, [], {}
    for name in ("base", "reference", "agent"):
        dest = output / name
        dest.mkdir()
        for path, data in [*files.items(), *((g["path"], g["content"]) for g in generated)]:
            target = dest / relative(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        executables(dest, execs)
        if name != "base":
            text = metadata["patch"] if name == "reference" else attempt["generated_patch"]
            if name == "agent" and not (text or "").strip():
                # The literal submission is "no change": the agent condition is the base tree.
                require(plan.get("agent_patch_empty") is True and plan["allowed_paths"]["agent"] == [],
                        CODE, "empty agent submission must be declared by the plan")
            else:
                require(name != "agent" or plan.get("agent_patch_empty") is not True, CODE,
                        "plan declares an empty submission")
                method = apply_patch(dest, output / (name + ".patch"), text, plan["allowed_paths"][name],
                                     fallback=name == "agent" and plan.get("agent_patch_fallback") is True)
                if method != "git-apply":
                    agent_method = method
        if name == "agent" and plan.get("reset_test_files") is True:
            agent_reset = reset_test_files(dest, files, plan["allowed_paths"]["test"])
        apply_patch(dest, output / (name + "-test.patch"), metadata["test_patch"], plan["allowed_paths"]["test"])
        conditions[name] = inventory(dest)
        patch_text = "" if name == "base" else (metadata["patch"] if name == "reference" else attempt["generated_patch"])
        cond_execs[name] = sorted(set(execs) | new_executables(patch_text) | new_executables(metadata["test_patch"]))
        check_modes(dest, cond_execs[name])
    for path in plan["allowed_paths"]["test"]:
        require(len({v.get(path) for v in conditions.values()}) == 1, CODE, "conditions used different tests")  # absent = deleted
    result = {"schema": "swe-prepared-v1", "plan_sha256": plan_sha256, "task": plan["task"],
              "image": plan["image"], "required_tests": roles, "test_modules": plan["test_modules"],
              "source_blobs": len(files), "source_bytes": sum(map(len, files.values())),
              "conditions": conditions, "runtime": inventory(wheels), "provider_target": attempt["target"],
              "admitted": False, "original_provider_runtime": False}
    if layout is not None:
        result.update(runtime_layout=layout, source_roots=runtime_spec["source_roots"],
                      plugins=runtime_spec["plugins"], wheels=len(plan["wheels"]))
    if plan.get("agent_patch_empty") is True:
        result["agent_patch_empty"] = True
    if completion is not None:
        result["test_id_completion"] = completion["map"]
    if generated:
        result["generated_files"] = plan["generated_files"]
    if agent_method:
        result["agent_patch_method"] = agent_method
    if any(v != execs for v in cond_execs.values()):  # a patch created an executable file
        result["condition_executables"] = cond_execs
    if agent_reset:
        result["agent_test_files_reset"] = agent_reset
    if execs:  # absent key = no executables, so earlier payloads keep their bytes
        result["executables"] = execs
    if transformed := sorted(b["name"] for b in plan.get("exact_blobs", [])):
        result["exact_blobs"] = transformed
    built = time.perf_counter()
    sha = publish(output / "prepared.json", result)
    finished = time.perf_counter()
    publish(output / "preparation-receipt.json", {"sha256": sha, "wall_seconds": finished - started,
        "phases": {"source": source_done - started, "case": case_done - source_done,
                   "build": built - case_done, "publication": finished - built}})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    args = p.parse_args()
    try:
        result = prepare(args.root, args.plan, args.plan_sha256, args.output)
        print(json.dumps({k: result[k] for k in ("task", "source_blobs", "source_bytes", "admitted")}))
    except (SenseError, OSError, ValueError, KeyError, TypeError, tarfile.TarError,
            zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)

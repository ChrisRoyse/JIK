"""Real retained manifest replay for policy scope and census-seal boundaries.

No network or acquisition. Changed metadata cases are explicitly isolated
negative controls, never training data or newly discovered provider state.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census
import fsv_stage0
from ledger import Ledger, digest_file, read_entries, verify_chain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predecessor(name, revision):
    code = subprocess.check_output(["git", "show", f"{revision}:scripts/census/{name}.py"], cwd=Path(__file__).resolve().parents[2])
    module = types.ModuleType("predecessor_" + name)
    module.__file__ = str(Path(__file__).with_name(name + ".py"))
    exec(compile(code, "git:" + revision + ":" + name, "exec"), module.__dict__)
    return module, hashlib.sha256(code).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--predecessor", default="4b89ed8")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks, calls, artifacts = [], [], []
    repo = Path(__file__).resolve().parents[2]
    version = "snapshot-boundary"
    base = args.source_root / "census" / "candidates-v1"
    actual = json.loads((base / "census.json").read_bytes())
    selected = ("swe-agent-trajectories", "web-audio-mpj")
    real_paths = [base / "census.json", args.source_root / "ledger" / "ledger.jsonl",
                  *(base / s / "manifest.jsonl" for s in selected)]
    original = {str(p): sha(p) for p in real_paths}
    cfg = json.loads((repo / "config/sources.candidates.json").read_bytes())
    cfg["census_version"] = version
    cfg["sources"] = [s for s in cfg["sources"] if s["id"] in selected]
    court = next(s for s in json.loads((repo / "config/sources.p0.json").read_bytes())["sources"] if s["id"] == "cuyahoga-eighth-district")
    cfg["sources"].append(court)
    cfgpath = args.output / "scope-control-config.json"
    cfgpath.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    manifests = {s: [json.loads(line) for line in (base / s / "manifest.jsonl").read_bytes().splitlines()] for s in selected}

    def check(name, ok, **evidence):
        checks.append({"boundary": name, "pass": bool(ok), **evidence})
        if not ok:
            (args.output / "failed-checks.json").write_text(json.dumps(checks, indent=2))
            raise AssertionError(name)

    def replay(source, window):
        calls.append(source["id"])
        assert source["id"] in selected, "excluded/unretained source reached resolution"
        return copy.deepcopy(actual["sources"][source["id"]]["info"]), copy.deepcopy(manifests[source["id"]])

    def invoke(module, argv, name, resolver=False):
        output = io.StringIO()
        error = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [module.__file__, *map(str, argv)]))
            stack.enter_context(contextlib.redirect_stdout(output))
            if resolver:
                stack.enter_context(patch.object(module, "resolve", replay))
            try:
                result = module.main()
            except (SystemExit, RuntimeError) as exc:
                result, error = 1, str(exc)
        artifact = args.output / (name + ".json")
        artifact.write_text(json.dumps({"returncode": result, "error": error, "stdout": output.getvalue()}, indent=2) + "\n", encoding="utf-8")
        artifacts.append(artifact)
        return result, error

    old_census, old_census_sha = predecessor("census", args.predecessor)
    old_fsv, old_fsv_sha = predecessor("fsv_stage0", args.predecessor)
    oldroot, root = args.output / "old-scope", args.output / "correct-scope"
    rc, err = invoke(old_census, ["--root", oldroot, "--config", cfgpath], "old-exclusion", True)
    check("old_exclusion_defect_reproduced", rc != 0 and err.startswith("MNLM_CENSUS_INCOMPLETE"), error=err)
    rc, err = invoke(census, ["--root", root, "--config", cfgpath], "correct-exclusion", True)
    check("excluded_scope_now_completes", rc == 0 and err is None)
    cdir = root / "census" / version
    retained = json.loads((cdir / "census.json").read_bytes())
    from training_scope import POLICY_SHA256
    check("operator_policy_bound", retained.get("training_scope_sha256") == POLICY_SHA256)
    check("only_effective_sources_persisted", set(retained["sources"]) == set(selected)
          and not (cdir / court["id"]).exists() and court["id"] not in calls, resolver_calls=calls.copy())
    for source in selected:
        check("real_manifest_" + source, (cdir / source / "manifest.jsonl").read_bytes() == (base / source / "manifest.jsonl").read_bytes())
    seal = json.loads((cdir / "SEAL.json").read_bytes())
    entries = read_entries(root / "ledger/ledger.jsonl")
    exact = entries[seal["ledger_seq"] - 1]
    cdig, size = digest_file(cdir / "census.json")
    check("independent_seal_position", exact["hash"] == seal["ledger_head"] and exact["payload_digests"]["census_blake3"] == cdig == seal["census_blake3"]
          and exact["payload_digests"]["census_bytes"] == size and verify_chain(root / "ledger/ledger.jsonl")["ok"])
    rc, _ = invoke(fsv_stage0, ["--root", root, "--version", version, "--issue", "18-valid"], "valid-snapshot")
    check("valid_snapshot_accepted", rc == 0)

    # Scope refusals must precede creating an output root or ledger.
    for label, value, extra, code in (
        ("unknown-only", cfg, ["--only", "not-in-this-config"], "MNLM_CENSUS_UNKNOWN_SOURCE"),
        ("empty-policy-scope", {**cfg, "sources": [court]}, [], "MNLM_CENSUS_EMPTY_SCOPE"),
        ("duplicate-sources", {**cfg, "sources": cfg["sources"] + [cfg["sources"][0]]}, [], "MNLM_CENSUS_DUPLICATE_SOURCE")):
        p = args.output / (label + "-config.json")
        p.write_text(json.dumps(value), encoding="utf-8")
        dest = args.output / label
        before_calls = len(calls)
        rc, err = invoke(census, ["--root", dest, "--config", p, *extra], label, True)
        check(label, rc != 0 and err.startswith(code) and not dest.exists() and len(calls) == before_calls, error=err)

    def clone(name):
        dest = args.output / name
        files = [Path("ledger/ledger.jsonl"), Path("census") / version / "census.json", Path("census") / version / "SEAL.json",
                 *(Path("census") / version / s / "manifest.jsonl" for s in selected)]
        for relative in files:
            p = dest / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((root / relative).read_bytes())
        return dest

    later = clone("later-source-attestation")
    Ledger(later / "ledger/ledger.jsonl", "fsv-later-generation").append("census.source", selected[0],
            {"manifest_blake3": "0" * 64, "entries": 1, "bytes_total": 1})
    rc, _ = invoke(old_fsv, ["--root", later, "--version", version, "--issue", "18-old"], "old-later-entry")
    check("old_historical_false_rejection_reproduced", rc != 0)
    rc, _ = invoke(fsv_stage0, ["--root", later, "--version", version, "--issue", "18-fixed"], "fixed-later-entry")
    check("later_attestation_cannot_change_old_snapshot", rc == 0)

    mixed = clone("mixed-snapshots")
    mixed_cdir = mixed / "census" / version
    newer = copy.deepcopy(retained)
    newer["window_end_utc"] = "2026-10-16"  # explicit negative-control change
    (mixed_cdir / "census.json").write_text(json.dumps(newer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    newdig, newsize = digest_file(mixed_cdir / "census.json")
    Ledger(mixed / "ledger/ledger.jsonl", "fsv-newer-artifact").append("census.artifact", version,
            {"census_blake3": newdig, "census_bytes": newsize, "discovery_complete": True})
    mixed_seal = {**seal, "census_blake3": newdig}  # old POSITION + new digest
    (mixed_cdir / "SEAL.json").write_text(json.dumps(mixed_seal), encoding="utf-8")
    rc, _ = invoke(old_fsv, ["--root", mixed, "--version", version, "--issue", "18-old"], "old-mixed-snapshot")
    check("old_mixed_snapshot_false_accept_reproduced", rc == 0)
    rc, _ = invoke(fsv_stage0, ["--root", mixed, "--version", version, "--issue", "18-fixed"], "fixed-mixed-snapshot")
    check("mixed_snapshot_refused", rc != 0)

    prior = (cdir / selected[0] / "manifest.jsonl").read_bytes()
    Ledger(root / "ledger/ledger.jsonl", "fsv-other-source-snapshot").append("census.source", selected[0],
            {"manifest_blake3": "0" * 64, "entries": 1, "bytes_total": 1})
    before_calls = len(calls)
    rc, err = invoke(census, ["--root", root, "--config", cfgpath, "--only", selected[1]], "only-recensus", True)
    check("only_retains_other_source", rc == 0 and err is None and (cdir / selected[0] / "manifest.jsonl").read_bytes() == prior
          and calls[before_calls:] == [selected[1]])
    rc, _ = invoke(fsv_stage0, ["--root", root, "--version", version, "--issue", "18-recensus"], "only-readback")
    check("only_recensus_binds_inherited_manifest_not_latest_source_event", rc == 0)
    check("live_inputs_unchanged", all(sha(Path(p)) == h for p, h in original.items()), inputs=original)
    report = {"acceptance": "author-checked", "checks": checks, "passed": len(checks), "failed": 0,
              "wall_seconds": time.perf_counter() - started, "manifest_entries": sum(len(v) for v in manifests.values()),
              "source_bytes": sum((base / s / "manifest.jsonl").stat().st_size for s in selected),
              "predecessor": {"revision": args.predecessor, "census_sha256": old_census_sha, "fsv_sha256": old_fsv_sha},
              "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(census.__file__), Path(fsv_stage0.__file__), Path(__file__).with_name("census_scope.py"))},
              "limits": "Offline replay of retained real metadata; not provider rediscovery, payload acquisition, or revised-base completion."}
    rp = args.output / "report.json"
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Only the exact bounded evidence tree just created, never a data-root scan.
    manifest = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("acceptance", "passed", "failed", "wall_seconds", "manifest_entries")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Real native population -> atom lineage -> existing Stage 3 consumer proof."""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "lens")]
from native_atoms import build, open_atoms, audit, split, Population, Store, file_hash
from senses.contract import CHANNELS, SenseError, canonical, seal_record, field
from senses.instruments import contracts
from senses.pipeline import read_bound
from universal import ChannelReader, measure_universal, coverage_report
from slotio import slot_digest


def run(popdir, mh, output, near, near_sha256):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    root = output / "atoms"
    extra = ["--near", str(near), "--near-sha256", near_sha256] if near else []
    proc = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("native_atoms.py")),
                           "--population", str(popdir), "--sha256", mh, "--output", str(root), *extra],
                          capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (output / "build.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    checks = []

    def check(name, ok, **details):
        checks.append({"boundary": name, "pass": bool(ok), **details})

    def refuses(name, call, expected):
        code = None
        try:
            call()
        except SenseError as exc:
            code = exc.code
        check(name, code == expected, code=code)

    am = open_atoms(root, file_hash(root / "manifest.json"))
    if near:
        check("near_annotations_bound_and_retained", am["near_annotations"]["manifest_sha256"] == near_sha256
              and file_hash(root / "near.sqlite") == file_hash(near / "near.sqlite")
              and am["near_annotations"]["observations_removed"] == 0)
    pop = Population(popdir, mh)
    source_before = file_hash(popdir / "native.sqlite")
    retained_before = file_hash(root / "atoms.sqlite")
    store = Store(root / "atoms.sqlite")
    db = sqlite3.connect((root / "lineage.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    result = audit(pop, store, db)
    check("cold_source_atom_lineage_readback", result == am["census"], census=result)
    check("complete_approved_population", result["parents"] == 4146 and result["atoms"] == 4185
          and result["kinds"] == {"tool_exchange": 4142, "recorded_audio": 2, "chapter_text": 41})
    # Independent SQL/source byte reconstruction, not the production atomizer.
    parents = {r["atom_id"]: r for r in pop.records()}
    raw = {a: json.loads(b) for a, b in store.db.execute("SELECT atom,body FROM records")}
    all_equal, audio_frames = True, 0
    for parent_id, p in parents.items():
        children = [raw[a] for a, in db.execute("SELECT atom FROM lineage WHERE parent=? ORDER BY ordinal", (parent_id,))]
        k = next(f["value"] for f in p["channels"]["structure"]["items"] if f["name"] == "kind")
        if k == "chapter_text":
            reconstructed = b"".join(c["channels"]["surface"]["items"][0]["text"].encode() for c in children)
            all_equal &= reconstructed == p["channels"]["surface"]["items"][0]["text"].encode()
        else:
            all_equal &= len(children) == 1 and children[0] == p
        audio_frames += sum(len(c["channels"]["perception"]["items"]) for c in children)
    check("independent_bytes_and_observed_units_conserved", all_equal)
    check("all_audio_frames_retained_without_text", audio_frames == 6224
          and all(r["channels"]["surface"]["state"] == "absent" for r in raw.values() if r["locator"]["source"] == "web-audio-mpj"))
    check("derivation_did_not_add_sensory_channels", all(
        {c for c in r["channels"] if r["channels"][c]["state"] == "present"}
        == {c for c in parents[parent]["channels"] if parents[parent]["channels"][c]["state"] == "present"}
        for atom, parent in db.execute("SELECT atom,parent FROM lineage") for r in [raw[atom]]))
    check("observation_store_has_zero_claimed_lenses", store.records_only and not store.instruments
          and store.db.execute("SELECT count(*) FROM slots").fetchone()[0] == 0)
    db.close()
    store.close()
    units = sorted({r["unit"] for r in raw.values()})
    policy = {"schema": "ucr-view-v1", "question": "native-interface", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": units, "label_aliases": [],
              "rationale": "retention/interface proof only; no labels, information claim or admission"}
    policies = output / "policies.json"
    policies.write_bytes(canonical({"native-interface": policy}) + b"\n")
    reader = ChannelReader(root / "atoms.sqlite", policies, result["ucr_generation"])
    same, sample, by_source = True, [], collections.Counter()
    for r in raw.values():
        loc = r["locator"]
        row = {"source": loc["source"], "shard": loc["shard"], "row": loc["row"], "span": loc["span"],
               "text": "\n".join(x["text"] for x in r["channels"]["surface"]["items"])}
        v, contexts, binding = reader.read(row, reader.policy("native-interface"))
        same &= v == r and binding["atom"] == r["atom_id"] and all(not context for context in contexts.values())
        if by_source[loc["source"]] < 2:
            sample.append(row)
            by_source[loc["source"]] += 1
    check("every_atom_resolves_through_existing_channel_reader", same, rows=len(raw))
    bad = {**sample[0], "text": "not the retained source text"}
    refuses("forged_text_cannot_enter_consumer", lambda: reader.read(bad, policy), "MNLM_UCR_SURFACE_BINDING")
    audio_row = next(r for r in sample if r["source"] == "web-audio-mpj")
    masked = copy.deepcopy(policy)
    masked["allow_channels"].remove("perception")
    v, _, _ = reader.read(audio_row, masked)
    check("audio_mask_remains_explicit_absence", v["channels"]["perception"]["state"] == "absent")
    text_row = next(r for r in sample if r["source"] == "ebible-english")
    ambiguous = {k: v for k, v in text_row.items() if k != "span"}
    refuses("chapter_children_require_exact_span", lambda: reader.read(ambiguous, policy), "MNLM_UCR_LOCATOR_AMBIGUOUS")
    reader.close()
    (output / "cal").mkdir()
    (output / "cal" / "native-interface.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in sample))
    registry = {"lenses": {}}
    for spec in contracts():
        entry = measure_universal(spec, {"native-interface": sample}, output / "slots", {}, root / "atoms.sqlite", policies)
        entry["spec"] = spec
        registry["lenses"][spec["name"]] = entry
        item = entry["families"]["native-interface"]
        cold = slot_digest(Path(item["slot"]))
        check("cold_slot_" + spec["channel"], cold["digest"] == item["digest"] and cold["n"] == len(sample))
    coverage = coverage_report(registry, {"native-interface": {"n": len(sample)}})
    registry["lenses"]["second-text-embedding-control"] = registry["lenses"]["ucr-surface-v1"]
    check("multiple_text_embeddings_count_once", coverage_report(registry, {"native-interface": {"n": len(sample)}})["questions"] == coverage["questions"])
    check("interface_slots_are_not_admitted_senses", not coverage["served"]
          and coverage["questions"]["native-interface"]["atoms_ge5_admitted"] == 0)
    (output / "registry.json").write_bytes(canonical(registry) + b"\n")
    for name, sql, code in (
        ("span", "UPDATE lineage SET end=end-1 WHERE atom=(SELECT atom FROM lineage LIMIT 1)", "MNLM_ATOMS_CONSERVATION"),
        ("order", "DELETE FROM edges WHERE rowid=(SELECT min(rowid) FROM edges)", "MNLM_NATIVE_ATOM_ORDER"),
        ("parent", "UPDATE lineage SET parent='unretained-parent' WHERE atom=(SELECT atom FROM lineage LIMIT 1)", "MNLM_NATIVE_ATOM_CLOSURE"),
    ):
        dest = output / ("corrupt-" + name)
        shutil.copytree(root, dest)
        db = sqlite3.connect(dest / "lineage.sqlite")
        db.execute(sql)
        db.commit()
        refuses(name + "_file_binding", lambda: open_atoms(dest, file_hash(dest / "manifest.json")), "MNLM_NATIVE_ATOM_DIGEST")
        s = Store(dest / "atoms.sqlite")
        try:
            refuses(name + "_internal_closure", lambda: audit(pop, s, db), code)
        finally:
            s.close()
            db.close()
    unknown = copy.deepcopy(next(iter(parents.values())))
    unknown["channels"]["structure"]["items"][0]["value"] = "unrecognised_source_unit"
    refuses("unsupported_units_refuse", lambda: list(split(seal_record(unknown))), "MNLM_NATIVE_ATOM_KIND")
    audio = copy.deepcopy(next(r for r in parents.values() if r["locator"]["source"] == "web-audio-mpj"))
    # Format control only: any future adapter may declare an indivisible typed
    # observation with its own coordinates, without a new source/lens branch.
    generic = copy.deepcopy(audio)
    generic["channels"]["structure"]["items"] = [field("kind", "observation", "category"), field("span_unit", "sample", "category")]
    seal_record(generic)
    parts = list(split(generic))
    check("generic_observation_adapter_needs_no_source_branch", len(parts) == 1 and parts[0][0] == generic and parts[0][1][2] == 'sample')
    generic["channels"]["structure"]["items"].pop()
    refuses("generic_observation_requires_coordinate_unit", lambda: list(split(seal_record(generic))), "MNLM_NATIVE_ATOM_SPAN")
    audio["channels"]["perception"]["items"].pop(0)
    refuses("missing_audio_frames_refuse", lambda: list(split(seal_record(audio))), "MNLM_NATIVE_AUDIO_SPAN")
    crash = subprocess.run([sys.executable, "-I", str(Path(__file__)), "--population", str(popdir), "--sha256", mh,
                            "--output", str(output / "crash"), "--crash", *extra], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    check("actual_death_before_publication", crash.returncode == 77 and (output / "crash" / "atoms.sqlite").is_file())
    refuses("interrupted_atoms_cannot_be_accepted", lambda: open_atoms(output / "crash", "0" * 64), "MNLM_NATIVE_ATOM_MANIFEST")
    check("input_and_atom_store_unchanged", file_hash(popdir / "native.sqlite") == source_before
          and file_hash(root / "atoms.sqlite") == retained_before)
    pop.close()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "coverage": coverage, "census": result,
              "atom_manifest_sha256": file_hash(root / "manifest.json"), "seconds": time.perf_counter() - start}
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    hashes = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256.json").write_bytes(canonical(hashes) + b"\n")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "seconds", "atom_manifest_sha256")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--crash", action="store_true")
    ap.add_argument("--near", type=Path)
    ap.add_argument("--near-sha256")
    args = ap.parse_args()
    if args.crash:
        build(args.population, args.sha256, args.output, fault=lambda _: os._exit(77), near=args.near, near_sha256=args.near_sha256)
    else:
        sys.exit(run(args.population, args.sha256, args.output, args.near, args.near_sha256))

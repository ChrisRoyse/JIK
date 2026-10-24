"""Three source/material/consumer boundaries on persisted real observations."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.source_quality import load_config, pinned, run
from text.source_material import material, project, AvailableField, IMAGE
from text.evaluation_multiscale import canonical, file_sha
from audit.source_quality_readback import audit_field, audit
from senses.trajectory_population import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "publication", "output"): p.add_argument("--"+n, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    cfg, parents = load_config(a.config); before = {}; chosen = {}; results = []
    for domain, meta in parents["source"]["sources"].items():
        src = pinned(Path(cfg["source"]["root"]), meta)
        geo = pinned(Path(cfg["multiscale"]["root"]), parents["multiscale"]["files"][domain])
        before[str(src)] = file_sha(src); before[str(geo)] = file_sha(geo)
        with src.open("rb") as sf, geo.open("rb") as gf:
            for line in sf:
                row = json.loads(line)
                for field in row["fields"]:
                    base = json.loads(next(gf)); matches = list(IMAGE.finditer(field["text"]))
                    residual = IMAGE.sub("", field["text"]).strip().lower()
                    kind = "image-only" if matches and residual in ("", "pmc") else (
                        "mixed" if matches else ("empty" if not residual else "prose"))
                    if kind not in chosen or len(field["text"]) < len(chosen[kind][1]["text"]):
                        if kind != "prose" or (base["graph"] and len(base["graph"]["nodes"]) >= 10):
                            chosen[kind] = row, field, base
    publish(a.output/"before.json", before)
    def check(name, ok, **details): results.append(dict(name=name, passed=bool(ok), **details))
    def negative(name, code, fn):
        observed = None
        try: fn()
        except ValueError as exc: observed = str(exc)
        check(name, observed is not None and code in observed, observed=observed, expected=code)
    check("B1 minimum real material cases exist", set(chosen) == {"image-only", "mixed", "empty", "prose"})
    for kind, (row, field, base) in chosen.items():
        annotation = material(field); projection = project(field, base["graph"], annotation)
        result = dict(document_id=row["id"], field_name=field["name"], domain=row["domain"],
                      source_record_sha256=row["source"]["record_sha256"], group_keys=row["group_keys"],
                      annotation=annotation, projection=projection)
        path = a.output/(kind+".json")
        publish(path, dict(row=row, field=field, base=base, result=result))
        cold = json.loads(path.read_bytes()); audit_field(cold["row"], cold["field"], cold["base"], cold["result"])
        check("B2 persisted "+kind+" routing", True, source_id=row["id"], proof_sha256=file_sha(path))
    image = json.loads((a.output/"image-only.json").read_bytes())
    field, base, result = image["field"], image["base"], image["result"]
    consumer = AvailableField(field, base["graph"], result["annotation"], result["projection"])
    view = consumer.view(len(field["text"]), base["graph"]["split_group"])
    publish(a.output/"image-only-consumer.json", view)
    check("B3 image filenames cannot become text meaning", not view["nodes"] and not view["meaning_asserted"])
    check("B3 image and text each require five native senses", result["annotation"]["native_panel_requirements"] == {"text":5,"image":5})
    bad = copy.deepcopy(field); bad["text"] += "changed"
    negative("B1 source mutation", "SOURCE_HASH", lambda: material(bad))
    bad = copy.deepcopy(result); bad["annotation"]["obstructions"] = []
    negative("B2 omitted missing-image span", "READBACK_ANNOTATION",
             lambda: audit_field(image["row"], field, base, bad))
    bad = copy.deepcopy(result); bad["projection"]["ready_node_ids"].append(base["graph"]["nodes"][0]["id"])
    negative("B3 promoting image to available text", "PROJECTION",
             lambda: AvailableField(field, base["graph"], result["annotation"], bad["projection"]))
    bad = copy.deepcopy(result); bad["annotation"]["grounded"] = True
    negative("B2 availability cannot assert grounding", "READBACK_ANNOTATION",
             lambda: audit_field(image["row"], field, base, bad))
    negative("B3 wrong split group", "GROUP", lambda: consumer.view(len(field["text"]), "other"))
    plain = json.loads((a.output/"prose.json").read_bytes())
    field, base, result = plain["field"], plain["base"], plain["result"]
    consumer = AvailableField(field, base["graph"], result["annotation"], result["projection"])
    cutoff = min(n["end"] for n in base["graph"]["nodes"])
    view = consumer.view(cutoff, base["graph"]["split_group"])
    publish(a.output/"causal-consumer.json", view)
    check("B3 partial parents remain hidden", all(n["end"] <= cutoff for n in view["nodes"])
          and len(view["nodes"]) < len(base["graph"]["nodes"]))
    bad = copy.deepcopy(result); nodes = [n for n in base["graph"]["nodes"] if n["level"] == "sentence"]
    bad["projection"]["next_edges"].append(dict(relation="next", **{"from":nodes[0]["id"],"to":nodes[-1]["id"]}))
    negative("B3 invented sequence bridge", "PROJECTION",
             lambda: AvailableField(field, base["graph"], result["annotation"], bad["projection"]))
    publish(a.output/"minimum-checks.json", results)
    if not all(r["passed"] for r in results): raise SystemExit("MNLM_MATERIAL_MINIMUM_FAILED")
    run(a.config, a.publication)
    cold = audit(a.config, a.publication); publish(a.output/"cold-readback.json", cold)
    check("B2 all source decisions cold-read back", cold["all_fields_read_back"], totals=cold["totals"])
    negative("B3 immutable publication", "EXISTS", lambda: run(a.config, a.publication))
    after = {name:file_sha(Path(name)) for name in before}; publish(a.output/"after.json", after)
    check("B1 original source and graphs unchanged", before == after)
    report = json.loads((a.publication/"report.json").read_bytes()); t = report["timing"]
    covered = sum(t[k] for k in ("input_seconds", "legacy_seconds", "material_seconds", "write_seconds"))
    check("PC37 at least 90 percent timing coverage", covered/t["wall_seconds"] >= .9,
          fraction=covered/t["wall_seconds"])
    publish(a.output/"fsv.json", dict(author_checked=True, checks=results,
                                    passed=all(r["passed"] for r in results), report_sha256=file_sha(a.publication/"report.json")))
    print(json.dumps(dict(checks=len(results), passed=all(r["passed"] for r in results))))
    if not all(r["passed"] for r in results): raise SystemExit(2)


if __name__ == "__main__": main()

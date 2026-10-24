"""Persisted three-boundary controls on real multiscale source and CUDA outputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.multiscale_reading_readback import population, plan, measurement, sha, check
from senses.multiscale_readings import visible


def rewrite(path, value):
    # Only FSV-owned shadows and receipts are rewritten; source roots are read-only.
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def tree(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def update_file(root, filename):
    r = json.loads((root/"report.json").read_bytes())
    r["files"][filename] = dict(sha256=sha(root/filename), bytes=(root/filename).stat().st_size)
    rewrite(root/"report.json", r)


def edit(root, filename, operation):
    d = json.loads((root/filename).read_bytes()); operation(d); rewrite(root/filename, d); update_file(root, filename)


def nested_update(root, name, filename, lane=False):
    p = root/"instruments.json"; obj = json.loads(p.read_bytes())
    item = obj[name] if lane else next(e for e in obj if e["name"] == name)
    item["files"][filename] = dict(sha256=sha(root/name/filename), bytes=(root/name/filename).stat().st_size)
    rewrite(p, obj); update_file(root, "instruments.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("population", "plan", "measurement", "models", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    a = p.parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS"); a.output.mkdir(parents=True)
    roots = dict(population=a.population, plan=a.plan, measurement=a.measurement)
    before = {n:tree(r) for n,r in roots.items()}; rewrite(a.output/"before.json", before)
    checks = []
    positive = measurement(a.measurement, sha(a.measurement/"report.json"), a.population, a.plan, a.models)
    rewrite(a.output/"cold-readback.json", positive); checks.append(dict(name="all_source_plan_geometry", passed=True))
    _, d = population(a.population, sha(a.population/"report.json")); causal_queries = 0
    for s in d["sources"]:
        for f in s["fields"]:
            nodes = [n for n in d["nodes"] if n["document_id"] == s["id"] and n["field_name"] == f["name"]]
            cuts = {0, len(f["text"])} | {c for n in nodes for c in (n["visible_after"]-1,n["visible_after"])}
            for c in cuts:
                got = visible(d, s["id"], f["name"], c, d["groups"][s["id"]])
                expected = sorted(n["id"] for n in nodes if all(b <= c for _,b in n["source_spans"]))
                check(got["node_ids"] == expected, "CAUSAL_NODES")
                check(got["edge_indices"] == [i for i,e in enumerate(d["edges"]) if e["from"] in expected and e["to"] in expected], "CAUSAL_EDGES")
                causal_queries += 1
    checks.append(dict(name="all_causal_boundaries", passed=True, queries=causal_queries))
    def negative(name, parent, mutate, reader):
        dest = a.output/name; shutil.copytree(parent, dest); mutate(dest)
        try:
            reader(dest)
        except ValueError as exc:
            check(str(exc).startswith("MNLM_"), "UNEXPECTED_EXCEPTION")
            checks.append(dict(name=name, passed=True, refusal=str(exc)))
        else:
            checks.append(dict(name=name, passed=False))
        rewrite(a.output/"progress.json", checks)
    popread = lambda r: population(r, sha(r/"report.json"))
    negative("changed-source-part", a.population,
             lambda r: edit(r, "nodes.json", lambda d: next(n for n in d if n["level"] == "physical_page")["source_spans"][0].__setitem__(0, 1)), popread)
    negative("dropped-source-edge", a.population,
             lambda r: edit(r, "edges.json", lambda d:d.pop()), popread)
    negative("wrong-content-unit", a.population,
             lambda r: edit(r, "nodes.json", lambda d:d[0]["unit_indices"].__setitem__(0, 88)), popread)
    negative("future-parent-exposure", a.population,
             lambda r: edit(r, "nodes.json", lambda d:d[-1].__setitem__("visible_after", 0)), popread)
    negative("false-admission", a.measurement,
             lambda r: rewrite(r/"report.json", dict(json.loads((r/"report.json").read_bytes()), scientific_admission=True)),
             lambda r: measurement(r, sha(r/"report.json"), a.population, a.plan, a.models))
    name = json.loads((a.plan/"instruments.json").read_bytes())[0]["name"]
    def lost_token(r):
        path = r/name/"input_ids.npy"; x = np.load(path); x[0,0] = (int(x[0,0])+1) % 30000
        np.save(path,x); nested_update(r,name,"input_ids.npy")
    negative("changed-native-token", a.plan, lost_token,
             lambda r: plan(r,sha(r/"report.json"),a.population,a.models))
    def bad_geometry(r):
        path = r/name/"edges.json"; x=json.loads(path.read_bytes()); x[0]["maxsim"]=-.9
        rewrite(path,x); nested_update(r,name,"edges.json",True)
    negative("wrong-edge-score", a.measurement, bad_geometry,
             lambda r: measurement(r,sha(r/"report.json"),a.population,a.plan,a.models))
    def mixed_lane(r):
        path = r/name/"edges.json"; x=json.loads(path.read_bytes()); x[0]["lens_id"]="0"*64
        rewrite(path,x); nested_update(r,name,"edges.json",True)
    negative("mixed-instrument", a.measurement, mixed_lane,
             lambda r: measurement(r,sha(r/"report.json"),a.population,a.plan,a.models))
    def altered_vector(r):
        path=r/name/"vectors.npy"; x=np.load(path); x[0,0]+=1
        np.save(path,x); nested_update(r,name,"vectors.npy",True)
    negative("changed-vector", a.measurement, altered_vector,
             lambda r: measurement(r,sha(r/"report.json"),a.population,a.plan,a.models))
    try:
        s=d["sources"][0]; visible(d,s["id"],s["fields"][0]["name"],0,"wrong-group")
    except ValueError as exc:
        checks.append(dict(name="wrong-causal-group", passed=str(exc)=="MNLM_MULTI_COLD_CAUSAL_GROUP"))
    else:
        checks.append(dict(name="wrong-causal-group", passed=False))
    # The first instrument fails on its altered weight bytes before later models
    # are read. Avoid copying immutable model corpora across archive filesystems.
    model_copy=a.output/"bad-model"; model_copy.mkdir(); model_before={}
    for entry in json.loads((a.plan/"instruments.json").read_bytes()):
        for file,h in entry["model_files"].items():
            check(sha(a.models/entry["name"]/file)==h,"MODEL_BEFORE")
            model_before[str(Path(entry["name"])/file)]=h
    dest=model_copy/name; dest.mkdir()
    for file in ("config.json", "tokenizer.json", "report.json"):
        shutil.copyfile(a.models/name/file,dest/file)
    (dest/"model.pt").write_bytes(b"corrupted negative control")
    try:
        plan(a.plan,sha(a.plan/"report.json"),a.population,model_copy)
    except ValueError as exc:
        checks.append(dict(name="changed-frozen-weights", passed=str(exc)=="MNLM_MULTI_COLD_MODEL_PIN"))
    else:
        checks.append(dict(name="changed-frozen-weights",passed=False))
    after={n:tree(r) for n,r in roots.items()}; rewrite(a.output/"after.json",after)
    checks.append(dict(name="source-and-models-unchanged",passed=before==after and all(sha(a.models/n)==h for n,h in model_before.items())))
    result=dict(acceptance="author-checked", checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                scientific_admission=False, boundaries=["source spans to exact content and node maps",
                "frozen identity and full native token windows", "typed geometry and future-safe consumption"])
    rewrite(a.output/"fsv.json",result); print(json.dumps(result)); check(result["passed"]==result["total"],"FSV_FAILED")


if __name__ == "__main__":
    main()

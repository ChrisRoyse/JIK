"""Independent source maps, native token coverage, typed geometry and causality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.evaluation_multiscale_readback import audit_field
from audit.page_attachment_readback import audit_case
from audit.commissioned_window_readback import validate_prepared, validate_native_ids
from senses.trajectory_population import publish


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_MULTI_COLD_"+code)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(root, pin, schema):
    check(sha(root/"report.json") == pin, "PIN")
    r = json.loads((root/"report.json").read_bytes())
    check(r["schema"] == schema and r["complete"] is True, "SCHEMA")
    check(all(r[k] is False for k in ("scientific_admission", "served", "meaning_asserted")), "CLAIMS")
    for name, meta in r["files"].items():
        check(Path(name).name == name and (root/name).stat().st_size == meta["bytes"]
              and sha(root/name) == meta["sha256"], "FILE")
    times = r["timing"]; phases = [v for k,v in times.items() if k != "wall_seconds"]
    check(all(np.isfinite(v) and v >= 0 for v in phases) and times["wall_seconds"] > 0
          and abs(sum(phases)/times["wall_seconds"]-1) < 1e-8, "TIMING")
    return r


def population(root, pin):
    r = load(root, pin, "multiscale-reading-population-v1")
    names = ("sources", "fields", "layouts", "attachments", "groups", "units", "nodes", "edges")
    check(set(r["files"]) == {n+".json" for n in names}, "FILES")
    d = {n: json.loads((root/(n+".json")).read_bytes()) for n in names}
    sources = {s["id"]: s for s in d["sources"]}; groups = d["groups"]
    check(len(sources) == len(d["sources"]) == r["documents"] == 6 and set(groups) == set(sources), "SOURCES")
    check(set(sources) == {s["document_id"] for s in r["selection"]}, "SELECTION")
    node_ref = {}; edge_ref = []; layout = {p["document_id"]:p for p in d["layouts"]}
    attach = {p["document_id"]:p for p in d["attachments"]}; seen_fields = set()
    for base in d["fields"]:
        doc = base["document_id"]; field_name = base["field_name"]
        field = next(f for f in sources[doc]["fields"] if f["name"] == field_name)
        check((doc, field_name) not in seen_fields, "DUPLICATE_FIELD")
        seen_fields.add((doc, field_name)); audit_field(base, sources[doc], field)
        if base["graph"] is None:
            continue
        for n in base["graph"]["nodes"]:
            node_ref[n["id"]] = (doc, field_name, n["level"], [[n["start"], n["end"]]])
        edge_ref.extend(base["graph"]["edges"])
        if doc in attach:
            audit_case(attach[doc], base, layout[doc], groups[doc], exhaustive=True)
            for n in attach[doc]["pages"]:
                node_ref[n["id"]] = (doc, field_name, "physical_page", n["source_spans"])
            edge_ref.extend(attach[doc]["edges"])
    check(seen_fields == {(s["id"], f["name"]) for s in d["sources"] for f in s["fields"]}, "FIELDS")
    texts = {}; touched = set(); by_node = {n["id"]:n for n in d["nodes"]}
    for i, u in enumerate(d["units"]):
        check(u["text"].strip() and hashlib.sha256(u["text"].encode()).hexdigest() == u["sha256"]
              and len(u["text"].encode()) == u["utf8_bytes"] and u["text"] not in texts, "UNIT")
        texts[u["text"]] = i
    check(len(by_node) == len(d["nodes"]) == len(node_ref) and set(by_node) == set(node_ref), "NODES")
    for key, node in by_node.items():
        doc, field_name, level, spans = node_ref[key]
        check((node["document_id"], node["field_name"], node["level"], node["source_spans"]) ==
              (doc, field_name, level, spans), "NODE_SOURCE")
        text = next(f["text"] for f in sources[doc]["fields"] if f["name"] == field_name)
        expected = [texts[text[a:b]] if text[a:b].strip() else None for a,b in spans]
        check(node["unit_indices"] == expected, "NODE_UNITS"); touched.update(i for i in expected if i is not None)
        check(node["overlap_group"] == groups[doc] and node["visible_after"] == max(b for _,b in spans), "NODE_VISIBILITY")
    check(touched == set(range(len(texts))), "UNIT_COVERAGE")
    check(len(edge_ref) == len(d["edges"]), "EDGES")
    for base, edge in zip(edge_ref, d["edges"]):
        left, right = by_node[base["from"]], by_node[base["to"]]
        expected = dict(base, overlap_group=groups[left["document_id"]],
                        visible_after=max(left["visible_after"], right["visible_after"]))
        check(edge == expected and left["overlap_group"] == right["overlap_group"], "EDGE")
    counts = dict(fields=len(d["fields"]), nodes=len(by_node), edges=len(edge_ref), units=len(texts),
                  unique_characters=sum(len(t) for t in texts), levels=dict(Counter(n["level"] for n in by_node.values())),
                  page_parts=sum(len(n["source_spans"]) for n in by_node.values() if n["level"] == "physical_page"),
                  empty_parts=sum(i is None for n in by_node.values() for i in n["unit_indices"]))
    check(all(r[k] == v for k,v in counts.items()) and r["sealed_evaluation"] is False
          and r["page_join"] == "none:each-disjoint-source-part-retained", "COUNTS")
    return r, d


def plan(root, pin, pop_root, models):
    from tokenizers import Tokenizer
    r = load(root, pin, "multiscale-native-plan-v1")
    _, d = population(pop_root, r["population_sha256"])
    check(set(r["files"]) == {"instruments.json"}, "PLAN_FILES")
    entries = json.loads((root/"instruments.json").read_bytes()); names = [e["name"] for e in entries]
    check(len(names) == len(set(names)) == r["instruments"], "INSTRUMENTS")
    for e in entries:
        check(Path(e["name"]).name == e["name"] and e["name"] not in ("", ".", ".."), "INSTRUMENT_NAME")
        mr = models/e["name"]; folder = root/e["name"]
        check(set(e["model_files"]) == {"model.pt", "tokenizer.json", "report.json", "config.json"}, "MODEL_FILES")
        for n,h in e["model_files"].items():
            check(sha(mr/n) == h, "MODEL_PIN")
        model_report = json.loads((mr/"report.json").read_bytes())
        check(model_report["name"] == e["name"] and model_report["weights_sha256"] == e["model_files"]["model.pt"]
              and model_report["tokenizer_sha256"] == e["model_files"]["tokenizer.json"]
              and model_report["config"] == e["config"], "MODEL_IDENTITY")
        for n,m in e["files"].items():
            check(Path(n).name == n and sha(folder/n) == m["sha256"] and (folder/n).stat().st_size == m["bytes"], "PLAN_DATA")
        check(set(e["files"]) == {"input_ids.npy", "offsets.npy", "traces.json"}, "PLAN_DATA_NAMES")
        ids = np.load(folder/"input_ids.npy", allow_pickle=False); offsets = np.load(folder/"offsets.npy", allow_pickle=False)
        validate_native_ids(ids, 128, e["config"]["vocab"])
        check(offsets.dtype == np.int64 and offsets.shape == (len(d["units"])+1,)
              and offsets[0] == 0 and offsets[-1] == len(ids) and (np.diff(offsets)>0).all(), "OFFSETS")
        tok = Tokenizer.from_file(str(mr/"tokenizer.json")); tok.no_padding(); tok.no_truncation()
        traces = json.loads((folder/"traces.json").read_bytes()); check(len(traces) == len(d["units"]), "TRACE_COUNT")
        tokens = 0
        for i,u in enumerate(d["units"]):
            enc = tok.encode(u["text"], add_special_tokens=False)
            tokens += validate_prepared(traces[i], u["text"], enc, ids[offsets[i]:offsets[i+1]], 128)
        check(e["windows"] == len(ids) and e["tokens"] == tokens, "PLAN_COUNTS")
    return r, entries, d


def measurement(root, pin, pop_root, plan_root, models):
    r = load(root, pin, "multiscale-native-readings-v1")
    pr, entries, d = plan(plan_root, r["plan_sha256"], pop_root, models)
    check(r["population_sha256"] == pr["population_sha256"] and set(r["files"]) ==
          {"instruments.json", "gpu-before.txt", "desktop-residents.json"}, "MEASUREMENT_SOURCE")
    lanes = json.loads((root/"instruments.json").read_bytes())
    check(set(lanes) == {e["name"] for e in entries}, "LANES")
    by_node = {n["id"]:n for n in d["nodes"]}; total_pairs = 0
    for e in entries:
        lane = lanes[e["name"]]; contract = lane["contract"]
        identity = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        check(lane["lens_id"] == identity and contract["model_files"] == e["model_files"]
              and contract["name"] == e["name"] and contract["device"] == "cuda"
              and contract["window_reduction"] == "none" and contract["batch"] == 32, "LENS_ID")
        folder = root/e["name"]
        for n,m in lane["files"].items():
            check(Path(n).name == n and sha(folder/n) == m["sha256"] and (folder/n).stat().st_size == m["bytes"], "LANE_DATA")
        check(set(lane["files"]) == {"vectors.npy", "edges.json"}, "LANE_FILES")
        v = np.load(folder/"vectors.npy", allow_pickle=False)
        check(v.dtype == np.float32 and v.shape == (e["windows"], e["config"]["width"])
              and np.isfinite(v).all() and np.allclose(np.linalg.norm(v, axis=1), 1, rtol=0, atol=3e-6), "VALUES")
        offsets = np.load(plan_root/e["name"]/"offsets.npy", allow_pickle=False)
        geometry = json.loads((folder/"edges.json").read_bytes()); check(len(geometry) == len(d["edges"]), "GEOMETRY_COUNT")
        comparisons = 0
        def index(node):
            return [j for i in node["unit_indices"] if i is not None for j in range(int(offsets[i]), int(offsets[i+1]))]
        for edge, score in zip(d["edges"], geometry):
            check(score["from"] == edge["from"] and score["to"] == edge["to"] and score["lens_id"] == identity
                  and score["visible_after"] == edge["visible_after"] and score["overlap_group"] == edge["overlap_group"], "GEOMETRY_BINDING")
            ix, iy = index(by_node[edge["from"]]), index(by_node[edge["to"]]); comparisons += len(ix)*len(iy)
            if not ix or not iy:
                check(score["status"] == "no_lexical_reading" and score["maxsim"] is None, "EMPTY_GEOMETRY")
                continue
            # Scalar float64 dot products, separate from producer's float32 GEMM.
            mx = [max(float(np.dot(v[i].astype(np.float64), v[j].astype(np.float64))) for j in iy) for i in ix]
            my = [max(float(np.dot(v[i].astype(np.float64), v[j].astype(np.float64))) for i in ix) for j in iy]
            expected = [sum(mx)/len(mx), sum(my)/len(my)]
            check(score["status"] == "measured_geometry_only" and np.allclose(score["directed_maxsim"], expected, rtol=0, atol=2e-6)
                  and abs(score["maxsim"]-min(expected)) <= 2e-6, "GEOMETRY_VALUE")
        check(comparisons == lane["window_pairs"] and lane["frozen_unchanged"] is True
              and lane["repeated_windows_equal"] == e["windows"], "GEOMETRY_COST")
        total_pairs += comparisons
    check(r["instruments"] == len(entries) and r["window_pairs"] == total_pairs, "TOTALS")
    return dict(acceptance="author-checked", population_sha256=pr["population_sha256"], measurement_sha256=pin,
                instruments=len(entries), all_edges_recomputed=True, window_pairs=total_pairs,
                nodes=len(d["nodes"]), edges=len(d["edges"]), scientific_admission=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("population", "plan", "measurement", "models", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    a = p.parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS")
    result = measurement(a.measurement, a.report_sha256, a.population, a.plan, a.models)
    publish(a.output, result); print(json.dumps(result))

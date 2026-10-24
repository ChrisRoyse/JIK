"""Physical source, native graph, cutoff and restart boundaries on four real clips."""
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.perception_corpus_readback import inspect_clip, pinned, sha, need, read
from senses.perception_corpus import sources, binding, runtime, read_item, encoded
from senses.perception_decode import build
from senses.perception_native import NativePrefix


def main(a):
    started = time.perf_counter()
    report = pinned(a.report, a.report_pin)
    a.output.mkdir(parents=True, exist_ok=False)
    plan, roster = sources(a)
    bound = binding(a, runtime("ffprobe"))
    need(report["binding"] == bound and len(report["results"]) == 4
         and report["complete"] and not report["corpus_complete"], "WITNESS")
    originals = a.acquisition_report.parent/"members"
    phases, proofs, controls = {}, [], []
    phases["source_bind"] = time.perf_counter()-started
    for entry in report["results"]:
        t = time.perf_counter()
        clip = entry["clip"]
        row, result = roster[clip]
        root = a.report.parent/"clips"/clip
        counts = inspect_clip(root, entry, row, result, bound, originals, replay=True)
        before = read_item(root, entry["report_pin"], row, result, bound, source_root=originals)
        prefix_root = a.output/("prefix-"+clip)
        decoded = build(root, entry["report_pin"], clip, [1, 10], prefix_root)
        prefix = NativePrefix(prefix_root, expected=sha(prefix_root/"manifest.json"),
                              episodes=root, episode_pin=entry["report_pin"])
        graph = json.loads((root/"public"/(clip+".json")).read_bytes())
        request = dict(clock_id=graph["clock"]["id"], split_group=graph["split_group"],
                       through=[1, 10], required_media=["image", "audio"])
        consumed = list(prefix.consume(**request))
        eligible = [n for n in graph["nodes"] if Fraction(*n["available_after"]) <= Fraction(1, 10)]
        need({x["node"]["id"] for x in consumed} == {n["id"] for n in eligible},
             "REAL_CUTOFF_COVERAGE")
        proofs.append(dict(clip=clip, counts=counts, native_replay=True,
            prefix_pin=sha(prefix_root/"manifest.json"), consumed_nodes=len(consumed),
            value_bytes=sum(x["values"].nbytes for x in consumed), decode=decoded))
        for label, changes, code in (
            ("wrong-clock", dict(clock_id="foreign"), "CLOCK"),
            ("wrong-split", dict(split_group="foreign"), "SPLIT"),
            ("future", dict(through=[1, 1]), "UNDECODED_FUTURE"),
            ("text-absent", dict(required_media=["text"]), "MISSING_MODALITY")):
            try:
                list(prefix.consume(**{**request, **changes}))
            except ValueError as exc:
                need(str(exc) == "MNLM_NATIVE_PREFIX_"+code, "CUTOFF_CONTROL_REASON")
                controls.append(dict(clip=clip, control=label, error=str(exc)))
            else:
                raise AssertionError(label+" accepted")
        need(read_item(root, entry["report_pin"], row, result, bound, source_root=originals)
             == before, "SOURCE_UNCHANGED")
        phases[clip] = time.perf_counter()-t

    t = time.perf_counter()
    entry = report["results"][0]
    clip = entry["clip"]
    row, result = roster[clip]
    root = a.report.parent/"clips"/clip
    shadow = a.output/"shadow"
    shadow.mkdir()
    base = json.loads((root/"report.json").read_bytes())
    for name in base["files"]:
        target = shadow/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root/name, target)  # Never mutate the original hard link.
    graph_path = shadow/"public"/(clip+".json")
    graph = json.loads(graph_path.read_bytes())

    def refuse(label, receipt, expected, graph_value=None, changed_row=None):
        if graph_value is not None:
            graph_path.write_bytes(encoded(graph_value))
            receipt["files"][f"public/{clip}.json"] = dict(
                bytes=graph_path.stat().st_size, sha256=sha(graph_path))
        (shadow/"report.json").write_bytes(encoded(receipt))
        changed_entry = {**entry, "report_pin": sha(shadow/"report.json")}
        try:
            inspect_clip(shadow, changed_entry, changed_row or row, result, bound, originals)
        except ValueError as exc:
            need(str(exc) == expected, "FAULT_REASON:"+label+":"+str(exc))
            controls.append(dict(clip=clip, control=label, error=str(exc)))
        else:
            raise AssertionError(label+" accepted")
        graph_path.write_bytes(encoded(graph))

    g = copy.deepcopy(graph); g["nodes"].pop()
    refuse("missing-native-node", copy.deepcopy(base), "MNLM_PERCEPTION_READBACK_NATIVE_COORDINATES", g)
    g = copy.deepcopy(graph); g["nodes"][0]["pts"] += 1
    refuse("shifted-native-time", copy.deepcopy(base), "MNLM_PERCEPTION_READBACK_NATIVE_COORDINATES", g)
    g = copy.deepcopy(graph); g["contexts"][0]["members"].append(graph["nodes"][-1]["id"])
    refuse("future-context-member", copy.deepcopy(base), "MNLM_PERCEPTION_READBACK_CONTEXT_CLOSURE", g)
    g = copy.deepcopy(graph); g["answer"] = "forbidden label control"
    refuse("public-label", copy.deepcopy(base), "MNLM_PERCEPTION_READBACK_PUBLIC_SCHEMA", g)
    g = copy.deepcopy(graph); g["clock"]["physical_sync_calibrated"] = True
    refuse("invented-sync", copy.deepcopy(base), "MNLM_PERCEPTION_READBACK_CLOCK", g)
    r = copy.deepcopy(base); r["role"] = "held_out" if row["role"] != "held_out" else "increment_one"
    refuse("changed-role", r, "MNLM_CORPUS_READBACK_CLIP_IDENTITY")
    media = shadow/"media"/(clip+".mp4")
    with media.open("r+b") as f:
        byte = f.read(1); f.seek(0); f.write(bytes([byte[0]^1]))
    refuse("actual-source-byte", copy.deepcopy(base), "MNLM_CORPUS_READBACK_ARTIFACT")
    shutil.copyfile(root/"media"/(clip+".mp4"), media)
    (shadow/"report.json").write_bytes(encoded(base))
    inspect_clip(shadow, entry, row, result, bound, originals)
    need(sha(root/"report.json") == entry["report_pin"], "ORIGINAL_REPORT_UNCHANGED")
    phases["graph_and_source_faults"] = time.perf_counter()-t
    t = time.perf_counter()
    for label, mutate, code, complete in (
        ("missing-corpus-member", lambda v: v["results"].pop(), "COMPLETE_ROSTER", False),
        ("premature-completion", lambda v: v.update(corpus_complete=True), "COMPLETE_SCOPE", False),
        ("partial-serving", lambda v: None, "CORPUS_INCOMPLETE", True),
        ("wrong-plan-pin", lambda v: v["binding"].update(plan_pin="0"*64), "PIN", False)):
        v = copy.deepcopy(report); mutate(v)
        path = a.report.parent/("fault-"+label+".json")
        with path.open("xb") as f:
            f.write(encoded(v))
        try:
            read(SimpleNamespace(report=path, report_pin=sha(path), plan=a.plan,
                acquisition_report=a.acquisition_report, require_complete=complete,
                replay=[], output=a.output/(label+"-unexpected.json")))
        except ValueError as exc:
            need(str(exc) == "MNLM_CORPUS_READBACK_"+code, "CORPUS_FAULT_REASON")
            controls.append(dict(control=label, error=str(exc), artifact=path.name))
        else:
            raise AssertionError(label+" accepted")
    phases["corpus_completion_faults"] = time.perf_counter()-t
    value = dict(author_checked=True, report_pin=a.report_pin, proofs=proofs, controls=controls,
        phases=phases, seconds=time.perf_counter()-started, corpus_complete=False,
        scientific_admission=False, training_ready=False, meaning_asserted=False)
    (a.output/"report.json").write_bytes(encoded(value))
    print(json.dumps(dict(pin=sha(a.output/"report.json"), clips=len(proofs),
                          controls=len(controls), seconds=value["seconds"])), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "acquisition-report", "source-readback", "report", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--report-pin", required=True)
    main(ap.parse_args())

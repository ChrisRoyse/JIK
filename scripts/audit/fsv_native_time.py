"""Three-boundary proof using real retained recordings and recorded robot steps."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "canon")]
from senses.contract import SenseError, canonical, seal_record
from senses.pipeline import file_hash, write_records
from native_time import BoundViews, build
from native_time_readback import audit_graph
from native_time_population import inputs, source_records, publish, readback, next_step


def run(plan, audio_root, output):
    output.mkdir(parents=True, exist_ok=False)
    before = {e["sha256"]: file_hash(Path(e["path"])) for e in plan["exports"]}
    chosen, steps = [], 0
    for record in source_records(inputs(plan)):
        if record["locator"]["source"] == "bridge-v2-rlds":
            if steps < 2:
                chosen.append(record)
            steps += 1
        else:
            chosen.append(record)
    assert len(chosen) == 6 and steps == 23702
    source = output / "proof.ucr.jsonl"
    m = write_records(source, chosen, {"parent_exports": plan["exports"], "scope": "two first actual steps and both actual audio/text correspondences"})
    proof = {"schema": "native-time-plan-v1", "exports": [{"path": str(source), "sha256": m["sha256"],
             "manifest_sha256": file_hash(source.with_suffix(".manifest.json")), "count": m["count"], "bytes": source.stat().st_size}]}
    (output / "plan.json").write_bytes(canonical(proof))
    publication = publish(proof, output / "publication")
    cold = readback(proof, output / "publication")
    (output / "cold-readback.json").write_bytes(canonical(cold))
    checks = []

    def check(name, condition, **detail):
        checks.append({"name": name, "passed": bool(condition), **detail})

    def refuses(name, fn, code):
        try:
            fn()
            check(name, False, expected=code, observed="accepted")
        except SenseError as exc:
            check(name, exc.code == code, expected=code, observed=exc.code)

    graphs = [json.loads(line) for line in (output / "publication/views.jsonl").read_bytes().splitlines()]
    audio = [(r, g) for r, g in zip(chosen, graphs) if g["clock"] and g["clock"]["unit"] == "sample"]
    robot = [(r, g) for r, g in zip(chosen, graphs) if g["clock"] and g["clock"]["unit"] == "step"]
    texts = [(r, g) for r, g in zip(chosen, graphs) if g["clock"] is None]
    check("B1 exact source export and interval closure", cold["counts"]["audio_frame"] == 6224 and len(texts) == 2)
    import soundfile as sf
    source_geometry = []
    for r, g in audio:
        p = audio_root / r["locator"]["shard"]
        with sf.SoundFile(p) as wav:
            source_geometry.append({"sha256": file_hash(p), "samples": len(wav), "rate": wav.samplerate, "channels": wav.channels})
            check("B1 waveform header matches retained source " + p.name,
                  source_geometry[-1]["sha256"] == r["locator"]["record_digest"]
                  and len(wav) == r["locator"]["span"][1] and g["clock"]["seconds_per_tick"] == [1, wav.samplerate])
            last = g["nodes"][-1]
            wav.seek(last["span"][0])
            check("B1 native final short frame retained " + p.name,
                  len(wav.read(last["span"][1] - last["span"][0])) == last["span"][1] - last["span"][0])
    r, g = audio[0]
    bound = BoundViews(r, g)
    frame = g["nodes"][1]
    args = {"clock_id": g["clock"]["id"], "through": frame["span"][1], "split_group": r["unit"]}
    read = bound.read(frame["id"], **args)
    check("B2 fine view exposes exactly one original native member", len(read["members"]) == 1
          and read["members"][0]["value"] == r["channels"]["perception"]["items"][0]
          and read["view"]["required_native_media"] == ["audio"])
    refuses("B2 full recording hidden before complete interval", lambda: bound.read(g["nodes"][0]["id"], **args), "MNLM_TIME_FUTURE")
    refuses("B2 cross-recording cutoff refuses", lambda: bound.read(frame["id"], **{**args, "clock_id": audio[1][1]["clock"]["id"]}), "MNLM_TIME_CLOCK")
    refuses("B2 foreign split group refuses", lambda: bound.read(frame["id"], **{**args, "split_group": "0"*64}), "MNLM_TIME_SPLIT")
    refuses("B2 floating cutoff refuses", lambda: bound.read(frame["id"], **{**args, "through": .02}), "MNLM_TIME_SELECTION")
    altered = copy.deepcopy(g); altered["nodes"][1]["span"][1] += 1
    refuses("B1 altered native interval refuses", lambda: audit_graph(r, altered), "MNLM_TIME_READBACK_MEMBERS")
    altered2 = copy.deepcopy(g); altered2["nodes"].pop()
    refuses("B1 missing final interval refuses", lambda: audit_graph(r, altered2), "MNLM_TIME_READBACK_COVERAGE")
    altered3 = copy.deepcopy(g); altered3["nodes"][1]["members"][0]["role"] = "spoken_transcript"
    refuses("B2 invented transcript role refuses", lambda: audit_graph(r, altered3), "MNLM_TIME_READBACK_MEMBERS")
    rr, gg = robot[0]
    badclock = copy.deepcopy(gg); badclock["clock"]["seconds_per_tick"] = [1, 30]
    refuses("B1 invented frame rate for ordinal steps refuses", lambda: audit_graph(rr, badclock), "MNLM_TIME_READBACK_CLOCK")
    check("B2 ordinal co-observation is not physical synchronization", all(x[1]["clock"]["seconds_per_tick"] is None for x in robot)
          and all(m["role"] != "spoken_transcript" for x in robot for m in x[1]["nodes"][0]["members"]))
    claimed = copy.deepcopy(texts[0][1]); claimed["clock"] = copy.deepcopy(g["clock"])
    refuses("B2 chapter correspondence cannot inherit audio clock", lambda: audit_graph(texts[0][0], claimed), "MNLM_TIME_READBACK_ALIGNMENT")
    previous = {}
    next_step(gg, previous)
    gap = copy.deepcopy(robot[1][1]); gap["nodes"][0]["span"] = [3, 4]
    check("B2 ordinal gaps never create adjacency bridges", next_step(gap, previous) is None)
    badrecord = copy.deepcopy(r); badrecord["channels"]["perception"]["items"].pop()
    badrecord = seal_record(badrecord)
    refuses("B1 incomplete source coverage refuses producer", lambda: build(badrecord), "MNLM_TIME_COVERAGE")
    sourcecopy, graphcopy = copy.deepcopy(r), copy.deepcopy(g)
    copied = BoundViews(sourcecopy, graphcopy)
    sourcecopy["channels"]["perception"]["items"].clear(); graphcopy["nodes"].clear()
    check("B2 caller mutation cannot widen a bound reader", copied.read(frame["id"], **args) == read)
    shadow = output / "shadow"
    shutil.copytree(output / "publication", shadow)
    original = (shadow / "views.jsonl").read_bytes()
    (shadow / "views.jsonl").write_bytes(original + b"\n")
    refuses("B3 modified publication refuses cold consumer", lambda: readback(proof, shadow), "MNLM_TIME_READBACK_DIGEST")
    (shadow / "views.jsonl").write_bytes(original)
    raw_report = (shadow / "report.json").read_bytes()
    promoted = json.loads(raw_report); promoted["scientific_admission"] = True
    (shadow / "report.json").write_bytes(canonical(promoted))
    refuses("B3 native views cannot promote themselves to meaning", lambda: readback(proof, shadow), "MNLM_TIME_READBACK_CLAIM")
    (shadow / "report.json").write_bytes(raw_report)
    check("B3 restored publication has exact original readback", readback(proof, shadow)["report_sha256"] == cold["report_sha256"])
    after = {e["sha256"]: file_hash(Path(e["path"])) for e in plan["exports"]}
    check("B1 original exports unchanged", before == after and all(k == v for k, v in before.items()))
    report = {"author_checked": True, "passed": all(c["passed"] for c in checks), "checks": checks,
              "before": before, "after": after, "source_geometry": source_geometry,
              "publication": publication, "cold_readback": cold, "scientific_admission": False}
    (output / "fsv.json").write_bytes(canonical(report))
    print(canonical({"passed": report["passed"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]]}).decode())
    return report["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sys.exit(0 if run(json.loads(a.plan.read_bytes()), a.audio_root, a.out) else 1)

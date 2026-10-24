"""Cold source-to-target readback; independent ordinal and audio boundary sweep."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.perception_episodes_readback import graph_read
from senses.perception_train_census import load_annotations, TASKS, ANNOTATION
from senses.perception_targets import SCHEMA, FLAGS, POLICY, CUT_PIN, README_PIN, VISUALISATION_PIN
from senses.perception_corpus import PLAN_PIN


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_TARGET_READBACK_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def pinned(path, pin):
    need(sha(path) == pin, "PIN")
    return json.loads(Path(path).read_bytes())


def rational(x):
    return [x.numerator, x.denominator]


def inspect(value, annotation, graph, cut):
    need(value["schema"] == SCHEMA and value["clip"] == graph["clip"]
         and value["flags"] == FLAGS and value["policy"] == POLICY, "IDENTITY")
    need(value["provider_metadata"] == annotation["metadata"]
         and value["annotation_sha256"] == hashlib.sha256(raw(annotation)).hexdigest(),
         "ANNOTATION")
    images = {n["ordinal"]: n for n in graph["nodes"] if n["modality"] == "image"}
    audio = sorted((n for n in graph["nodes"] if n["modality"] == "audio"),
                   key=lambda n: n["ordinal"])
    n = len(images)
    need(set(images) == set(range(n)) and value["native_frame_count"] == n, "FRAME_COUNT")
    need(set(value["targets"]) == set(TASKS), "TASK_ROSTER")
    if type(cut) is int and -1 <= cut <= n:
        if cut == -1:
            time_limit = max(Fraction(*x["available_after"]) for x in graph["nodes"])
        else:
            time_limit = Fraction(*(images[cut]["span"][0] if cut < n else images[n-1]["span"][1]))
        qa = dict(status="bound", provider_cut=cut,
                  scope="full_source_no_provider_cut" if cut == -1 else "first_excluded_frame",
                  through=rational(time_limit), clock_id=graph["clock"]["id"],
                  split_group=graph["split_group"], online_forecast=False,
                  withheld_future=cut != -1 and cut < n)
    else:
        qa = dict(status="unresolved", code="QA_CUT_INVALID", provider_cut=cut)
    need(value["qa"] == qa, "QA_CUTOFF")
    # Sweep all requested event endpoints once; producer uses binary searches.
    boundaries = set()
    for task in ("action_localisation", "sound_localisation"):
        for e in annotation[task]:
            a, b = e["frame_ids"]
            if 0 <= a < b <= n:
                boundaries.add(Fraction(*images[a]["span"][0]))
                boundaries.add(Fraction(*(images[b]["span"][0] if b < n else images[n-1]["span"][1])))
    starts = [Fraction(*x["span"][0]) for x in audio]
    ends = [Fraction(*x["span"][1]) for x in audio]
    sweep, left, right = {}, 0, 0
    for boundary in sorted(boundaries):
        while left < len(audio) and ends[left] <= boundary:
            left += 1
        while right < len(audio) and starts[right] < boundary:
            right += 1
        sweep[boundary] = (left, right)
    counts, objects = Counter(), set()
    for task in TASKS:
        rows = value["targets"][task]
        need([r["provider"] for r in rows] == annotation[task], "PROVIDER_CONTENT")
        counts[task] = len(rows)
        counts[task+":unresolved"] = 0
        for row in rows:
            e, actual = row["provider"], row["native"]
            if task in ("object_tracking", "point_tracking"):
                ids = e["frame_ids"]
                stamps = e.get("timestamps", [None]*len(ids))
                need(len(stamps) == len(ids), "TRACK_LENGTH")
                coords = e["bounding_boxes"] if task == "object_tracking" else e["points"]
                need((len(coords) == len(ids) and all(len(p) == 4 for p in coords))
                     if task == "object_tracking" else
                     (len(coords) == 2 and all(len(p) == len(ids) for p in coords)),
                     "TRACK_COORDINATES")
                points = []
                for i, stamp in zip(ids, stamps):
                    if type(i) is not int or i not in images:
                        p = dict(status="unresolved", code="FRAME_OUT_OF_RANGE", ordinal=i)
                    elif stamp is not None and (type(stamp) is not int or stamp < 0):
                        p = dict(status="unresolved", code="TIMESTAMP_INVALID", ordinal=i)
                    else:
                        frame = images[i]
                        p = dict(status="bound", ordinal=i, node=frame["id"], span=frame["span"],
                            available_after=frame["available_after"],
                            timestamp_delta_us=None if stamp is None else
                            rational(Fraction(*frame["span"][0])*1000000-stamp))
                    points.append(p)
                    counts[task+":points"] += 1
                    counts[task+":unresolved_points"] += p["status"] != "bound"
                expect = dict(status="bound" if points and
                              all(p["status"] == "bound" for p in points) else "unresolved",
                              points=points)
                if task == "object_tracking" and expect["status"] == "bound":
                    objects.add(e["id"])
            elif task in ("action_localisation", "sound_localisation"):
                a, b = e["frame_ids"]
                stamps = e["timestamps"]
                if not all(type(i) is int for i in (a, b)) or not 0 <= a < b <= n:
                    expect = dict(status="unresolved", code="FRAME_INTERVAL_INVALID")
                elif (len(stamps) != 2 or not all(type(i) is int for i in stamps)
                      or not 0 <= stamps[0] < stamps[1]):
                    expect = dict(status="unresolved", code="TIMESTAMP_INTERVAL_INVALID")
                else:
                    start = Fraction(*images[a]["span"][0])
                    end = Fraction(*(images[b]["span"][0] if b < n else images[n-1]["span"][1]))
                    expect = dict(status="bound", frame_range=[a, b],
                        span=[rational(start), rational(end)], first_node=images[a]["id"],
                        last_node=images[b-1]["id"], available_after=images[b-1]["available_after"],
                        timestamp_delta_us=[rational(start*1000000-stamps[0]),
                                            rational(end*1000000-stamps[1])],
                        audio_overlap_range=[sweep[start][0], sweep[end][1]],
                        audio_relation="container_presentation_overlap_only")
            elif task == "mc_question":
                expect = dict(status="bound" if type(e["answer_id"]) is int and
                    0 <= e["answer_id"] < len(e["options"]) and qa["status"] == "bound"
                    else "unresolved", observation_scope="qa_prefix")
            else:
                expect = dict(status="bound" if e["answers"] and
                    all(i in objects for i in e["answers"]) and qa["status"] == "bound"
                    else "unresolved", object_track_ids=e["answers"])
            need(actual == expect, "MAPPING:"+task+":"+str(e["id"]))
            counts[task+":unresolved"] += expect["status"] != "bound"
    counts["qa_cut"] = qa["status"] == "bound" and cut >= 0
    return dict(counts)


def run(a):
    started = time.perf_counter()
    report = pinned(a.targets/"report.json", a.target_pin)
    corpus = pinned(a.corpus, a.corpus_pin)
    plan = pinned(a.plan, PLAN_PIN)
    annotations, member_pin = load_annotations(a.annotations)
    cuts = pinned(a.cuts, CUT_PIN)
    code = {n: sha(Path(__file__).parents[1]/n) for n in
            ("senses/perception_targets.py", "senses/perception_train_census.py",
             "senses/perception_native.py", "senses/perception_episodes.py",
             "senses/perception_corpus.py")}
    bound = dict(corpus_pin=a.corpus_pin, plan_pin=PLAN_PIN, annotation=ANNOTATION,
        annotation_member_pin=member_pin, cut_pin=CUT_PIN, provider_readme=README_PIN,
        provider_visualisation=VISUALISATION_PIN, policy=POLICY, code=code)
    need(report["binding"] == bound and report["flags"] == FLAGS
         and report["complete"] and not report["errors"], "REPORT")
    all_rows = {r["clip"]: r for r in corpus["results"]}
    requested = report["requested"]
    selected = set(requested)
    need([r["clip"] for r in corpus["results"]] == [r["clip"] for r in plan["splits"]]
         and [r["role"] for r in corpus["results"]] == [r["role"] for r in plan["splits"]]
         and len(all_rows) == 512 and corpus["corpus_complete"], "CORPUS")
    need(requested == [r["clip"] for r in report["results"]]
         == [c for c in all_rows if c in selected]
         and report["corpus_complete"] == (len(requested) == 512), "ROSTER")
    phases, checked, counts = dict(setup=time.perf_counter()-started), [], Counter()
    for entry in report["results"]:
        tick = time.perf_counter()
        clip = entry["clip"]
        root = a.corpus.parent/"clips"/clip
        source_row = all_rows[clip]
        receipt = pinned(root/"report.json", source_row["report_pin"])
        need(entry["role"] == source_row["role"] == receipt["role"], "ROLE")
        for name, file in receipt["files"].items():
            need((root/name).stat().st_size == file["bytes"]
                 and sha(root/name) == file["sha256"], "NATIVE_ARTIFACT")
        graph = json.loads((root/"public"/(clip+".json")).read_bytes())
        probe = json.loads((root/"probes"/(clip+".json")).read_bytes())
        graph_read(graph, probe, clip, graph["blob_sha256"], "train", exposure="native_timing_only")
        path = a.targets/"private"/(clip+".json")
        value = pinned(path, entry["pin"])
        native_bound = dict(report=source_row["report_pin"], media=graph["blob_sha256"],
            graph=receipt["files"][f"public/{clip}.json"]["sha256"],
            probe=receipt["files"][f"probes/{clip}.json"]["sha256"])
        need(path.stat().st_size == entry["bytes"] and value["binding"] == bound
             and value["source"] == native_bound and value["role"] == entry["role"], "TARGET_BINDING")
        found = inspect(value, annotations[clip], graph, cuts.get(clip))
        need(entry["counts"] == {t: found[t] for t in TASKS}
             and entry["unresolved"] == {t: found[t+":unresolved"] for t in TASKS}
             and entry["qa_status"] == value["qa"]["status"], "COUNTS")
        checked.append(dict(clip=clip, role=entry["role"], pin=entry["pin"], counts=found))
        counts.update(found)
        phases[clip] = time.perf_counter()-tick
    seconds = time.perf_counter()-started
    value = dict(author_checked=True, target_pin=a.target_pin, corpus_pin=a.corpus_pin,
        reader_pin=sha(__file__), checked=checked, counts=dict(counts), flags=FLAGS,
        seconds=seconds, phases=phases, phase_coverage=sum(phases.values())/seconds)
    with a.output.open("xb") as f:
        f.write(raw(value))
    print(json.dumps(dict(pin=sha(a.output), clips=len(checked), counts=dict(counts), seconds=seconds)))
    return value


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("targets", "corpus", "plan", "annotations", "cuts", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    for name in ("target-pin", "corpus-pin"):
        ap.add_argument("--"+name, required=True)
    run(ap.parse_args())

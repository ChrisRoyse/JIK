"""Private provider targets bound to native ordinals; answer-free QA intake requests.

PC-04/07/28/32: parse the pinned annotation archive once, index each native clock
once, retain unresolved coordinates. Labels never enter the public sensory graph.
"""
import argparse
from bisect import bisect_left, bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.perception_corpus import encoded, pinned, save, sha, PLAN_PIN
from senses.perception_native import source
from senses.perception_train_census import load_annotations, ANNOTATION, TASKS

SCHEMA = "perception-private-native-targets-v1"
CUT_PIN = "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4"
README_PIN = "9f739a07616e3f60af0723800d813ca87953496ba9e288fc62b4bfd7578afa85"
VISUALISATION_PIN = "4707275b12bace37c4cfc718125792b9236f30f9e78de9f7c3277c70a643c8fe"
FLAGS = dict(meaning_asserted=False, scientific_admission=False, training_ready=False,
             world_facts_established=False, physical_sync_calibrated=False,
             sealed_evaluation=False, source_annotation_targets=True)
POLICY = dict(frame_coordinates="zero_based_decoded_video_ordinals",
    point_coordinates="provider_normalised_y_then_x_arrays;_preserved",
    box_coordinates="provider_normalised_x1_y1_x2_y2_rows;_preserved",
    event_end="exclusive_as_provider_visualisation_range",
    timestamps="retain_provider_microseconds_and_native_delta;_never_replace_native_pts",
    audio_coordinates="container_overlap_only;_separate_provider_wav_equivalence_unestablished",
    qa_cut="nonnegative_first_excluded_frame;_-1_full_source;_never_clip_bad_index",
    grounding="provider_annotation_about_recording;_not_independent_world_verification")


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_TARGET_"+code)


def rat(x):
    return [x.numerator, x.denominator]


def code_pins():
    return {n: sha(Path(__file__).parents[1]/n) for n in
            ("senses/perception_targets.py", "senses/perception_train_census.py",
             "senses/perception_native.py", "senses/perception_episodes.py",
             "senses/perception_corpus.py")}


class NativeClock:
    def __init__(self, graph):
        self.graph = graph
        self.frames = sorted((n for n in graph["nodes"] if n["modality"] == "image"),
                             key=lambda n: n["ordinal"])
        self.audio = sorted((n for n in graph["nodes"] if n["modality"] == "audio"),
                            key=lambda n: n["ordinal"])
        for nodes in (self.frames, self.audio):
            need(nodes and [n["ordinal"] for n in nodes] == list(range(len(nodes))), "NATIVE_ORDER")
            need(all(Fraction(*a["span"][1]) <= Fraction(*b["span"][0])
                     for a, b in zip(nodes, nodes[1:])), "NATIVE_OVERLAP")
        self.starts = [Fraction(*n["span"][0]) for n in self.audio]
        self.ends = [Fraction(*n["span"][1]) for n in self.audio]

    def point(self, ordinal, timestamp=None):
        if type(ordinal) is not int or not 0 <= ordinal < len(self.frames):
            return dict(status="unresolved", code="FRAME_OUT_OF_RANGE", ordinal=ordinal)
        node = self.frames[ordinal]
        delta = None
        if timestamp is not None:
            if type(timestamp) is not int or timestamp < 0:
                return dict(status="unresolved", code="TIMESTAMP_INVALID", ordinal=ordinal)
            delta = rat(Fraction(*node["span"][0])*1000000-timestamp)
        return dict(status="bound", ordinal=ordinal, node=node["id"], span=node["span"],
                    available_after=node["available_after"], timestamp_delta_us=delta)

    def event(self, event):
        indices, timestamps = event.get("frame_ids"), event.get("timestamps")
        if not (type(indices) is list and len(indices) == 2
                and all(type(v) is int for v in indices)
                and 0 <= indices[0] < indices[1] <= len(self.frames)):
            return dict(status="unresolved", code="FRAME_INTERVAL_INVALID")
        if not (type(timestamps) is list and len(timestamps) == 2
                and all(type(v) is int for v in timestamps)
                and 0 <= timestamps[0] < timestamps[1]):
            return dict(status="unresolved", code="TIMESTAMP_INTERVAL_INVALID")
        a, b = indices
        start = Fraction(*self.frames[a]["span"][0])
        end = Fraction(*self.frames[b]["span"][0]) if b < len(self.frames) else Fraction(*self.frames[-1]["span"][1])
        lo, hi = bisect_right(self.ends, start), bisect_left(self.starts, end)
        return dict(status="bound", frame_range=indices, span=[rat(start), rat(end)],
            first_node=self.frames[a]["id"], last_node=self.frames[b-1]["id"],
            available_after=self.frames[b-1]["available_after"],
            timestamp_delta_us=[rat(start*1000000-timestamps[0]), rat(end*1000000-timestamps[1])],
            audio_overlap_range=[lo, hi], audio_relation="container_presentation_overlap_only")

    def qa(self, cut):
        if type(cut) is not int or cut < -1 or cut > len(self.frames):
            return dict(status="unresolved", code="QA_CUT_INVALID", provider_cut=cut)
        if cut == -1:
            end = max(Fraction(*n["available_after"]) for n in self.graph["nodes"])
            scope = "full_source_no_provider_cut"
        elif cut == len(self.frames):
            end = Fraction(*self.frames[-1]["span"][1])
            scope = "first_excluded_frame"
        else:
            end = Fraction(*self.frames[cut]["span"][0])
            scope = "first_excluded_frame"
        return dict(status="bound", provider_cut=cut, scope=scope, through=rat(end),
                    clock_id=self.graph["clock"]["id"], split_group=self.graph["split_group"],
                    online_forecast=False, withheld_future=cut != -1 and cut < len(self.frames))


def align(ann, graph, cut):
    need(set(ann) == {"metadata", *TASKS} and ann["metadata"]["video_id"] == graph["clip"]
         and ann["metadata"]["split"] == "train", "ANNOTATION_IDENTITY")
    clock = NativeClock(graph)
    result = {}
    objects = {e["id"] for e in ann["object_tracking"]}
    need(len(objects) == len(ann["object_tracking"]), "OBJECT_IDS")
    qa = clock.qa(cut)
    for task in TASKS:
        rows = []
        supported = ({r["provider"]["id"] for r in result["object_tracking"]
                      if r["native"]["status"] == "bound"} if task == "grounded_question" else set())
        need(type(ann[task]) is list and len({e["id"] for e in ann[task]}) == len(ann[task]), "TARGET_IDS")
        for event in ann[task]:
            if task in ("action_localisation", "sound_localisation"):
                native = clock.event(event)
            elif task in ("object_tracking", "point_tracking"):
                ids = event["frame_ids"]
                timestamps = event.get("timestamps", [None]*len(ids))
                need(type(ids) is list and len(timestamps) == len(ids), "TRACK_SHAPE")
                coordinates = event["bounding_boxes" if task == "object_tracking" else "points"]
                if task == "object_tracking":
                    need(len(coordinates) == len(ids) and all(len(p) == 4 for p in coordinates),
                         "TRACK_COORDINATE_COUNT")
                else:
                    need(len(coordinates) == 2 and all(len(p) == len(ids) for p in coordinates),
                         "TRACK_COORDINATE_COUNT")
                points = [clock.point(i, t) for i, t in zip(ids, timestamps)]
                native = dict(status="bound" if points and all(p["status"] == "bound" for p in points)
                              else "unresolved", points=points)
            elif task == "grounded_question":
                native = dict(status="bound" if event["answers"]
                              and all(i in supported for i in event["answers"])
                              and qa["status"] == "bound"
                              else "unresolved", object_track_ids=event["answers"])
            else:
                native = dict(status="bound" if type(event["answer_id"]) is int
                              and 0 <= event["answer_id"] < len(event["options"])
                              and qa["status"] == "bound" else "unresolved",
                              observation_scope="qa_prefix")
            rows.append(dict(provider=event, native=native))
        result[task] = rows
    return dict(schema=SCHEMA, clip=graph["clip"], annotation_sha256=hashlib.sha256(encoded(ann)).hexdigest(),
        provider_metadata=ann["metadata"], policy=POLICY, flags=FLAGS,
        native_frame_count=len(clock.frames), qa=qa, targets=result)


class Corpus:
    def __init__(self, report, pin, plan):
        self.path = Path(report)
        self.report = pinned(report, pin)
        self.plan = pinned(plan, PLAN_PIN)
        need(self.report["schema"] == "perception-native-corpus-v1" and self.report["corpus_complete"]
             and self.report["complete"] and not self.report["errors"]
             and self.report["binding"]["plan_pin"] == PLAN_PIN, "CORPUS")
        need([r["clip"] for r in self.report["results"]] == [r["clip"] for r in self.plan["splits"]]
             and len(self.report["results"]) == 512, "CORPUS_ROSTER")
        self.rows = {r["clip"]: r for r in self.report["results"]}
        need([r["role"] for r in self.report["results"]] == [r["role"] for r in self.plan["splits"]], "ROLES")

    def read(self, clip):
        need(clip in self.rows, "CLIP")
        entry = self.rows[clip]
        root = self.path.parent/"clips"/clip
        graph, streams, bound = source(root, entry["report_pin"], clip)
        return graph, streams, bound


def run(a):
    started = time.perf_counter()
    corpus = Corpus(a.corpus, a.corpus_pin, a.plan)
    ann, member_pin = load_annotations(a.annotations)
    cuts = pinned(a.cuts, CUT_PIN)
    need(sha(a.provider_readme) == README_PIN and sha(a.provider_visualisation) == VISUALISATION_PIN,
         "PROVIDER_METHOD")
    wanted = list(corpus.rows) if a.clips is None else a.clips
    selected = set(wanted)
    need(wanted and len(selected) == len(wanted) and selected <= set(corpus.rows), "REQUEST_ROSTER")
    wanted = [c for c in corpus.rows if c in selected]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"private").mkdir()
    bound = dict(corpus_pin=a.corpus_pin, plan_pin=PLAN_PIN, annotation=ANNOTATION,
        annotation_member_pin=member_pin, cut_pin=CUT_PIN, provider_readme=README_PIN,
        provider_visualisation=VISUALISATION_PIN, policy=POLICY,
        code=code_pins())
    results, errors, phases = [], [], dict(setup=time.perf_counter()-started)
    for clip in wanted:
        tick = time.perf_counter()
        try:
            graph, _, native_bound = corpus.read(clip)
            target = align(ann[clip], graph, cuts.get(clip))
            target.update(source=native_bound, role=corpus.rows[clip]["role"], binding=bound)
            path = a.output/"private"/(clip+".json")
            pin = save(path, target)
            counts = {t: len(v) for t, v in target["targets"].items()}
            unresolved = {t: sum(r["native"]["status"] != "bound" for r in v)
                          for t, v in target["targets"].items()}
            results.append(dict(clip=clip, role=target["role"], pin=pin, bytes=path.stat().st_size,
                                counts=counts, unresolved=unresolved, qa_status=target["qa"]["status"]))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(dict(clip=clip, error=str(exc), type=type(exc).__name__))
        phases[clip] = time.perf_counter()-tick
        print(json.dumps(dict(clip=clip, ok=not errors or errors[-1]["clip"] != clip,
                              seconds=phases[clip])), flush=True)
    seconds = time.perf_counter()-started
    report = dict(schema=SCHEMA, binding=bound, requested=wanted,
        complete=not errors and len(results) == len(wanted),
        corpus_complete=not errors and len(results) == 512, results=results, errors=errors,
        flags=FLAGS, phases=phases, seconds=seconds, phase_coverage=sum(phases.values())/seconds)
    pin = save(a.output/"report.json", report)
    print(json.dumps(dict(pin=pin, clips=len(results), errors=len(errors), seconds=seconds)))
    return 0 if report["complete"] else 2


class Targets:
    """Private targets; the sole sensory request method emits no labels or answers."""
    def __init__(self, root, *, expected, corpus, require_complete=True):
        self.root, self.corpus = Path(root), corpus
        self.report = pinned(self.root/"report.json", expected)
        need(self.report["schema"] == SCHEMA and self.report["complete"] and not self.report["errors"]
             and self.report["flags"] == FLAGS
             and self.report["binding"]["corpus_pin"] == sha(corpus.path), "REPORT_BINDING")
        need(self.report["binding"]["code"] == code_pins()
             and self.report["binding"]["policy"] == POLICY
             and self.report["binding"]["plan_pin"] == PLAN_PIN
             and self.report["binding"]["annotation"] == ANNOTATION
             and self.report["binding"]["cut_pin"] == CUT_PIN
             and self.report["binding"]["provider_readme"] == README_PIN
             and self.report["binding"]["provider_visualisation"] == VISUALISATION_PIN,
             "METHOD_BINDING")
        need(self.report["requested"] == [r["clip"] for r in self.report["results"]], "TARGET_ROSTER")
        requested = set(self.report["requested"])
        need(self.report["requested"] == [c for c in corpus.rows if c in requested]
             and self.report["corpus_complete"] == (len(self.report["results"]) == 512),
             "TARGET_ROSTER")
        need(not require_complete or self.report["corpus_complete"], "TARGET_CORPUS_INCOMPLETE")
        self.entries = {r["clip"]: r for r in self.report["results"]}

    def read(self, clip):
        need(clip in self.entries, "TARGET_MISSING")
        entry = self.entries[clip]
        value = pinned(self.root/"private"/(clip+".json"), entry["pin"])
        need(value["binding"] == self.report["binding"] and value["policy"] == POLICY
             and value["flags"] == FLAGS and value["clip"] == clip
             and value["role"] == entry["role"] == self.corpus.rows[clip]["role"], "TARGET_BINDING")
        graph, _, bound = self.corpus.read(clip)
        need(value["source"] == bound, "TARGET_SOURCE")
        return value, graph

    def qa_request(self, clip, *, role, through=None):
        value, graph = self.read(clip)
        need(role == value["role"], "REQUEST_ROLE")
        cut = value["qa"]
        need(cut["status"] == "bound", "QA_CUT_UNRESOLVED")
        limit = Fraction(*cut["through"])
        need(through is None or type(through) is list and len(through) == 2
             and all(type(v) is int for v in through) and through[1] > 0, "REQUEST_TIME")
        chosen = limit if through is None else Fraction(*through)
        need(chosen >= 0 and chosen <= limit, "QA_FUTURE")
        return dict(clock_id=graph["clock"]["id"], split_group=graph["split_group"],
                    through=rat(chosen), required_media=["image", "audio"])

    def require_target(self, clip, *, role, task, annotation_id):
        value, _ = self.read(clip)
        need(role == value["role"], "REQUEST_ROLE")
        need(task in TASKS, "TARGET_TASK")
        rows = [r for r in value["targets"][task] if r["provider"]["id"] == annotation_id]
        need(len(rows) == 1 and rows[0]["native"]["status"] == "bound", "TARGET_UNRESOLVED")
        return rows[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "plan", "annotations", "cuts", "provider-readme",
                 "provider-visualisation", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--corpus-pin", required=True)
    ap.add_argument("--clips", nargs="+")
    raise SystemExit(run(ap.parse_args()))

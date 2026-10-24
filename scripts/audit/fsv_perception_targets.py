"""Real annotation, decoded-cutoff and persisted-reader boundaries; no model claims."""
import argparse
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.perception_targets_readback import inspect, run as cold
from senses.perception_targets import Corpus, Targets, align, FLAGS
from senses.perception_corpus import pinned, sha, save
from senses.perception_train_census import load_annotations
from senses.perception_decode import build
from senses.perception_native import NativePrefix

WITNESSES = {"video_5930", "video_6074", "video_4792", "video_6322", "video_8625"}


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_TARGET_FSV_"+code)


def main(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin, a.plan)
    targets = Targets(a.targets, expected=a.target_pin, corpus=corpus, require_complete=False)
    need(set(targets.entries) == WITNESSES, "WITNESS_ROSTER")
    annotations, _ = load_annotations(a.annotations)
    cuts = json.loads(a.cuts.read_bytes())
    phases, proofs, controls = dict(setup=time.perf_counter()-started), [], []

    def refused(name, fn, code):
        try:
            fn()
        except ValueError as exc:
            need(str(exc).startswith(code), "WRONG_REFUSAL:"+name+":"+str(exc))
            controls.append(dict(control=name, error=str(exc)))
            return
        raise ValueError("MNLM_TARGET_FSV_ACCEPTED:"+name)

    for clip, entry in targets.entries.items():
        tick = time.perf_counter()
        value, graph = targets.read(clip)
        found = inspect(value, annotations[clip], graph, cuts[clip])
        before = dict(target_pin=entry["pin"], source=value["source"], counts=found,
                      provider_cut=cuts[clip], qa=value["qa"])
        role = entry["role"]
        chosen = None if clip == "video_6322" else [1, 10]
        request = targets.qa_request(clip, role=role, through=chosen)
        need(set(request) == {"clock_id", "split_group", "through", "required_media"},
             "ANSWER_LEAK")
        episode_root = corpus.path.parent/"clips"/clip
        episode_pin = corpus.rows[clip]["report_pin"]
        prefix_root = a.output/("prefix-"+clip)
        build(episode_root, episode_pin, clip, request["through"], prefix_root)
        prefix_pin = sha(prefix_root/"manifest.json")
        consumer = NativePrefix(prefix_root, expected=prefix_pin,
                                episodes=episode_root, episode_pin=episode_pin)
        seen, totals, digest = [], Counter(), hashlib.sha256()
        for item in consumer.consume(**request):
            node, values = item["node"], item["values"]
            need(not values.flags.writeable and not item["meaning_asserted"], "MEDIA_FLAGS")
            need(Fraction(*node["available_after"]) <= Fraction(*request["through"]), "FUTURE")
            digest.update(values.tobytes())
            totals[item["modality"]] += 1
            seen.append(node["id"])
        expected = sorted((n for n in graph["nodes"]
                           if Fraction(*n["available_after"]) <= Fraction(*request["through"])),
                          key=lambda n: (Fraction(*n["span"][0]), n["stream"], n["ordinal"]))
        need(seen == [n["id"] for n in expected] and all(totals[m] > 0 for m in ("image", "audio")),
             "OBSERVATION_ROSTER")
        if cuts[clip] >= 0:
            need(totals["image"] <= cuts[clip]
                 and all(n["ordinal"] < cuts[clip] for n in expected if n["modality"] == "image"),
                 "EXCLUDED_FRAME")
            if request["through"] == value["qa"]["through"]:
                need(totals["image"] == cuts[clip], "FULL_QA_PREFIX")
        maximum = Fraction(*value["qa"]["through"])+1
        refused(clip+":qa-future", lambda: targets.qa_request(clip, role=role,
                through=[maximum.numerator, maximum.denominator]), "MNLM_NATIVE_TARGET_QA_FUTURE")
        refused(clip+":role", lambda: targets.qa_request(clip, role="wrong"),
                "MNLM_NATIVE_TARGET_REQUEST_ROLE")
        for field, replacement, code in (
                ("clock_id", "wrong", "MNLM_NATIVE_PREFIX_CLOCK"),
                ("split_group", "wrong", "MNLM_NATIVE_PREFIX_SPLIT"),
                ("required_media", ["image", "text"], "MNLM_NATIVE_PREFIX_MISSING_MODALITY"),
                ("through", [maximum.numerator, maximum.denominator], "MNLM_NATIVE_PREFIX_UNDECODED_FUTURE")):
            bad = dict(request)
            bad[field] = replacement
            refused(clip+":"+field, lambda b=bad: next(consumer.consume(**b)), code)
        need(sha(a.targets/"private"/(clip+".json")) == before["target_pin"], "SOURCE_MUTATED")
        proofs.append(dict(clip=clip, before=before, action=request,
            after=dict(prefix_pin=prefix_pin, counts=dict(totals), payload_sha256=digest.hexdigest(),
                       request_contains_answers=False, eligible_native_nodes_only=True)))
        phases[clip] = time.perf_counter()-tick
        print(json.dumps(dict(clip=clip, counts=dict(totals), seconds=phases[clip])), flush=True)
    tick = time.perf_counter()
    bad_clip = "video_4792"
    value, graph = targets.read(bad_clip)
    invalid = next(r for r in value["targets"]["object_tracking"] if r["native"]["status"] != "bound")
    refused("real-out-of-range-track", lambda: targets.require_target(bad_clip,
            role=value["role"], task="object_tracking", annotation_id=invalid["provider"]["id"]),
            "MNLM_NATIVE_TARGET_TARGET_UNRESOLVED")
    # Independent reader catches semantically wrong mappings even after re-signing.
    for label, mutate, code in (
            ("qa-limit", lambda v: v["qa"].update(through=[999, 1]), "MNLM_TARGET_READBACK_QA_CUTOFF"),
            ("provider-label", lambda v: v["targets"]["object_tracking"][0]["provider"].update(label="changed"),
             "MNLM_TARGET_READBACK_PROVIDER_CONTENT"),
            ("invented-frame", lambda v: v["targets"]["object_tracking"][0]["native"]["points"][0].update(node="invented"),
             "MNLM_TARGET_READBACK_MAPPING"),
            ("native-audio-range", lambda v: v["targets"]["sound_localisation"][0]["native"].update(audio_overlap_range=[0, 0]),
             "MNLM_TARGET_READBACK_MAPPING")):
        changed = copy.deepcopy(value)
        mutate(changed)
        path = a.output/("resigned-"+label+".json")
        pin = save(path, changed)
        refused(label, lambda p=path, h=pin: inspect(pinned(p, h), annotations[bad_clip],
                graph, cuts[bad_clip]), code)
    # Missing/bad cuts remain unresolved, never clamped to the last frame.
    for cut in (None, -2, value["native_frame_count"]+1, True):
        mapped = align(annotations[bad_clip], graph, cut)
        need(mapped["qa"]["status"] == "unresolved" and
             all(r["native"]["status"] == "unresolved" for t in ("mc_question", "grounded_question")
                 for r in mapped["targets"][t]), "BAD_CUT")
        pin = save(a.output/("bad-cut-"+str(cut)+".json"), mapped)
        controls.append(dict(control="bad-cut:"+str(cut), pin=pin, code=mapped["qa"]["code"]))
    changed = copy.deepcopy(annotations[bad_clip])
    changed["grounded_question"] = [dict(id=-1, answers=[invalid["provider"]["id"]])]
    dependency = align(changed, graph, cuts[bad_clip])
    need(dependency["targets"]["grounded_question"][0]["native"]["status"] == "unresolved",
         "UNRESOLVED_DEPENDENCY")
    controls.append(dict(control="grounded-answer-dependency",
                         pin=save(a.output/"dependency-control.json", dependency)))
    changed = copy.deepcopy(annotations["video_8625"])
    changed["point_tracking"][0]["points"][0].pop()
    _, point_graph = targets.read("video_8625")
    refused("point-coordinate-count", lambda: align(changed, point_graph, cuts["video_8625"]),
            "MNLM_NATIVE_TARGET_TRACK_COORDINATE_COUNT")
    # Actual persisted-byte corruption, isolated from the immutable originals.
    shadow = a.output/"corrupt"
    (shadow/"private").mkdir(parents=True)
    for c in targets.entries:
        (shadow/"private"/(c+".json")).write_bytes((a.targets/"private"/(c+".json")).read_bytes())
    (shadow/"report.json").write_bytes((a.targets/"report.json").read_bytes())
    path = shadow/"private"/(bad_clip+".json")
    path.write_bytes(path.read_bytes()+b" ")
    corrupt = Targets(shadow, expected=a.target_pin, corpus=corpus, require_complete=False)
    refused("private-byte", lambda: corrupt.read(bad_clip), "MNLM_NATIVE_CORPUS_PIN")
    refused("incomplete-corpus", lambda: Targets(a.targets, expected=a.target_pin, corpus=corpus),
            "MNLM_NATIVE_TARGET_TARGET_CORPUS_INCOMPLETE")
    phases["negative_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    # Re-open everything from persisted files after the real path ran.
    args = copy.copy(a)
    args.output = a.output/"cold.json"
    readback = cold(args)
    need(len(readback["checked"]) == len(WITNESSES), "COLD_COUNT")
    phases["cold_readback"] = time.perf_counter()-tick
    seconds = time.perf_counter()-started
    result = dict(author_checked=True, target_pin=a.target_pin, corpus_pin=a.corpus_pin,
        code_pin=sha(__file__), proofs=proofs, controls=controls, flags=FLAGS,
        cold_pin=sha(args.output), phases=phases, seconds=seconds,
        phase_coverage=sum(phases.values())/seconds)
    pin = save(a.output/"report.json", result)
    print(json.dumps(dict(pin=pin, clips=len(proofs), controls=len(controls), seconds=seconds)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("targets", "corpus", "plan", "annotations", "cuts", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    for name in ("target-pin", "corpus-pin"):
        ap.add_argument("--"+name, required=True)
    main(ap.parse_args())

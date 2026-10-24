"""Private sound-target correspondence attachments; original annotations stay unchanged."""
import argparse
from bisect import bisect_left, bisect_right
from fractions import Fraction
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.audio_correspondence import Alignment, need
from senses.contract import digest
from senses.perception_corpus import pinned, save, sha
from senses.perception_targets import Corpus, Targets, NativeClock


class SoundCorrespondence:
    def __init__(self, targets, alignment, clips):
        self.targets, self.alignment = targets, alignment
        self.clips = {}
        for clip in clips:
            value, graph = targets.read(clip)
            need(clip in alignment.records and
                 digest(graph) == digest(alignment.native[clip][0]), "TARGET_SOURCE")
            clock = NativeClock(graph)
            need(all(a == b for a, b in zip(clock.ends[:-1], clock.starts[1:])), "AUDIO_CLOCK_GAP")
            events = {r["provider"]["id"]: r for r in value["targets"]["sound_localisation"]}
            for row in events.values():
                need(row["native"] == clock.event(row["provider"]), "TARGET_COORDINATES")
            self.clips[clip] = (value, clock, events)

    def bind(self, clip, annotation_id, *, target_pin):
        need(clip in self.clips, "TARGET_CLIP")
        need(target_pin == self.targets.entries[clip]["pin"], "TARGET_SOURCE")
        value, clock, events = self.clips[clip]
        need(annotation_id in events, "TARGET_ID")
        event = events[annotation_id]
        native = event["native"]
        base = dict(clip=clip, annotation_id=annotation_id, target_sha256=self.targets.entries[clip]["pin"],
            original_target_digest=digest(event), native_interval=native,
            interpretation="provider_annotation_about_recording",
            source_waveforms_identical=False, annotation_truth_established=False,
            scientific_admission=False, training_ready=False, physical_sync_calibrated=False,
            online_feedback_eligible=False, alignment_scope="retrospective_full_source_audit")
        try:
            need(native["status"] == "bound", "TARGET_UNRESOLVED")
            record = self.alignment.records[clip]
            need(record["summary"]["correspondence_supported"], "CORRESPONDENCE_UNRESOLVED")
            # Frame-index annotations already describe the container clock. Do not silently
            # reinterpret their coordinates as WAV times when a different offset is measured.
            need(all(w["lag_samples"] == 0 for w in record["windows"] if w["window_qualified"]),
                 "NONZERO_MAPPING_REQUIRES_PROVIDER_CLOCK")
            start, end = map(lambda r: Fraction(*r), native["span"])
            need(clock.starts[0] <= start < end <= clock.ends[-1], "TARGET_OUTSIDE_AUDIO")
            i, j = bisect_right(clock.ends, start), bisect_left(clock.starts, end)-1
            offsets = self.alignment.native[clip][2]
            rate = record["rate"]
            a = (start-clock.starts[i])*rate
            b = (end-clock.starts[j])*rate
            first = offsets[i]+a.numerator//a.denominator
            last = offsets[j]-(-b.numerator//b.denominator)
            graph = clock.graph
            projection = self.alignment.project(clip, [first, last],
                wav_sha256=record["sources"]["wav"]["source_sha256"],
                mp4_sha256=graph["blob_sha256"], clock_id=graph["clock"]["id"],
                split_group=graph["split_group"], required_channels=list(range(record["channels"])))
            return dict(**base, status="covered", correspondence=projection,
                rounding="include_every_native_sample_overlapping_frame_interval")
        except ValueError as exc:
            return dict(**base, status="unresolved", code=str(exc))


def main(a):
    started = time.perf_counter()
    alignment = Alignment(a.alignment, a.alignment_pin)
    report = pinned(a.targets/"report.json", a.targets_pin)
    corpus = Corpus(a.corpus, report["binding"]["corpus_pin"], a.plan)
    targets = Targets(a.targets, expected=a.targets_pin, corpus=corpus)
    bound = SoundCorrespondence(targets, alignment, list(alignment.records))
    phases = dict(open=time.perf_counter()-started)
    tick = time.perf_counter()
    rows = [bound.bind(clip, key, target_pin=targets.entries[clip]["pin"])
            for clip, (_, _, events) in bound.clips.items() for key in events]
    phases["targets"] = time.perf_counter()-tick
    controls = []
    tick = time.perf_counter()
    for clip, (_, clock, events) in bound.clips.items():
        def refuse(name, operation, expected):
            try:
                operation()
            except ValueError as exc:
                need(str(exc) == expected, "TARGET_CONTROL")
                controls.append(dict(name=clip+":"+name, code=str(exc)))
            else:
                need(False, "TARGET_CONTROL_ACCEPTED:"+name)
        refuse("missing-target", lambda: bound.bind(clip, "unrecorded-target",
               target_pin=targets.entries[clip]["pin"]),
               "MNLM_AUDIO_ALIGNMENT_TARGET_ID")
        original = dict(bound.clips[clip][2])
        other = next(c for c in bound.clips if c != clip)
        # Real other-source annotation, retaining its original identity.
        refuse("foreign-target", lambda: bound.bind(clip, next(iter(events)),
               target_pin=targets.entries[other]["pin"]),
               "MNLM_AUDIO_ALIGNMENT_TARGET_SOURCE")
        need(bound.clips[clip][2] == original, "TARGET_CONTROL_MUTATION")
        for row in alignment.records[clip]["windows"]:
            if row["window_qualified"]:
                continue
            request = alignment.records[clip]
            refuse("coverage-hole-"+str(row["index"]),
                lambda row=row, request=request: alignment.project(clip, row["wav_samples"],
                    wav_sha256=request["sources"]["wav"]["source_sha256"],
                    mp4_sha256=clock.graph["blob_sha256"], clock_id=clock.graph["clock"]["id"],
                    split_group=clock.graph["split_group"], required_channels=list(range(request["channels"]))),
                "MNLM_AUDIO_ALIGNMENT_UNCOVERED_INTERVAL")
    phases["controls"] = time.perf_counter()-tick
    result = dict(schema="private-sound-correspondence-v1", author_checked=True,
        alignment_pin=a.alignment_pin, targets_pin=a.targets_pin, targets=str(a.targets),
        rows=rows, covered=sum(r["status"] == "covered" for r in rows),
        unresolved=sum(r["status"] == "unresolved" for r in rows), controls=controls,
        scientific_admission=False, training_ready=False, physical_sync_calibrated=False,
        online_feedback_eligible=False, phases=phases, seconds=time.perf_counter()-started,
        code_pin=sha(Path(__file__)))
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    pin = save(a.output, result)
    import json
    print(json.dumps(dict(pin=pin, covered=result["covered"], unresolved=result["unresolved"],
                         controls=len(controls), seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment", "targets", "corpus", "plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--alignment-pin", required=True)
    p.add_argument("--targets-pin", required=True)
    main(p.parse_args())

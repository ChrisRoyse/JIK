"""Cold, independently selected native/annotation readback of speech-event memory."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import wave
from xml.dom import minidom
import zipfile

STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.episodic_memory import canonical, digest
from claims.timed_speech_memory import SpeechEvidence, SpeechMemory
from senses.timed_speech import AmiSource, FLAGS, NITE, SpeechViews, need, save
from senses.perception_corpus import sha
from audit.timed_speech_witness import readout, request


def xml_readback(source, event):
    with zipfile.ZipFile(source/"ami_public_manual_1.6.2.zip") as archive:
        document = minidom.parseString(archive.read(event["text"]["words_file"]))
        ordered = [e for e in document.documentElement.childNodes if e.nodeType == e.ELEMENT_NODE]
        ids = {e.getAttribute("nite:id"): i for i, e in enumerate(ordered)}
        names = re.findall(r"id\(([^)]+)\)", event["text"]["reference"])
        selected = ordered[ids[names[0]]:ids[names[-1]]+1]
        rows = []
        for e in selected:
            attrs = dict(e.attributes.items())
            attrs[NITE+"id"] = attrs.pop("nite:id")
            text = "".join(c.data for c in e.childNodes if c.nodeType == c.TEXT_NODE) or None
            lo, hi = Fraction(attrs["starttime"]), Fraction(attrs["endtime"])
            rows.append(dict(tag=e.tagName, attributes=attrs, text=text,
                             span=[[lo.numerator, lo.denominator], [hi.numerator, hi.denominator]]))
        need(rows == event["text"]["words"], "COLD_XML")
        # Independently resolve the segment and channel link, not just its words.
        segdoc = minidom.parseString(archive.read(event["text"]["segments_file"]))
        seg = next(e for e in segdoc.getElementsByTagName("segment")
                   if e.getAttribute("nite:id") == event["segment"][NITE+"id"])
        need(seg.getAttribute("channel") == event["speaker"]["channel"]
             and seg.getElementsByTagName("nite:child")[0].getAttribute("href") ==
             event["text"]["reference"], "COLD_SEGMENT")
        meta = minidom.parseString(archive.read("corpusResources/meetings.xml"))
        meeting = next(e for e in meta.getElementsByTagName("meeting")
                       if e.getAttribute("observation") == event["meeting"])
        person = next(e for e in meeting.getElementsByTagName("speaker")
                      if e.getAttribute("nxt_agent") == event["agent"])
        attrs = dict(person.attributes.items())
        attrs[NITE+"id"] = attrs.pop("nite:id")
        need(attrs == event["speaker"], "COLD_SPEAKER_MAPPING")
    return len(rows)


def decoded_readback(source, view, output):
    event = view.manifest["event"]
    image, audio = event["image"], event["audio"]
    first, last = image["frames"][0]["pts"], image["frames"][-1]["pts"]
    # Different selection: timestamps, not decode ordinals. Original PTS preserved.
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
           "-err_detect", "explode", "-copyts", "-noautorotate", "-i", str(source/image["file"]),
           "-map", "0:0", "-vf", f"select=gte(pts\\,{first})*lte(pts\\,{last}),format=rgb24",
           "-frames:v", str(len(image["frames"])), "-fps_mode", "passthrough",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with output.open("xb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
    need(result.returncode == 0 and not result.stderr, "COLD_STRICT_DECODE")
    need(sha(output) == view.manifest["files"]["frames.rgb"]["sha256"]
         and output.stat().st_size == view.manifest["files"]["frames.rgb"]["bytes"], "COLD_PIXELS")
    # Original sample bytes, via WAV seek; compare with the downstream consumer bytes.
    with wave.open(str(source/audio["file"])) as w:
        need(w.getframerate() == audio["rate"] and w.getnchannels() == audio["channels"],
             "COLD_AUDIO_FORMAT")
        w.setpos(audio["start_sample"])
        raw = w.readframes(audio["stop_sample"]-audio["start_sample"])
    need(hashlib.sha256(raw).hexdigest() == view.manifest["files"]["audio.pcm"]["sha256"],
         "COLD_SAMPLES")
    return dict(frames=len(image["frames"]), samples=len(raw)//2, pixels=sha(output),
                pcm=hashlib.sha256(raw).hexdigest())


def main(a):
    need(sha(a.run/"report.json") == a.report_pin, "COLD_REPORT")
    report = json.loads((a.run/"report.json").read_bytes())
    a.output.mkdir(exist_ok=False)
    # An independent full file pass precedes the public consumer's own binding.
    source_root = Path(report["source"])
    manifest = json.loads((source_root/"manifest.json").read_bytes())
    source_bytes = 0
    for name, expected in manifest["files"].items():
        h = hashlib.sha256()
        with (source_root/name).open("rb") as f:
            while raw := f.read(1024**2):
                source_bytes += len(raw)
                h.update(raw)
        need(h.hexdigest() == expected["sha256"], "COLD_SOURCE_BYTES")
    source = AmiSource(source_root, report["source_pin"])
    views = [SpeechViews(source, a.run/r["directory"], r["pin"]) for r in report["views"]]
    evidence = SpeechEvidence(source, views)
    phases = dict(import_source_bind=time.perf_counter()-STARTED)
    tick = time.perf_counter()
    rows = []
    for i, view in enumerate(views):
        event = view.manifest["event"]
        rows.append(dict(segment=event["segment"][NITE+"id"],
                         annotation_rows=xml_readback(source_root, event),
                         **decoded_readback(source_root, view, a.output/f"{i}.rgb")))
    phases["independent_xml_pcm_pixels"] = time.perf_counter()-tick
    tick = time.perf_counter()
    # SQL chain recalculation before the production restore path.
    for name in ("memory.sqlite", "rollback.sqlite"):
        previous = "0"*64
        with sqlite3.connect("file:"+str(a.run/name)+"?mode=ro", uri=True) as db:
            for count, (seq, prev, raw, pin) in enumerate(db.execute(
                    "SELECT seq,previous,body,hash FROM events ORDER BY seq")):
                need(seq == count and prev == previous, "COLD_CHAIN_ORDER")
                value = json.loads(raw)
                need(canonical(value) == raw and
                     digest(dict(seq=seq, previous=prev, body=value)) == pin, "COLD_CHAIN_HASH")
                previous = pin
    current = SpeechMemory(a.run/"memory.sqlite", evidence, create=False,
                           retained=report["final"]["retained"])
    need(current.snapshot() == report["final"], "COLD_FINAL")
    retained = json.loads((a.run/"rollback-retained.json").read_bytes())
    rollback = SpeechMemory(a.run/"rollback.sqlite", evidence, create=False, retained=retained)
    actual = [readout(rollback.consume(m["id"], **request(m))) for m in report["moments"]]
    need(actual == report["rollback_readouts"], "COLD_CONSUMER")
    controls = []

    def deny(name, operation, code):
        try:
            operation()
        except ValueError as exc:
            need(str(exc) == code, "COLD_CONTROL_REASON:"+name+":"+str(exc))
            controls.append(dict(name=name, error=str(exc)))
        else:
            raise ValueError("MNLM_TIMED_SPEECH_COLD_CONTROL_ACCEPTED:"+name)

    first = report["moments"][0]
    deny("withdrawn", lambda: current.consume(first["id"], **request(first)),
         "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    deny("prospective-annotation", lambda: rollback.consume(first["id"],
        **(request(first) | dict(through=first["content"]["at"]))),
        "MNLM_TIMED_SPEECH_FUTURE_ANNOTATION")
    deny("meaning-admission", lambda: rollback.consume(first["id"],
        **(request(first) | dict(purpose="training"))), "MNLM_TIMED_SPEECH_ADMISSION_REQUIRED")
    deny("invented-why", lambda: evidence.assess(evidence.unresolved(first)),
         "MNLM_TIMED_SPEECH_WHY_UNRESOLVED")
    need(not current.snapshot()["memory"]["grounded_world_claims"], "COLD_WORLD_PROMOTION")
    current.close()
    rollback.close()
    bad = a.output/"tampered.sqlite"
    shutil.copy2(a.run/"memory.sqlite", bad)
    with sqlite3.connect(bad) as db:
        db.execute("UPDATE events SET hash=? WHERE seq=0", ("f"*64,))
    deny("journal-byte-tamper", lambda: SpeechMemory(bad, evidence, create=False,
        retained=report["final"]["retained"]), "MNLM_CODATA_JOURNAL_HASH")
    need(sha(Path("/zfs/archive/mnlm/raw/identity-v1/profile.md")) == report["identity_pin"],
         "COLD_IDENTITY")
    phases["journal_consumer_controls"] = time.perf_counter()-tick
    result = dict(report=a.report_pin, source=report["source_pin"], source_bytes=source_bytes,
        readbacks=rows, controls=controls, exact_consumer_replay=True, identity_pin=report["identity_pin"],
        phases=phases, seconds=time.perf_counter()-STARTED, flags=FLAGS,
        code=sha(Path(__file__)), acceptance="author-checked only")
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "COLD_TIMING")
    save(a.output/"report.json", result)
    print(json.dumps(dict(report=sha(a.output/"report.json"), controls=len(controls),
                          seconds=result["seconds"], exact_consumer_replay=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--report-pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())

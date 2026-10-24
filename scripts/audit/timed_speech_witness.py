"""Real AMI speech -> native views -> episodic evidence -> correction and rollback."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.timed_speech_memory import SpeechEvidence, SpeechMemory
from senses.timed_speech import AmiSource, FLAGS, NITE, SpeechViews, decode, need, pair, rational, save
from senses.perception_corpus import sha


def readout(result):
    return dict(head=result["head"], moment=result["moment"], provenance=result["provenance"],
        image=dict(shape=list(result["image"].shape),
                   sha256=hashlib.sha256(result["image"].tobytes()).hexdigest()),
        audio=dict(shape=list(result["audio"].shape),
                   sha256=hashlib.sha256(result["audio"].tobytes()).hexdigest()),
        text=result["text"], flags=result["flags"])


def request(moment):
    return dict(clock_id=moment["source"]["clock_id"], speaker=moment["content"]["who"],
                through=moment["event_time"]["available_after"])


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    identity = Path("/zfs/archive/mnlm/raw/identity-v1/profile.md")
    identity_before = sha(identity)
    source = AmiSource(a.source, a.source_pin)
    phases = dict(import_and_source=time.perf_counter()-STARTED)
    tick = time.perf_counter()
    records, views = [], []
    for event in source.events():
        key = event["segment"][NITE+"id"]
        receipt = decode(source, event, a.output/key)
        records.append(receipt)
        views.append(SpeechViews(source, a.output/key, receipt["pin"]))
    phases["decode_and_reopen_views"] = time.perf_counter()-tick
    evidence = SpeechEvidence(source, views)
    path = a.output/"memory.sqlite"
    session = SpeechMemory(path, evidence, create=True)
    controls, boundaries, readouts, moments = [], [], [], []
    tick = time.perf_counter()

    def act(op, ep, **kw):
        return session.apply(dict(op=op, episode=ep, **kw),
                             expected_head=session.retained()["head"])

    def promote(ep):
        act("check_source", ep["id"])
        act("promote", ep["id"], receipt=session.memory.state[ep["id"]]["receipt"]["id"])

    def deny(name, operation, code):
        before = (sha(path), session.snapshot())
        try:
            operation()
        except (ValueError, TypeError) as exc:
            need(str(exc) == code, "CONTROL_REASON:"+name+":"+str(exc))
            controls.append(dict(name=name, error=str(exc)))
        else:
            raise ValueError("MNLM_TIMED_SPEECH_CONTROL_ACCEPTED:"+name)
        need(before == (sha(path), session.snapshot()), "CONTROL_MUTATED:"+name)

    for event in source.events():
        key = event["segment"][NITE+"id"]
        before = session.snapshot()
        parents = evidence.parents(key)
        word = next(w for w in event["text"]["words"] if
                    rational(w["span"][0]) < rational(w["span"][1]))
        at = pair((rational(word["span"][0])+rational(word["span"][1]))/2)
        moment = evidence.moment(key, at)
        why = evidence.unresolved(moment)
        for ep in (*parents, moment, why):
            act("intake", ep)
        deny(key+"-unchecked-language", lambda: promote(parents[2]),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        for ep in (*parents, moment):
            promote(ep)
        moments.append(moment)
        deny(key+"-why", lambda: act("check_source", why["id"]),
             "MNLM_TIMED_SPEECH_WHY_UNRESOLVED")
        req = request(moment)
        result = session.consume(moment["id"], **req)
        need(not result["image"].flags.writeable and not result["audio"].flags.writeable,
             "MUTABLE_PAYLOAD")
        readouts.append(readout(result))
        boundaries.append(dict(segment=key, before=before, after=session.snapshot(),
            source_event=event["id"], view=evidence.views[key].pin))
        for name, changes, code in (
            ("clock", dict(clock_id="foreign"), "CLOCK"),
            ("speaker", dict(speaker={"nxt_agent": "foreign"}), "SPEAKER"),
            ("future", dict(through=moment["content"]["at"]), "FUTURE_ANNOTATION"),
            ("training", dict(purpose="training"), "ADMISSION_REQUIRED"),
            ("bad-time", dict(through=[True, 1]), "TIME")):
            deny(key+"-"+name, lambda changes=changes: session.consume(
                moment["id"], **(req | changes)), "MNLM_TIMED_SPEECH_"+code)
        forged = copy.deepcopy(parents[2])
        forged["content"]["native"]["surface"] += " invented"
        deny(key+"-invented-text", lambda: act("intake", forged),
             "MNLM_TIMED_SPEECH_EPISODE_BINDING")
        tail = pair((rational(event["text"]["words"][-1]["span"][1])+
                     rational(event["support"][1]))/2)
        deny(key+"-silence-is-not-text", lambda: evidence.moment(key, tail),
             "MNLM_TIMED_SPEECH_MISSING_ACTIVE_FRAME_OR_LANGUAGE")
    phases["intake_check_promote_consume_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    prior = session.snapshot()
    prior_readouts = [readout(session.consume(m["id"], **request(m))) for m in moments]
    stale = SpeechMemory(path, evidence, create=False, retained=prior["retained"])
    with sqlite3.connect(a.output/"rollback.sqlite") as db:
        session.journal.db.backup(db)
    save(a.output/"rollback-retained.json", prior["retained"])
    # First speaker's audio withdrawal must remove the dependent transcript and event.
    first = moments[0]
    audio = first["evidence"][1]
    act("invalidate", audio, reason="FSV source support withdrawal; original bytes retained")
    deny("stale-reader", lambda: stale.consume(first["id"], **request(first)),
         "MNLM_CODATA_JOURNAL_STALE")
    stale.close()
    for key in (audio, first["evidence"][2], first["id"]):
        need(session.memory.state[key]["grounded_scope"] is None, "DEPENDENT_INVALIDATION")
    deny("withdrawn-event", lambda: session.consume(first["id"], **request(first)),
         "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    unaffected = readout(session.consume(moments[-1]["id"], **request(moments[-1])))
    need(unaffected["image"] == prior_readouts[-1]["image"]
         and unaffected["text"] == prior_readouts[-1]["text"], "UNRELATED_EVENT_CHANGED")
    act("retire", audio)
    invalidated = session.snapshot()
    retained = session.retained()
    session.close()
    session = SpeechMemory(path, evidence, create=False, retained=retained)
    need(session.snapshot() == invalidated, "CORRECTION_RESTART")
    deny("withdrawal-survives-restart", lambda: session.consume(first["id"], **request(first)),
         "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    rollback = SpeechMemory(a.output/"rollback.sqlite", evidence, create=False,
                            retained=prior["retained"])
    need(rollback.snapshot() == prior, "ROLLBACK_STATE")
    need([readout(rollback.consume(m["id"], **request(m))) for m in moments] == prior_readouts,
         "ROLLBACK_NATIVE_READBACK")
    rollback.close()
    boundaries.append(dict(action="withdraw-and-retire-audio", before=prior, after=invalidated,
                           rollback_exact=True, unaffected=unaffected))
    # Same-size transcript corruption, without modifying any retained view.
    bad = a.output/"bad-view"
    bad.mkdir()
    chosen = views[0]
    changed_manifest = chosen.manifest
    changed_manifest["event"]["text"]["surface"] = "invented"
    need(chosen.manifest["event"]["text"]["surface"] != "invented", "VIEW_ALIAS")
    for name in (*chosen.manifest["files"], "manifest.json"):
        if name == "text.json":
            raw = bytearray((chosen.root/name).read_bytes())
            pos = raw.index(b"surface")
            raw[pos] = ord("S")
            (bad/name).write_bytes(raw)
        else:
            (bad/name).symlink_to(chosen.root/name)
    deny("corrupt-derived-text", lambda: SpeechViews(source, bad, chosen.pin),
         "MNLM_TIMED_SPEECH_VIEW_BYTES")
    deny("missing-derived-view", lambda: SpeechViews(source, a.output/"absent", chosen.pin),
         "MNLM_TIMED_SPEECH_VIEW_MANIFEST")
    deny("unknown-segment", lambda: evidence.moment("missing", [0, 1]),
         "MNLM_TIMED_SPEECH_UNKNOWN_SEGMENT")
    phases["correction_recovery_negative_control"] = time.perf_counter()-tick
    need(sha(identity) == identity_before, "IDENTITY_CHANGED")
    tick = time.perf_counter()
    final = session.snapshot()
    session.close()
    report = dict(source=str(a.source), source_pin=a.source_pin, views=records,
        counts=source.counts, moments=moments, readouts=readouts,
        rollback_readouts=prior_readouts, final=final, boundaries=boundaries, controls=controls,
        identity_pin=identity_before, source_quality=source.quality, flags=FLAGS,
        code={name: sha(Path(__file__).parents[1]/name) for name in (
            "senses/timed_speech.py", "claims/timed_speech_memory.py",
            "audit/timed_speech_witness.py")},
        phases=phases, acceptance="author-checked only")
    phases["report_setup"] = time.perf_counter()-tick
    report["seconds"] = time.perf_counter()-STARTED
    report["phase_coverage"] = sum(phases.values())/report["seconds"]
    need(report["phase_coverage"] >= .9, "TIMING_COVERAGE")
    save(a.output/"report.json", report)
    print(json.dumps(dict(report=sha(a.output/"report.json"), views=len(views),
        controls=len(controls), seconds=report["seconds"], flags=FLAGS)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())

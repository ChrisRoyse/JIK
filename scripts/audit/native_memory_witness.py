"""Real native-bank -> shared temporal memory -> correction/restart witness."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
from claims.native_evidence import NativeEvidence, need, pair, rational
from claims.native_memory import NativeMemory
from senses.perception_corpus import save, sha


def main(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checker = NativeEvidence(a.bank, a.report_pin, a.inputs, a.corpus)
    phases = dict(open_sources=time.perf_counter()-started)
    path = a.output/"memory.sqlite"
    session = NativeMemory(path, checker, create=True)
    controls, boundaries, readouts, episodes = [], [], [], {}

    def act(op, ep, **kw):
        return session.apply(dict(op=op, episode=ep, **kw),
                             expected_head=session.retained()["head"])

    def promote(ep):
        act("check_source", ep["id"])
        act("promote", ep["id"], receipt=session.memory.state[ep["id"]]["receipt"]["id"])

    def deny(name, operation, code, unchanged=True):
        before = (sha(path), session.snapshot()) if unchanged else None
        try:
            operation()
        except (ValueError, TypeError, sqlite3.OperationalError) as exc:
            need(str(exc) == code, "CONTROL_REASON:"+name+":"+str(exc))
            controls.append(dict(control=name, error=str(exc)))
        else:
            raise ValueError("MNLM_NATIVE_MEMORY_CONTROL_ACCEPTED:"+name)
        if unchanged:
            need(before == (sha(path), session.snapshot()), "CONTROL_MUTATED:"+name)

    def request(moment):
        s = moment["source"]
        return dict(clock_id=s["clock_id"], split_group=s["split_group"],
                    source_pin=s["source"]["media"], through=[2, 1])

    for clip in a.clips:
        tick = time.perf_counter()
        before = session.snapshot()
        image, audio = (checker.measurement(clip, m) for m in ("image", "audio"))
        span = image["event_time"]["support"]
        at = pair((rational(span[0])+rational(span[1]))/2)
        moment = checker.moment(clip, at)
        why = checker.unresolved(moment)
        episodes[clip] = dict(image=image, audio=audio, moment=moment, why=why)
        for ep in (image, audio, moment, why):
            act("intake", ep)
        intake = session.snapshot()
        deny(clip+"-ungrounded-parent", lambda: act("check_source", moment["id"]),
             "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        promote(image)
        promote(audio)
        promote(moment)
        deny(clip+"-unresolved-why", lambda: act("check_source", why["id"]),
             "MNLM_NATIVE_MEMORY_WHY_UNRESOLVED")
        deny(clip+"-borrowed-receipt", lambda: act("promote", why["id"],
             receipt=session.memory.state[image["id"]]["receipt"]["id"]),
             "MNLM_MEMORY_NO_MATCHING_CHECKED_RECEIPT")
        req = request(moment)
        result = session.consume(moment["id"], **req)
        rows = []
        for reading in result["readings"]:
            raw = reading["values"].tobytes()
            need(not reading["values"].flags.writeable and
                 hashlib.sha256(raw).hexdigest() == reading["entry"]["sha256"], "CONSUMER_BYTES")
            rows.append(dict(reference=reading["reference"], medium=reading["medium"],
                instrument=reading["name"], shape=list(reading["values"].shape),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        readouts.append(dict(clip=clip, head=result["head"], moment=moment, rows=rows))
        for label, kwargs, code in (
            ("clock", dict(clock_id="foreign"), "CLOCK"),
            ("split", dict(split_group="foreign"), "SPLIT"),
            ("source", dict(source_pin="0"*64), "SOURCE"),
            ("future-context", dict(through=at), "FUTURE_CONTEXT"),
            ("timed-text-absent", dict(required_media=("image", "audio", "text")),
             "MISSING_TIMED_TEXT_OR_MEDIUM"),
            ("training", dict(purpose="training"), "NATIVE_ADMISSION_REQUIRED"),
            ("meaning", dict(purpose="assert_meaning"), "NATIVE_ADMISSION_REQUIRED"),
        ):
            deny(clip+"-"+label, lambda kw=kwargs: session.consume(
                moment["id"], **{**req, **kw}), "MNLM_NATIVE_MEMORY_"+code)
        tampered = copy.deepcopy(moment)
        tampered["content"]["physical_sync_calibrated"] = True
        deny(clip+"-invented-accuracy", lambda: checker.validate_episode(tampered),
             "MNLM_NATIVE_MEMORY_EPISODE_BINDING")
        claimed = copy.deepcopy(image)
        claimed["claim_kind"] = "event_occurrence"
        deny(clip+"-world-claim", lambda: checker.validate_episode(claimed),
             "MNLM_NATIVE_MEMORY_UNSUPPORTED_CLAIM")
        admitted = copy.deepcopy(image)
        admitted["content"]["scientific_admission"] = True
        deny(clip+"-false-admission", lambda: checker.validate_episode(admitted),
             "MNLM_NATIVE_MEMORY_EPISODE_BINDING")
        deny(clip+"-immutable-values", lambda: result["readings"][0]["values"].__setitem__(
             (0, 0), 0), "assignment destination is read-only")
        boundaries.append(dict(name="intake-"+clip, before=before, after_intake=intake,
                               after_checked=session.snapshot()))
        save(a.output/("boundary-"+clip+".json"), boundaries[-1])
        phases["intake-"+clip] = time.perf_counter()-tick

    tick = time.perf_counter()
    first, second = [episodes[c] for c in a.clips]
    crossed = copy.deepcopy(first["moment"])
    crossed["content"]["members"][1] = second["moment"]["content"]["members"][1]
    deny("cross-clip-member", lambda: checker.validate_episode(crossed),
         "MNLM_NATIVE_MEMORY_EPISODE_BINDING")
    before = session.snapshot()
    before_head = session.retained()
    rollback_path = a.output/"before-withdrawal.sqlite"
    shutil.copy2(path, rollback_path)
    stale = NativeMemory(path, checker, create=False, retained=before_head)
    old_receipt = session.memory.state[first["moment"]["id"]]["receipt"]["id"]
    act("invalidate", first["audio"]["id"], reason="operator withdraws this source measurement")
    withdrawn = session.snapshot()
    for key in ("audio", "moment", "why"):
        need(withdrawn["states"][first[key]["id"]]["review"] == "needs_review",
             "DEPENDENT_INVALIDATION")
    need(all(before["states"][e["id"]] == withdrawn["states"][e["id"]]
             for e in second.values()), "UNRELATED_CLIP_CHANGED")
    deny("withdrawn-consumer", lambda: session.consume(first["moment"]["id"],
         **request(first["moment"])), "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    deny("old-receipt", lambda: act("promote", first["moment"]["id"], receipt=old_receipt),
         "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    deny("stale-writer", lambda: stale.apply(dict(op="check_source", episode=second["image"]["id"]),
         expected_head=before_head["head"]), "MNLM_CODATA_JOURNAL_STALE")
    stale.close()
    deny("optimistic-conflict", lambda: session.apply(
         dict(op="check_source", episode=second["image"]["id"]),
         expected_head=before_head["head"]), "MNLM_NATIVE_MEMORY_WRITER_CONFLICT")
    act("retire", first["audio"]["id"])
    deny("retired-parent", lambda: act("check_source", first["moment"]["id"]),
         "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    act("reactivate", first["audio"]["id"])
    deny("reactivation-not-recheck", lambda: act("check_source", first["moment"]["id"]),
         "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    promote(first["audio"])
    deny("parent-recheck-not-child", lambda: session.consume(first["moment"]["id"],
         **request(first["moment"])), "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    # Leave the withdrawn moment invalid in the authoritative final journal.
    after = session.snapshot()
    boundary = dict(name="withdrawal", before=before, after_withdrawal=withdrawn, after=after)
    boundaries.append(boundary)
    save(a.output/"boundary-withdrawal.json", boundary)
    retained = session.retained()
    session.close()
    session = NativeMemory(path, checker, create=False, retained=retained)
    need(session.snapshot() == after, "RESTART_STATE")
    deny("missing-retained-head", lambda: NativeMemory(path, checker, create=False),
         "MNLM_NATIVE_MEMORY_RETAINED_HEAD_REQUIRED")
    deny("wrong-retained-contract", lambda: NativeMemory(path, checker, create=False,
         retained={**retained, "contract": "0"*64}), "MNLM_CODATA_JOURNAL_RETAINED_CONTRACT")
    deny("wrong-memory-head", lambda: NativeMemory(path, checker, create=False,
         retained={**retained, "memory_head": "0"*64}), "MNLM_NATIVE_MEMORY_MEMORY_HEAD")
    deny("restart-withdrawal", lambda: session.consume(first["moment"]["id"],
         **request(first["moment"])), "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    need(len(session.consume(second["moment"]["id"], **request(second["moment"]))["readings"])
         == len(readouts[1]["rows"]), "RESTART_RETAINED_READINGS")
    rollback = NativeMemory(rollback_path, checker, create=False, retained=before_head)
    need(rollback.snapshot() == before and
         len(rollback.consume(first["moment"]["id"], **request(first["moment"]))["readings"])
         == len(readouts[0]["rows"]), "ROLLBACK")
    rollback.close()
    deny("rollback-under-new-head", lambda: NativeMemory(rollback_path, checker,
         create=False, retained=retained), "MNLM_CODATA_JOURNAL_HEAD")
    for label in ("truncated", "corrupt"):
        target = a.output/(label+".sqlite")
        shutil.copy2(path, target)
        with sqlite3.connect(target) as db:
            if label == "truncated":
                db.execute("DELETE FROM events WHERE seq=(SELECT max(seq) FROM events)")
            else:
                db.execute("UPDATE events SET body=? WHERE seq=0", (b"{}",))
        deny(label, lambda p=target: NativeMemory(p, checker, create=False, retained=retained),
             "MNLM_CODATA_JOURNAL_"+("HEAD" if label == "truncated" else "HASH"))
    failed_path = a.output/"failed-publication.sqlite"
    shutil.copy2(path, failed_path)
    failed = NativeMemory(failed_path, checker, create=False, retained=retained)
    failed.journal.db.execute("PRAGMA query_only=ON")
    before_failed = sha(failed_path)
    deny("publication-readonly", lambda: failed.apply(
         dict(op="check_source", episode=second["image"]["id"]), expected_head=retained["head"]),
         "attempt to write a readonly database")
    deny("failed-handle-poisoned", lambda: failed.consume(second["moment"]["id"],
         **request(second["moment"])), "MNLM_NATIVE_MEMORY_RELOAD_REQUIRED")
    failed.close()
    need(sha(failed_path) == before_failed, "FAILED_PUBLICATION_CHANGED_DB")
    recovered = NativeMemory(failed_path, checker, create=False, retained=retained)
    need(recovered.snapshot() == after, "FAILED_PUBLICATION_RELOAD")
    recovered.close()
    phases["correction_restart"] = time.perf_counter()-tick

    tick = time.perf_counter()
    # Corruption probes use symlinked scratch trees and replace only one leaf.
    # Original source and bank artifacts are never written.
    bank_copy = a.output/"corrupt-bank"
    bank_copy.mkdir()
    for p in a.bank.iterdir():
        if p.is_dir():
            (bank_copy/p.name).mkdir()
            for leaf in p.iterdir():
                (bank_copy/p.name/leaf.name).symlink_to(leaf)
        else:
            (bank_copy/p.name).symlink_to(p)
    leaf = next((bank_copy/"image").glob("*.bin"))
    original = sha(leaf)
    leaf.unlink()
    leaf.write_bytes(b"corrupt")
    deny("corrupt-reading", lambda: NativeEvidence(bank_copy, a.report_pin, a.inputs, a.corpus),
         "MNLM_VISUAL_BANK_READING_BYTES")
    need(sha(a.bank/"image"/leaf.name) == original, "ORIGINAL_BANK_CHANGED")
    source_copy = a.output/"corrupt-corpus"
    (source_copy/"clips").mkdir(parents=True)
    for clip in a.clips:
        target = source_copy/"clips"/clip
        target.mkdir()
        original_root = a.corpus/"clips"/clip
        for p in original_root.iterdir():
            if p.name == "media":
                (target/"media").mkdir()
                for leaf in p.iterdir():
                    (target/"media"/leaf.name).symlink_to(leaf)
            else:
                (target/p.name).symlink_to(p, target_is_directory=p.is_dir())
    leaf = source_copy/"clips"/a.clips[0]/"media"/(a.clips[0]+".mp4")
    original = sha(leaf)
    leaf.unlink()
    leaf.write_bytes(b"corrupt")
    deny("corrupt-source", lambda: NativeEvidence(a.bank, a.report_pin, a.inputs, source_copy),
         "MNLM_NATIVE_PREFIX_SOURCE_BYTES")
    need(sha(a.corpus/"clips"/a.clips[0]/"media"/leaf.name) == original,
         "ORIGINAL_SOURCE_CHANGED")
    phases["source_corruption_controls"] = time.perf_counter()-tick
    session.close()
    result = dict(schema="native-temporal-memory-witness-v1", author_checked=True,
        bank=str(a.bank), report_pin=a.report_pin, inputs=str(a.inputs), corpus=str(a.corpus),
        clips=a.clips, evidence_pin=checker.pin, retained=retained, rollback_retained=before_head,
        journal_sha256=sha(path), rollback_sha256=sha(rollback_path),
        snapshot=after, readouts=readouts, boundaries=[b["name"] for b in boundaries],
        controls=controls, phases=phases, scientific_admission=False, training_ready=False,
        physical_sync_calibrated=False, timed_text_present=False, useful_learning_demonstrated=False,
        code_pin=sha(Path(__file__)), seconds=time.perf_counter()-started)
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    pin = save(a.output/"report.json", result)
    print(json.dumps(dict(report_pin=pin, readings=sum(len(r["rows"]) for r in readouts),
        bytes=sum(x["bytes"] for r in readouts for x in r["rows"]),
        controls=len(controls), seconds=result["seconds"], phase_coverage=result["phase_coverage"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "inputs", "corpus", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--report-pin", required=True)
    p.add_argument("--clips", nargs=2, required=True)
    main(p.parse_args())

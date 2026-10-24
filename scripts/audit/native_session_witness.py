"""Real complete episodes through durable native model operations; no admission."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

START = time.perf_counter()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import torch
from audit.timed_world_witness import bundle, fingerprint
from claims.timed_speech_memory import SpeechMemory
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.episode_input import EpisodeWorldInputs
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_episode_session import NativeEpisodeSession, FLAGS
from model.native_identity_world import NativeIdentityWorld, SensoryIdentityCore
from model.timed_world_input import TimedWorldInputs
from senses.contract import digest
from senses.perception_corpus import pinned, sha
from senses.timed_readings import Bank, Inputs, need
from senses.timed_speech import save


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    source = Inputs(a.inputs, a.input_pin, a.source)
    manifest = pinned(a.bank/"bank.json", a.bank_pin)
    bank = Bank(a.bank, a.bank_pin, source,
                {n: digest(v) for n, v in manifest["instruments"].items()})
    retained = pinned(a.inputs/"rollback-retained.json", a.retained_pin)
    need(sha(a.inputs/"rollback.sqlite") == a.memory_pin, "MEMORY_PIN")
    shutil.copy2(a.inputs/"rollback.sqlite", a.output/"memory.sqlite")
    memory = SpeechMemory(a.output/"memory.sqlite", source.evidence, create=False, retained=retained)
    cfg = ModelConfig()
    selected = TimedWorldInputs(bank, memory, cfg)
    inputs = EpisodeWorldInputs(selected)
    identity = LoopInputs(a.identity_population, a.identity_native, a.identity_source)
    previous = pinned(a.previous/"report.json", a.previous_pin)
    need(previous["binding"] == inputs.binding, "PRIOR_BINDING")
    phases = dict(source_seconds=time.perf_counter()-START)
    controls, calls = [], []

    def deny(name, fn, expected):
        try:
            fn()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            need(actual == expected, "CONTROL:"+name+":"+actual)
            controls.append(dict(name=name, refused=actual))
        else:
            raise ValueError("MNLM_NATIVE_SESSION_CONTROL_ACCEPTED:"+name)

    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                      "--format=csv,noheader"], text=True).strip(), "GPU_BUSY")
    take_lease("28", "native-episode-session")
    session = None
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261020)
        core = IdentityLoopPredictor(cfg, identity.specs,
            identity.batch(identity.standard_units, cfg, "cpu"), source_id=identity.source_id)
        model = NativeIdentityWorld(SensoryIdentityCore(core, inputs.specs, inputs.temporal_catalogue),
            selected.batch([0, 2], "cpu"), world_source_id=inputs.binding).cuda().eval()

        def open_session(path, *, create=False, retained=None):
            return NativeEpisodeSession(path, model, inputs, checkpoint=a.previous/"initial",
                checkpoint_pin=previous["initial"], create=create, retained=retained)

        if a.cold:
            observed = pinned(a.cold/"report.json", a.cold_pin)
            session = open_session(a.cold/"session.sqlite", retained=observed["after"]["retained"])
            need(session.snapshot() == observed["after"], "FRESH_PROCESS_EXACT")
            need(sha(a.cold/"session.sqlite") == observed["files"]["session.sqlite"], "READ_ONLY_REPLAY")
            phases["fresh_process_replay_seconds"] = time.perf_counter()-tick
            elapsed = time.perf_counter()-START
            save(a.output/"cold-readback.json", dict(acceptance="author-checked", report=a.cold_pin,
                snapshot=session.snapshot(), exact=True, flags=FLAGS, phases=phases, seconds=elapsed,
                phase_coverage=sum(phases.values())/elapsed))
            print(json.dumps(dict(cold_exact=True, seconds=elapsed)), flush=True)
            return

        path = a.output/"session.sqlite"
        session = open_session(path, create=True)
        before = session.snapshot()
        save(a.output/"before.json", before)
        phases["initialize_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        for phase in range(3):
            for context in range(2):
                name = "AMI:"+"AB"[context]
                index = 2*context + int(phase > 0)
                untouched = session.snapshot()["contexts"]["AMI:"+"BA"[context]]
                old = session.retained()
                output = session.advance(name, index, expected_head=old["head"])
                snap = session.snapshot()
                need(snap["contexts"]["AMI:"+"BA"[context]] == untouched, "CONTEXT_ISOLATION")
                need(not session.journal.cache, "UNBOUNDED_CACHE")
                calls.append(dict(phase=phase, context=name, index=index,
                    fingerprint=fingerprint(bundle(output)), after=snap))
                deny("stale_request_"+str(phase)+str(context),
                     lambda: session.advance(name, index, expected_head=old["head"]),
                     "MNLM_NATIVE_SESSION_WRITER_CONFLICT")
                print(json.dumps(dict(phase=phase, context=name, head=snap["retained"]["head"])), flush=True)
            save(a.output/("phase-"+str(phase)+".json"), session.snapshot())
            if phase == 0:
                session.close()
                shutil.copy2(path, a.output/"rollback.sqlite")
                session = open_session(path, retained=calls[-1]["after"]["retained"])
                need(session.snapshot() == calls[-1]["after"], "RESTART_EXACT")
        complete = session.snapshot()
        save(a.output/"calls.json", calls)
        save(a.output/"complete.json", complete)
        phases["operations_restart_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        deny("unadmitted_training", lambda: session.advance("AMI:A", 1,
             expected_head=session.retained()["head"], purpose="training"),
             "MNLM_NATIVE_SESSION_ADMISSION_REQUIRED")
        deny("unadmitted_serving", lambda: session.advance("AMI:A", 1,
             expected_head=session.retained()["head"], purpose="serving"),
             "MNLM_NATIVE_SESSION_ADMISSION_REQUIRED")
        deny("unattested_action_echo", lambda: session.advance("AMI:A", 1,
             expected_head=session.retained()["head"]), "MNLM_NATIVE_SESSION_ECHO_UNGROUNDED")
        need(session.snapshot() == complete, "REFUSAL_CHANGED_STATE")
        session.close()
        session = open_session(path, retained=complete["retained"])
        need(session.snapshot() == complete, "COLD_RESTART_EXACT")
        session.close()

        # Same committed predecessor, stale writer rejected before publishing any context.
        rollback = open_session(a.output/"rollback.sqlite", retained=calls[1]["after"]["retained"])
        peer = open_session(a.output/"rollback.sqlite", retained=calls[1]["after"]["retained"])
        prior_peer = {k: (s.sequence, s.last_event) for k, s in peer.states.items()}
        restored = rollback.advance("AMI:A", 1, expected_head=rollback.retained()["head"])
        need(fingerprint(bundle(restored)) == calls[2]["fingerprint"], "ROLLBACK_OUTPUT_EXACT")
        deny("stale_writer", lambda: peer.advance("AMI:B", 3,
             expected_head=calls[1]["after"]["retained"]["head"]), "MNLM_CODATA_JOURNAL_STALE")
        need({k: (s.sequence, s.last_event) for k, s in peer.states.items()} == prior_peer,
             "STALE_WRITER_CHANGED_CONTEXT")
        rollback_retained = rollback.retained()
        rollback.close(); peer.close()

        fault_path = a.output/"append-fault.sqlite"
        shutil.copy2(a.output/"rollback.sqlite", fault_path)
        fault = open_session(fault_path, retained=rollback_retained)
        prior_states = dict(fault.states)
        before_counts = tuple(fault.journal.db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0]
                              for table in ("events", "payloads"))
        fault.journal.db.execute("CREATE TRIGGER injected BEFORE INSERT ON events BEGIN "
                                "SELECT RAISE(ABORT, 'injected append failure'); END")
        deny("append_failure", lambda: fault.advance("AMI:B", 3, expected_head=rollback_retained["head"]),
             "injected append failure")
        need(all(fault.states[k] is s for k, s in prior_states.items()), "FAILED_WRITE_PUBLISHED_RAM")
        need(before_counts == tuple(fault.journal.db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0]
                                     for table in ("events", "payloads")), "FAILED_WRITE_PUBLISHED_BYTES")
        deny("poisoned_handle", fault.retained, "MNLM_NATIVE_SESSION_RELOAD")
        fault.close()
        deny("closed_poisoned_connection", lambda: fault.journal.db.execute("SELECT 1"),
             "Cannot operate on a closed database.")

        for kind in ("truncated", "corrupt"):
            target = a.output/(kind+".sqlite")
            shutil.copy2(path, target)
            with sqlite3.connect(target) as db:
                if kind == "truncated":
                    db.execute("DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)")
                else:
                    db.execute("UPDATE payloads SET data=? WHERE hash=(SELECT hash FROM payloads LIMIT 1)",
                               (b"{}",))
            deny(kind, lambda: open_session(target, retained=complete["retained"]),
                 "MNLM_CODATA_JOURNAL_HEAD" if kind == "truncated" else "MNLM_CODATA_JOURNAL_PAYLOAD")

        session = open_session(path, retained=complete["retained"])
        moment = inputs.moments[0]
        parent = memory.memory.episodes[moment["id"]]["evidence"][1]
        memory.apply(dict(op="invalidate", episode=parent, reason="FSV source withdrawal"),
                     expected_head=memory.retained()["head"])
        deny("withdrawn_source", session.retained, "MNLM_TIMED_MODEL_STALE")
        session.close()
        save(a.output/"withdrawn-memory.json", memory.snapshot())
        phases["controls_seconds"] = time.perf_counter()-tick
        seconds = time.perf_counter()-START
        report = dict(acceptance="author-checked", input_pin=a.input_pin, bank_pin=a.bank_pin,
            previous_pin=a.previous_pin, checkpoint=previous["initial"], binding=inputs.binding,
            standard=model.standard_id, world=model.world_id, temporal_id=model.core.temporal_id,
            memory_pin=a.memory_pin, retained_pin=a.retained_pin,
            before=before, after=complete, flags=FLAGS, operations=len(calls), controls=controls,
            phases=phases, seconds=seconds, phase_coverage=sum(phases.values())/seconds,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            files={p.name: sha(p) for p in [path, a.output/"calls.json", a.output/"complete.json"]},
            code={n: sha(Path(__file__).resolve().parents[1]/n) for n in
                  ("model/native_episode_session.py", "audit/native_session_witness.py")})
        need(report["phase_coverage"] >= .9, "PHASE_COVERAGE")
        save(a.output/"report.json", report)
        print(json.dumps(dict(report=sha(a.output/"report.json"), controls=len(controls), seconds=seconds)), flush=True)
    finally:
        if session is not None:
            session.close()
        memory.close()
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("inputs", "source", "bank", "output", "identity-population", "identity-native",
              "identity-source", "previous"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("input-pin", "bank-pin", "memory-pin", "retained-pin", "previous-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    main(p.parse_args())

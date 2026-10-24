"""Physical source/native/world-loop proof; no learning or sense-admission claim."""
import argparse
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
import torch
from audit.fsv_identity_loop import configure
from claims.codata_source import CodataSource, check, sha
from claims.episodic_memory import canonical, digest, ZERO
from claims.scientific_memory import ScientificMemory
from lens.lenscommon import run_cli, take_lease
from model import loop_checkpoint, world_checkpoint
from model.scientific_input import ScientificInputs
from model.scientific_loop import ScientificLoop
from model.world_loop import WorldLoop
from senses.contract import SenseError

CONTEXTS = ("science:electron", "science:planck")
QUANTITIES = ("electron mass", "Planck constant")


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def independent_contexts(path, source, snapshot):
    """Raw SQL projection, independent of ScientificLoop's state/replay methods."""
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    head, count, edition, initialized = ZERO, 0, None, False
    states, contexts, dirty, advances = {}, {}, set(), 0
    try:
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            assert raw == canonical(body) and seq == count and previous == head
            assert pin == digest(dict(seq=seq, previous=previous, body=body))
            kind = body["kind"]
            if kind == "memory":
                event = body["record"]["event"]
                op = event["op"]
                if op == "intake":
                    states[event["episode"]["id"]] = None
                elif op == "promote":
                    states[event["episode"]] = event["receipt"]
                elif op == "invalidate":
                    for key in event["affected"]:
                        states[key] = None
                    dirty.update(c for c, v in contexts.items() if set(v["support"]) & set(event["affected"]))
                else:
                    assert op == "check_source"
            elif kind == "adopt":
                assert body["prior"] == edition
                edition = body["edition"]
                dirty.update(c for c, v in contexts.items() if v["edition"] not in (None, edition))
            elif kind == "world_init":
                assert not initialized and digest(body["contract"]) == snapshot["loop_contract"]
                initialized = True
                contexts = {c: dict(sequence=0, last_event=None, edition=None, support={})
                            for c in body["contract"]["contexts"]}
            elif kind == "world_reset":
                c = body["context"]
                assert c in dirty and body["prior_sequence"] == contexts[c]["sequence"]
                contexts[c] = dict(sequence=0, last_event=None, edition=None, support={})
                dirty.remove(c)
            else:
                assert kind == "world_advance"
                request = body["request"]
                c, quantity = request["context"], request["quantity"]
                assert c not in dirty and request["prior_head"] == head
                assert request["edition"] == edition and request["sequence"] == contexts[c]["sequence"]
                assert request["phase"] == contexts[c]["sequence"] % 5
                ep = source.episode(edition, quantity)
                assert request["episode"] == ep["id"] and request["checked_receipt"] == source.assess(ep)
                assert states[ep["id"]] == request["checked_receipt"]["id"]
                assert request["origin"] == ("execution_receipt" if request["phase"] == 3 else "source_observation")
                hidden = base64.b64decode(body["hidden"], validate=True)
                assert len(hidden) == 6*32*4 and sha(hidden) == body["hidden_sha256"]
                contexts[c].update(sequence=request["sequence"]+1, last_event=digest(request),
                                   edition=edition, hidden_sha256=sha(hidden))
                contexts[c]["support"][ep["id"]] = request["checked_receipt"]
                advances += 1
            head, count = pin, count+1
        assert head == snapshot["head"] and count == snapshot["events"]
        for c, v in contexts.items():
            expected = dict(**v, needs_review=c in dirty, world=snapshot["world"])
            assert expected == snapshot["contexts"][c]
        return dict(events=count, advances=advances, contexts=len(contexts), head=head, exact=True)
    finally:
        db.close()


def prefix_copy(source_path, destination, source, retained):
    """Publish a separate historical development view; leave the retained ledger intact."""
    original = ScientificMemory(source_path, source, expected_head=retained["head"],
                                expected_contract=retained["contract"])
    before = sha(source_path.read_bytes())
    with sqlite3.connect(destination) as target:
        original.db.backup(target)
        rows = target.execute("SELECT seq,previous,body FROM events ORDER BY seq").fetchall()
        cut, prefix = next((seq, previous) for seq, previous, raw in rows
                           if json.loads(raw).get("kind") == "adopt" and json.loads(raw)["edition"] == "2022")
        target.execute("DELETE FROM events WHERE seq>=?", (cut,))
    original.close()
    check(sha(source_path.read_bytes()) == before, "RETAINED_SOURCE_CHANGED")
    return dict(parent_head=retained["head"], parent_bytes=before, prefix_head=prefix, prefix_events=cut)


def setup(args):
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv"], text=True)
    take_lease("28", "scientific-intrinsic-loop")
    configure()
    source = CodataSource(args.source, args.source_pin)
    inputs = ScientificInputs(source, args.population, args.native,
                              population_pin=args.population_pin, native_pin=args.native_pin)
    return source, inputs, resident


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "population", "native", "base", "science", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("source-pin", "population-pin", "native-pin", "base-pin", "science-run-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--run-pin")
    args = p.parse_args()
    started = time.perf_counter()
    source, inputs, resident = setup(args)
    phases = dict(source_native_readback=time.perf_counter()-started)
    if args.cold:
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["source"] == source.pin and run["world_source"] == inputs.source_id, "RUN_SOURCE")
        session = ScientificLoop(args.output/"science.sqlite3", source, inputs, args.output/"model",
                                 **run["open"], device="cuda", neural_replay=True)
        snapshot = json.loads((args.output/"snapshot.json").read_bytes())
        check(snapshot == session.snapshot(), "COLD_SNAPSHOT")
        independent = independent_contexts(args.output/"science.sqlite3", source, snapshot)
        session.model.check_reference()
        replayed = session.replayed
        session.close()
        write(args.output/"cold-readback.json", dict(complete=True, acceptance="author-checked",
              independent=independent, neural_replayed=replayed, source_windows=inputs.window_readback,
              wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(complete=True, cold=True, neural_replayed=replayed)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"gpu-before.txt").write_text(resident, encoding="utf-8")
    tick = time.perf_counter()
    check(sha((args.base/"manifest.json").read_bytes()) == args.base_pin, "BASE_PIN")
    baseline = json.loads((args.base/"manifest.json").read_bytes())
    core, _, _ = loop_checkpoint.load(args.base/"model", expected_checkpoint=baseline["model"],
                                     expected_standard=baseline["standard"], device="cuda")
    check(core.specs == inputs.specs, "INSTRUMENT_JOIN")
    torch.manual_seed(20261020)
    model = WorldLoop(core, inputs.world_bank(core.config, "cuda"), world_source_id=inputs.source_id).cuda().eval()
    pin = world_checkpoint.save(args.output/"model", model, tuple(model.start(c) for c in CONTEXTS))
    phases["model_construction_publication"] = time.perf_counter()-tick
    tick = time.perf_counter()
    check(sha((args.science/"run.json").read_bytes()) == args.science_run_pin, "SCIENCE_RUN_PIN")
    retained = json.loads((args.science/"run.json").read_bytes())
    lineage = prefix_copy(args.science/"science.sqlite3", args.output/"science.sqlite3", source, retained)
    kw = dict(model_pin=pin, standard=model.standard_id, world=model.world_id, contexts=CONTEXTS,
              expected_head=lineage["prefix_head"], expected_contract=retained["contract"])
    session = ScientificLoop(args.output/"science.sqlite3", source, inputs, args.output/"model",
                             **kw, device="cuda", attach=True)
    controls, boundaries = [], []

    def refuse(label, code, fn, *, same=True):
        before = digest(session.snapshot()) if same else None
        try:
            fn()
        except (ValueError, SenseError) as exc:
            check(getattr(exc, "code", str(exc)) == code, "WRONG_REFUSAL_"+label)
        else:
            raise AssertionError(label+" did not refuse")
        if same:
            check(digest(session.snapshot()) == before, "REFUSAL_MUTATED")
        controls.append(dict(name=label, code=code, unchanged=same))

    def cycle():
        hashes = {c: [] for c in CONTEXTS}
        for phase in range(5):
            for c, q in zip(CONTEXTS, QUANTITIES):
                other = next(x for x in CONTEXTS if x != c)
                unchanged = session.states[other].hidden.detach().clone()
                receipt = session.resolve(q)["receipt"] if phase == 3 else None
                if phase == 3:
                    refuse("generated_echo_"+c+"_"+session.edition, "MNLM_CODATA_LOOP_ECHO_RECEIPT",
                           lambda: session.advance(c, q, receipt="model-self-confidence"))
                out = session.advance(c, q, receipt=receipt)
                check(torch.equal(unchanged, session.states[other].hidden), "CONTEXT_CROSSTALK")
                check(all(bool(session.allowed_cache[v].all()) for v in out["world_owner"].values()), "WITHDRAWN_REGION")
                hashes[c].append(sha(session.states[c].hidden.detach().cpu().numpy().tobytes()))
        session.model.check_reference()
        return hashes

    before = session.snapshot()
    old = cycle()
    boundaries.append(dict(name="source-to-native-to-context", before=digest(before), after=digest(session.snapshot()),
                           action="two interleaved five-phase loops on the adopted 2018 publication",
                           identity_pin=session.model.standard_id, world_pin=session.model.world_id))
    prior_query = session.resolve("electron mass")
    session.adopt("2022")
    check(all(session.needs_review(c) for c in CONTEXTS), "ADOPTION_REVIEW")
    refuse("edition_requires_review", "MNLM_CODATA_LOOP_CONTEXT_NEEDS_REVIEW",
           lambda: session.advance(CONTEXTS[0], QUANTITIES[0]))
    for c in CONTEXTS:
        session.reset(c, reason="Explicitly rebind this development context to the adopted 2022 publication.")
    new = cycle()
    current_query = session.resolve("electron mass")
    check(old[CONTEXTS[0]][:3] == new[CONTEXTS[0]][:3]
          and old[CONTEXTS[0]][3:] != new[CONTEXTS[0]][3:], "REVISION_INPUT_EFFECT")
    check(old[CONTEXTS[1]] == new[CONTEXTS[1]], "UNCHANGED_REFERENCE_EFFECT")
    boundaries.append(dict(name="published-revision", before=prior_query, after=current_query,
                           action="adopt 2022, review and replay both contexts", old_hidden=old, new_hidden=new))
    ep = source.episode("2022", "electron mass")
    before = digest(session.snapshot())
    session.apply_memory(dict(op="invalidate", episode=ep["id"], reason="Withdrawal exercise on retained publication evidence."))
    check(session.needs_review(CONTEXTS[0]) and not session.needs_review(CONTEXTS[1]), "WITHDRAWAL_SCOPE")
    mask = inputs.allowed(session, "cuda")
    check(not bool(mask[inputs.world_index[inputs.unit("row", "2022", "electron mass")]]), "WITHDRAWAL_REGION")
    refuse("withdrawn_context", "MNLM_CODATA_LOOP_CONTEXT_NEEDS_REVIEW",
           lambda: session.advance(CONTEXTS[0], QUANTITIES[0]))
    # Restoring the same receipt does not erase the consumed-evidence review requirement.
    session.apply_memory(dict(op="check_source", episode=ep["id"]))
    session.apply_memory(dict(op="promote", episode=ep["id"], receipt=source.assess(ep)["id"]))
    refuse("rechecked_context_still_dirty", "MNLM_CODATA_LOOP_CONTEXT_NEEDS_REVIEW",
           lambda: session.advance(CONTEXTS[0], QUANTITIES[0]))
    session.reset(CONTEXTS[0], reason="Reviewed the withdrawal and rebuilt the context from checked source evidence.")
    for phase in range(5):
        session.advance(CONTEXTS[0], QUANTITIES[0],
                        receipt=source.assess(ep)["id"] if phase == 3 else None)
    boundaries.append(dict(name="withdrawal-and-review", before=before, after=digest(session.snapshot()),
                           action="withdraw, refuse, recheck, still refuse, explicitly reset and replay",
                           withdrawn_world_region_excluded=True, unrelated_context_preserved=True))
    phases["connected_cycles_revision_withdrawal"] = time.perf_counter()-tick
    tick = time.perf_counter()
    native = inputs.batch([inputs.unit("name", "2022", "electron mass")], model.config, "cuda")
    allowed = inputs.allowed(session, "cuda")
    def direct(ns=native, state=None, permit=allowed):
        return model(ns, (state or model.start("negative"),), allowed=permit, phase=0,
                     event_ids=["0"*64], evidence_origin=["source_observation"])
    refuse("foreign_world_context", "MNLM_WORLD_CONTEXT_BANK",
           lambda: direct(state=replace(model.start("negative"), world_id="0"*64)))
    refuse("missing_native_lane", "MNLM_LOOP_INPUT_ROSTER", lambda: direct(dict(list(native.items())[:-1])))
    refuse("empty_allowed_bank", "MNLM_WORLD_ALLOWED", lambda: direct(permit=torch.zeros_like(allowed)))
    loadkw = dict(expected=pin, standard=model.standard_id, world=model.world_id, device="cuda")
    refuse("foreign_world_pin", "MNLM_WORLD_CHECKPOINT_CONTRACT",
           lambda: world_checkpoint.load(args.output/"model", **dict(loadkw, world="0"*64)))
    corrupt = args.output/"corrupt-model"
    shutil.copytree(args.output/"model", corrupt)
    raw = bytearray((corrupt/"world.pt").read_bytes())
    raw[-1] ^= 1
    (corrupt/"world.pt").write_bytes(raw)
    refuse("corrupt_world_weights", "MNLM_WORLD_CHECKPOINT_BYTES", lambda: world_checkpoint.load(corrupt, **loadkw))
    (corrupt/"COMMITTED").unlink()
    refuse("interrupted_model_publication", "MNLM_WORLD_CHECKPOINT_INCOMPLETE",
           lambda: world_checkpoint.load(corrupt, **loadkw))
    final = session.snapshot()
    openkw = dict(kw, expected_head=session.head, expected_loop_contract=session.loop_pin)
    shadow = ScientificLoop(args.output/"science.sqlite3", source, inputs, args.output/"model", **openkw, device="cuda")
    session.advance(CONTEXTS[1], QUANTITIES[1])
    refuse("stale_live_context", "MNLM_CODATA_STALE_LEDGER",
           lambda: shadow.advance(CONTEXTS[1], QUANTITIES[1]))
    shadow.close()
    final = session.snapshot()
    openkw["expected_head"] = session.head
    session.close()
    # Negative ledger copies are contained within this create-only proof directory.
    for label, sql, error in (
        ("truncated-ledger", "DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)", "MNLM_CODATA_HEAD_PIN"),
        ("corrupt-ledger", "UPDATE events SET hash='"+"0"*64+"' WHERE seq=(SELECT MAX(seq) FROM events)",
         "MNLM_CODATA_CHAIN_HASH")):
        path = args.output/(label+".sqlite3")
        shutil.copyfile(args.output/"science.sqlite3", path)
        with sqlite3.connect(path) as db:
            db.execute(sql)
        refuse(label, error, lambda: ScientificLoop(path, source, inputs, args.output/"model",
                                                    **openkw, device="cuda"), same=False)
    phases["negative_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    restored = ScientificLoop(args.output/"science.sqlite3", source, inputs, args.output/"model",
                              **openkw, device="cuda", neural_replay=True)
    check(restored.snapshot() == final, "RESTART_SNAPSHOT")
    independent = independent_contexts(args.output/"science.sqlite3", source, final)
    check(restored.replayed == independent["advances"] == 26, "REPLAY_COUNT")
    restored.model.check_reference()
    boundaries.append(dict(name="durable-restart", before=digest(final), after=digest(restored.snapshot()),
                           action="fresh object reconstructs all source and context events; reruns every neural step",
                           independent=independent))
    restored.close()
    phases["independent_neural_readback"] = time.perf_counter()-tick
    write(args.output/"snapshot.json", final)
    write(args.output/"boundaries.json", boundaries)
    write(args.output/"run.json", dict(source=source.pin, world_source=inputs.source_id,
          base=args.base_pin, scientific_parent=args.science_run_pin, lineage=lineage, open=openkw))
    report = dict(complete=True, acceptance="author-checked", measured=dict(
                  unique_observations=len(inputs.units), world_rows=len(inputs.world_units),
                  parameters=sum(p.numel() for p in model.parameters()), context_readback=independent,
                  source_windows=inputs.window_readback, controls=len(controls)),
                  controls=controls, phase_seconds=phases, wall_seconds=time.perf_counter()-started,
                  scientific_admission=False, useful_capability_gain_demonstrated=False, serving=False,
                  interpretation="present-day development replay of published editions; no historical prediction",
                  files={n: sha((args.output/n).read_bytes()) for n in (
                      "snapshot.json", "run.json", "boundaries.json", "science.sqlite3")})
    write(args.output/"report.json", report)
    print(json.dumps(dict(complete=True, advances=independent["advances"], controls=len(controls),
                         wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    sys.exit(run_cli(main))

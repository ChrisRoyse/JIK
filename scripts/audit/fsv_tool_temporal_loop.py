"""Real tool constellations through intrinsic loops, delayed echoes and cold recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_temporal_readback import readback as source_readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest, ZERO
from claims.tool_temporal_source import ToolSource, write
from lens.lenscommon import take_lease, run_cli
from model import world_checkpoint
from model.constellation import score
from model.constellation_world import ConstellationWorld
from model.loop_session import output_digest
from model.loop_training_checkpoint import tensor_fingerprint, put
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def parameters(model):
    return digest(tensor_fingerprint(dict(model.named_parameters())))


def states_digest(states):
    return {c: dict(sequence=s.sequence, last=s.last_event,
                    hidden=sha(s.hidden.detach().cpu().numpy().tobytes())) for c, s in states.items()}


def native_readback(inputs, native):
    import numpy as np
    spans = inputs.population.spans
    result = {}
    for spec in inputs.specs:
        with np.load(native/spec.name/"slots.npz", allow_pickle=False) as z:
            x, offsets = z["vectors"], z["offsets"]
            reconstructed, membership = [], []
            for span in spans:
                atom = span["atom"]
                reconstructed.extend(x[offsets[atom]:offsets[atom+1]])
                membership.extend([atom]*int(offsets[atom+1]-offsets[atom]))
            expected = np.array(reconstructed, dtype=np.float32)
            actual, ptr = inputs.arrays[spec.name]
            assert actual.tobytes() == expected.tobytes()
            assert tuple(membership) == inputs.window_atoms[spec.name]
            assert len(ptr) == len(inputs.refs)+1
            result[spec.name] = dict(windows=len(actual), dimensions=actual.shape[1],
                maximum_message_windows=int(np.diff(ptr).max()), sha256=sha(actual.tobytes()),
                atom_membership_sha256=digest(membership))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("raw", "population", "native", "parent", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("population-pin", "native-pin", "parent-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--run-pin")
    args = p.parse_args()
    start = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv"]).decode()
    take_lease("28", "tool-temporal-intrinsic-loop")
    configure()
    import torch
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                       population_pin=args.population_pin, native_pin=args.native_pin)
    phases = dict(source_native_readback=time.perf_counter()-start)
    tick = time.perf_counter()
    code_root = Path(__file__).resolve().parents[1]
    code = {n: sha((code_root/n).read_bytes()) for n in (
        "audit/fsv_tool_temporal_loop.py", "model/tool_temporal_input.py",
        "model/constellation_world.py", "model/constellation.py", "claims/tool_temporal_source.py")}
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "TOOL_PARENT_PIN")
    parent = json.loads((args.parent/"run.json").read_bytes())["open"]
    tasks = [e["task"] for e in source.episodes if e["status"] == "ready" and e["partition"] == "withheld"][:2]
    contexts = {f"tool:{i}": task for i, task in enumerate(tasks)}
    def load_model(path, pin, world):
        model, states, _ = world_checkpoint.load(path, expected=pin, standard=parent["standard"],
                                               world=world, device="cuda")
        return ConstellationWorld.from_world(model).eval(), {s.context: s for s in states}
    if args.cold:
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "TOOL_RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["population"] == args.population_pin and run["native"] == args.native_pin
              and run["parent"] == args.parent_pin and run["contexts"] == contexts
              and run["code"] == code, "TOOL_RUN_JOIN")
        check(all(sha((args.output/n).read_bytes()) == pin for n, pin in run["files"].items()),
              "TOOL_RUN_FILES")
        model, states = load_model(args.output/"initial", run["initial"], run["world"])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"gpu-before.txt").write_text(resident, encoding="utf-8")
        base, _, _ = world_checkpoint.load(args.parent/"model", expected=parent["model_pin"],
            standard=parent["standard"], world=parent["world"], device="cuda")
        check(base.specs == inputs.specs, "TOOL_INSTRUMENT_IDENTITY")
        baseline_parameters = parameters(base)
        model = ConstellationWorld(base.core, inputs.world_bank(base.config, "cuda"),
                                   world_source_id=inputs.source_id).to("cuda").eval()
        model.query.load_state_dict(base.query.state_dict())
        model.place.load_state_dict(base.place.state_dict())
        check(parameters(model) == baseline_parameters, "TOOL_PARAMETERS_CHANGED")
        states = {c: model.start(c) for c in contexts}
        initial = world_checkpoint.save(args.output/"initial", model, tuple(states.values()))
        run = dict(parent=args.parent_pin, population=args.population_pin, native=args.native_pin,
                   contexts=contexts, initial=initial, world=model.world_id, standard=model.standard_id,
                   parameters=baseline_parameters, training=False, scientific_admission=False, code=code)
    phases["load_bank_and_initial_checkpoint"] = time.perf_counter()-tick
    tick = time.perf_counter()
    allowed = torch.ones(len(inputs.world_units), dtype=torch.bool, device="cuda")
    records, committed, head = [], {}, ZERO
    expected = list(map(json.loads, (args.output/"events.jsonl").read_bytes().splitlines())) if args.cold else None
    def advance(context, phase, exchange):
        nonlocal head
        task = contexts[context]
        action = 2+exchange*2
        turn = 1 if phase == 0 else action if phase in (1, 2) else action+1
        ref = inputs.population.lookup[task, turn]
        receipt = source.receipt(task, turn) if phase == 3 else None
        if phase == 2 and exchange == 2:
            # Validate the complete available view before making the commitment.
            inputs.prefix(task, action, partition="withheld", observed_through=action,
                          config=model.config, device="cuda")
        origin = "generated_proposal" if phase in (1, 2) else "source_observation"
        request = dict(context=context, task=task, turn=turn, phase=phase, reference=ref,
            receipt=receipt, prior_sequence=states[context].sequence, origin=origin, prior_head=head)
        native = inputs.observed(ref, model.config, "cuda")
        with torch.no_grad():
            out = model(native.slots, (states[context],), allowed=allowed, phase=phase,
                        event_ids=[digest(request)], evidence_origin=[origin])
        states[context] = out["states"][0]
        body = dict(request=request, output=output_digest(out), state=states_digest(states)[context])
        if phase == 2 and exchange == 2:
            body["prediction"] = out["predicted_constellation"].pack(model.specs, model.config)
            body["region"] = out["predicted_constellation"].regions[0]
            committed[context] = body
        body["previous"] = head
        body["sequence"] = len(records)
        body["hash"] = digest(body)
        if expected is not None:
            check(body == expected[len(records)], "TOOL_EVENT_REPLAY")
        else:
            put(args.output/f"event-{len(records):03}.json", canonical(body))
        records.append(body)
        head = body["hash"]
    for exchange in range(2):
        for phase in range(5):
            for context in contexts:
                advance(context, phase, exchange)
    for phase in range(3):
        for context in contexts:
            advance(context, phase, 2)
    pending_states = states_digest(states)
    if not args.cold:
        put(args.output/"committed.json", canonical(committed))
        pending_pin = world_checkpoint.save(args.output/"pending", model, tuple(states.values()))
        model, states = load_model(args.output/"pending", pending_pin, model.world_id)
        check(states_digest(states) == pending_states, "TOOL_PENDING_RECOVERY")
        run.update(pending=pending_pin, committed=sha((args.output/"committed.json").read_bytes()))
    else:
        check(len(expected) == len(records)+4, "TOOL_PENDING_EVENT_COUNT")
        check(json.loads((args.output/"committed.json").read_bytes()) == committed
              and sha((args.output/"committed.json").read_bytes()) == run["committed"], "TOOL_COMMITMENT_REPLAY")
    assessments = {}
    for context, task in contexts.items():
        receipt = source.receipt(task, 7)
        inputs.feedback(task, 6, receipt=receipt, config=model.config, device="cuda")
        advance(context, 3, 2)
        advance(context, 4, 2)
        assessments[context] = score(committed[context]["prediction"], inputs,
            inputs.population.lookup[task, 5]["unit"], inputs.population.lookup[task, 7]["unit"], model.config)
    check(parameters(model) == run["parameters"], "TOOL_PARAMETERS_CHANGED")
    model.check_reference()
    snapshot = dict(states=states_digest(states), head=head, events=len(records), scores=assessments)
    phases["intrinsic_forecast_and_echo"] = time.perf_counter()-tick
    tick = time.perf_counter()
    controls = []
    def refuse(name, code, fn):
        before = states_digest(states)
        try:
            fn()
        except (ValueError, SenseError) as e:
            check(getattr(e, "code", str(e)) == code, "TOOL_CONTROL_"+name)
        else:
            raise AssertionError(name+" did not refuse")
        check(states_digest(states) == before, "TOOL_CONTROL_MUTATION")
        controls.append(dict(name=name, code=code))
    task = tasks[0]
    prefix = list(inputs.population.prefix(task, 6, partition="withheld", observed_through=6))
    for label, replacement in (("future", inputs.population.lookup[task, 7]),
                               ("other_task", inputs.population.lookup[tasks[1], 6])):
        altered = [*prefix[:-1], replacement]
        refuse(label, "MNLM_CODATA_TOOL_PREFIX_JOIN", lambda: inputs.prefix(task, 6,
            partition="withheld", observed_through=6, references=altered, config=model.config, device="cuda"))
    refuse("target_in_region_bank", "MNLM_CODATA_TOOL_BANK_CUTOFF", lambda: inputs.world_bank(
        model.config, "cuda", references=(*inputs.population.initial_bank(), inputs.population.lookup[task, 7])))
    refuse("self_scored_echo", "MNLM_CODATA_TOOL_ECHO_RECEIPT", lambda: inputs.feedback(
        task, 6, receipt=dict(id="model-confidence"), config=model.config, device="cuda"))
    refuse("meaning_without_admission", "MNLM_CODATA_TOOL_SENSE_ADMISSION_MISSING", inputs.require_meaning)
    native = inputs.observed(prefix[-1], model.config, "cuda").slots
    broken = dict(native)
    broken.pop(next(iter(broken)))
    refuse("missing_instrument", "MNLM_CODATA_CONSTELLATION_ROSTER", lambda: model(
        broken, (states["tool:0"],), allowed=allowed, phase=0, event_ids=["negative-control"],
        evidence_origin=["source_observation"]))
    phases["cutoff_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    source_check = source_readback(args.raw, args.population)
    native_check = native_readback(inputs, args.native)
    phases["independent_source_native"] = time.perf_counter()-tick
    tick = time.perf_counter()
    if args.cold:
        check(len(expected) == len(records) and head == run["head"], "TOOL_COLD_EVENT_COUNT")
        check(all(json.loads((args.output/f"event-{i:03}.json").read_bytes()) == record
                  for i, record in enumerate(records)), "TOOL_DURABLE_EVENTS")
        check(snapshot == json.loads((args.output/"snapshot.json").read_bytes()), "TOOL_COLD_STATE")
        write(args.output/"cold-readback.json", dict(complete=True, snapshot=digest(snapshot),
            source=source_check, native=native_check, events=len(records),
            wall_seconds=time.perf_counter()-start, acceptance="author-checked"))
    else:
        write(args.output/"snapshot.json", snapshot)
        put(args.output/"events.jsonl", b"".join(canonical(r)+b"\n" for r in records))
        run.update(head=head, files={n: sha((args.output/n).read_bytes())
                   for n in ("events.jsonl", "snapshot.json", "committed.json")})
        write(args.output/"run.json", run)
        phases["publish_and_rehash"] = time.perf_counter()-tick
        wall = time.perf_counter()-start
        coverage = sum(phases.values())/wall
        check(coverage >= .9, "TOOL_TIMING_COVERAGE")
        write(args.output/"report.json", dict(complete=True, acceptance="author-checked",
            source=source_check, native=native_check, controls=controls, phases_seconds=phases,
            wall_seconds=wall, timing_coverage=coverage, events=len(records),
            parameters_unchanged=True, training=False,
            files={n: sha((args.output/n).read_bytes()) for n in ("run.json", "events.jsonl", "snapshot.json", "committed.json")}))
    print(json.dumps(dict(complete=True, cold=args.cold, events=len(records), controls=len(controls),
                          head=head, wall_seconds=time.perf_counter()-start)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

"""Withdraw observed regions, repair the bank, and learn from new clean contexts."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import write
from audit.bank_recovery_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_observation import ProcessObservations
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model import constellation_training_checkpoint as checkpoint
from model.constellation_learning import trainable, frozen_pin
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.process_input import ProcessInputs
from model.tool_growth import GrowingToolLoop, parameters, state_pin
from model.tool_kernel import ToolKernel
from model.tool_learning import ToolLearningData
from model.tool_learner import start
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def learn(session, root, train, evaluate, refuse):
    data = ToolLearningData(session, train=train, evaluate=evaluate)
    refuse("withdrawn_predecessor", "TOOL_LEARNER_RETENTION", lambda:
           ToolLearningData(session, train=train, evaluate=evaluate,
                            continuation=session.decisions[-1]["candidate"]))
    model, opt, continuity = start(data)
    check(continuity is None and data.optimizer_offset == 0, "RECOVERY_LEARNER_BASELINE")
    before = tensor_fingerprint(dict(model.named_parameters()))
    parent = session.active_checkpoint
    initial = checkpoint.save(root/"recovery-start", model, opt, data, step=0, parent=parent)
    losses = []
    def step(m, o):
        o.zero_grad(set_to_none=True)
        loss, details = data.loss(m)
        check(bool(torch.isfinite(loss)), "RECOVERY_FINITE_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable(m)], 1.)
        o.step()
        return dict(loss=float(loss.detach()), details=details)
    for i in range(1, data.plan["steps"]+1):
        losses.append(step(model, opt))
        if i == 4:
            mid = checkpoint.save(root/"recovery-mid", model, opt, data, step=i, parent=parent)
    recovered, restored_opt, _ = checkpoint.load(root/"recovery-mid", data, expected=mid)
    for i in range(4, data.plan["steps"]):
        check(step(recovered, restored_opt) == losses[i], "RECOVERY_OPTIMIZER_STEPS")
    check(fingerprint(model,opt) == fingerprint(recovered,restored_opt), "RECOVERY_OPTIMIZER_RESTART")
    after = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in before if before[n] != after[n]]
    check(changed and all(n.startswith(("query.", "place.")) for n in changed)
          and frozen_pin(model) == data.frozen, "RECOVERY_FROZEN_CORE")
    name = "recovery-increment"
    pin = checkpoint.save(root/"candidates"/name, model, opt, data, step=data.plan["steps"], parent=parent)
    decision = session.consider(name, pin=pin, train=train, evaluate=evaluate, parent=parent)
    refuse("unadmitted_recovery_candidate", "TOOL_CANDIDATE_NOT_SELECTED", lambda: session.require_selection(name))
    return dict(start=initial, midpoint=mid, decision=decision, losses=losses, changed=changed,
                optimizer_steps=data.plan["steps"], exact_restart=True, continuation=None)


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "kernel", "observations",
                 "execution-population", "execution-native", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin", "observation-pin",
                 "execution-population-pin", "execution-native-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--grow-observed", action="store_true")
    a = p.parse_args()
    begin = time.perf_counter()
    gpu = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "bank-recovery")
    configure()
    source = ToolSource(a.raw)
    inputs = ToolInputs(source,a.population,a.native,population_pin=a.population_pin,native_pin=a.native_pin)
    observations = ProcessObservations(a.observations,a.observation_pin,source)
    native = ProcessInputs(observations,inputs,a.execution_population,a.execution_native,
        population_pin=a.execution_population_pin,native_pin=a.execution_native_pin)
    check(sha((a.parent/"run.json").read_bytes()) == a.parent_pin, "RECOVERY_PARENT")
    parent = json.loads((a.parent/"run.json").read_bytes())
    routing = ToolKernel(inputs,a.output/"initial-kernel",(a.output/"initial-kernel"/"COMMITTED").read_text())
    withheld = next(e["task"] for e in source.episodes if e["partition"] == "withheld" and e["status"] == "ready")
    contexts = {"train:1":"tonioo__sievelib-79","withheld:0":withheld}
    def open_session(**kw):
        return GrowingToolLoop(a.output/"recovery.sqlite3",source,inputs,a.parent/"initial",
            model_pin=parent["initial"],standard=parent["standard"],world=parent["world"],
            contexts=contexts,device="cuda",observations=observations,execution_inputs=native,
            routing=routing,growth_root=a.output/"growth",candidate_root=a.output/"candidates",**kw)
    phases = dict(source_native=time.perf_counter()-begin)
    tick = time.perf_counter()
    if a.cold:
        snapshot = json.loads((a.output/"recovery-snapshot.json").read_bytes())
        session = open_session(expected_head=snapshot["head"], expected_contract=snapshot["contract"])
        check(session.snapshot() == snapshot, "RECOVERY_COLD")
        direct = readback(inputs,native,a.output,a.observations)
        check(direct == json.loads((a.output/"recovery-report.json").read_bytes())["readback"], "RECOVERY_COLD_READ")
        session.close()
        phases["replay_readback"] = time.perf_counter()-tick
        write(a.output/"recovery-cold.json",dict(complete=True,phases=phases,wall_seconds=time.perf_counter()-begin))
        print(json.dumps(dict(complete=True,cold=True)))
        return
    with sqlite3.connect(f"file:{(a.output/'experience.sqlite3').as_posix()}?mode=ro",uri=True) as origin:
        with sqlite3.connect(a.output/"recovery.sqlite3") as branch:
            origin.backup(branch)
    base = json.loads((a.output/"snapshot.json").read_bytes())
    session = open_session(expected_head=base["head"],expected_contract=base["contract"])
    check(session.snapshot() == base, "RECOVERY_BRANCH")
    phases["restore_prefix"] = time.perf_counter()-tick
    controls, boundaries = [], []
    def refuse(name,code,action):
        before = digest(session.snapshot())
        try:
            action()
        except (ValueError,SenseError) as error:
            check(code in str(error), "RECOVERY_CONTROL_CODE_"+name+": "+str(error))
        else:
            raise AssertionError("RECOVERY_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "RECOVERY_CONTROL_STATE_"+name)
        controls.append(dict(name=name,code=code))
    def boundary(action):
        before = copy.deepcopy(session.snapshot())
        event = action()
        boundaries.append(dict(before=before,action=event,after=copy.deepcopy(session.snapshot())))
        return event
    tick = time.perf_counter()
    # Leave an actual committed forecast pending when withdrawal arrives.
    dev = next(e["task"] for e in source.episodes if e["partition"] == "development" and e["status"] == "ready")
    session.start_context("pending:0",dev)
    for sequence in range(13):
        phase, exchange = sequence%5, sequence//5
        session.advance("pending:0",receipt=source.receipt(dev,3+2*exchange) if phase == 3 else None)
    pending = session.pending["pending:0"]
    old_train = [k for k,v in session.forecasts.items() if v["record"]["request"]["context"] == "train:1"]
    first = next(iter(session.executions))
    boundary(lambda: session.invalidate(session.executions[first]["episode"],reason="FSV shared occurrence withdrawn"))
    check(session.forecasts[pending]["invalidated"], "RECOVERY_PENDING_INVALIDATION")
    broken = session.snapshot()
    refuse("recovery_requires_pinned_head","TOOL_RECOVERY_HEAD_REQUIRED",lambda: open_session(recovery=True))
    refuse("unavailable_bank","TOOL_EXECUTION_UNAVAILABLE",session._bank_checked)
    refuse("dirty_context","TOOL_CONTEXT_REVIEW",lambda: session.advance("train:1"))
    refuse("normal_cold_refuses","TOOL_EXECUTION_UNAVAILABLE",lambda:
           open_session(expected_head=broken["head"],expected_contract=broken["contract"]))
    session.close()
    session = open_session(recovery=True,expected_head=broken["head"],expected_contract=broken["contract"])
    check(session.snapshot() == broken, "RECOVERY_EXPLICIT_OPEN")
    new_train = next(e["task"] for e in source.episodes if e["partition"] == "train" and e["status"] == "ready"
                     and e["task"] not in session.contexts.values())
    refuse("repair_mode_not_operational","TOOL_EXECUTION_UNAVAILABLE",
           lambda: session.start_context("new:train",new_train))
    # Corruption cannot masquerade as an invalidation.
    ep = session.bank[inputs.world_units[0]]
    saved = copy.deepcopy(session.memory.state[ep])
    session.memory.state[ep]["receipt"] = None
    refuse("corrupt_retained_receipt","TOOL_EVIDENCE_UNAVAILABLE",lambda: session.repair_bank("bad"))
    session.memory.state[ep] = saved
    before_parameters, before_state = parameters(session.model), state_pin(session.states)
    repair = boundary(lambda: session.repair_bank("withdrawn-observations"))
    check(len(repair["removed"]) == 2 and len(session.bank) == 60
          and parameters(session.model) == before_parameters and state_pin(session.states) == before_state
          and session.forecasts[pending]["status"] == "canceled" and not session.pending
          and session.dirty == set(broken["dirty"]), "RECOVERY_REPAIR_BOUNDARY")
    session._bank_checked()
    refuse("duplicate_repair","TOOL_GROWTH_DUPLICATE",lambda: session.repair_bank("withdrawn-observations"))
    refuse("unneeded_repair","TOOL_RECOVERY_NOT_NEEDED",lambda: session.repair_bank("empty"))
    for c in broken["dirty"]:
        refuse("dirty_after_repair_"+c,"TOOL_CONTEXT_REVIEW",lambda: session.advance(c))
    refuse("withdrawn_learning","TOOL_LEARNING_INELIGIBLE",lambda: session.eligible(first))
    refuse("canceled_learning","TOOL_LEARNING_INELIGIBLE",lambda: session.eligible(pending))
    refuse("withdrawn_growth","TOOL_LEARNING_INELIGIBLE",lambda: session.grow("revival",[first]))
    phases["withdraw_repair"] = time.perf_counter()-tick
    tick = time.perf_counter()
    new_eval = next(e["task"] for e in source.episodes if e["partition"] == "withheld" and e["status"] == "ready"
                    and e["task"] not in session.contexts.values())
    for c,t in (("new:train",new_train),("new:withheld",new_eval)):
        boundary(lambda: session.start_context(c,t))
    refuse("duplicate_context","TOOL_CONTEXT_EXISTS",lambda: session.start_context("new:train",new_train))
    refuse("duplicate_task","TOOL_CONTEXT_TASK",lambda: session.start_context("alias",new_train))
    for exchange in range(3):
        for phase in range(5):
            for c,t in (("new:train",new_train),("new:withheld",new_eval)):
                session.advance(c,receipt=source.receipt(t,3+2*exchange) if phase == 3 else None)
    train = [k for k,v in session.forecasts.items() if v["record"]["request"]["context"] == "new:train"]
    evaluate = [k for k,v in session.forecasts.items()
                if v["record"]["request"]["context"] in ("withheld:0","new:withheld") and not v["invalidated"]]
    check(len(train) == 1 and len(evaluate) == 2, "RECOVERY_FRESH_ROSTER")
    for k in train+evaluate:
        session.eligible(k)
    retired = {int(u) for u in repair["removed"]}
    for k in train+evaluate:
        desc = session.forecasts[k]["record"]["region_context"]["definition"]
        check(not retired.intersection(desc["units"]), "RECOVERY_RETIRED_REGION_USED")
    check(state_pin({c:session.states[c] for c in broken["states"]}) == before_state, "RECOVERY_DIRTY_STATE_DRIFT")
    phases["new_contexts"] = time.perf_counter()-tick
    tick = time.perf_counter()
    before_learning = copy.deepcopy(session.snapshot())
    training = learn(session,a.output,train,evaluate,refuse)
    boundaries.append(dict(before=before_learning,action=training["decision"],after=session.snapshot()))
    check(parameters(session.model) == before_parameters, "RECOVERY_ACTIVE_PARAMETERS")
    write(a.output/"recovery-training.json",training)
    snapshot = session.snapshot()
    session.close()
    phases["learning"] = time.perf_counter()-tick
    write(a.output/"recovery-snapshot.json",snapshot)
    tick = time.perf_counter()
    direct = readback(inputs,native,a.output,a.observations)
    phases["independent_readback"] = time.perf_counter()-tick
    report = dict(complete=True,controls=controls,boundaries=boundaries,readback=direct,phases=phases,
                  wall_seconds=time.perf_counter()-begin,scientific_admission=False,useful_capability=False)
    report["timing_coverage"] = sum(phases.values())/report["wall_seconds"]
    check(report["timing_coverage"] >= .9, "RECOVERY_TIMING")
    write(a.output/"recovery-report.json",report)
    print(json.dumps(dict(complete=True,controls=len(controls),wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

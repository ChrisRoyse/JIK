"""Independent saved-byte audit of quarantined parameters and Adam continuity."""
import io

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        a = value.cpu().contiguous().numpy()
        return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=sha(a.tobytes()))
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(v) for v in value]
    return value


def parameters(saved):
    core, extra = saved[2:4]
    values = {**{"core."+k: torch.from_numpy(a) for k, a in core.items()
                 if not k.startswith("ref_")},
              **{k: torch.from_numpy(a) for k, a in extra.items() if not k.startswith("world_")}}
    return digest(tensor_tree(values))


def optimizer(root, m, cumulative, saved):
    raw = (root/"training.pt").read_bytes()
    check(len(raw) == m["bytes"] and sha(raw) == m["sha256"], "LEARNER_READ_OPTIMIZER_BYTES")
    value = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")["optimizer"]
    check(m["schema"] == "constellation-readout-learning-v2"
          and m["optimizer_steps"] == cumulative and len(value["param_groups"]) == 1,
          "LEARNER_READ_CURSOR")
    group = value["param_groups"][0]
    ids = group["params"]
    check(m["names"] == [k for k in saved[3] if k.startswith(("query.", "place."))],
          "LEARNER_READ_PARAMETER_ORDER")
    check(len(ids) == len(set(ids)) == len(m["names"]) == 24
          and set(value["state"]) == (set(ids) if cumulative else set()),
          "LEARNER_READ_OPTIMIZER_ROSTER")
    check(group["lr"] == m["binding"]["plan"]["learning_rate"]
          and group["weight_decay"] == m["binding"]["plan"]["weight_decay"]
          and group["betas"] == (.9, .999) and group["eps"] == 1e-8,
          "LEARNER_READ_OPTIMIZER_RECIPE")
    for state in value["state"].values():
        check(set(state) == {"step", "exp_avg", "exp_avg_sq"}
              and float(state["step"]) == cumulative
              and all(bool(torch.isfinite(v).all()) for v in state.values()),
              "LEARNER_READ_OPTIMIZER_STATE")
    for identity, name in zip(ids, m["names"]):
        if cumulative:
            for key in ("exp_avg", "exp_avg_sq"):
                a = value["state"][identity][key].numpy()
                check(a.shape == saved[3][name].shape and a.dtype == saved[3][name].dtype,
                      "LEARNER_READ_OPTIMIZER_SHAPE")
    return value


def inspect(run, body, proof, active, history, checkpoint, manifest):
    generation = proof["generation"]
    offset = proof["optimizer_offset"]
    result = {}
    stages = (("start", run/f"start-{generation}", proof["start_pin"], 0),
              ("mid", run/f"mid-{generation}", None, 4),
              ("final", run/"candidates"/body["candidate"], body["pin"], 8))
    for name, root, pin, step in stages:
        m = manifest(root, pin or (root/"COMMITTED").read_text())
        saved = checkpoint(root/"model", m["model"])
        check(m["step"] == step and m["parent"] == body["parent"]
              and digest(m["binding"]) == body["binding"]
              and saved[0]["world"] == active[0]["world"]
              and saved[0]["standard"] == active[0]["standard"]
              and saved[2].keys() == active[2].keys()
              and all(np.array_equal(a, active[2][k]) for k, a in saved[2].items())
              and all(np.array_equal(a, active[3][k]) for k, a in saved[3].items()
                      if k.startswith("world_")), "LEARNER_READ_FROZEN")
        opt = optimizer(root, m, offset+step, saved)
        check(m["fingerprint"]["optimizer"] == digest(tensor_tree(opt)),
              "LEARNER_READ_OPTIMIZER_PIN")
        result[name] = m, saved, opt
    sm, start, so = result["start"]
    fm, final, fo = result["final"]
    check(parameters(start) == proof["before_parameters"]
          and parameters(final) == proof["after_parameters"]
          and proof["optimizer_steps"] == offset+8, "LEARNER_READ_PARAMETER_PIN")
    if "continuation" in body:
        name = body["continuation"]
        check(history and name == next(reversed(history)), "LEARNER_READ_PREDECESSOR")
        pb, pm, ps, po = history[name]
        expected = dict(candidate=name, pin=pb["pin"], world=pm["binding"]["world"],
                        optimizer_steps=pm["optimizer_steps"])
        check(sm["binding"]["continuation"] == expected and offset == pm["optimizer_steps"]
              and parameters(start) == parameters(ps) and tensor_tree(so) == tensor_tree(po),
              "LEARNER_READ_TRANSPORT")
        check(all(set(pb[k]) <= set(body[k]) for k in ("train", "evaluate"))
              and all(sm["binding"]["evidence"].get(k) == v
                      for k, v in pm["binding"]["evidence"].items()), "LEARNER_READ_RETENTION")
        measured = dict(parameters=parameters(start), optimizer=digest(tensor_tree(so)))
        check(proof["continuity"] == dict(predecessor=expected, old_world=expected["world"],
              world=active[0]["world"], before=measured, after=measured), "LEARNER_READ_PROOF")
    else:
        check(offset == 0 and not so["state"] and parameters(start) == parameters(active)
              and proof["continuity"] is None, "LEARNER_READ_INITIAL")
    history[body["candidate"]] = body, fm, final, fo
    return start

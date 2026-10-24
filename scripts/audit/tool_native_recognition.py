"""Independent native-window region scoring and direct-path failure controls."""
import base64
from dataclasses import replace

import numpy as np
import torch

from claims.codata_source import check


def context(inputs, record, core, extra, hidden):
    hidden = np.frombuffer(base64.b64decode(hidden, validate=True), dtype="<f4").astype(np.float64)
    hidden = hidden.reshape(len(inputs.specs), -1)
    mode = record.get("recognition")
    check(mode in ("typed-native-window-context-v1", "typed-native-causal-prefix-v1"),
          "RECOGNITION_READ_CONTRACT")
    if mode == "typed-native-window-context-v1":
        return hidden
    request = record["request"]
    refs = [inputs.population.lookup[request["task"], t] for t in range(1, request["turn"])]
    check(record["region_context"]["definition"]["history"] == refs, "RECOGNITION_READ_HISTORY")
    result = hidden.copy()
    for lane, spec in enumerate(inputs.specs):
        name = spec.name
        a, ptr = inputs.arrays[name]
        w, b = core[f"projectors.{name}.weight"], core[f"projectors.{name}.bias"]
        qw, qb = extra[f"query.{name}.weight"].astype(np.float64), extra[f"query.{name}.bias"]
        state = np.zeros(hidden.shape[1], dtype=np.float64)
        for ref in refs:
            u = ref["unit"]
            projected = a[ptr[u]:ptr[u+1]].astype(np.float64)@w.astype(np.float64).T+b
            projected /= np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-12)
            for window in projected:
                state = np.tanh((state+window)@qw.T+qb)
        result[lane] += state
    return result


def scores(inputs, record, core, extra, hidden, *, unit=None):
    hidden = context(inputs, record, core, extra, hidden)
    units = record["region_context"]["definition"]["units"]
    unit = record["request"]["reference"]["unit"] if unit is None else unit
    result = {}
    def norm(v):
        return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    for i, spec in enumerate(inputs.specs):
        name = spec.name
        a, ptr = inputs.arrays[name]
        weight = core[f"projectors.{name}.weight"].astype(np.float64)
        bias = core[f"projectors.{name}.bias"].astype(np.float64)
        observation = a[ptr[unit]:ptr[unit+1]].astype(np.float64)
        observed = norm(observation@weight.T+bias)
        q = norm((observed+norm(hidden[i]))@extra[f"query.{name}.weight"].astype(np.float64).T
                 + extra[f"query.{name}.bias"])
        values = []
        for candidate in units:
            native = a[ptr[candidate]:ptr[candidate+1]].astype(np.float64)
            keys = norm(native@weight.T+bias)
            # Each observed window contributes scalar evidence, without pooling vectors.
            values.append(float((q@keys.T*10).max(axis=1).mean()))
        result[name] = np.asarray(values)
    selected = units[int(sum(result.values()).argmax())]
    return result, selected


def controls(data, model, key, generation, refuse):
    from model.native import NativeInput
    e = data.examples[key]
    view = data.prediction_model(model, key)
    name = model.specs[0].name
    original = e["native"][name]
    try:
        e["native"][name] = NativeInput(original.identity, original.rows,
            torch.zeros_like(original.values), torch.ones_like(original.padding),
            torch.ones_like(original.absent))
        refuse(f"recognition_absent_{generation}", "CONSTELLATION_RECOGNITION_ABSENT",
               lambda: data.forward(model, key))
        e["native"][name] = replace(original, identity="0"*64)
        refuse(f"recognition_wrong_instrument_{generation}", "MNLM_MODEL_IDENTITY",
               lambda: data.forward(model, key))
        e["native"][name] = replace(original, padding=torch.ones_like(original.padding))
        refuse(f"recognition_invalid_mask_{generation}", "MNLM_MODEL_MASK",
               lambda: data.forward(model, key))
    finally:
        e["native"][name] = original
    config = view.config
    try:
        view.config = replace(config, max_attention_elements=1)
        refuse(f"recognition_budget_{generation}", "CONSTELLATION_SCORE_BUDGET",
               lambda: data.forward(model, key))
    finally:
        view.config = config


def inspect_start(data, model, key, output):
    e = data.examples[key]
    hidden = base64.b64encode(output["states"][0].hidden.detach().cpu().numpy().tobytes()).decode()
    core = {k: v.detach().cpu().numpy() for k, v in model.core.state_dict().items()}
    extra = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
             if k.startswith(("query.", "place."))}
    expected, selected = scores(data.session.inputs, e["record"], core, extra, hidden)
    check(selected == int(output["predicted_constellation"].regions[0].split(":")[0]),
          "RECOGNITION_READ_REGION")
    errors = {n: float(np.max(np.abs(v-output["regions"][n][0].detach().cpu().numpy())))
              for n, v in expected.items()}
    check(max(errors.values()) < 2e-5, "RECOGNITION_READ_SCORES")
    request = e["request"]
    earlier = data.session.inputs.population.lookup[request["task"], request["turn"]-2]["unit"]
    changed, alternate = scores(data.session.inputs, e["record"], core, extra, hidden, unit=earlier)
    deltas = {n: float(np.max(np.abs(v-changed[n]))) for n, v in expected.items()}
    check(max(deltas.values()) > 1e-6, "RECOGNITION_NATIVE_PATH_INACTIVE")
    return dict(forecast=key, selected=selected, earlier_action=earlier,
                earlier_action_selected=alternate, fixed_state=hidden,
                independent_errors=errors, direct_native_deltas=deltas)

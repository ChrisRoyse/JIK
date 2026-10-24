"""Actual learned context boundaries; independent recurrence readback."""
import base64
from dataclasses import replace

import numpy as np
import torch

from claims.codata_source import check
from audit.tool_native_recognition import context as world_context


def controls(data, model, key, output, generation, refuse):
    from model.constellation_history import interpret
    e = data.examples[key]
    view = data.prediction_model(model, key)
    history = view.context_history
    first = history.frames[0]
    name = model.specs[0].name
    wrong = replace(first, slots={**first.slots, name: replace(first.slots[name], identity="0"*64)})
    missing = replace(first, slots={n: s for n,s in first.slots.items() if n != name})
    original = first.slots[name]
    absent = replace(first, slots={**first.slots, name: replace(original,
        values=torch.zeros_like(original.values), padding=torch.ones_like(original.padding),
        absent=torch.ones_like(original.absent))})
    variants = [
        ("absent", None, "TOOL_LEARNING_HISTORY"),
        ("foreign", replace(history, context="foreign-context"), "CONSTELLATION_HISTORY_CONTEXT"),
        ("missing_frame", replace(history, frames=history.frames[:-1]), "CONSTELLATION_HISTORY_ROSTER"),
        ("wrong_row", replace(history, rows=("foreign-row",)+history.rows[1:]), "CONSTELLATION_HISTORY_FRAME"),
        ("wrong_instrument", replace(history, frames=(wrong,)+history.frames[1:]), "MNLM_MODEL_IDENTITY"),
        ("missing_instrument", replace(history, frames=(missing,)+history.frames[1:]), "CONSTELLATION_ROSTER"),
        ("absent_instrument", replace(history, frames=(absent,)+history.frames[1:]), "CONSTELLATION_HISTORY_FRAME"),
    ]
    try:
        for name, value, code in variants:
            view.context_history = value
            refuse(f"history_{name}_{generation}", code, lambda: data.forward(model, key))
    finally:
        view.context_history = history
    config = view.config
    size = sum(s.values.numel() for frame in history.frames for s in frame.slots.values())
    hidden = torch.stack([s.hidden for s in output["states"]])
    try:
        view.config = replace(config, max_native_elements=size-1)
        refuse(f"history_budget_{generation}", "CONSTELLATION_HISTORY_BUDGET",
               lambda: interpret(view, hidden, (e["state"],), history))
    finally:
        view.config = config


def inspect(data, model, key, output):
    e = data.examples[key]
    state = base64.b64encode(output["states"][0].hidden.detach().cpu().numpy().tobytes()).decode()
    arrays = lambda module, prefix: {n: p.detach().cpu().numpy() for n,p in module.state_dict().items()
                                    if n.startswith(prefix)}
    expected = world_context(data.session.inputs, e["record"], arrays(model.core, "projectors."),
                             arrays(model, "query."), state)
    actual = output["placement_hidden"][0].detach().cpu().numpy()
    check(np.max(np.abs(expected-actual)) < 2e-6, "HISTORY_READ_CONTEXT")
    # This derivative isolates the history branch: fixed identity states have no
    # query dependency; direct current-action recognition is not in this scalar.
    params = tuple(model.query.parameters())
    gradients = torch.autograd.grad(output["placement_hidden"].square().mean(), params, retain_graph=True)
    check(all(bool(torch.isfinite(g).all()) and bool((g != 0).any()) for g in gradients),
          "HISTORY_GRADIENT_DISCONNECTED")
    return dict(forecast=key, context=actual.tolist(), fixed_identity=state,
                max_error=float(np.max(np.abs(expected-actual))),
                gradient_norms=[float(torch.linalg.vector_norm(g)) for g in gradients])


def read(inputs, records, hidden, start, final, proof):
    result = []
    for sample in proof["history"]:
        key = sample["forecast"]
        check(sample["fixed_identity"] == hidden[key], "HISTORY_READ_IDENTITY")
        before = world_context(inputs, records[key], start[2], start[3], hidden[key])
        after = world_context(inputs, records[key], final[2], final[3], hidden[key])
        check(np.max(np.abs(before-np.asarray(sample["context"]))) < 2e-6, "HISTORY_READ_START")
        changes = np.linalg.norm(after-before, axis=1)
        check(bool((changes > 0).all()), "HISTORY_READ_NOT_LEARNED")
        result.append(dict(forecast=key, context_changes=changes.tolist()))
    check({r["forecast"] for r in result} == set(proof["decision"]["train"]), "HISTORY_READ_ROSTER")
    return result

"""Physical restart and refusal checks for full-document constellation learning."""
import json

from claims.codata_source import check
from model.loop_training_checkpoint import fingerprint
from model.text_dynamics import frozen_pin, update
from model.text_dynamics_checkpoint import load
from model.text_sequence_data import PURPOSE


def checkpoint_model(root):
    return json.loads((root/"manifest.json").read_bytes())["model"]


def training_document(data, examples):
    group = next(g for g in data.groups if g["split"] == "train")
    lookup = {e["left"]: e for e in examples}
    return group, [lookup[i] for i in group["indices"][:-1]]


def next_update(root, pin, model, data, evidence, examples):
    optimizer, _ = load(root, model, data, expected=pin, evidence=evidence, purpose=PURPOSE)
    group, rows = training_document(data, examples)
    value = update(model, optimizer, data, rows, group, gradient_probe=True)
    return dict(update=value, fingerprint=fingerprint(model, optimizer))


def controls(root, pin, model, optimizer, data, evidence, examples):
    results = []
    group, rows = training_document(data, examples)
    before = fingerprint(model, optimizer)
    def deny(name, function, code):
        try:
            function()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            check(actual == "MNLM_CODATA_"+code, "DYNAMICS_CONTROL_CODE:"+name+":"+actual)
            results.append(dict(name=name, code=actual))
        else:
            raise ValueError("MNLM_DYNAMICS_CONTROL_ACCEPTED:"+name)
    deny("saved_hidden_state", lambda: data.loss(model, rows[0]), "DYNAMICS_SAVED_STATE_FORBIDDEN")
    development = next(g for g in data.groups if g["split"] == "development")
    deny("development_gradient", lambda: data.sequence([], development), "DYNAMICS_DEVELOPMENT_GRADIENT")
    deny("reordered_sequence", lambda: data.sequence(list(reversed(rows)), group), "DYNAMICS_COMPLETE_SEQUENCE")
    deny("incomplete_sequence", lambda: data.sequence(rows[1:], group), "DYNAMICS_COMPLETE_SEQUENCE")
    changed = [{**rows[0], "label": -1}]+rows[1:]
    deny("changed_receipt", lambda: data.sequence(changed, group), "DYNAMICS_RECEIPT")
    deny("serving", lambda: load(root, model, data, expected=pin, evidence=evidence, purpose="serving"),
         "DYNAMICS_ADMISSION_REQUIRED")
    deny("wrong_checkpoint_pin", lambda: load(root, model, data, expected="0"*64,
         evidence=evidence, purpose=PURPOSE), "DYNAMICS_CHECKPOINT_PIN")
    check(fingerprint(model, optimizer) == before, "DYNAMICS_FAILED_CONTROL_MUTATION")
    return results


def frozen_control(model):
    import torch
    reference = model.core.identity.ref_values_0
    before = reference.detach().clone()
    old = frozen_pin(model)
    with torch.no_grad():
        reference.view(-1)[0].add_(.125)
    try:
        check(frozen_pin(model) != old, "DYNAMICS_FROZEN_CONTROL_INVISIBLE")
        try:
            model.check_reference()
        except Exception as exc:
            check(getattr(exc, "code", str(exc)) == "MNLM_LOOP_STANDARD_DRIFT", "DYNAMICS_STANDARD_CONTROL_CODE")
        else:
            raise ValueError("MNLM_DYNAMICS_CHANGED_STANDARD_ACCEPTED")
    finally:
        with torch.no_grad():
            reference.copy_(before)
    model.check_reference()
    check(frozen_pin(model) == old, "DYNAMICS_STANDARD_CONTROL_RESTORE")
    return dict(changed_standard_rejected=True, restored_exactly=True)

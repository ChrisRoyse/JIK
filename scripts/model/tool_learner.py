"""Continue a committed quarantined learner without selecting it for service."""
import copy
import json
from types import SimpleNamespace

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model import constellation_training_checkpoint as checkpoint
from model.constellation_learning import frozen_pin
from model.loop_training_checkpoint import tensor_fingerprint


def bind(data, name):
    """Authenticate the immediate predecessor and retain all of its evidence."""
    from model.tool_learning import candidate_path
    session = data.session
    check(session.decisions and session.decisions[-1]["candidate"] == name,
          "TOOL_LEARNER_PREDECESSOR")
    decision = session.decisions[-1]
    root = candidate_path(session, name)
    check((root/"manifest.json").is_file() and (root/"COMMITTED").is_file(),
          "TOOL_LEARNER_INCOMPLETE")
    check((root/"manifest.json").stat().st_size <= 1 << 20, "LEARNING_CHECKPOINT_BUDGET")
    raw = (root/"manifest.json").read_bytes()
    m = json.loads(raw)
    check(sha(raw) == decision["pin"] == (root/"COMMITTED").read_text()
          and raw == canonical(m), "TOOL_LEARNER_PIN")
    prior = m["binding"]
    check(digest(prior) == decision["binding"] and m["parent"] == decision["parent"]
          and m["step"] == prior["plan"]["steps"] and m["serving"] is False,
          "TOOL_LEARNER_COMPLETENESS")
    check(all(prior[k] == data.binding[k] for k in ("contract", "standard", "code"))
          and {k: v for k, v in prior["plan"].items() if k not in ("train", "evaluate")} ==
              {k: v for k, v in data.plan.items() if k not in ("train", "evaluate")},
          "TOOL_LEARNER_CONTRACT")
    check(all(set(prior["plan"][k]) <= set(data.plan[k]) for k in ("train", "evaluate")),
          "TOOL_LEARNER_RETENTION")
    check(all(data.binding["evidence"].get(k) == v for k, v in prior["evidence"].items()),
          "TOOL_LEARNER_EVIDENCE")
    offset = prior.get("continuation", {}).get("optimizer_steps", 0)
    check(type(offset) is int and offset >= 0 and type(m.get("optimizer_steps")) is int
          and m["optimizer_steps"] == offset+m["step"], "LEARNING_CURSOR")
    data.optimizer_offset = m["optimizer_steps"]
    data.continuation_name = name
    data.predecessor = (root, m, decision["pin"])
    data.binding["continuation"] = dict(candidate=name, pin=decision["pin"],
        world=prior["world"], optimizer_steps=data.optimizer_offset)


def start(data):
    """Load exact historical state, then extend only its frozen reference bank."""
    from model.tool_growth import transport, parameters
    data.fresh()
    if data.continuation_name is None:
        model = copy.deepcopy(data.model)
        optimizer = checkpoint.optimizer_for(model, data.plan)
        checkpoint.validate(model, optimizer, data, 0)
        return model, optimizer, None
    root, m, pin = data.predecessor
    # The historical binding is authenticated against the committed decision by bind().
    historical = SimpleNamespace(binding=m["binding"], frozen=m["frozen"],
        plan=m["binding"]["plan"], session=data.session, fresh=data.fresh,
        optimizer_offset=m["binding"].get("continuation", {}).get("optimizer_steps", 0),
        model=SimpleNamespace(standard_id=m["binding"]["standard"], world_id=m["binding"]["world"]))
    old, old_optimizer, restored = checkpoint.load(root, historical, expected=pin)
    check(restored == m and old.world_source_id in (
              data.model.world_source_id, *getattr(data.session.inputs, "source_ancestry", ()))
          and old.specs == data.model.specs, "TOOL_LEARNER_SOURCE")
    before = dict(parameters=parameters(old),
                  optimizer=digest(tensor_fingerprint(old_optimizer.state_dict())))
    model, _ = transport(old, data.session.inputs, {})
    check(frozen_pin(model) == data.frozen, "TOOL_LEARNER_FROZEN_DRIFT")
    optimizer = checkpoint.optimizer_for(model, data.plan)
    optimizer.load_state_dict(copy.deepcopy(old_optimizer.state_dict()))
    checkpoint.validate(model, optimizer, data, 0)
    after = dict(parameters=parameters(model),
                 optimizer=digest(tensor_fingerprint(optimizer.state_dict())))
    check(before == after, "TOOL_LEARNER_TRANSPORT_DRIFT")
    return model, optimizer, dict(predecessor=data.binding["continuation"],
        old_world=old.world_id, world=model.world_id, before=before, after=after)

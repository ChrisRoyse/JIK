"""One cold read of a sealed native bank and explicit unit-disjoint training plan."""
from contextlib import ExitStack
from pathlib import Path

from senses.contract import digest, require, validate_record
from senses.trajectory_population import read_pinned
from prefix_panel import load as load_panel
from model.contracts import Instrument, ModelConfig
from model.native import from_arrays
from production.native_rows import mapped_rows, one, value_hash
from model.train_sensory import byte_targets
from model.retention import validate_split

SCHEMA = "sensory-continual-plan-v1"


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {"schema", "purpose", "panel_sha256", "records_sha256",
                "rows", "channels", "model", "prefix_tokens", "seed", "base_steps", "extension_steps", "learning_rate"}
            and plan["schema"] == SCHEMA and plan["purpose"] == "engineering_retention_boundary",
            "MNLM_RETENTION_PLAN", "pin an engineering experiment before training")
    validate_split(plan["rows"])
    config = ModelConfig(**plan["model"])
    config.validate()
    require(config.vocabulary == 258 and type(plan["prefix_tokens"]) is int
            and 0 < plan["prefix_tokens"] <= config.max_prefix
            and type(plan["seed"]) is int and 0 <= plan["seed"] < 2**32
            and all(type(plan[k]) is int and 0 < plan[k] <= 1000 for k in ("base_steps", "extension_steps"))
            and type(plan["learning_rate"]) in (int, float) and 0 < plan["learning_rate"] <= .01
            and isinstance(plan["channels"], list) and len(plan["channels"]) >= 6
            and len(set(plan["channels"])) == len(plan["channels"]),
            "MNLM_RETENTION_PLAN", "bounded recipe and five base instruments plus one extension required")
    return config


def prepare(panel, records_path, plan):
    config = validate_plan(plan)
    manifest, rows = load_panel(panel, plan["panel_sha256"])
    records = read_pinned(records_path, plan["records_sha256"])
    indexed = {r["atom"]: r for r in rows}
    require(all(c in manifest["slots"] for c in plan["channels"]),
            "MNLM_RETENTION_BINDING", "declared channel not in frozen bank")
    specs = tuple(Instrument.from_slot(manifest["slots"][c]["contract"]["name"], manifest["slots"][c])
                  for c in plan["channels"])
    batches, receipts = [], []
    for r in plan["rows"]:
        atom = r["atom"]
        require(atom in indexed and atom in records, "MNLM_RETENTION_BINDING", "missing declared observation")
        current = validate_record(records[atom])
        binding = indexed[atom]["binding"]
        require(current["atom_id"] == atom and current["unit"] == r["unit"]
                and binding["current"] == atom and binding["future_visible"] is False
                and binding["outcomes_visible"] is False,
                "MNLM_RETENTION_BINDING", "native/source unit, cutoff or output exclusions differ")
        past = binding["visible_past"]
        require(all(k in records and records[k]["unit"] == r["unit"] for k in past),
                "MNLM_RETENTION_BINDING", "past observation crosses source units")
        text = " ".join(item["text"] for item in current["channels"]["surface"]["items"])
        prefix, targets = byte_targets([{"text": text}], plan["prefix_tokens"], "cpu")
        batches.append({**r, "native": {}, "prefix": prefix, "targets": targets})
    # Header maps and row positions are invariant across the row loop (PC-07).
    positions = {r["atom"]: i for i, r in enumerate(rows)}
    total = 0
    with ExitStack() as stack:
        arrays = {c: stack.enter_context(mapped_rows(Path(panel) / (c + ".npz"))) for c in plan["channels"]}
        for b in batches:
            i = positions[b["atom"]]
            for c, spec in zip(plan["channels"], specs):
                a = arrays[c]
                width = max(1, int(a["offsets"][i + 1] - a["offsets"][i]))
                total += width * spec.dims * 4
                require(width <= config.max_native_tokens and total <= 64 << 20,
                        "MNLM_RETENTION_INPUT_BUDGET", "native extent/cache exceeds fixed budget; no truncation")
                selected = one(a, i)
                b["native"][spec.name] = from_arrays(spec, selected, [b["atom"]], config)
                receipts.append({"instrument": spec.identity, "atom": b["atom"],
                                 "native_value_sha256": value_hash(selected), "tokens": width})
    return config, specs, batches, {"native_value_bytes": total, "slots": receipts,
                                    "source_sha256": {"panel": plan["panel_sha256"], "records": plan["records_sha256"]}}


def on_device(batches, device):
    return [{**b, "native": {n: x.to(device) for n, x in b["native"].items()},
             "prefix": b["prefix"].to(device), "targets": b["targets"].to(device)} for b in batches]

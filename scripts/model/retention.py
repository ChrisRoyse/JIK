"""Finite held-out behavioral gate for engineering decoder updates, never CMI.

Frozen old weights do not imply preserved behavior with a new branch active.
Every retained unit must avoid regression; averaging cannot hide a harmed unit.
This gate certifies only the declared finite benchmark, not universal retention.
"""
import math

import torch

from senses.contract import canonical, digest, require
from model.checkpoint import state_hash

ROLES = ("old_train", "new_train", "retention", "acquisition")
POLICY = {"metric": "per_unit_mean_target_log_loss_bits", "retention_max_increase": 0,
          "acquisition": "strict_mean_improvement", "scope": "declared_finite_benchmark_only"}


def validate_split(rows):
    require(isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows),
            "MNLM_RETENTION_SPLIT", "declare training and held-out units before fitting")
    require(all(set(r) == {"atom", "unit", "role"} and r["role"] in ROLES
                and all(isinstance(r[k], str) and r[k] for k in ("atom", "unit")) for r in rows)
            and {r["role"] for r in rows} == set(ROLES)
            and len({r["atom"] for r in rows}) == len(rows),
            "MNLM_RETENTION_SPLIT", "all four roles and unique observation identities required")
    roles = {}
    for r in rows:
        roles.setdefault(r["unit"], set()).add(r["role"])
    require(all(len(v) == 1 for v in roles.values()), "MNLM_RETENTION_LEAKAGE",
            "a source unit or its siblings cannot cross training/held-out roles")


def tensor_hash(t):
    import hashlib
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(canonical([a.dtype.str, list(a.shape)]) + a.tobytes()).hexdigest()


def batch_hash(batches):
    return digest([{**{k: b[k] for k in ("atom", "unit", "role")},
                    "prefix": tensor_hash(b["prefix"]), "targets": tensor_hash(b["targets"]),
                    "native": {n: {"identity": x.identity, "rows": x.rows,
                        "tensors": [None if t is None else tensor_hash(t)
                                    for t in (x.values, x.padding, x.absent, x.indices)]}
                               for n, x in b["native"].items()}} for b in batches])


def evaluate(model, batches):
    """Use every registered branch and the original unmodified observations."""
    before, results = state_hash(model), []
    with torch.inference_mode():
        for b in batches:
            if b["role"] not in ("retention", "acquisition"):
                continue
            native = {s.name: b["native"][s.name] for s in model.specs}
            require(all(x.rows == (b["atom"],) for x in native.values()),
                    "MNLM_RETENTION_BINDING", "held-out input rows changed")
            logits = model(b["prefix"], native)["logits"]
            target = b["targets"]
            require(target.shape == logits.shape[:2] and target.dtype == torch.int64
                    and target.device == logits.device
                    and bool(((target == -100) | ((target >= 0) & (target < logits.shape[-1]))).all())
                    and bool((target != -100).any()), "MNLM_RETENTION_TARGET", "invalid held-out targets")
            valid = target != -100
            logp = logits.double().log_softmax(-1)
            losses = -logp.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)[valid] / math.log(2)
            require(bool(torch.isfinite(losses).all()), "MNLM_RETENTION_NONFINITE", "nonfinite held-out loss")
            results.append({"atom": b["atom"], "unit": b["unit"], "role": b["role"],
                            "tokens": losses.numel(), "loss_bits": losses.cpu().tolist(),
                            "logits_sha256": tensor_hash(logits)})
    require(before == state_hash(model), "MNLM_RETENTION_MUTATION", "evaluation changed model state")
    return results


def decide(plan_rows, baseline, candidate):
    validate_split(plan_rows)
    expected = {r["atom"]: r for r in plan_rows if r["role"] in ("retention", "acquisition")}
    maps = []
    for measurements in (baseline, candidate):
        require(isinstance(measurements, list) and len(measurements) == len(expected)
                and {r.get("atom") for r in measurements} == set(expected),
                "MNLM_RETENTION_COVERAGE", "measure every declared held-out observation exactly once")
        mapped = {}
        for r in measurements:
            require(all(r.get(k) == expected[r["atom"]][k] for k in ("atom", "unit", "role"))
                    and type(r.get("tokens")) is int and r["tokens"] > 0
                    and isinstance(r.get("loss_bits"), list) and len(r["loss_bits"]) == r["tokens"],
                    "MNLM_RETENTION_BINDING", "metric does not bind to declared unit/target extent")
            require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in r["loss_bits"]),
                    "MNLM_RETENTION_NONFINITE", "finite nonnegative per-token losses required")
            mapped[r["atom"]] = r
        maps.append(mapped)
    units = {}
    for atom, declared in expected.items():
        a, b = (m[atom] for m in maps)
        require(a["tokens"] == b["tokens"], "MNLM_RETENTION_BINDING", "target extent changed between versions")
        u = units.setdefault(declared["unit"], {"role": declared["role"], "baseline": [], "candidate": []})
        u["baseline"].extend(a["loss_bits"])
        u["candidate"].extend(b["loss_bits"])
    metrics = [{"unit": unit, "role": u["role"], "tokens": len(u["baseline"]),
                "baseline_bits": math.fsum(u["baseline"]) / len(u["baseline"]),
                "candidate_bits": math.fsum(u["candidate"]) / len(u["candidate"])}
               for unit, u in sorted(units.items())]
    for m in metrics:
        m["delta_bits"] = m["candidate_bits"] - m["baseline_bits"]
    regressions = [m["unit"] for m in metrics if m["role"] == "retention" and m["delta_bits"] > 0]
    gains = [m["delta_bits"] for m in metrics if m["role"] == "acquisition"]
    delta = math.fsum(gains) / len(gains)
    reasons = (["MNLM_RETENTION_REGRESSION"] if regressions else [])
    if delta >= 0:
        reasons.append("MNLM_RETENTION_NO_ACQUISITION_GAIN")
    return {"policy": POLICY, "eligible_on_finite_benchmark": not reasons, "reasons": reasons,
            "regressed_units": regressions, "acquisition_mean_delta_bits": delta, "units": metrics,
            "production_admission": False, "meaning_asserted": False, "stage4_accepted": False,
            "population_retention_proven": False}


def assess(base, candidate, base_manifest, candidate_manifest, plan, batches):
    require(candidate_manifest["parent"] == digest(base_manifest)
            and base.config == candidate.config and candidate.specs[:-1] == base.specs,
            "MNLM_RETENTION_LINEAGE", "candidate must extend this exact parent by one instrument")
    require(base_manifest["input_sha256"] == candidate_manifest["input_sha256"]
            == digest({"plan": plan, "batch_sha256": batch_hash(batches)}),
            "MNLM_RETENTION_BINDING", "source, split, targets or native measurements changed")
    require(base_manifest["state_sha256"] == state_hash(base)
            and candidate_manifest["state_sha256"] == state_hash(candidate),
            "MNLM_RETENTION_LINEAGE", "evaluate exact cold checkpoint states")
    old, new = base.state_dict(), candidate.state_dict()
    require(all(torch.equal(t, new[n]) for n, t in old.items()), "MNLM_RETENTION_OLD_WEIGHTS",
            "this frozen-extension contract cannot change earlier parameters")
    baseline = evaluate(base, batches)
    measured = evaluate(candidate, batches)
    decision = decide(plan["rows"], baseline, measured)
    return {"schema": "sensory-retention-assessment-v1", "plan_sha256": digest(plan),
            "baseline_sha256": digest(base_manifest), "candidate_sha256": digest(candidate_manifest),
            "selected_checkpoint_sha256": digest(candidate_manifest if decision["eligible_on_finite_benchmark"]
                                                   else base_manifest),
            "baseline": baseline, "candidate": measured, "decision": decision}

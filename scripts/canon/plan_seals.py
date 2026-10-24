"""Complete extraction-plan publication over append-only shard manifests (#19).

Plan history is grouped once per pass. A seal binds the exact history and member
rows, so interrupted manifest appends cannot look like a complete archive.
"""
import hashlib
import json

from contracts import ContractError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def plan_id(row):
    return row.get("plan_shard", row["shard"])


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(plan_id(row), []).append(row)
    return groups


def make_seal(history, results):
    members = sorted(r["shard"] for r in results)
    if not results or any(r.get("error") for r in results) or len(set(members)) != len(members):
        raise ContractError("MNLM_CANON_PLAN_UNSEALED", "cannot publish failed, empty or duplicate-member plan")
    current = {r["shard"]: r for r in history}
    body = {"schema": "canon-plan-seal-v1", "plan_shard": plan_id(results[0]),
            "history_sha256": digest(history), "members": members,
            "members_sha256": digest([current[m] for m in members])}
    return {**body, "sha256": digest(body)}


def accepted(history, seal=None):
    """None means incomplete; callers may rebuild, but readers must refuse it."""
    current = {r["shard"]: r for r in history}
    if not history:
        return None
    # Legacy shard hashes prove individual content, not that an archive was
    # fully extracted. Rebuild/recertify the named plan to obtain a completion seal.
    if not isinstance(seal, dict) or seal.get("schema") != "canon-plan-seal-v1":
        return None
    body = {k: v for k, v in seal.items() if k != "sha256"}
    members = seal.get("members")
    if (seal.get("sha256") != digest(body) or seal.get("history_sha256") != digest(history)
            or seal.get("plan_shard") != plan_id(history[0]) or not isinstance(members, list)
            or not all(isinstance(m, str) for m in members) or len(set(members)) != len(members)):
        return None
    rows = [current[m] for m in members if m in current]
    if (not rows or len(rows) != len(members) or any(r.get("error") for r in rows)
            or {r["shard"] for r in current.values() if not r.get("error")} != set(members)
            or seal.get("members_sha256") != digest(rows)):
        return None
    return rows


def read_seals(directory):
    from common import read_jsonl
    seals = {}
    for row in read_jsonl(directory / "plans.jsonl"):
        if not isinstance(row, dict) or not isinstance(row.get("plan_shard"), str):
            raise ContractError("MNLM_CANON_PLAN_UNSEALED", "malformed extraction plan seal")
        seals[row["plan_shard"]] = row
    return seals

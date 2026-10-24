"""Frozen input scope for bounded inference, before output hashes exist (#16).

An engineering plan never grants production admission. Runtime, source, policy,
row and instrument drift require a new plan; checkpoints cannot be relabelled.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re

from senses.contract import canonical, digest, require
from senses.scope import check_source
from training_scope import POLICY_SHA256
from production.gate import require_admitted

BASE = Path(__file__).resolve().parents[1]
CODE = ("production/input_plan.py", "production/session.py", "production/stream_slots.py",
        "production/atom_cache.py", "production/cached_session.py", "production/native_rows.py",
        "production/stream.py", "production/publication.py", "production/gate.py",
        "lens/learned.py", "lens/learned_ucr.py", "lens/text_windows.py", "lens/universal.py", "lens/slotio.py",
        "lens/slot_layout.py", "lens/audio_encoder.py", "lens/audio_ucr.py", "lens/lenscommon.py",
        "lens/catalogue.py", "lens/measure.py", "lens/image_ucr.py", "senses/pixels.py",
        "lens/vision_encoder.py", "lens/vision_assets.py", "lens/vision_ucr.py",
        "senses/contract.py", "senses/views.py", "senses/store.py", "senses/blobs.py",
        "senses/instruments.py", "senses/scope.py", "canon/training_scope.py")
KEY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
TIERS = {"deterministic", "cheap-semantic", "panel", "multi"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def environment():
    versions = {}
    for name in ("numpy", "torch", "transformers", "scipy", "soundfile", "blake3", "tokenizers", "Pillow", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def code_hashes():
    return {n: sha(BASE / n) for n in CODE}


def positive(n):
    return type(n) is int and n > 0


def validate(plan, admission_root=None, ledger=None):
    require(plan.get("schema") == "tier-input-plan-v1", "MNLM_STREAM_PLAN", "unknown input schema")
    require(plan.get("purpose") in ("engineering_boundary", "production"), "MNLM_STREAM_PLAN", "purpose required")
    if plan["purpose"] == "production":
        require_admitted(admission_root, ledger, plan.get("required_native_media"))
        require(False, "MNLM_STAGE4_SCOPE_UNCOMMISSIONED", "accepted revised atom/tier scope is still required")
    require(plan.get("storage") == "raw-f32", "MNLM_MEASURE_QUANTIZER", "raw native slots only")
    require(plan.get("training_scope_sha256") == POLICY_SHA256, "MNLM_MEASURE_SCOPE", "source policy changed")
    require(plan.get("code_sha256") == code_hashes(), "MNLM_STREAM_CODE_DRIFT", "freeze a new executable plan")
    require(plan.get("environment") == environment(), "MNLM_STREAM_RUNTIME_DRIFT", "runtime differs from input plan")
    for name in ("batch_rows", "slot_budget_bytes", "metadata_budget_bytes", "blob_budget_bytes", "work_budget_bytes"):
        require(positive(plan.get(name)), "MNLM_STREAM_PLAN", f"positive {name} required")
    require(isinstance(plan.get("generation"), str) and HEX.fullmatch(plan["generation"]),
            "MNLM_STREAM_PLAN", "exact source generation required")
    instruments, shards = plan.get("instruments"), plan.get("shards")
    require(isinstance(instruments, dict) and bool(instruments) and isinstance(shards, dict) and bool(shards),
            "MNLM_STREAM_PLAN", "named instrument and shard scope required")
    for name, instrument in instruments.items():
        require(KEY.fullmatch(name) and instrument["spec"]["name"] == name
                and instrument["spec"].get("module") in ("universal", "learned_ucr", "audio_ucr", "image_ucr", "vision_ucr")
                and instrument["spec"].get("inputs") == "ucr", "MNLM_STREAM_INSTRUMENT", "unsupported UCR instrument")
        require(HEX.fullmatch(instrument.get("weights_sha256", "")), "MNLM_STREAM_INSTRUMENT", "frozen weights required")
        require(instrument.get("tier") in TIERS, "MNLM_STREAM_TIER", "unknown tier")
        require(positive(instrument.get("slot_budget_bytes")) and instrument["slot_budget_bytes"] <= plan["slot_budget_bytes"],
                "MNLM_MEASURE_STORAGE_BUDGET", "instrument must fit the frozen slot ceiling")
    modules = {i["spec"]["module"] for i in instruments.values()}
    needed = {"numpy", "blake3"}
    if modules & {"learned_ucr", "audio_ucr"}:
        needed.update(("torch", "transformers", "scipy", "tokenizers"))
    if "audio_ucr" in modules:
        needed.add("soundfile")
    if modules & {"image_ucr", "vision_ucr"}:
        needed.add("Pillow")
    if "vision_ucr" in modules:
        needed.update(("torch", "transformers", "safetensors", "scipy", "tokenizers"))
    missing = sorted(n for n in needed if plan["environment"]["packages"].get(n) is None)
    require(not missing, "MNLM_STREAM_RUNTIME_MISSING",
            f"install the declared runtime dependencies before freezing a new plan: {missing}; audio: config/requirements-audio.txt")
    planned, row_bytes, slot_bytes, seen = 0, 0, 0, set()
    for key, shard in shards.items():
        require(KEY.fullmatch(key) and positive(shard.get("n")) and shard.get("family"),
                "MNLM_STREAM_PLAN", "invalid shard identity/count/policy")
        require(shard.get("tier") in TIERS and isinstance(shard.get("instruments"), list)
                and bool(shard["instruments"]) and len(set(shard["instruments"])) == len(shard["instruments"]),
                "MNLM_STREAM_TIER", "separate, unique instruments required")
        require(all(n in instruments and instruments[n]["tier"] == shard["tier"] for n in shard["instruments"]),
                "MNLM_STREAM_TIER", "shard instrument outside assigned tier")
        require(HEX.fullmatch(shard.get("rows_sha256", "")) and positive(shard.get("rows_bytes")),
                "MNLM_STREAM_PLAN", "input byte count and digest required")
        planned += len(shard["instruments"])
        slot_bytes += sum(instruments[n]["slot_budget_bytes"] for n in shard["instruments"])
        row_bytes += shard["rows_bytes"]
        seen.update(shard["instruments"])
    require(seen == set(instruments), "MNLM_STREAM_PLAN", "unused instrument in frozen plan")
    require(plan["blob_budget_bytes"] >= row_bytes + slot_bytes + planned * plan["metadata_budget_bytes"],
            "MNLM_MEASURE_STORAGE_BUDGET", "content budget smaller than fixed component reservations")
    # Staging plus final content copies, one raw+temporary slot, and bounded metadata.
    reserve = 2 * plan["blob_budget_bytes"] + 2 * plan["slot_budget_bytes"] + (planned + 4) * plan["metadata_budget_bytes"]
    if "cache" in plan:
        cache = plan["cache"]
        require(isinstance(cache, dict) and set(cache) == {"schema", "max_db_bytes"}
                and cache["schema"] == "atom-cache-v1" and positive(cache["max_db_bytes"])
                and cache["max_db_bytes"] >= 65536 and cache["max_db_bytes"] % 4096 == 0,
                "MNLM_CACHE_CONFIG", "fixed SQLite page budget required")
        reserve += 2 * cache["max_db_bytes"] + 4 * plan["metadata_budget_bytes"]
    require(reserve <= plan["work_budget_bytes"], "MNLM_MEASURE_STORAGE_BUDGET", "work budget excludes retained/scratch bytes")
    return planned


def read_rows(path, shard):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == shard["rows_bytes"], "MNLM_STREAM_INPUT", "row file size changed")
    raw = path.read_bytes()  # one explicitly byte-bounded shard, never the corpus
    require(hashlib.sha256(raw).hexdigest() == shard["rows_sha256"], "MNLM_STREAM_INPUT", "row digest changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == shard["n"] and [r.get("i") for r in rows] == list(range(len(rows))),
            "MNLM_MEASURE_ROWS", "row count/order mismatch")
    keys = []
    for r in rows:
        check_source(r["source"])
        require(isinstance(r.get("atom"), str) and HEX.fullmatch(r["atom"]), "MNLM_STREAM_INPUT", "physical atom required")
        keys.append(r["atom"])
    require(len(set(keys)) == len(keys), "MNLM_MEASURE_ROWS", "duplicate atom within shard")
    return rows, raw


def input_bindings(plan, reader, rows_root, blobs=None):
    require(reader.generation == plan["generation"] and reader.policy_hash == plan["policies_sha256"],
            "MNLM_STREAM_INPUT", "source generation or policy changed")
    reader.store.verify_chain()
    seen, checked_blobs = set(), set()
    for key, shard in plan["shards"].items():
        rows, _ = read_rows(Path(rows_root) / (key + ".jsonl"), shard)
        policy = reader.policy(shard["family"])
        for row in rows:
            view, _, binding = reader.read(row, policy)
            require(binding["atom"] == row["atom"], "MNLM_STREAM_INPUT", "locator resolves to a different atom")
            # The same atom may appear in different tiers, never twice for one instrument.
            for name in shard["instruments"]:
                item = (row["atom"], name)
                require(item not in seen, "MNLM_STREAM_DUPLICATE", "atom/instrument is planned more than once")
                seen.add(item)
            modalities = {"audio" if plan["instruments"][n]["spec"]["module"] == "audio_ucr" else "image"
                          for n in shard["instruments"] if plan["instruments"][n]["spec"]["module"] in ("audio_ucr", "image_ucr", "vision_ucr")}
            if modalities:
                for percept in view["channels"]["perception"]["items"]:
                    if percept["modality"] in modalities and percept["sha256"] not in checked_blobs:
                        require(blobs is not None, "MNLM_UCR_BLOBS_REQUIRED", "perception plan needs retained source bytes")
                        blobs.read(percept["sha256"], percept["media_type"])
                        checked_blobs.add(percept["sha256"])
    return len(seen)

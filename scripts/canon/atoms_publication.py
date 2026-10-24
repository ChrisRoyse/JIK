"""Stage 2 publication binds the expected jobs before work and seals completion last.

O(job descriptors + parent bytes + atom bytes); canonical manifests cached once
per source, canonical data hashed by the writer's existing streaming pass.
"""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import blake3
import numpy as np

from artifact_seals import Audit
from common import manifest_rows, read_jsonl
from training_scope import POLICY_SHA256, require_training_source


def require(ok, code, detail):
    if not ok:
        from atomise import AtomError
        raise AtomError(code, f"{detail}; rebuild Stage 2 from accepted Stage 1 inputs in a fresh tree")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def identity(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha(path):
    try:
        with Path(path).open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError as exc:
        require(False, "MNLM_ATOMS_UNSEALED", f"{path}: {exc}")


def contract():
    from atomise import CONTRACT_ID, CODE_ATOMIZER_ID, SEGMENTER_ID, PY_VERSION, WRITER_SHA256
    return dict(contract=CONTRACT_ID, writer_sha256=WRITER_SHA256, code_atomizer=CODE_ATOMIZER_ID,
                segmenter=SEGMENTER_ID, python=PY_VERSION)


def check_contract(row):
    require(all(row.get(k) == v for k, v in contract().items()), "MNLM_ATOMS_CONTRACT", "stale or missing parser identity")


def component(value):
    require(isinstance(value, str) and value not in ("", ".", "..") and not any(c in value for c in '/\\:'),
            "MNLM_ATOMS_PLAN", "unsafe source/shard component")
    return value


def descriptor(job):
    rows = job["unique_rows"]
    require(all(type(r) is int and 0 <= r <= np.iinfo(np.int32).max for r in rows) and len(rows) == len(set(rows)),
            "MNLM_ATOMS_ROWS", "selected rows must be unique nonnegative int32 identities")
    require(set(map(int, job["cuts"])) <= set(rows), "MNLM_ATOMS_ROWS", "cut outside selected rows")
    return dict(source=component(job["source"]), shard=component(job["shard"]), docs=len(rows),
                rows_sha256=identity(sorted(rows)), cuts_sha256=identity({str(k): v for k, v in job["cuts"].items()}),
                canon_blake3=job.get("canon_blake3"))


def sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish(path, body):
    obj = {**body, "id": identity(body)}
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("xb") as f:
            f.write(canonical(obj) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # no overwrite, including a racing publisher
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return obj


def read_seal(path):
    try:
        obj = json.loads(path.read_bytes())
        require(obj.get("id") == identity({k: v for k, v in obj.items() if k != "id"}),
                "MNLM_ATOMS_SEAL", str(path))
        return obj
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        require(False, "MNLM_ATOMS_UNSEALED", f"{path}: {exc}")


def begin(paths, jobs, stage1):
    adir = paths.out / "atoms" / paths.version
    require(jobs and not (adir.exists() and any(adir.iterdir())), "MNLM_ATOMS_TREE_EXISTS", "empty new atom tree required")
    require(stage1 == Audit(paths).require("spans"), "MNLM_ATOMS_PARENT", "Stage 1 changed since planning")
    manifests, expected, keys = {}, [], set()
    for job in jobs:
        d = descriptor(job)
        require_training_source(d["source"])
        key = (d["source"], d["shard"])
        require(key not in keys, "MNLM_ATOMS_PLAN", f"duplicate job {key}")
        keys.add(key)
        require(Path(job["out_dir"]).resolve() == (adir / d["source"]).resolve()
                and Path(job["canon_dir"]).resolve() == paths.canon(d["source"]).resolve(),
                "MNLM_ATOMS_PLAN", f"job path outside generation: {key}")
        if d["source"] not in manifests:
            manifests[d["source"]] = manifest_rows(paths.canon(d["source"]))
        m = manifests[d["source"]].get(d["shard"], {})
        require(d["canon_blake3"] and m.get("out_blake3") == d["canon_blake3"],
                "MNLM_ATOMS_CANON_CHANGED", str(key))
        expected.append({**d, "canonical_manifest": m})
    body = dict(schema="atoms-plan-v1", version=paths.version, canon_version=paths.canon_version,
                policy=POLICY_SHA256, stage1=stage1, parser=contract(), jobs=expected)
    adir.mkdir(parents=True, exist_ok=True)
    return publish(adir / "plan.json", body)


def inventory(paths, plan):
    adir = paths.out / "atoms" / paths.version
    require(plan.get("schema") == "atoms-plan-v1" and plan.get("version") == paths.version
            and plan.get("canon_version") == paths.canon_version and plan.get("policy") == POLICY_SHA256,
            "MNLM_ATOMS_PLAN", "generation or training policy mismatch")
    require(plan.get("parser") == contract(), "MNLM_ATOMS_CONTRACT", "generation parser changed")
    require(plan.get("stage1") == Audit(paths).require("spans"), "MNLM_ATOMS_PARENT", "Stage 1 parent changed")
    expected = {(j["source"], j["shard"]): j for j in plan["jobs"]}
    rows = read_jsonl(adir / "manifest.jsonl")
    keys = [(r.get("source"), r.get("shard")) for r in rows]
    require(expected and len(expected) == len(plan["jobs"]) and len(keys) == len(set(keys)) and set(keys) == set(expected),
            "MNLM_ATOMS_INCOMPLETE", "manifest does not equal the complete expected shard set")
    files, selected = {}, []
    for r in rows:
        check_contract(r)
        source, shard = component(r["source"]), component(r["shard"])
        require_training_source(source)
        d = expected[(source, shard)]
        require(not r.get("error") and all(r.get(k) == d[k] for k in ("docs", "rows_sha256", "cuts_sha256", "canon_blake3")),
                "MNLM_ATOMS_INCOMPLETE", f"incomplete or different job {source}/{shard}")
        ap = adir / source / (shard + ".atoms.npz")
        check_arrays(ap, r)
        files[f"{source}/{shard}.atoms.npz"] = sha(ap)
        selected.append(dict(source=source, shard=shard, canon=str(paths.canon(source) / (shard + ".jsonl.zst")),
            atoms=str(ap), atoms_sha256=files[f"{source}/{shard}.atoms.npz"],
            manifest=d["canonical_manifest"], atom_manifest=r))
    return files, selected


def check_arrays(path, row):
    from atomise import ARRAYS
    types = dict(row="int32", start="int32", end="int32", role="uint8", unit="uint8", cut_of="int64")
    try:
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == set(ARRAYS), "MNLM_ATOMS_ARRAYS", "array inventory")
            arrays = {k: z[k] for k in ARRAYS}
        require(all(arrays[k].ndim == 1 and arrays[k].dtype == np.dtype(types[k]) and len(arrays[k]) == row["atoms"] for k in ARRAYS),
                "MNLM_ATOMS_ARRAYS", "array shape/dtype/count")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        require(False, "MNLM_ATOMS_ARRAYS", f"{path}: {exc}")
    h = blake3.blake3()
    for k in ARRAYS:
        h.update(arrays[k].tobytes())
    require(h.hexdigest() == row.get("arrays_blake3"), "MNLM_ATOMS_ARRAYS", "array digest")
    require(bool((arrays["row"] >= 0).all()) and bool((arrays["row"][1:] >= arrays["row"][:-1]).all())
            and bool(((arrays["role"] >= 1) & (arrays["role"] <= 7)).all())
            and bool((arrays["unit"] <= 5).all()) and bool((arrays["cut_of"] >= -1).all())
            and bool(((arrays["role"] == 4) == (arrays["cut_of"] >= 0)).all()),
            "MNLM_ATOMS_ARRAYS", "row order or role/unit/cut identity")
    return arrays


def finish(paths):
    adir = paths.out / "atoms" / paths.version
    plan = read_seal(adir / "plan.json")
    files, _ = inventory(paths, plan)
    for rel in (*files, "manifest.jsonl", "atoms.summary.json"):
        with (adir / rel).open("r+b") as f:
            os.fsync(f.fileno())
        sync_dir((adir / rel).parent)
    return publish(adir / "complete.json", dict(schema="atoms-complete-v1", plan=plan["id"],
        manifest_sha256=sha(adir / "manifest.jsonl"), summary_sha256=sha(adir / "atoms.summary.json"), files=files))


def accepted(paths):
    adir = paths.out / "atoms" / paths.version
    seal = read_seal(adir / "complete.json")
    plan = read_seal(adir / "plan.json")
    require(seal.get("schema") == "atoms-complete-v1" and seal.get("plan") == plan["id"]
            and seal.get("manifest_sha256") == sha(adir / "manifest.jsonl")
            and seal.get("summary_sha256") == sha(adir / "atoms.summary.json"),
            "MNLM_ATOMS_SEAL", "completion metadata changed")
    files, selected = inventory(paths, plan)
    require(seal.get("files") == files, "MNLM_ATOMS_SEAL", "completed atom bytes changed")
    return seal, selected

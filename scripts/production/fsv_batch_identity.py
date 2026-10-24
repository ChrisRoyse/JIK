"""Measure atom identity across real batch layouts before enabling cache reuse.

This is an engineering diagnostic, not a lens-admission assay. All native values
and window traces are retained; no tolerance turns a changed value into equality.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical
from production.input_plan import code_hashes, environment, sha
from production.session import Session
from universal import ChannelReader
from lenscommon import take_lease, release_lease, LEASE

LAYOUTS = {"whole": [[0, 1, 2, 3]], "repeat": [[0, 1, 2, 3]],
           "single": [[0], [1], [2], [3]], "pairs": [[0, 1], [2, 3]], "reverse": [[3, 2, 1, 0]]}


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def one(arrays, i):
    result = {k: arrays[k][i:i + 1].copy() for k in ("absent", "windows")}
    if "X" in arrays:
        result["X"] = arrays["X"][i:i + 1].copy()
    else:
        off = "offsets" if "tokens" in arrays else "indptr"
        start, stop = map(int, arrays[off][i:i + 2])
        result[off] = np.asarray([0, stop - start], np.int64)
        for key in (("tokens",) if off == "offsets" else ("data", "indices")):
            result[key] = arrays[key][start:stop].copy()
        if off == "indptr":
            result["shape"] = np.asarray([1, arrays["shape"][1]], np.int64)
    return result


def unpadded(detail):
    out = copy.deepcopy(detail)
    for w in out["text_windows"]["windows"]:
        selected = [i for i, m in enumerate(w["attention_mask"]) if m]
        for key in ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping", "token_type_ids"):
            if key in w:
                w[key] = [w[key][i] for i in selected]
    return out


def compare(a, b):
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        layout = set(left.files) == set(right.files) and all(
            left[k].shape == right[k].shape and left[k].dtype == right[k].dtype for k in left.files)
        equal = layout and all(left[k].tobytes() == right[k].tobytes() for k in left.files)
        error = max((float(np.abs(left[k] - right[k]).max(initial=0)) for k in left.files
                     if left[k].dtype.kind == "f"), default=0.0) if layout else None
    x, y = [json.loads(p.with_suffix(".json").read_bytes()) for p in (a, b)]
    return {"native_layout_identical": layout, "native_byte_identical": equal, "max_absolute_error": error,
            "view_binding_identical": x["binding"] == y["binding"],
            "window_record_identical": x["detail"] == y["detail"],
            "unpadded_window_record_identical": unpadded(x["detail"]) == unpadded(y["detail"])}


def probe_controls(root):
    """Counterexamples at the comparator/identity boundary, never training data."""
    from measure import probe_equal, lens_id
    import scipy.sparse as sp
    dense = root / "ucr-all-minilm-l6-v2"
    with np.load(dense / "whole" / "0.npz", allow_pickle=False) as z:
        left, right = z["X"].copy(), z["X"].copy()
    left[0, 0], right[0, 0] = 0.0, -0.0
    signed_zero = not probe_equal(left, right)
    right[0, 0] = np.nan
    nonfinite = not probe_equal(right, right)
    sparse = root / "ucr-opensearch-sparse-doc-v2" / "whole" / "0.npz"
    with np.load(sparse, allow_pickle=False) as z:
        a = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        b = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(1, a.shape[1] + 1))
        sparse_shape = not probe_equal(a, b)
    contract = json.loads((dense / "contract.json").read_bytes())["contract"]
    old = {k: v for k, v in contract.items() if k not in ("execution_unit", "forward_token_budget")}
    checks = {"signed_zero_refuses": signed_zero, "nonfinite_refuses": nonfinite,
              "sparse_shape_refuses": sparse_shape, "truncated_native_tuple_refuses": not probe_equal((left, left), (left,)),
              "execution_contract_changes_identity": lens_id(old) != lens_id(contract)}
    assert all(checks.values())
    write(root / "probe-controls.json", {"artificial_boundary_only": True, "checks": checks})
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-evidence", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-independent", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start, phases = time.monotonic(), {}
    plan = json.loads((a.input_evidence / "plan.json").read_bytes())
    policies = a.input_evidence / "policies.json"
    rows = [json.loads(line) for line in (a.input_evidence / "rows" / "window-0.jsonl").read_bytes().splitlines()][:4]
    before = sha(a.store)
    reader = ChannelReader(a.store, policies, generation=plan["generation"])
    code = code_hashes()
    write(a.out / "input.json", {"rows": rows, "source_sha256": before, "generation": reader.generation,
                                "policy_sha256": sha(policies), "code_sha256": code, "environment": environment(),
                                "layouts": LAYOUTS, "diagnostic_only": True})
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    write(a.out / "resident-before.json", {"nvidia_smi": resident})
    phases["preflight"] = time.monotonic() - start
    take_lease("16", "cross-plan batch identity diagnostic")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    phases["gpu_setup"] = time.monotonic() - start - phases["preflight"]
    phases["load"] = phases["measure_and_persist"] = phases["frozen_state"] = 0.0
    findings = []
    try:
        for name, item in plan["instruments"].items():
            at = time.monotonic()
            session = Session(item, reader, code)
            phases["load"] += time.monotonic() - at
            root = a.out / name
            root.mkdir()
            write(root / "contract.json", {"lens_id": session.lens_id, "contract": session.contract})
            try:
                at = time.monotonic()
                for label, groups in LAYOUTS.items():
                    dest = root / label
                    dest.mkdir()
                    for group in groups:
                        arrays, bindings, details = session.batch([rows[i] for i in group], reader.policy("window-boundary"))
                        for j, original in enumerate(group):
                            with (dest / f"{original}.npz").open("xb") as f:
                                np.savez(f, **one(arrays, j))
                            write(dest / f"{original}.json", {"binding": bindings[j], "detail": details[j]})
                phases["measure_and_persist"] += time.monotonic() - at
            finally:
                at = time.monotonic()
                session.close()
                phases["frozen_state"] += time.monotonic() - at
            at = time.monotonic()
            for label in LAYOUTS:
                if label == "whole":
                    continue
                for i in range(len(rows)):
                    facts = compare(root / "whole" / f"{i}.npz", root / label / f"{i}.npz")
                    findings.append({"instrument": name, "layout": label, "atom": rows[i]["atom"], **facts})
                    if label == "repeat":
                        assert facts["native_byte_identical"] and facts["window_record_identical"]
            phases["cold_comparison"] = phases.get("cold_comparison", 0.0) + time.monotonic() - at
    finally:
        reader.close()
        release_lease()
    assert sha(a.store) == before and not LEASE.exists()
    at = time.monotonic()
    controls = probe_controls(a.out)
    phases["probe_controls"] = time.monotonic() - at
    report = {"acceptance": "author-checked only", "stage4_accepted": False, "findings": findings,
              "probe_controls": controls,
              "all_layouts_byte_identical": all(f["native_byte_identical"] and f["window_record_identical"] for f in findings),
              "source_before": before, "source_after": sha(a.store), "phases_seconds": phases,
              "wall_seconds": time.monotonic() - start}
    write(a.out / "report.json", report)
    write(a.out / "SHA256.json", {str(p.relative_to(a.out)): sha(p) for p in sorted(a.out.rglob("*")) if p.is_file()})
    print(json.dumps({"comparisons": len(findings), "native_mismatches": sum(not f["native_byte_identical"] for f in findings),
                      "window_record_mismatches": sum(not f["window_record_identical"] for f in findings),
                      "report_sha256": sha(a.out / "report.json")}))
    if a.require_independent:
        assert report["all_layouts_byte_identical"], "atom identity still depends on batch layout"


if __name__ == "__main__":
    main()

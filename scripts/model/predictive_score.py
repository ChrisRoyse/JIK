"""Per-instrument errors for committed native forecasts, not a truth oracle."""
import base64

import numpy as np

from claims.codata_source import check, sha


def pack(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    check(a.dtype == np.float32 and a.ndim == 1 and np.isfinite(a).all(), "PREDICTION_VECTOR")
    raw = a.tobytes()
    return dict(size=len(a), dtype="float32", sha256=sha(raw), base64=base64.b64encode(raw).decode())


def unpack(value, dims):
    check(set(value) == {"size", "dtype", "sha256", "base64"} and value["size"] == dims
          and value["dtype"] == "float32" and len(value["base64"]) <= dims*8, "PREDICTION_SHAPE")
    raw = base64.b64decode(value["base64"], validate=True)
    check(len(raw) == dims*4 and sha(raw) == value["sha256"], "PREDICTION_BYTES")
    a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    check(np.isfinite(a).all() and abs(np.linalg.norm(a)-1) < 1e-4, "PREDICTION_NORM")
    return a/np.linalg.norm(a)


def score(predictions, inputs, baseline_unit, target_unit):
    check(set(predictions) == {s.name for s in inputs.specs}, "PREDICTION_ROSTER")
    results = {}
    for spec in inputs.specs:
        prediction = unpack(predictions[spec.name], spec.dims)
        vectors, ptr = inputs.arrays[spec.name]
        baseline = vectors[ptr[baseline_unit]:ptr[baseline_unit+1]].astype(np.float64)
        target = vectors[ptr[target_unit]:ptr[target_unit+1]].astype(np.float64)
        baseline /= np.linalg.norm(baseline, axis=1, keepdims=True)
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        # Retain every target window, and never combine distinct instruments.
        predicted = np.clip(target@prediction, -1, 1)
        persistent = np.clip((target@baseline.T).max(axis=1), -1, 1)
        loss, reference = float((1-predicted).mean()), float((1-persistent).mean())
        results[spec.name] = dict(target_windows=len(target), baseline_windows=len(baseline),
                                 prediction_cosines=predicted.tolist(), persistence_cosines=persistent.tolist(),
                                 prediction_loss=loss, persistence_loss=reference, delta=loss-reference)
    return dict(metric="mean_target_window_cosine_distance_per_instrument",
                baseline="last_observed_source_row_full_native_windows",
                instruments=results, better_on_every_instrument=all(v["delta"] < 0 for v in results.values()),
                calibrated_probability=False, establishes_empirical_truth=False)

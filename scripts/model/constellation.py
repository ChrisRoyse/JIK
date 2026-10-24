"""Typed, ordered native constellations on both sides of the predictor."""
import base64
from dataclasses import dataclass, replace

import numpy as np

from claims.codata_source import check, sha
from model.native import NativeInput


@dataclass(frozen=True)
class Constellation:
    slots: dict[str, NativeInput]
    origin: str
    regions: tuple[str, ...] | None = None

    def validate(self, specs, config):
        check(self.origin in ("observed", "predicted"), "CONSTELLATION_ORIGIN")
        check(set(self.slots) == {s.name for s in specs}, "CONSTELLATION_ROSTER")
        rows, device, elements = None, None, 0
        for spec in specs:
            slot = self.slots[spec.name]
            # Predictions may carry gradients. Observations must remain frozen.
            (replace(slot, values=slot.values.detach()) if self.origin == "predicted" else slot).check(spec, config)
            check((rows is None or slot.rows == rows) and (device is None or slot.values.device == device),
                  "CONSTELLATION_ALIGNMENT")
            rows, device = slot.rows, slot.values.device
            elements += slot.values.numel()
        check(elements <= config.max_native_elements, "CONSTELLATION_BUDGET")
        check(self.regions is None if self.origin == "observed" else
              isinstance(self.regions, tuple) and len(self.regions) == len(rows)
              and all(isinstance(r, str) and r for r in self.regions), "CONSTELLATION_REGIONS")
        return self

    def pack(self, specs, config, row=0):
        self.validate(specs, config)
        check(self.origin == "predicted" and type(row) is int
              and 0 <= row < len(self.regions), "CONSTELLATION_PREDICTION")
        result = {}
        for spec in specs:
            slot = self.slots[spec.name]
            check(not bool(slot.absent[row]) and slot.indices is None, "CONSTELLATION_REQUIRED_SLOT")
            a = slot.values[row, ~slot.padding[row]].detach().cpu().contiguous().numpy().astype("<f4")
            check(np.isfinite(a).all() and np.all(np.abs(np.linalg.norm(a, axis=1)-1) < 1e-4),
                  "CONSTELLATION_NORM")
            raw = a.tobytes()
            result[spec.name] = dict(instrument=spec.identity, shape=list(a.shape), dtype="float32-le",
                                     sha256=sha(raw), base64=base64.b64encode(raw).decode())
        return result


def unpack(value, spec, *, max_windows):
    check(isinstance(value, dict) and set(value) == {"instrument", "shape", "dtype", "sha256", "base64"}
          and value["instrument"] == spec.identity and value["dtype"] == "float32-le",
          "CONSTELLATION_SLOT_IDENTITY")
    shape = value["shape"]
    check(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int for n in shape)
          and 0 < shape[0] <= max_windows and shape[1] == spec.dims, "CONSTELLATION_SHAPE")
    size = shape[0]*shape[1]*4
    check(isinstance(value["base64"], str) and len(value["base64"]) == 4*((size+2)//3),
          "CONSTELLATION_PAYLOAD_SIZE")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except ValueError:
        check(False, "CONSTELLATION_BYTES")
    check(len(raw) == size and sha(raw) == value["sha256"], "CONSTELLATION_BYTES")
    a = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(shape)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    check(np.isfinite(a).all() and np.all(np.abs(norms-1) < 1e-4), "CONSTELLATION_NORM")
    return a/norms


def ordered_distance(cosines):
    """Unit-cost insertion/deletion, cosine-distance substitution; native order retained."""
    rows, cols = cosines.shape
    previous = np.arange(cols+1, dtype=np.float64)
    for i in range(rows):
        current = np.empty(cols+1, dtype=np.float64)
        current[0] = i+1
        for j in range(cols):
            current[j+1] = min(previous[j+1]+1, current[j]+1, previous[j]+1-cosines[i, j])
        previous = current
    return float(previous[-1]/max(rows, cols))


def score(predictions, inputs, baseline_unit, target_unit, config, *, target_inputs=None):
    target_inputs = inputs if target_inputs is None else target_inputs
    check(target_inputs.specs == inputs.specs, "CONSTELLATION_TARGET_INSTRUMENTS")
    check(set(predictions) == {s.name for s in inputs.specs}, "CONSTELLATION_ROSTER")
    results = {}
    for spec in inputs.specs:
        pred = unpack(predictions[spec.name], spec, max_windows=config.max_native_tokens)
        vectors, ptr = inputs.arrays[spec.name]
        old = vectors[ptr[baseline_unit]:ptr[baseline_unit+1]].astype(np.float64)
        targets, target_ptr = target_inputs.arrays[spec.name]
        target = targets[target_ptr[target_unit]:target_ptr[target_unit+1]].astype(np.float64)
        check(len(old) > 0 and len(target) > 0 and
              len(target)*(len(old)+len(pred)) <= config.max_attention_elements, "CONSTELLATION_SCORE_BUDGET")
        old /= np.linalg.norm(old, axis=1, keepdims=True)
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        predicted, persistent = np.clip(target@pred.T, -1, 1), np.clip(target@old.T, -1, 1)
        loss, reference = ordered_distance(predicted), ordered_distance(persistent)
        results[spec.name] = dict(instrument=spec.identity, target_windows=len(target),
            baseline_windows=len(old), prediction_windows=len(pred), prediction_cosines=predicted.tolist(),
            persistence_cosines=persistent.tolist(), prediction_loss=loss, persistence_loss=reference,
            delta=loss-reference)
    return dict(metric="per_instrument_ordered_cosine_edit_distance_v1",
        baseline="last_observed_source_row_full_native_windows", instruments=results,
        better_on_every_instrument=all(v["delta"] < 0 for v in results.values()),
        calibrated_probability=False, establishes_empirical_truth=False)

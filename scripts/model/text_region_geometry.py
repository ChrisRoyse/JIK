"""Train-only per-slot spherical placement, retaining the fixed native region book."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check, sha
from model.constellation import Constellation
from model.native import NativeInput
from model.native_identity_world import NativeIdentityWorld
from model.text_content_scale import build_content
from model.text_sigreg import SigregData
from senses.contract import digest
from text.constellation_corpus import file_sha


def spherical_place(base, raw, radius):
    check(base.shape == raw.shape and radius.shape == base.shape[:-1]+(1,)
          and bool(torch.isfinite(base).all()) and bool(torch.isfinite(raw).all())
          and bool(torch.isfinite(radius).all()) and bool(((radius >= 0) & (radius <= math.pi)).all()),
          "GEOMETRY_BOUNDS")
    tangent = raw-(raw*base).sum(-1, keepdim=True)*base
    length = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    # RMS makes the angular parameter independent of the native dimensionality.
    angle = radius*torch.tanh(length/math.sqrt(base.shape[-1]))
    ratio = torch.where(length > 1e-7, torch.sin(angle)/length.clamp_min(1e-7),
                        radius/math.sqrt(base.shape[-1]))
    return F.normalize(torch.cos(angle)*base+ratio*tangent, dim=-1)


class GeometryWorld(NativeIdentityWorld):
    def __init__(self, core, references, *, world_source_id, radii, mode):
        super().__init__(core, references, world_source_id=world_source_id)
        check(mode in ("fixed", "derived"), "GEOMETRY_MODE")
        values = torch.tensor(radii, dtype=torch.float32)
        check(values.shape == (len(self.specs), len(next(iter(self.reference_rows.values()))))
              and bool(torch.isfinite(values).all()) and bool(((values >= 0) & (values <= math.pi)).all()),
              "GEOMETRY_BOUNDS")
        self.register_buffer("placement_angles", values)
        self.angle_pin = sha(values.numpy().astype("<f4").tobytes())
        self.placement_mode = mode

    def check_geometry(self):
        check(sha(self.placement_angles.detach().cpu().numpy().astype("<f4").tobytes()) == self.angle_pin,
              "GEOMETRY_CHANGED")

    def check_reference(self):
        super().check_reference()
        self.check_geometry()

    def _placement(self, hidden, selected, event_ids):
        if self.placement_mode == "fixed":
            return super()._placement(hidden, selected, event_ids)
        check(self.placement_mode == "derived", "GEOMETRY_MODE")
        rows = next(iter(self.reference_rows.values()))
        slots = {}
        for lane, spec in enumerate(self.specs):
            book = getattr(self, f"world_values_{lane}")[selected]
            padding = getattr(self, f"world_padding_{lane}")[selected]
            check(book.shape[1] == 1 and not bool(padding.any()), "GEOMETRY_SINGLE_WINDOW_SCOPE")
            base = F.normalize(book, dim=-1)
            state = hidden[:, lane, None, :]+self.core.projectors[spec.name](book)
            radius = self.placement_angles[lane, selected][:, None, None]
            predicted = spherical_place(base, self.place[spec.name](state), radius)
            slots[spec.name] = NativeInput(spec.identity, tuple("prediction:"+e for e in event_ids),
                predicted, padding, torch.zeros(len(selected), dtype=torch.bool, device=predicted.device))
        return Constellation(slots, "predicted", tuple(rows[j] for j in selected.tolist())).validate(self.specs, self.config)


class GeometryData(SigregData):
    def __init__(self, corpus, bank, config, mode):
        super().__init__(corpus, bank, config, .02)
        check(mode in ("fixed", "derived"), "GEOMETRY_MODE")
        check(all(len(a) == 1 for values in self.native.values() for a in values), "GEOMETRY_SINGLE_WINDOW_SCOPE")
        fit = [i for g in self.groups if g["split"] == "train" for i in g["indices"]]
        check(fit == list(self.fit) and len(fit) == 107 and len(self.refs) == 8, "GEOMETRY_TRAIN_ONLY")
        radii = np.zeros((len(self.lanes), len(self.refs)), np.float64)
        for lane, name in enumerate(self.lanes):
            native = self.native[name]
            for index in fit:
                region = self.book["codes"][index]
                base, target = (native[i][0].astype(np.float64) for i in (self.refs[region], index))
                base, target = base/np.linalg.norm(base), target/np.linalg.norm(target)
                dot = float(np.clip(base@target, -1, 1))
                angle = float(np.arctan2(np.linalg.norm(target-dot*base), dot))
                radii[lane, region] = max(radii[lane, region], angle)
        # Outward float32 rounding is numerical containment, not a tuned margin.
        radii = np.where(radii == 0, 0., np.nextafter(radii.astype(np.float32), np.float32(np.inf))).astype(np.float32)
        self.mode = mode
        self.geometry = dict(method="train_only_max_assigned_native_angle", fit=fit,
            exemplars=self.refs, lanes=list(self.lanes), angles=radii.tolist(),
            native_dot_products=642, stored_bytes=int(radii.nbytes),
            native_sha256=sha(radii.astype("<f4").tobytes()), mode=mode,
            placement="spherical_tangent_RMS_angle", view="reuse_fixed_native_region_bank",
            runtime_acceptance_thresholds_changed=False,
            code=file_sha(Path(__file__)),
            driver=file_sha(Path(__file__).parents[1]/"audit/text_geometry_learning.py"))
        self.bind = {**self.bind, "geometry": self.geometry}
        self.pin = digest(self.bind)

    def view(self, model, left):
        indices = self.region_indices(left)
        rows = [self.sources[i][0]["atom_id"] for i in indices]
        check(isinstance(model, GeometryWorld) and model.placement_mode == self.mode
              and all(list(r) == rows for r in model.reference_rows.values())
              and model.angle_pin == self.geometry["native_sha256"], "GEOMETRY_MODEL_BINDING")
        return model, dict(schema="text-private-native-regions-v1", context=self.by_index[left]["context"],
            left=left, through=self.sources[left][0]["char_end"], shared=model.world_id, rows=rows, input=self.inputs.binding)

    def representation_controls(self, model):
        from audit.text_geometry_controls import controls
        return {**super().representation_controls(model), "geometry": controls(model, self)}


def build_geometry(data, args):
    old = build_content(data, args)
    before = {n: v.detach().clone() for n, v in old.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        model = GeometryWorld(old.core, old.references(), world_source_id=old.world_source_id,
            radii=data.geometry["angles"], mode=data.mode)
    model.to(old.core.initial.device)
    keys = model.load_state_dict(before, strict=False)
    check(keys.missing_keys == ["placement_angles"] and not keys.unexpected_keys
          and all(torch.equal(v, model.state_dict()[n]) for n, v in before.items()), "GEOMETRY_INITIAL_TENSORS")
    model.eval()
    model.check_reference()
    return model

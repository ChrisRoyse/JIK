"""Publication bindings for an assay generation (PC-04/07).

Small manifests checked once before a reader consumes reports. Slot bytes are hashed
once before each assay. A new registry, calibration or instrument invalidates old
results; partial reruns may not silently relabel results from another generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import blake3

from lenscommon import LensError, RETIRED_QUESTIONS, read_family
import vinfo


def digest(path):
    return blake3.blake3(Path(path).read_bytes()).hexdigest()


def instrument_digest():
    here = Path(__file__).resolve().parent
    h = blake3.blake3()
    for name in ("assay.py", "vinfo.py", "ksg.py", "slotio.py", "slot_layout.py", "slot_binding.py", "multi_view.py", "admission_state.py", "lenscommon.py", "catalogue.py", "universal.py", "learned_ucr.py", "learned.py", "text_windows.py", "audio_ucr.py", "audio_encoder.py"):
        h.update(name.encode())
        h.update((here / name).read_bytes())
    h.update(b"image_ucr.py")
    h.update((here / "image_ucr.py").read_bytes())
    for name in ("vision_encoder.py", "vision_ucr.py", "vision_assets.py"):
        h.update(name.encode())
        h.update((here / name).read_bytes())
    for name in ("contract.py", "instruments.py", "store.py", "views.py", "blobs.py", "pixels.py"):
        h.update(name.encode())
        h.update((here.parent / "senses" / name).read_bytes())
    from training_scope import POLICY_PATH
    h.update(POLICY_PATH.read_bytes())
    h.update((here.parent / "canon" / "training_scope.py").read_bytes())
    return h.hexdigest()


def bindings(root, fam, registry):
    _, summary = read_family(root / "cal", fam)
    return {"registry_blake3": registry["registry_blake3_before_ledger"],
            "calibration_blake3": summary["file_blake3"],
            "instrument": vinfo.INSTRUMENT, "instrument_blake3": instrument_digest()}


def validate_report(root, fam, rep, registry):
    if rep.get("inputs") != bindings(root, fam, registry):
        raise LensError("MNLM_ASSAY_STALE", f"{fam}: registry, calibration or instrument changed; re-assay this population")
    if rep.get("conditional_estimator", {}).get("instrument") != vinfo.INSTRUMENT:
        raise LensError("MNLM_ASSAY_STALE", f"{fam}: old instrument")
    if (rep.get("served") is not False or rep.get("shannon_conditional_certified") is not False
            or rep.get("sufficiency", {}).get("sufficient") is not False):
        raise LensError("MNLM_ASSAY_UNCERTIFIED", f"{fam}: this diagnostic cannot certify admission or sufficiency")
    for nm in rep["admission"]["panel"]:
        ent = registry["lenses"].get(nm, {})
        if ent.get("error") or ent.get("spec", {}).get("retired") or fam not in ent.get("families", {}):
            raise LensError("MNLM_PANEL_RETIRED_LENS_ADMITTED", f"{fam}/{nm}: no current registered measurement")


def load_reports(root, registry, admission):
    if admission.get("registry_blake3") != registry["registry_blake3_before_ledger"]:
        raise LensError("MNLM_ASSAY_STALE", "admission and registry describe different generations")
    reports = {}
    for fam, dg in admission["digests"].items():
        p = root / "assay" / f"{fam}.json"
        if not p.is_file() or digest(p) != dg:
            raise LensError("MNLM_ASSAY_RECORD_TORN", f"{fam}: assay differs from admission manifest")
        if fam in RETIRED_QUESTIONS:
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        validate_report(root, fam, rep, registry)
        reports[fam] = rep
    return reports


def validate_ledger(ledger_path, registry, admission=None):
    from ledger import verify_chain

    verify_chain(ledger_path)
    wanted = {registry["ledger_seq"]}
    if admission is not None:
        wanted.add(admission["ledger_seq"])
    entries = {}
    with Path(ledger_path).open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"] in wanted:
                entries[e["seq"]] = e
    e = entries.get(registry["ledger_seq"], {})
    if (e.get("hash") != registry.get("ledger_head") or e.get("kind") != "lenses.measure"
            or e.get("payload_digests", {}).get("registry_blake3") != registry["registry_blake3_before_ledger"]):
        raise LensError("MNLM_REGISTRY_UNSEALED", "registry is not bound to the recorded measurement ledger entry")
    if admission is not None:
        e = entries.get(admission["ledger_seq"], {})
        body = {k: v for k, v in admission.items() if k not in ("ledger_seq", "ledger_head")}
        if e.get("hash") != admission.get("ledger_head") or e.get("kind") != "lenses.admission" or e.get("payload_digests") != body:
            raise LensError("MNLM_ASSAY_UNSEALED", "admission does not match its ledger entry")


def load_panel(root, registry, admission, ledger_path):
    """Bind a published panel to its body, exact admission and ledger entry."""
    panel = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in panel.items() if k not in ("panel_blake3", "ledger_seq", "ledger_head")}
    dg = blake3.blake3((json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode()).hexdigest()
    inv = panel.get("invariants", {})
    if (dg != panel.get("panel_blake3") or inv.get("registry_blake3") != registry["registry_blake3_before_ledger"]
            or inv.get("admission_ledger_seq") != admission["ledger_seq"]):
        raise LensError("MNLM_PANEL_STALE", "panel body or its registry/admission binding changed; freeze a new panel")
    with Path(ledger_path).open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"] == panel.get("ledger_seq"):
                if (e.get("hash") == panel.get("ledger_head") and e.get("kind") == "lenses.panel"
                        and e.get("payload_digests", {}).get("panel_blake3") == dg):
                    return panel
                break
    raise LensError("MNLM_PANEL_UNSEALED", "panel does not match its ledger entry")

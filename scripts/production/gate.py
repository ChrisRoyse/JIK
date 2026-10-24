"""Stage 4 admission boundary. Read-only and evaluated before any output or GPU lease."""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lenscommon import LensError, sealed_registry

NATIVE_MEDIA = {"text", "image", "audio"}
MIN_NATIVE_SENSES = 5


def media_scope(required_media):
    if (not isinstance(required_media, (list, tuple)) or not required_media
            or any(not isinstance(m, str) or m not in NATIVE_MEDIA for m in required_media)
            or len(set(required_media)) != len(required_media)):
        raise LensError("MNLM_STAGE4_MEDIA_SCOPE", "declare every analyzed native medium: text/image/audio")
    return tuple(required_media)


def require_native_floor(panel, required_media):
    """Necessary count gate AFTER admission validation; never certifies D1/D2/D3.

    O(panel instruments + required media). Names and UCR channel labels do not
    establish native capability. Identical weights cannot count twice within a
    medium. A future multimodal instrument needs a pinned native branch contract
    for each medium; a caption or transcript is still a text reading.
    """
    required_media = media_scope(required_media)
    by_medium = {m: set() for m in required_media}
    for lens in panel["lenses"]:
        contract = lens.get("contract", {}); medium = contract.get("modality")
        if medium in by_medium:
            weights = contract.get("weights_sha256", "")
            if not isinstance(weights, str) or not re.fullmatch(r"[0-9a-f]{64}", weights):
                raise LensError("MNLM_STAGE4_NATIVE_IDENTITY", "native sense requires pinned weights")
            by_medium[medium].add(weights)
    counts = {m: len(ids) for m, ids in by_medium.items()}
    if any(n < MIN_NATIVE_SENSES for n in counts.values()):
        raise LensError("MNLM_STAGE4_NATIVE_FLOOR", f"at least five admitted native senses per medium required: {counts}")
    return counts


def require_admitted(root, ledger, required_media=None):
    try:
        root = Path(root)
        panel = json.loads((root / "panel.json").read_text(encoding="utf-8"))
        if (panel.get("contract") != "universal-panel-v1"
                or panel.get("gates", {}).get("shannon_conditional_certified") is not True
                or panel.get("gates", {}).get("families_served_ge3") is not True
                or len(panel.get("lenses", [])) < 5):
            raise LensError("MNLM_STAGE4_PANEL_UNADMITTED",
                            "Stage 3 has no certified universal panel; commission conditional information and five-sense coverage first")
        media_scope(required_media)
        # Engineering-only consumers do not use the diagnostic assay runtime.
        # Keep the production gate closed before importing its report readers.
        try:
            from admission_state import load_panel, load_reports, validate_ledger
        except ImportError as exc:
            raise LensError("MNLM_STAGE4_RUNTIME_MISSING",
                            "Restore the declared Stage 3 admission runtime before validating reports: " + str(exc)) from exc
        registry = sealed_registry(root)
        admission = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8"))
        validate_ledger(Path(ledger), registry, admission)
        load_reports(root, registry, admission)
        # This remains closed while the only registered assay is diagnostic.
        accepted = load_panel(root, registry, admission, Path(ledger))
        require_native_floor(accepted, required_media)
        return accepted
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LensError("MNLM_STAGE4_ADMISSION_MISSING", f"Complete the bound Stage 3 publication: {exc}") from exc

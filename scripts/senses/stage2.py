"""UCR integration for accepted Stage 2 shards; explicit source scope, no discovery."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from atoms_publication import accepted
from .contract import require
from .pipeline import export_canonical, write_records


def export_stage2(paths, output, sources=None):
    seal, shards = accepted(paths)
    require(sources is None or (sources and set(sources) <= {s["source"] for s in shards}),
            "MNLM_UCR_SOURCE_MISSING", "requested source absent from completed Stage 2 tree; rebuild the intended scope")
    selected = [s for s in shards if sources is None or s["source"] in sources]
    require(bool(selected), "MNLM_UCR_STAGE2", "empty source selection")
    return write_records(output, export_canonical({"shards": selected}),
                         {"kind": "stage2_atoms", "version": paths.version, "completion": seal["id"], "shards": selected})

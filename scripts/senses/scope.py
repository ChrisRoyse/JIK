"""Bind universal ingestion to the project's current training scope."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from training_scope import POLICY, allowed
from .contract import require


def check_source(source):
    require(allowed(source), "MNLM_SOURCE_EXCLUDED", f"{source}: excluded by operator; preserve historical evidence only")

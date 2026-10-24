"""Resolve the operator's source policy before census creates any state."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from training_scope import allowed


def effective_sources(config, only=None):
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SystemExit("MNLM_CENSUS_EMPTY_SCOPE: provide configured sources")
    if any(not isinstance(s, dict) or not isinstance(s.get("id"), str) or not s["id"] for s in sources):
        raise SystemExit("MNLM_CENSUS_SOURCE_SCHEMA: source identities must be nonempty strings")
    names = [s["id"] for s in sources]
    if len(names) != len(set(names)):
        raise SystemExit("MNLM_CENSUS_DUPLICATE_SOURCE: configuration identities must be unique")
    selected = [s for s in sources if allowed(s["id"])]
    if not selected:
        raise SystemExit("MNLM_CENSUS_EMPTY_SCOPE: every configured source is excluded by operator policy")
    if only is not None and only not in {s["id"] for s in selected}:
        raise SystemExit(f"MNLM_CENSUS_UNKNOWN_SOURCE: {only} is not in the effective configured scope")
    return selected

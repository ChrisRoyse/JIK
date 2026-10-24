"""Current operator exclusions, applied before any new training-path mutation."""
import json
import hashlib
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "training-scope.json"
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
POLICY_SHA256 = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def allowed(source):
    return source not in POLICY["excluded_sources"]


def require_training_source(source, *, experiment=None):
    if experiment is not None:
        scope = POLICY.get("authorized_experiments", {}).get(experiment)
        if scope is None or source not in scope["sources"]:
            raise RuntimeError(f"MNLM_SOURCE_EXPERIMENT_SCOPE: {experiment}: {source}")
        return
    if not allowed(source):
        raise RuntimeError(f"MNLM_SOURCE_EXCLUDED: {source}: {POLICY['excluded_sources'][source]}")


def require_training_question(question):
    if question in POLICY["excluded_questions"]:
        raise RuntimeError(f"MNLM_SOURCE_EXCLUDED: {question}: excluded by the operator; historical evidence remains readable")

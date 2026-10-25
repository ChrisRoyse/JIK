"""Optional bounded optimizer predicates with the original complete-answer refusals."""
from contextlib import contextmanager

from model import answer_full_history as learner
from model import native_device_transport as transport
from model.batched_finite_scope import batched_finiteness
from model.native_placement_checked import state_census, validate_optimizer
from text.constellation_corpus import need

ORIGINAL = validate_optimizer
ACTIVE = False


class OptimizerChecks:
    def __init__(self):
        self.closed = False
        self.records = []

    def __call__(self, model, optimizer, policy, step):
        need(not self.closed, 'ANSWER_OPTIMIZER_CLOSED')
        state_census(model, optimizer, step)
        try:
            with batched_finiteness() as checks:
                transport.validate_optimizer(model, optimizer, policy, step)
        except Exception:
            # Reproduce original error precedence when several defects coexist.
            # If the original unexpectedly accepts, the fast refusal still stands.
            ORIGINAL(model, optimizer, policy, step)
            raise
        self.records.append(dict(step=step, states=len(optimizer.state), checks=checks.report()))


@contextmanager
def batched_optimizer():
    global ACTIVE
    need(not ACTIVE and learner.validate_optimizer is ORIGINAL, 'ANSWER_OPTIMIZER_ALREADY_INSTALLED')
    checker = OptimizerChecks()
    ACTIVE = True
    learner.validate_optimizer = checker
    try:
        yield checker
    finally:
        intact = learner.validate_optimizer is checker
        learner.validate_optimizer = ORIGINAL
        ACTIVE = False
        checker.closed = True
        need(intact, 'ANSWER_OPTIMIZER_OWNERSHIP_CHANGED')

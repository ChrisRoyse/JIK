"""Native placement updates require an intact optimizer state for every parameter."""
import torch

from model import native_placement_training as base
from model.loop_training_checkpoint import validate_optimizer as validate_existing
from text.constellation_corpus import need

NativePlacementForecaster=base.NativePlacementForecaster
ARCHITECTURES=base.ARCHITECTURES
optimizer_for=base.optimizer_for


def state_census(model,optimizer,step):
    need(type(step) is int and step>=0,'PLACEMENT_OPTIMIZER_STEP')
    need(type(optimizer) is torch.optim.AdamW,'PLACEMENT_OPTIMIZER_TYPE')
    parameters=tuple(model.parameters())
    expected={id(p) for p in parameters} if step else set()
    present={id(p) for p in optimizer.state}
    need(len(expected)==(len(parameters) if step else 0)
         and len(optimizer.state)==len(expected) and present==expected,
         'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')


def validate_optimizer(model,optimizer,policy,step):
    state_census(model,optimizer,step)
    validate_existing(model,optimizer,policy,step)


def update(model,optimizer,item,step,policy,*,target):
    # The existing real path retains its recipe, finite-buffer and cursor checks.
    # The census must run before that path can allocate gradients or update state.
    state_census(model,optimizer,step)
    result=base.update(model,optimizer,item,step,policy,target=target)
    state_census(model,optimizer,step+1)
    return result

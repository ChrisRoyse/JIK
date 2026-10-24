"""Reject changed head weights before entering checkpoint recomputation.

This guards the scalar training objective consumed by answer_full_history.
The recomputation's own version check remains a second boundary. No tensor,
operator, precision or gradient is changed.
"""
from contextlib import contextmanager
from types import MethodType

from model.native_answer_sequence import versions
from model.transient_answer_attention import transient_attention
from text.constellation_corpus import need


@contextmanager
def guarded_transient_attention(model, *, heads=False):
    with transient_attention(model,heads=heads) as streams:
        if not heads:
            yield streams
            return
        forecaster=model.forecaster; checkpoint_loss=forecaster.losses
        def loss(this,*args,**kwargs):
            pin=versions(this);result=checkpoint_loss(*args,**kwargs)
            def before_backward(gradient):
                need(versions(this)==pin,'TRANSIENT_HEAD_WEIGHTS_CHANGED')
                return gradient
            if result['loss'].requires_grad:result['loss'].register_hook(before_backward)
            return result
        forecaster.losses=MethodType(loss,forecaster)
        try:yield streams
        finally:forecaster.losses=checkpoint_loss

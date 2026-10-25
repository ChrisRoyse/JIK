"""Observe registered tensor identities/counters without building a state_dict.

The traversal is live on every call, including shared-module aliases. Serialization
hooks or custom state are refused; the numerical model and its buffers are untouched.
"""
from contextlib import contextmanager

from torch import nn
from model import answer_full_history, guarded_transient_answer, native_answer_sequence
from model import native_stochastic_placement, transient_answer_attention
from text.constellation_corpus import need


def versions(model):
    result=[]
    def visit(module):
        kind=type(module)
        need(kind.state_dict is nn.Module.state_dict and 'state_dict' not in vars(module)
             and kind._save_to_state_dict is nn.Module._save_to_state_dict
             and '_save_to_state_dict' not in vars(module)
             and kind.get_extra_state is nn.Module.get_extra_state
             and 'get_extra_state' not in vars(module)
             and not module._state_dict_pre_hooks and not module._state_dict_hooks,
             'DIRECT_VERSIONS_CUSTOM_SERIALIZATION')
        for value in module._parameters.values():
            if value is not None:result.append((id(value),value._version))
        for name,value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                result.append((id(value),value._version))
        for child in module._modules.values():
            if child is not None:visit(child)
    visit(model)
    return tuple(result)


BINDINGS=((answer_full_history,'versions'),(guarded_transient_answer,'versions'),
          (native_answer_sequence,'versions'),(native_stochastic_placement,'versions'),
          (transient_answer_attention,'model_versions'))
ORIGINALS=tuple(getattr(module,name) for module,name in BINDINGS)
ACTIVE=False


@contextmanager
def direct_versions():
    global ACTIVE
    need(not ACTIVE and all(getattr(module,name) is original
         for (module,name),original in zip(BINDINGS,ORIGINALS)), 'DIRECT_VERSIONS_ALREADY_INSTALLED')
    ACTIVE=True
    for module,name in BINDINGS:setattr(module,name,versions)
    try:yield
    finally:
        intact=all(getattr(module,name) is versions for module,name in BINDINGS)
        for (module,name),original in zip(BINDINGS,ORIGINALS):setattr(module,name,original)
        ACTIVE=False
        need(intact,'DIRECT_VERSIONS_OWNERSHIP_CHANGED')

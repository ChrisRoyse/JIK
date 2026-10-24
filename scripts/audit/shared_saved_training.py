"""Exact original complete-answer updates with weakly shared saved-tensor handles."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import weakref

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import transient_answer_training as candidate
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model.guarded_transient_answer import guarded_transient_attention
from model.native_population_training import fingerprint
from model.saved_gradient_storage import SavedStorage
from model.shared_saved_storage import SharedSavedStorage
from text.constellation_corpus import need


def storage_controls(state):
    item, _ = candidate.previous.source.prior.item(state, state['train'][0])
    native = next(iter(item['context'][0].values())); pin = fingerprint(native)
    tensor = native.clone().requires_grad_(); store = SharedSavedStorage(state['policy']['working_bytes'], 'cpu')
    one = store.pack(tensor); two = store.pack(tensor)
    need(one is two and store.packs == 2 and store.created_handles == 1, 'SHARED_HANDLE_NOT_REUSED')
    alias = tensor.transpose(0, 1); other = store.pack(alias)
    need(other is not one and len(store.storages) == 1 and store.live_handles == 2,
         'SHARED_HANDLE_ALIAS_LAYOUT')
    del one; gc.collect(); need(store.current > 0 and store.live_handles == 2, 'SHARED_HANDLE_EARLY_RELEASE')
    with torch.no_grad(): tensor.add_(1)
    fresh = store.pack(tensor)
    need(fresh is not two and torch.equal(store.unpack(fresh),tensor), 'SHARED_HANDLE_VERSION_REUSED')
    result = dict(mutation=rejected(lambda:store.unpack(two), 'GRADIENT_STORAGE_MUTATED'),
                  alias_mutation=rejected(lambda:store.unpack(other), 'GRADIENT_STORAGE_MUTATED'))
    # Storage reassignment is a cache miss even when PyTorch's version does not move.
    version = tensor._version; tensor.data = tensor.detach().clone()+1
    replacement = store.pack(tensor)
    need(tensor._version == version and replacement is not fresh and
         torch.equal(store.unpack(replacement),tensor), 'SHARED_HANDLE_STORAGE_REPLACEMENT')
    source_ref = weakref.ref(tensor)
    del fresh, replacement, two, other, alias, tensor; gc.collect()
    need(source_ref() is None and store.current == 0 and store.live_handles == 0 and not store.handles,
         'SHARED_HANDLE_RETAINED')
    result['physical_memory'] = rejected(lambda:SharedSavedStorage(1,'cpu').pack(native), 'GRADIENT_STORAGE_MEMORY')
    probes = []
    for kind in (SavedStorage, SharedSavedStorage):
        tensor = native.clone().requires_grad_(); storage = kind(state['policy']['working_bytes'],'cpu')
        with storage.hooks(): value = (tensor*tensor).sum()+(tensor*tensor).sum()
        value.backward(); gradient = tensor.grad.clone(); del value; gc.collect()
        need(storage.current == 0, 'SHARED_HANDLE_GRAPH_RETAINED')
        if kind is SharedSavedStorage:
            need(storage.reused_handles > 0 and not storage.handles and storage.live_handles == 0,
                 'SHARED_HANDLE_GRAPH_NOT_REUSED')
        probes.append(fingerprint(gradient)); del tensor, gradient
    need(probes[0] == probes[1] and fingerprint(native) == pin, 'SHARED_HANDLE_GRAPH_CHANGED')
    return dict(**result, same_object_shared=True, aliases_separate=True, version_repack_distinct=True,
                storage_replacement_distinct=True, weak_original_released=True, all_handles_released=True,
                real_native_gradient=probes[0], native_unchanged=True)


def run(a):
    # Process-local opt-in only. Existing pinned callers and files are untouched.
    modules = (candidate, candidate.previous, candidate.learner)
    need(all(m.SavedStorage is SavedStorage for m in modules), 'SHARED_HANDLE_FACTORY_POLICY')
    old_controls = candidate.controls
    def controls(state, heads):
        return dict(original=old_controls(state,heads), shared_handles=storage_controls(state))
    candidate.NEW = (*candidate.NEW, 'model/guarded_transient_answer.py', 'model/shared_saved_storage.py',
                     'audit/shared_saved_training.py')
    candidate.transient_attention = guarded_transient_attention; candidate.controls = controls
    try:
        for module in modules: module.SavedStorage = SharedSavedStorage
        candidate.run(a)
    finally:
        for module in modules: module.SavedStorage = SavedStorage
        candidate.controls = old_controls


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=candidate.previous.source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--architecture',choices=tuple(candidate.PINS),required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--long-document',type=int);p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin')
    a=p.parse_args();a.heads=True
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

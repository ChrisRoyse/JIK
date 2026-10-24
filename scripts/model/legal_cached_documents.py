"""Complete native documents joined to separately authenticated future labels."""
import time

import numpy as np
import torch

from model.native_population_training import fingerprint
from text.constellation_corpus import need


def read_document(data, reader, cache, document, *, side, fit):
    start, stop = map(int, data.documents[document])
    began = time.perf_counter()
    labels = cache.document(document, side=side, fit=fit, expected_bounds=(start, stop))
    phases = dict(target_join_seconds=time.perf_counter()-began); began = time.perf_counter()
    uids = data.mapping[start:stop].tolist(); frames = []
    for lo in range(0, len(uids), 128): frames.extend(reader.frames(uids[lo:lo+128]))
    phases['native_read_seconds'] = time.perf_counter()-began; began = time.perf_counter()
    need(labels['target_uids'] == uids[1:], 'CACHED_DOCUMENT_TARGET_ADDRESSES')
    for name in data.roster:
        need(torch.equal(labels['counts'][name], torch.tensor([len(f[name]) for f in frames[1:]], dtype=torch.int64)),
             'CACHED_DOCUMENT_NATIVE_COUNT')
    frames = tuple(frames)
    item = dict(document=document, side=side, frames=frames, codes=labels['codes'], counts=labels['counts'],
                native_pin=fingerprint(frames), label_pin=fingerprint((labels['codes'], labels['counts'])),
                uids=uids, source_start=start, source_stop=stop, cache=cache.binding)
    phases['native_and_label_fingerprints_seconds'] = time.perf_counter()-began
    phases.update(atoms=len(frames), targets=len(frames)-1,
                  native_windows={n:sum(len(f[n]) for f in frames) for n in data.roster},
                  native_bytes=sum(x.numel()*x.element_size() for f in frames for x in f.values()))
    return item, phases


def current_codes(cache, uids):
    result = [cache.take(uids[lo:lo+128])[0] for lo in range(0, len(uids), 128)]
    return torch.cat(result).numpy()


def categorical_baselines(cache, continuation, uids):
    current = current_codes(cache, uids); hist = np.asarray(cache.baselines['joint'], dtype=np.int64)
    marginal = np.broadcast_to(np.asarray(cache.baselines['majority_codes']), current.shape).copy()
    # One coherent joint mode and a greedy conditional prefix mode are both explicit.
    joint = np.broadcast_to(np.asarray(np.unravel_index(hist.argmax(), hist.shape)), current.shape).copy()
    root = int(hist.sum((1, 2)).argmax()); middle = int(hist[root].sum(1).argmax())
    conditional = np.broadcast_to([root, middle, int(hist[root, middle].argmax())], current.shape).copy()
    transition = np.asarray(cache.baselines['root_transition']).argmax(1)[current[:, 0]]
    root_transition = np.full(current.shape, -1, dtype=np.int64); root_transition[:, 0] = transition
    continuation.fresh(); modal_uids = np.asarray(continuation.modal[uids], dtype=np.int64)
    available = modal_uids >= 0; retrieval = np.full(current.shape, -1, dtype=np.int64)
    if bool(available.any()): retrieval[available] = current_codes(cache, modal_uids[available].tolist())
    continuation.fresh()
    return dict(copy_current=current, marginal_mode=marginal, joint_mode=joint,
                conditional_mode=conditional, root_transition=root_transition,
                exact_input_successor=retrieval, exact_input_or_copy=np.where(available[:, None], retrieval, current)), modal_uids

"""Candidate CUDA fp32 instrument runtime with fixed window-forward geometry.

All native overflow windows survive. External I/O cohort shape is not an input;
only an attested window's tokens select its declared padding bucket. Admission
requires byte-exact cohort/reversal/singleton/duplicate and cold replay probes.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from lens.learned import LearnedLens
import text_windows
from text.constellation_corpus import canonical, digest, file_sha, need

RUNTIME = 'canonical-complete-native-windows-cuda-fp32-v1'
BATCH = 32


def widths(maximum):
    return sorted({v for v in (32, 64, 128, 256, 512, maximum) if v <= maximum})


def contract(lens):
    return dict(lens.contract(), runtime=RUNTIME, kind='dense-window-multi', shape=f'MultiWindows({lens.dim})',
        window_reduction='none; ordered complete tokenizer windows', execution_unit='fixed-native-window-buckets',
        max_windows=None, batch=BATCH, widths=widths(lens.spec['max_len']),
        filler='copy first actual row in this physical batch; discard declared filler outputs only',
        norm='unit per native window', torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(0), platform=sys.platform,
        code=dict(runtime=file_sha(Path(__file__)), learned=file_sha(Path(sys.modules[LearnedLens.__module__].__file__)),
                  provenance=file_sha(Path(text_windows.__file__))))


def tokenize(lens, texts):
    need(lens.kind == 'dense' and lens.tok.is_fast and len(texts) > 0, 'CANONICAL_INPUT')
    prefix = lens.spec.get('prefix', '')
    enc = lens.tok([prefix+t for t in texts], truncation=True, max_length=lens.spec['max_len'],
        return_overflowing_tokens=True, padding=False, return_offsets_mapping=True, return_special_tokens_mask=True)
    mapping = enc.pop('overflow_to_sample_mapping')
    need(mapping == sorted(mapping) and set(mapping) == set(range(len(texts))), 'CANONICAL_MAPPING')
    enc['offset_mapping'] = [[list(span) for span in window] for window in enc['offset_mapping']]
    allowed = widths(lens.spec['max_len'])
    for i, ids in enumerate(enc['input_ids']):
        width = next(w for w in allowed if w >= len(ids)); padding = width-len(ids)
        for key in enc:
            fill = lens.tok.pad_token_id if key == 'input_ids' else [0, 0] if key == 'offset_mapping' else 1 if key == 'special_tokens_mask' else 0
            enc[key][i] = list(enc[key][i])+[fill]*padding
    traces = text_windows.build(texts, prefix, enc, mapping, 'dense')
    return {key: enc[key] for key in ('input_ids', 'attention_mask', 'token_type_ids') if key in enc}, mapping, traces


@torch.inference_mode()
def forward(lens, texts):
    start = tick = time.perf_counter(); enc, mapping, traces = tokenize(lens, texts)
    phases = dict(tokenize=time.perf_counter()-tick); tick = time.perf_counter()
    groups = defaultdict(list)
    for i, ids in enumerate(enc['input_ids']): groups[len(ids)].append(i)
    values = np.empty((len(mapping), lens.dim), dtype=np.float32); shapes = []; padded = 0
    for width, indices in sorted(groups.items()):
        for start_row in range(0, len(indices), BATCH):
            actual = indices[start_row:start_row+BATCH]
            physical = actual+[actual[0]]*(BATCH-len(actual))
            batch = {key: torch.tensor([rows[i] for i in physical], dtype=torch.long, device='cuda')
                     for key, rows in enc.items()}
            native = lens._windows(batch).astype(np.float32)
            norms = np.linalg.norm(native, axis=1, keepdims=True)
            need(np.isfinite(native).all() and (norms > 0).all(), 'CANONICAL_NONFINITE')
            native /= norms
            values[actual] = native[:len(actual)]
            shapes.append(dict(rows=BATCH, width=width, actual=len(actual))); padded += (BATCH-len(actual))*width
    phases['forward'] = time.perf_counter()-tick
    bounds = np.r_[np.int64(0), np.cumsum(np.bincount(mapping, minlength=len(texts)), dtype=np.int64)]
    need(len(values) == bounds[-1] and all(len(t['windows']) == b-a for t, a, b in zip(traces, bounds, bounds[1:])),
         'CANONICAL_ALL_WINDOWS')
    return values, bounds, traces, dict(phases_seconds=phases, seconds=time.perf_counter()-start,
        shapes=shapes, padding_copy_tokens=padded, source_windows=len(mapping),
        active_source_tokens=sum(sum(w['attention_mask']) for t in traces for w in t['windows']))


def exact_reference(values, bounds, traces, other_values, other_bounds, other_traces, order):
    need(len(other_bounds) == len(order)+1 and len(other_traces) == len(order), 'CANONICAL_CONTEXT_COUNT')
    for j, i in enumerate(order):
        need(other_traces[j] == traces[i], 'CANONICAL_TRACE_DRIFT')
        need(np.array_equal(values[bounds[i]:bounds[i+1]], other_values[other_bounds[j]:other_bounds[j+1]]),
             'CANONICAL_BATCH_DRIFT')
    return True

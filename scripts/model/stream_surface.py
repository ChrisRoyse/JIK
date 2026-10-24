"""Complete attested surfaces through bounded causal decoder windows, without truncation."""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.native import from_arrays
from model.network import SensoryLanguageModel

import numpy as np

BOS, EOS = 256, 257


def schedule(raw, *, width=32, stride=16):
    need(isinstance(raw, bytes) and raw and 0 < stride <= width, 'SURFACE_SCHEDULE')
    tokens = [BOS, *raw, EOS]
    for start in range(0, len(raw) + 1, stride):
        end = min(len(raw) + 1, start + stride)
        left = max(0, start - (width - stride))
        prefix = tokens[left:end]
        labels = [-100] * (start - left) + tokens[start + 1:end + 1]
        yield dict(offset=left, scored=[start, end], prefix=prefix, labels=labels)


def position_batch(offsets, length, width):
    phase = offsets.to(torch.float32)[:, None] + torch.arange(length, device=offsets.device)[None, :]
    frequency = torch.exp(torch.arange(0, width, 2, device=offsets.device) * (-math.log(10000) / width))
    phase = phase[..., None] * frequency
    out = torch.empty((*phase.shape[:-1], width), device=offsets.device)
    out[..., 0::2], out[..., 1::2] = phase.sin(), phase.cos()
    return out


class StreamSurfaceDecoder(SensoryLanguageModel):
    """Identical trainable architecture; source byte positions continue across windows."""
    def forward(self, prefix, native, *, offsets):
        memories, (instruments, channels) = self.memories(native)
        need(prefix.dtype == torch.int64 and prefix.ndim == 2 and prefix.shape[0] == len(offsets)
             and 0 < prefix.shape[1] <= self.config.max_prefix
             and prefix.device == self.tokens.weight.device
             and bool(((prefix >= 0) & (prefix < self.config.vocabulary)).all()), 'SURFACE_PREFIX')
        need(offsets.shape == (prefix.shape[0],) and offsets.dtype == torch.int64
             and offsets.device == prefix.device and bool(((offsets >= 0) & (offsets < 2**24)).all()),
             'SURFACE_POSITION')
        need(prefix.shape[0] == len(next(iter(native.values())).rows), 'SURFACE_ROWS')
        h = self.tokens(prefix) + position_batch(offsets, prefix.shape[1], self.config.width)
        for block in self.blocks:
            h = block(h, memories, native, [s.name for s in self.specs])
        logits = self.output(h)
        need(bool(torch.isfinite(logits).all()), 'SURFACE_NONFINITE')
        return dict(logits=logits, instrument_count=instruments, channel_count=channels,
                    meaning_asserted=False, purpose='engineering_complete_surface_decoder')


class ConnectedSurface:
    """Only an eligible journal task can release a source surface for decoder training."""
    def __init__(self, session, key, *, binding):
        self.session, self.key, self.binding = session, key, binding
        self.selection = session.capability_selection(key, side='targets', binding=binding)
        matches = [e for e in session.capability_sources.values() if e.native.pin == binding]
        need(len(matches) == 1, 'SURFACE_NATIVE_SOURCE')
        self.evidence = matches[0]
        self.evidence.fresh()
        self.fields = [f['id'] for f in self.selection['fields']]

    def field(self, key, config, device):
        selection = self.session.capability_selection(self.key, side='targets', binding=self.binding)
        need(selection == self.selection and key in self.fields, 'SURFACE_TARGET_ELIGIBILITY')
        native = self.evidence.native
        field = native.population.fields[key]
        row = digest(dict(task=self.key, field=key, binding=self.binding, purpose='surface_training'))
        slots = {}
        for spec in native.specs:
            values, _ = native.field(key, spec.name)
            slots[spec.name] = from_arrays(spec, dict(tokens=values,
                offsets=np.array([0, len(values)], dtype=np.int64), absent=np.array([False])), (row,), config).to(device)
        raw = field['text'].encode('utf-8')
        need(sha(raw) == field['text_sha256'], 'SURFACE_SOURCE_BYTES')
        return raw, slots


def batches(raw, native, config, *, batch_size=8, stride=16):
    """Batch bounded windows from one surface; each original byte is scored once."""
    need(type(batch_size) is int and batch_size > 0, 'SURFACE_BATCH')
    pending = []
    for item in schedule(raw, width=config.max_prefix, stride=stride):
        pending.append(item)
        if len(pending) == batch_size:
            yield make_batch(pending, native, config)
            pending = []
    if pending: yield make_batch(pending, native, config)


def make_batch(items, native, config):
    device = next(iter(native.values())).values.device
    b, width = len(items), max(len(i['prefix']) for i in items)
    need(sum(x.values.numel() for x in native.values()) * b <= config.max_native_elements,
         'SURFACE_NATIVE_BUDGET')
    prefix = torch.zeros((b, width), dtype=torch.int64, device=device)
    labels = torch.full_like(prefix, -100)
    for i, item in enumerate(items):
        prefix[i, :len(item['prefix'])] = torch.tensor(item['prefix'], device=device)
        labels[i, :len(item['labels'])] = torch.tensor(item['labels'], device=device)
    rows = tuple(digest(dict(source=next(iter(native.values())).rows, scored=i['scored'])) for i in items)
    repeated = {n: replace(x, rows=rows, values=x.values.expand(b, -1, -1),
        padding=x.padding.expand(b, -1), absent=x.absent.expand(b),
        indices=None if x.indices is None else x.indices.expand(b, -1)) for n, x in native.items()}
    offsets = torch.tensor([i['offset'] for i in items], device=device, dtype=torch.int64)
    return prefix, labels, offsets, repeated, items


def whole_surface_loss(model, raw, native, *, backward=False):
    total, count, windows, ranges = 0., 0, 0, []
    for prefix, labels, offsets, slots, items in batches(raw, native, model.config):
        output = model(prefix, slots, offsets=offsets)
        loss = F.cross_entropy(output['logits'].reshape(-1, model.config.vocabulary),
                               labels.reshape(-1), ignore_index=-100, reduction='sum') / (len(raw) + 1)
        need(bool(torch.isfinite(loss)), 'SURFACE_LOSS_FINITE')
        if backward: loss.backward()
        total += float(loss.detach())
        count += sum(i['scored'][1] - i['scored'][0] for i in items)
        windows += len(items)
        ranges.extend(i['scored'] for i in items)
    need(count == len(raw) + 1 and ranges[0][0] == 0 and ranges[-1][1] == count
         and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), 'SURFACE_INCOMPLETE')
    return dict(loss=total, scored_positions=count, windows=windows, ranges=ranges,
                utf8_bytes=len(raw), sha256=sha(raw), eos_scored_once=True)

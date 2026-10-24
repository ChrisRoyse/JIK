"""Predict a region and hidden native count from explicitly visible instruments.

This auxiliary independent-atom task is distinct from next-atom prediction.
It accepts no hidden reading, hidden count, target code or source identifier.
"""
import torch
from torch.nn import functional as F

from model.native_region_predictor import NativeRegionPredictor
from text.constellation_corpus import need


class NativeReciprocalRegion(NativeRegionPredictor):
    def visible_layout(self, frames, hidden, visible):
        need(hidden in self.roster and isinstance(visible, (tuple, list)) and visible
             and len(set(visible)) == len(visible) and hidden not in visible
             and set(visible) <= set(self.roster), 'RECIPROCAL_REGION_VISIBLE_SCOPE')
        need(isinstance(frames, (tuple, list)) and 0 < len(frames) <= 64,
             'RECIPROCAL_REGION_PHYSICAL_TILE')
        names = tuple(n for n in self.roster if n in visible)
        dimensions = {s.name: s.dims for s in self.encoder.specs}
        device = self.encoder.query.device; groups = {}; tokens = elements = attention = 0
        for i, frame in enumerate(frames):
            need(set(frame) == set(names), 'RECIPROCAL_REGION_TARGET_VISIBLE_OR_MISSING')
            counts = []
            for name in names:
                value = frame[name]
                need(isinstance(value, torch.Tensor) and value.ndim == 2 and len(value) > 0
                     and value.shape[1] == dimensions[name] and value.dtype == torch.float32
                     and value.device == device and not value.requires_grad and bool(torch.isfinite(value).all()),
                     'RECIPROCAL_REGION_NATIVE')
                counts.append(len(value)); elements += value.numel()
            size = sum(counts)+len(self.roster)
            tokens += size; attention += self.config['heads']*size*size
            groups.setdefault(tuple(counts), []).append(i)
        need(tokens <= self.config['max_native_tokens'] and elements <= self.config['max_native_elements']
             and attention <= self.config['max_attention_elements'], 'RECIPROCAL_REGION_MEMORY')
        return names, [(counts, groups[counts]) for counts in sorted(groups)]

    def encode_visible(self, frames, *, hidden, visible):
        names, groups = self.visible_layout(frames, hidden, visible)
        encoder = self.encoder; device = encoder.query.device
        origin = encoder.position(torch.zeros(1, dtype=torch.int64, device=device))[0]
        result = [None]*len(frames)
        for counts, indices in groups:
            sizes = dict(zip(names, counts)); positions = []; queries = []; cursor = 0
            for lane, name in enumerate(self.roster):
                if name in sizes:
                    native = torch.stack([frames[i][name] for i in indices])
                    local = encoder.position(torch.arange(sizes[name], device=device))
                    positions.append(self.projectors[name](native)+encoder.kind[lane]+origin+local)
                    cursor += sizes[name]
                queries.append(cursor)
                query = encoder.query[lane]+encoder.kind[lane]+origin
                positions.append(query[None, None].expand(len(indices), 1, -1)); cursor += 1
            # The instrument axis is represented by typed attention positions;
            # independent atoms occupy distinct batch rows, never a shared history.
            value = torch.cat(positions, 1)
            mask = torch.zeros((cursor, cursor), dtype=torch.bool, device=device)
            for block in encoder.blocks:
                value = block(value, mask)
            selected = encoder.norm(value)[:, torch.tensor(queries, dtype=torch.int64, device=device)]
            for i, row in zip(indices, selected):
                result[i] = row
        output = torch.stack(result); self.check_hidden(output)
        return output

    def loss_withheld(self, context, codes, count, *, hidden):
        self.check_hidden(context)
        need(hidden in self.roster and isinstance(codes, torch.Tensor) and codes.device == context.device
             and isinstance(count, torch.Tensor) and count.shape == (len(context),)
             and count.dtype == torch.int64 and count.device == context.device
             and bool(((count >= 1) & (count <= self.max_windows[hidden])).all()),
             'RECIPROCAL_REGION_LOSS_TARGET')
        logits, _ = self.region(context, codes)
        region = tuple(F.cross_entropy(score.double(), codes[:, level]) for level, score in enumerate(logits))
        counts = self.counts[hidden](context[:, self.roster.index(hidden)])
        count_loss = F.cross_entropy(counts.double(), count-1)
        return dict(loss=torch.stack(region).mean()+count_loss, region=region, count=count_loss,
                    logits=logits, count_logits=counts)

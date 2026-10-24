"""Complete-target byte codec and fail-closed greedy rendering for #25 diagnostics."""
import math
import torch
from torch.nn import functional as F

from model.surface_renderer import BOS, EOS, VOCAB


def encode_complete(texts, device, capacity=512):
    rows = []
    for text in texts:
        raw = text.encode("utf-8", "strict")
        if not text.strip():
            raise ValueError("MNLM_SURFACE_EMPTY_TARGET")
        if len(raw) > capacity:
            raise ValueError("MNLM_SURFACE_TARGET_CAPACITY")
        rows.append([BOS, *raw, EOS])
    if not rows:
        raise ValueError("MNLM_SURFACE_EMPTY_BATCH")
    ids = torch.full((len(rows), max(map(len, rows))), EOS, dtype=torch.long, device=device)
    pad = torch.ones_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        pad[i, :len(row)] = False
    return ids, pad


def losses(model, slots, texts, capacity=512):
    ids, pad = encode_complete(texts, slots[0].device, capacity)
    return losses_encoded(model, slots, ids, pad)


def losses_encoded(model, slots, ids, pad):
    """Consume already-pinned target bytes; no per-step text encoding."""
    logits = model(slots, ids[:, :-1], pad[:, :-1]).float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("MNLM_SURFACE_NONFINITE_LOGITS")
    target = ids[:, 1:].masked_fill(pad[:, 1:], -100)
    ce = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction="none")
    byte_nll = (ce * (target < 256) * (target >= 0)).sum(1)
    eos_nll = (ce * (target == EOS)).sum(1)
    counts = ((target >= 0) & (target < 256)).sum(1)
    return ce.sum()/((~pad[:, 1:]).sum()), byte_nll/math.log(2), eos_nll/math.log(2), counts


def decode_complete(row, capacity=512):
    """Retain raw bytes on refusal; never replace UTF-8 or omit illegal symbols."""
    row = list(map(int, row))
    if EOS not in row:
        return dict(status="MNLM_SURFACE_NO_EOS", text=None, tokens=row)
    content = row[:row.index(EOS)]
    if len(content) > capacity:
        return dict(status="MNLM_SURFACE_OUTPUT_CAPACITY", text=None, tokens=row)
    if any(b < 0 or b >= 256 for b in content):
        return dict(status="MNLM_SURFACE_INVALID_SYMBOL", text=None, tokens=row)
    try:
        text = bytes(content).decode("utf-8", "strict")
    except UnicodeDecodeError:
        return dict(status="MNLM_SURFACE_INVALID_UTF8", text=None, tokens=row)
    if not text.strip():
        return dict(status="MNLM_SURFACE_EMPTY_OUTPUT", text=None, tokens=row)
    return dict(status="complete_unasserted", text=text, tokens=row)


@torch.inference_mode()
def render_complete(model, slots, capacity=512):
    ids = torch.full((len(slots[0]), 1), BOS, dtype=torch.long, device=slots[0].device)
    done = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    for _ in range(capacity+1):  # a full-capacity target still needs its EOS
        logits = model(slots, ids, None)[:, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("MNLM_SURFACE_NONFINITE_LOGITS")
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        ids = torch.cat((ids, nxt[:, None]), 1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    return [decode_complete(row, capacity) for row in ids[:, 1:].tolist()]

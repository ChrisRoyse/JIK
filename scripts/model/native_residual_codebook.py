"""Learned residual-quantized index over complete, typed native windows.

Only the quantizer's own latent combines instrument evidence (spec 15/24).
Its finite code is a lossy derived index, never a replacement for native slots.
Reconstruction receives attested window counts; this is not a length forecaster.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Linear
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_prefix_transformer import NativePrefixTransformer
from senses.contract import digest
from text.constellation_corpus import file_sha, need


class NativeResidualCodebook(nn.Module):
    def __init__(self, dimensions, config, sizes):
        super().__init__()
        need(dimensions and all(type(d) is int and d > 0 for d in dimensions.values()), 'NATIVE_RQ_DIMENSIONS')
        need(sizes and all(type(c) is int and c > 1 for c in sizes), 'NATIVE_RQ_SIZES')
        self.dimensions, self.config, self.sizes = dict(dimensions), dict(config), tuple(sizes)
        self.roster = tuple(dimensions); h = config['width']
        self.encoder = NativePrefixTransformer(SimpleNamespace(**config),
            [SimpleNamespace(name=n, dims=d) for n, d in dimensions.items()])
        self.projectors = nn.ModuleDict({n: Linear(d, h) for n, d in dimensions.items()})
        # This concatenates learned quantizer latents, never original native coordinates.
        self.to_latent = nn.Sequential(Linear(len(dimensions)*h, h), nn.LayerNorm(h))
        self.books = nn.ParameterList(nn.Parameter(torch.randn(c, h)*.02) for c in sizes)
        self.decoders = nn.ModuleDict({n: nn.Sequential(Linear(2*h, 2*h), nn.GELU(), Linear(2*h, d))
                                      for n, d in dimensions.items()})
        self.register_buffer('initialized', torch.tensor(False))

    def validate(self, frame):
        need(set(frame) == set(self.roster), 'NATIVE_RQ_ROSTER')
        device = self.encoder.query.device
        for name, dims in self.dimensions.items():
            x = frame[name]
            need(isinstance(x, torch.Tensor) and x.ndim == 2 and len(x) > 0
                 and x.shape[1] == dims and x.dtype == torch.float32 and x.device == device
                 and bool(torch.isfinite(x).all()), 'NATIVE_RQ_NATIVE')

    def latent(self, frames):
        need(isinstance(frames, (tuple, list)) and len(frames) > 0, 'NATIVE_RQ_EMPTY')
        result = []
        for frame in frames:
            self.validate(frame)
            # Each atom's code is independent of the other atoms in the batch.
            hidden = self.encoder.encode((frame,), self.projectors)[0]
            result.append(self.to_latent(hidden.reshape(1, -1))[0])
        return torch.stack(result)

    @torch.no_grad()
    def initialize(self, frames):
        need(not bool(self.initialized), 'NATIVE_RQ_ALREADY_INITIALIZED')
        residual = self.latent(frames)
        need(len(residual) >= max(self.sizes), 'NATIVE_RQ_INITIALIZATION_SUPPORT')
        for book in self.books:
            chosen = [0]; distances = (residual-residual[0]).square().sum(-1)
            for _ in range(1, len(book)):
                distances[chosen] = -torch.inf
                idx = int(distances.argmax())
                need(bool(distances[idx] > 0), 'NATIVE_RQ_INITIALIZATION_COLLAPSED')
                chosen.append(idx)
                distances = torch.minimum(distances, (residual-residual[idx]).square().sum(-1))
            book.copy_(residual[chosen])
            ids = (residual[:, None]-book[None]).square().sum(-1).argmin(-1)
            residual = residual-book[ids]
        self.initialized.fill_(True)

    def quantize(self, latent):
        need(bool(self.initialized), 'NATIVE_RQ_UNINITIALIZED')
        residual = latent; pieces, codes, penalties = [], [], []
        for book in self.books:
            ids = (residual[:, None]-book[None]).square().sum(-1).argmin(-1)
            selected = book[ids]
            penalties.append(F.mse_loss(selected, residual.detach())+.25*F.mse_loss(residual, selected.detach()))
            pieces.append(selected); codes.append(ids)
            residual = residual-selected.detach()
        quantized = torch.stack(pieces).sum(0)
        straight_through = latent+(quantized-latent).detach()
        return torch.stack(codes, -1), straight_through, torch.stack(penalties).sum()

    def values(self, codes):
        need(codes.ndim == 2 and codes.shape[1] == len(self.sizes) and codes.dtype == torch.int64,
             'NATIVE_RQ_CODE_SHAPE')
        for level, size in enumerate(self.sizes):
            need(bool(((codes[:, level] >= 0) & (codes[:, level] < size)).all()), 'NATIVE_RQ_CODE_RANGE')
        return torch.stack([book[codes[:, level]] for level, book in enumerate(self.books)]).sum(0)

    def reconstruct(self, latent, counts):
        need(latent.ndim == 2 and len(latent) == len(counts), 'NATIVE_RQ_RECONSTRUCTION_BATCH')
        result = []
        for z, row in zip(latent, counts):
            need(set(row) == set(self.roster) and all(type(w) is int and w > 0 for w in row.values()),
                 'NATIVE_RQ_WINDOW_COUNTS')
            windows = sum(row.values())
            need(windows+len(row) <= self.config['max_native_tokens']
                 and sum(row[n]*self.dimensions[n] for n in row) <= self.config['max_native_elements'],
                 'NATIVE_RQ_RECONSTRUCTION_MEMORY')
            result.append({n: self.decoders[n](torch.cat((z[None].expand(w, -1),
                self.encoder.position(torch.arange(w, device=z.device))), -1)) for n, w in row.items()})
        return tuple(result)

    def forward(self, frames):
        latent = self.latent(frames); codes, quantized, penalty = self.quantize(latent)
        decoded = self.reconstruct(quantized, [{n: len(f[n]) for n in self.roster} for f in frames])
        lane_losses = {n: torch.stack([F.mse_loss(d[n], f[n]) for d, f in zip(decoded, frames)]).mean()
                       for n in self.roster}
        loss = torch.stack(list(lane_losses.values())).mean()+penalty
        return dict(codes=codes, latent=latent, decoded=decoded, per_lane=lane_losses, quantizer_loss=penalty, loss=loss)


class AutoregressiveSemanticHead(nn.Module):
    """P(c_l | typed context, c_<l); the current/future code never enters its logit."""
    def __init__(self, lanes, width, sizes):
        super().__init__(); self.sizes = tuple(sizes)
        self.lanes, self.width = lanes, width
        self.initial = Linear(lanes*width, width)
        self.start = nn.Parameter(torch.randn(width)*.02)
        self.cell = nn.GRUCell(width, width)
        self.embeddings = nn.ModuleList(nn.Embedding(c, width) for c in sizes[:-1])
        self.outputs = nn.ModuleList(Linear(width, c) for c in sizes)

    def forward(self, context, codes=None, *, generator=None):
        need(context.ndim == 3 and context.shape[1:] == (self.lanes, self.width), 'NATIVE_ID_CONTEXT')
        if codes is not None:
            need(codes.shape == (len(context), len(self.sizes)) and codes.dtype == torch.int64,
                 'NATIVE_ID_TARGET_SHAPE')
            for level, size in enumerate(self.sizes):
                need(bool(((codes[:, level] >= 0) & (codes[:, level] < size)).all()), 'NATIVE_ID_TARGET_RANGE')
        state = self.initial(context.flatten(1)); previous = self.start.expand(len(context), -1)
        logits, selected = [], []
        for level, output in enumerate(self.outputs):
            state = self.cell(previous, state); score = output(state); logits.append(score)
            if codes is not None: token = codes[:, level]
            elif generator is None: token = score.argmax(-1)
            else: token = torch.multinomial(score.softmax(-1), 1, generator=generator).squeeze(-1)
            selected.append(token)
            if level+1 < len(self.sizes): previous = self.embeddings[level](token)
        return tuple(logits), torch.stack(selected, -1)


def save_frozen(model, path, binding):
    need(bool(model.initialized) and binding.get('fitting') == 'TRAIN_unique_inputs_only'
         and all(binding.get(k) for k in ('source_pin', 'split_pin', 'bank_pin', 'fit_members_sha256')),
         'NATIVE_RQ_FIT_BINDING')
    path = Path(path); need(not path.exists(), 'NATIVE_RQ_IMMUTABLE_ARTIFACT')
    state = {n: x.detach().cpu().clone() for n, x in model.state_dict().items()}
    payload = dict(schema='native-residual-codebook-v1', dimensions=model.dimensions, config=model.config,
                   sizes=model.sizes, binding=binding, state=state, state_pin=tensor_fingerprint(state),
                   implementation=file_sha(Path(__file__)), kind='lossy_derived_semantic_index_not_a_sense')
    torch.save(payload, path); return file_sha(path)


class FrozenNativeCodebook:
    def __init__(self, path, pin, binding, device):
        need(file_sha(Path(path)) == pin, 'NATIVE_RQ_ARTIFACT_PIN')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        need(payload['schema'] == 'native-residual-codebook-v1' and payload['binding'] == binding
             and payload['implementation'] == file_sha(Path(__file__)), 'NATIVE_RQ_ARTIFACT_BINDING')
        need(tensor_fingerprint(payload['state']) == payload['state_pin'], 'NATIVE_RQ_STATE_PIN')
        self.model = NativeResidualCodebook(payload['dimensions'], payload['config'], payload['sizes']).to(device)
        self.model.load_state_dict(payload['state'], strict=True); self.model.eval().requires_grad_(False)
        self.pin, self.binding = pin, dict(binding)
        self._versions = tuple((id(x), x._version) for x in self.model.state_dict(keep_vars=True).values())

    def fresh(self):
        need(self._versions == tuple((id(x), x._version) for x in self.model.state_dict(keep_vars=True).values()),
             'NATIVE_RQ_FROZEN_CHANGED')

    @torch.no_grad()
    def encode(self, frames):
        self.fresh(); result = self.model.quantize(self.model.latent(frames))[0]; self.fresh(); return result

    @torch.no_grad()
    def decode(self, codes, counts):
        self.fresh(); result = self.model.reconstruct(self.model.values(codes), counts); self.fresh(); return result

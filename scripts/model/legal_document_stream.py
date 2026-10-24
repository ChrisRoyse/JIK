"""Bind streamed context to a complete source document; targets are a separate read."""
import numpy as np
import torch

from model.legal_learning_source import SourceTile
from model.native_prefix_stream import PrefixCache
from text.constellation_corpus import digest, canonical, need
from text.legal_continuations import SIDES


class LegalDocumentStream:
    def __init__(self, data, document, *, side, fit=False):
        need(type(document) is int and 0 <= document < len(data.documents)
             and side in SIDES and int(data.doc_sides[document]) == SIDES.index(side),
             'LEGAL_DOCUMENT_STREAM_SCOPE')
        need(type(fit) is bool and (not fit or side == 'train'), 'LEGAL_DOCUMENT_STREAM_FIT')
        self.data, self.document, self.side, self.fit = data, document, side, fit
        self.start, self.stop = map(int, data.documents[document]); self.cursor = self.start
        self.context = digest(canonical(dict(binding=data.binding, document=document, scope='complete_source_document')))
        self.state = None; self.pending = None; self.frames = None; self.frame_versions = None

    def prepare(self, atoms, *, device, memory_bytes=64 << 20):
        need(type(atoms) is int and atoms > 0 and self.cursor < self.stop and self.pending is None,
             'LEGAL_DOCUMENT_STREAM_CURSOR')
        end = min(self.cursor+atoms, self.stop)
        indices = list(range(self.cursor, end))
        native = self.data.observations(indices, side=self.side, fit=self.fit, memory_bytes=memory_bytes)
        frames = tuple({name: torch.from_numpy(native[name].values[native[name].offsets[j]:native[name].offsets[j+1]].copy()).to(device)
                        for name in self.data.roster} for j in range(len(indices)))
        tile = SourceTile(self.document, self.side, self.cursor, end, self.start, self.stop)
        self.pending = tile; self.frames = frames
        self.frame_versions = tuple((id(x), x._version) for f in frames for x in f.values())
        return frames, dict(context=self.context, offset=self.cursor-self.start, state=self.state), tile

    def accept(self, tile, state):
        need(isinstance(state, PrefixCache) and tile is self.pending and tile.start == self.cursor and state.context == self.context
             and state.next_atom == tile.stop-self.start, 'LEGAL_DOCUMENT_STREAM_STATE')
        need(self.frame_versions == tuple((id(x), x._version) for f in self.frames for x in f.values()),
             'LEGAL_DOCUMENT_STREAM_NATIVE_CHANGED')
        self.state = state; self.cursor = tile.stop; self.pending = None; self.frames = None; self.frame_versions = None

    def encode(self, stream, atoms, *, device, detach_history, memory_bytes=64 << 20):
        frames, arguments, tile = self.prepare(atoms, device=device, memory_bytes=memory_bytes)
        hidden, state = stream.step(frames, **arguments, detach_history=detach_history)
        self.accept(tile, state)
        return hidden, tile

    def targets(self, tile, *, memory_bytes=64 << 20):
        need(tile.document == self.document and tile.side == self.side and tile.document_start == self.start
             and tile.document_stop == self.stop and self.start <= tile.start < tile.stop <= self.cursor
             and self.pending is None, 'LEGAL_DOCUMENT_STREAM_TARGET_ORDER')
        left = np.arange(tile.start, min(tile.stop, self.stop-1), dtype='<i8'); right = left+1
        if not len(left): return dict(left=(), right=(), observations={})
        self.data.validate_next(left, right)
        observations = self.data.observations(right.tolist(), side=self.side, fit=self.fit, memory_bytes=memory_bytes)
        return dict(left=tuple(left.tolist()), right=tuple(right.tolist()), observations=observations)

    def close(self):
        need(self.pending is None and self.cursor == self.stop and self.state is not None
             and self.state.next_atom == self.stop-self.start, 'LEGAL_DOCUMENT_STREAM_INCOMPLETE')
        self.data.fresh()
        return dict(document=self.document, side=self.side, atoms=self.stop-self.start,
                    successors=self.stop-self.start-1, complete_forward_context=True,
                    context=self.context, source_claims_are_truth=False)

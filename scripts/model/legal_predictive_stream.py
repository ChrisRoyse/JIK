"""Certified native reads for the real document stream; successor targets stay separate."""
import numpy as np

from model.legal_document_stream import LegalDocumentStream
from model.legal_learning_source import SourceTile
from text.constellation_corpus import need


class CertifiedLegalDocumentStream(LegalDocumentStream):
    def __init__(self, data, reader, document, *, side, fit=False):
        super().__init__(data, document, side=side, fit=fit)
        need(reader.data is data and tuple(reader.roster) == tuple(data.roster),
             'LEGAL_PREDICTIVE_READER_BINDING')
        self.reader = reader

    def prepare(self, atoms, *, device, memory_bytes=16 << 20):
        need(type(atoms) is int and 0 < atoms <= 128 and self.cursor < self.stop
             and self.pending is None and memory_bytes == 16 << 20, 'LEGAL_PREDICTIVE_CURSOR')
        end = min(self.cursor+atoms, self.stop)
        indices = self.data.indices(list(range(self.cursor, end)), self.side, fit=self.fit)
        uids = self.data.mapping[indices].tolist()
        frames = tuple({name: x.to(device) for name, x in frame.items()}
                       for frame in self.reader.frames(uids))
        tile = SourceTile(self.document, self.side, self.cursor, end, self.start, self.stop)
        self.pending, self.frames = tile, frames
        self.frame_versions = tuple((id(x), x._version) for f in frames for x in f.values())
        return frames, dict(context=self.context, offset=self.cursor-self.start, state=self.state), tile

    def targets(self, tile, *, memory_bytes=16 << 20):
        need(tile.document == self.document and tile.side == self.side and tile.document_start == self.start
             and tile.document_stop == self.stop and self.start <= tile.start < tile.stop <= self.cursor
             and self.pending is None and memory_bytes == 16 << 20, 'LEGAL_PREDICTIVE_TARGET_ORDER')
        left = np.arange(tile.start, min(tile.stop, self.stop-1), dtype='<i8')
        right = left+1
        if not len(left):
            return dict(left=(), right=(), uids=(), frames=())
        self.data.validate_next(left, right)
        self.data.indices(right.tolist(), self.side, fit=self.fit)
        uids = self.data.mapping[right].tolist()
        return dict(left=tuple(left.tolist()), right=tuple(right.tolist()), uids=tuple(uids),
                    frames=self.reader.frames(uids))

"""Group complete-document native I/O by source UID, then restore text order."""
from model.legal_cached_documents import read_document as temporal_document
from text.constellation_corpus import need


def read_frames(reader, uids):
    need(isinstance(uids, (tuple, list)) and len(uids) > 0
         and all(type(u) is int and 0 <= u < reader.data.native.config['inputs'] for u in uids),
         'GROUPED_DOCUMENT_UID_SCOPE')
    # Only the index is sorted. Existing document output storage is reused;
    # each source call still enforces 128 atoms / 16 MiB and all freshness checks.
    order = sorted(range(len(uids)), key=uids.__getitem__)
    result = [None]*len(uids)
    for lo in range(0, len(order), 128):
        positions = order[lo:lo+128]
        frames = reader.frames([uids[i] for i in positions])
        need(len(frames) == len(positions), 'GROUPED_DOCUMENT_COUNT')
        for i, frame in zip(positions, frames): result[i] = frame
    need(all(f is not None for f in result), 'GROUPED_DOCUMENT_COVERAGE')
    reader.data.fresh()
    return tuple(result)


class DocumentReader:
    """Lazy adapter keeps the canonical target-join and document checks in use."""
    def __init__(self, reader, uids):
        self.reader, self.uids = reader, uids
        self.saved = None; self.cursor = 0

    def frames(self, uids):
        end = self.cursor+len(uids)
        need(uids == self.uids[self.cursor:end] and end <= len(self.uids),
             'GROUPED_DOCUMENT_CONSUMER_ORDER')
        if self.saved is None: self.saved = read_frames(self.reader, self.uids)
        result = self.saved[self.cursor:end]; self.cursor = end
        return result


def read_document(data, reader, cache, document, *, side, fit):
    start, stop = map(int, data.documents[document])
    adapter = DocumentReader(reader, data.mapping[start:stop].tolist())
    # The canonical function authenticates target/split authority before the
    # lazy adapter reads any native values; labels never enter the I/O schedule.
    item, phases = temporal_document(data, adapter, cache, document, side=side, fit=fit)
    need(adapter.cursor == len(adapter.uids), 'GROUPED_DOCUMENT_CONSUMER_COVERAGE')
    return item, phases

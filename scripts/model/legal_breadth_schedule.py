"""Match document/source/native-window shapes while varying distinct TRAIN cases."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

SEED = 'legal-history-breadth-20261023'
DTYPE = np.dtype([('full', '<i8'), ('repeated', '<i8'), ('layout', 'V32')])
POLICY = dict(seed=SEED, order='ascending_source_document', epochs=1,
              group='source_collection_and_complete_ordered_native_window_counts',
              exemplar='minimum_sha256_seed_document_then_document_index',
              labels_select_groups=False, optimizer_updates=0, native_encoder_calls=0)


def priority(document):
    return hashlib.sha256(f'{SEED}:{document}'.encode()).digest(), document


class Metadata:
    def __init__(self, data):
        ref = data.source.report['corpus']; root = Path(ref['path'])
        need(file_sha(root/'report.json') == ref['pin'], 'BREADTH_CORPUS_PIN')
        report = json.loads((root/'report.json').read_bytes())
        path = root/'documents.jsonl'; item = report['files']['documents.jsonl']
        need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'],
             'BREADTH_METADATA_PIN')
        with path.open('rb') as stream:
            self.rows = [json.loads(line) for line in stream]
        need(len(self.rows) == len(data.documents), 'BREADTH_METADATA_DOCUMENTS')
        for document, row in enumerate(self.rows):
            need((row['atom_start'], row['atom_end']) == tuple(data.documents[document])
                 and isinstance(row['source_path'], str) and row['source_path'],
                 'BREADTH_METADATA_BOUNDARY')
        self.binding = dict(corpus=ref['pin'], documents=item['sha256'])
        self.stamps = {p:stamp(p) for p in [root/'report.json', path]}

    def fresh(self):
        need(all(stamp(p) == s for p, s in self.stamps.items()), 'BREADTH_METADATA_CHANGED')


def document(data, metadata, counts, number, expected_bounds=None):
    need(type(number) is int and 0 <= number < len(data.documents)
         and int(data.doc_sides[number]) == 0, 'BREADTH_REQUIRES_TRAIN_DOCUMENT')
    start, stop = map(int, data.documents[number])
    need(stop-start >= 2 and (expected_bounds is None or tuple(expected_bounds) == (start, stop)),
         'BREADTH_DOCUMENT_BOUNDARY')
    uids = data.mapping[start:stop]
    value = np.asarray(counts[uids])
    need(value.dtype == np.uint8 and value.shape == (stop-start, len(data.roster))
         and bool((value > 0).all()), 'BREADTH_NATIVE_WINDOW_LAYOUT')
    return uids, value, metadata.rows[number]['source_path']


def layout_key(source, values):
    # Length framing keeps the source bytes and tensor dimensions unambiguous.
    raw = source.encode(); h = hashlib.sha256()
    h.update(len(raw).to_bytes(8, 'little')); h.update(raw)
    h.update(np.asarray(values.shape, dtype='<i8').tobytes()); h.update(values.tobytes())
    return h.digest()


def build(data, metadata, counts):
    full = np.flatnonzero(data.doc_sides == 0)
    schedule = np.empty(len(full), dtype=DTYPE); groups = {}
    for index, number in enumerate(full.tolist()):
        _, values, source = document(data, metadata, counts, number)
        key = layout_key(source, values)
        if key not in groups or priority(number) < priority(groups[key]): groups[key] = number
        schedule[index] = (number, 0, key)
    for index in range(len(schedule)):
        schedule['repeated'][index] = groups[schedule['layout'][index].tobytes()]
    return schedule


def validate_row(row, data, metadata, counts):
    left, right = int(row['full']), int(row['repeated'])
    a, x, source = document(data, metadata, counts, left)
    b, y, other = document(data, metadata, counts, right)
    need(source == other and x.shape == y.shape and np.array_equal(x, y),
         'BREADTH_SUBSTITUTION_SHAPE_OR_SOURCE')
    need(layout_key(source, x) == row['layout'].tobytes(), 'BREADTH_LAYOUT_DIGEST')
    return a, b, x, source


def recount(schedule, data, metadata, counts, presence):
    """Direct shape comparison for every step; independent reverse exemplar census."""
    full = np.flatnonzero(data.doc_sides == 0)
    need(schedule.dtype == DTYPE and schedule.shape == full.shape
         and np.array_equal(schedule['full'], full), 'BREADTH_FULL_COVERAGE')
    candidates = {}; group_sizes = Counter()
    # Reverse source order avoids depending on the producer's traversal/min update.
    for number in full[::-1].tolist():
        _, value, source = document(data, metadata, counts, number)
        key = layout_key(source, value)
        candidates.setdefault(key, []).append(number); group_sizes[key] += 1
    expected = {key: sorted(rows, key=priority)[0] for key, rows in candidates.items()}
    masks = [np.zeros(len(presence), dtype=np.bool_) for _ in range(2)]
    cases = [set(), set()]; sources = {}; targets = atoms = changed = positions_squared = 0
    native = np.zeros(len(data.roster), dtype=np.int64)
    for row in schedule:
        a, b, values, source = validate_row(row, data, metadata, counts)
        left, right = int(row['full']), int(row['repeated'])
        need(right == expected[row['layout'].tobytes()], 'BREADTH_EXEMPLAR_SELECTION')
        # The direct source intervals also independently establish target membership.
        start, stop = map(int, data.documents[left])
        data.validate_next(np.arange(start, stop-1, dtype='<i8'), np.arange(start+1, stop, dtype='<i8'))
        n = len(a); atoms += n; targets += n-1; changed += left != right
        windows = values.sum(axis=0, dtype=np.int64); native += windows
        positions = int(windows.sum()) + n*len(data.roster)
        positions_squared += positions*positions
        tally = sources.setdefault(source, dict(documents=0, atoms=0, targets=0, changed_documents=0))
        tally['documents'] += 1; tally['atoms'] += n; tally['targets'] += n-1
        tally['changed_documents'] += left != right
        for i, (uid, doc) in enumerate([(a, left), (b, right)]):
            masks[i][uid] = True; cases[i].add(int(data.components[doc]))
    support = data.report['support']['train']
    need(targets == support['successors'] and atoms == support['atoms']
         and len(full) == support['documents'], 'BREADTH_TRAIN_DENOMINATOR')
    need(np.array_equal(masks[0], (presence & 1) != 0), 'BREADTH_UNIQUE_TRAIN_COVERAGE')
    need(not bool((masks[1] & ((presence & 1) == 0)).any()), 'BREADTH_REPEATED_HELD_UID')
    data.fresh(); metadata.fresh()
    return dict(documents=len(full), atoms=atoms, targets=targets, updates_per_arm=len(full),
                distinct_documents=dict(full=len(full), repeated=len(expected)),
                distinct_inputs=dict(full=int(masks[0].sum()), repeated=int(masks[1].sum())),
                distinct_case_components=dict(full=len(cases[0]), repeated=len(cases[1])),
                groups=len(expected), singleton_groups=sum(n == 1 for n in group_sizes.values()),
                changed_documents=changed, source_collections=sources,
                native_windows=dict(zip(data.roster, native.tolist())),
                sum_attention_positions_squared=positions_squared,
                every_step_shape_exact=True, complete_full_train_coverage=True,
                target_labels_used_for_selection=False, both_arms_share_target_pretraining=True)

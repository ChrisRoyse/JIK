"""Cold legal reuse audit: full input bytes, reverse postings and contextual edges."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from text.constellation_corpus import digest, file_sha, need
from text.legal_corpus import LegalCorpus
from text.legal_reuse import bind, successor_inputs
from text.legal_reuse_input import ReuseInputs


def postings(mapping, offsets, occurrences):
    n = len(mapping); counts = np.diff(offsets)
    need(len(occurrences) == n and offsets[0] == 0 and offsets[-1] == n and (counts > 0).all(),
         'REUSE_POSTING_COVERAGE')
    need((occurrences >= 0).all() and (occurrences < n).all()
         and np.all(np.bincount(occurrences, minlength=n) == 1), 'REUSE_POSTING_PERMUTATION')
    need(np.array_equal(mapping[occurrences], np.repeat(np.arange(len(counts)), counts)),
         'REUSE_POSTING_MEMBERSHIP')
    return True


def transitions(documents, successors, n):
    # Independent boundary test: all N-D left addresses, strictly increasing,
    # with each pair entirely inside one document.
    need(len(successors) == n-len(documents) and (np.diff(successors) > 0).all()
         and (successors >= 0).all() and (successors < n-1).all(), 'REUSE_EDGE_COVERAGE')
    owners = np.searchsorted(documents[:, 0], successors, side='right')-1
    need((successors+1 < documents[owners, 1]).all(), 'REUSE_EDGE_CROSS_DOCUMENT')
    return True


def run(root, pin, output):
    start = tick = time.perf_counter(); cpu = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    bank = ReuseInputs(root, pin); r = bank.report; a = bank.arrays
    corpus = LegalCorpus(Path(r['corpus']['path']), r['corpus']['pin'])
    n, u = r['counts']['occurrences'], r['counts']['unique_inputs']
    need(n == corpus.report['counts']['atoms'], 'REUSE_CORPUS_COUNT')
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    raw_inputs = []; unique_bytes = 0
    for batch in bank.table.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            uid = len(raw_inputs); raw = row['text'].encode()
            need(row['index'] == uid and digest(raw) == row['text_sha256'] == a['hashes'][uid].tobytes().hex()
                 and row['occurrences'] == a['counts'][uid]
                 and row['first_occurrence'] == a['first'][uid], 'REUSE_UNIQUE_BINDING')
            raw_inputs.append(raw); unique_bytes += len(raw)
    need(len(raw_inputs) == u and unique_bytes == r['counts']['unique_text_bytes'], 'REUSE_UNIQUE_COUNT')
    phases['unique_input_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    counts = np.zeros(u, dtype=np.int64); first = np.full(u, n, dtype=np.int64)
    cursor = 0; text_bytes = 0; doc_index = 0; last = None; edge_count = 0
    columns = ['text', 'text_sha256', 'atom_id', 'doc', 'pos', 'previous', 'next']
    for batch in corpus.table.iter_batches(batch_size=8192, columns=columns):
        values = batch.to_pydict(); size = len(values['text']); ids = np.asarray(a['mapping'][cursor:cursor+size])
        need((ids >= 0).all() and (ids < u).all(), 'REUSE_MAPPING_RANGE')
        # add.at avoids a U-wide zero allocation per source batch.
        np.add.at(counts, ids, 1)
        np.minimum.at(first, ids, np.arange(cursor, cursor+size))
        for text, expected, atom, doc, pos, previous, following in zip(*(values[k] for k in columns)):
            uid = int(a['mapping'][cursor]); raw = text.encode()
            need(raw == raw_inputs[uid] and digest(raw) == expected, 'REUSE_OCCURRENCE_BYTES')
            meta = corpus.documents[doc_index]
            need(meta['atom_start'] <= cursor < meta['atom_end'] and meta['id'] == doc
                 and pos == cursor-meta['atom_start'], 'REUSE_OCCURRENCE_CONTEXT')
            if pos == 0:
                need(previous is None and last is None, 'REUSE_DOCUMENT_START')
            else:
                need(last[0] == previous and last[1] == atom, 'REUSE_SOURCE_SUCCESSOR'); edge_count += 1
            last = (atom, following); cursor += 1; text_bytes += len(raw)
            if cursor == meta['atom_end']:
                need(following is None, 'REUSE_DOCUMENT_END'); last = None; doc_index += 1
    need(cursor == n and doc_index == len(corpus.documents) and last is None
         and text_bytes == r['counts']['text_bytes'] and edge_count == r['counts']['successors'], 'REUSE_FULL_COVERAGE')
    need(np.array_equal(counts, a['counts']) and np.array_equal(first, a['first']), 'REUSE_MULTIPLICITY')
    phases['all_original_bytes_and_contexts'] = time.perf_counter()-tick; tick = time.perf_counter()
    postings(a['mapping'], a['offsets'], a['occurrences'])
    documents = np.asarray([(d['atom_start'], d['atom_end']) for d in corpus.documents])
    need(np.array_equal(documents, a['documents']), 'REUSE_DOCUMENT_INDEX')
    transitions(documents, a['successors'], n)
    # Independent sorting by input and next-input; count changes instead of
    # reusing the producer's structured np.unique implementation.
    left = a['mapping'][a['successors']]; right = a['mapping'][a['successors']+1]
    order = np.lexsort((right, left)); left, right = left[order], right[order]
    changed = np.r_[True, (left[1:] != left[:-1]) | (right[1:] != right[:-1])]
    starts = np.flatnonzero(changed); distinct_left = left[starts]
    input_starts = np.flatnonzero(np.r_[True, distinct_left[1:] != distinct_left[:-1]])
    alternatives = np.diff(np.r_[input_starts, len(starts)])
    repeat = np.diff(np.r_[starts, len(left)])
    continuation = dict(distinct_pairs=len(starts), repeated_pairs=int((repeat > 1).sum()),
        inputs_with_successors=len(input_starts), inputs_with_multiple_successors=int((alternatives > 1).sum()),
        maximum_distinct_successors=int(alternatives.max(initial=0)),
        distinct_successor_histogram={str(k): v for k, v in sorted(Counter(map(int, alternatives)).items())})
    need(continuation == r['continuation'], 'REUSE_CONTINUATION_RECOUNT')
    phases['all_postings_and_edges'] = time.perf_counter()-tick; tick = time.perf_counter()
    selected = sorted({0, u-1, int(np.argmax(counts)), *np.linspace(0, u-1, 32, dtype=np.int64).tolist()})
    selections = []
    for uid in selected:
        row = bank.input(uid); observed = bank.occurrences(uid)
        indices = sorted({int(observed[0]), int(observed[-1]), int(observed[len(observed)//2])})
        for i in indices:
            owner = int(np.searchsorted(documents[:, 0], i, side='right')-1)
            original = corpus.document(owner)[i-documents[owner, 0]]
            need(original['text'] == row['text'] and original['text_sha256'] == row['text_sha256'],
                 'REUSE_INDEXED_INPUT')
        selections.append(dict(unique=uid, input_sha256=row['text_sha256'], occurrences=len(observed), selected=indices))
    # Controls use an actual source prefix, with local addresses for the same
    # generic validators; corpus, bank and their published arrays remain immutable.
    _, small_map = np.unique(a['mapping'][:128], return_inverse=True)
    small_occ = np.argsort(small_map, kind='stable'); small_off = np.r_[0, np.cumsum(np.bincount(small_map))]
    duplicate = small_occ.copy(); duplicate[-1] = duplicate[0]
    wrong = small_occ.copy(); wrong[0], wrong[-1] = wrong[-1], wrong[0]
    small_docs = documents[:3].copy(); small_n = int(small_docs[-1, 1]); edges = successor_inputs(small_docs, small_n)
    crossing = edges.copy(); crossing[np.searchsorted(edges, small_docs[0, 1]-1)] = small_docs[0, 1]-1
    raw = raw_inputs[0]; expected = digest(raw)
    controls = dict(changed_text=rejected(lambda: bind(raw+b'!', expected)),
        merged_distinct_text=rejected(lambda: bind(raw, expected, raw+b'!')),
        missing_occurrence=rejected(lambda: postings(small_map, small_off, small_occ[:-1])),
        duplicated_occurrence=rejected(lambda: postings(small_map, small_off, duplicate)),
        wrong_occurrence=rejected(lambda: postings(small_map, small_off, wrong)),
        cross_document_successor=rejected(lambda: transitions(small_docs, crossing, small_n)),
        changed_report_pin=rejected(lambda: ReuseInputs(root, '0'*64)),
        implicit_training_admission=rejected(bank.training_authority))
    write_committed(output/'consumer-selections.json', selections)
    bank.fresh(); corpus.fresh()
    phases['indexed_consumers_and_controls'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'REUSE_READBACK_TIMING')
    result = dict(schema='legal-input-reuse-readback-v1', complete=True, source=dict(path=str(root), pin=pin),
        counts=r['counts'], continuation=continuation, all_input_bytes_checked=True, all_postings_checked=True,
        all_source_successors_checked=True, indexed_selections=len(selections), controls=controls,
        files={'consumer-selections.json': file_sha(output/'consumer-selections.json')},
        phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code_sha256=file_sha(Path(__file__)), number_status='measured', acceptance='author-checked',
        training_ready=False, scientific_admission=False, native_measurements=0, model_updates=0)
    result_pin = write_committed(output/'report.json', result)
    print(json.dumps(dict(pin=result_pin, counts=r['counts'], controls=controls, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.pin, a.output)

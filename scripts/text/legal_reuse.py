"""Exact legal input reuse with every source occurrence and successor retained.

PC-07/28/35: two narrow corpus passes, O(bytes + N log N), compact immutable
arrays and a byte cache. This is an input index, never a meaning/admission gate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import digest, file_sha, need
from text.legal_corpus import LegalCorpus

SCHEMA = 'legal-exact-input-reuse-v1'
GROUP = 4096
ARRAYS = ('mapping', 'hashes', 'first', 'counts', 'offsets', 'occurrences',
          'documents', 'successors')
UNIQUE_SCHEMA = pa.schema([('index', pa.int64()), ('text_sha256', pa.string()),
    ('first_occurrence', pa.int64()), ('occurrences', pa.int64()), ('text', pa.string())])


def bind(raw, expected, retained=None):
    need(digest(raw) == expected, 'REUSE_TEXT_HASH')
    need(retained is None or raw == retained, 'REUSE_DISTINCT_TEXT')
    return raw


def successor_inputs(documents, n):
    need(documents.ndim == 2 and documents.shape[1] == 2 and len(documents), 'REUSE_DOCUMENTS')
    need(documents[0, 0] == 0 and documents[-1, 1] == n
         and np.all(documents[:, 1] > documents[:, 0])
         and np.array_equal(documents[1:, 0], documents[:-1, 1]), 'REUSE_DOCUMENT_RANGES')
    available = np.ones(n, dtype=bool)
    available[documents[:, 1]-1] = False
    return np.flatnonzero(available)


def continuation_counts(mapping, successors):
    pairs = np.empty(len(successors), dtype=[('input', '<i8'), ('target', '<i8')])
    pairs['input'] = mapping[successors]
    pairs['target'] = mapping[successors+1]
    distinct, repetitions = np.unique(pairs, return_counts=True)
    input_ids, alternatives = np.unique(distinct['input'], return_counts=True)
    return dict(distinct_pairs=len(distinct), repeated_pairs=int((repetitions > 1).sum()),
        inputs_with_successors=len(input_ids), inputs_with_multiple_successors=int((alternatives > 1).sum()),
        maximum_distinct_successors=int(alternatives.max(initial=0)),
        distinct_successor_histogram=dict(sorted(Counter(map(int, alternatives)).items())))


def build(root, pin, output):
    start = tick = time.perf_counter(); cpu = time.process_time()
    need(not output.exists(), 'REUSE_EXISTS')
    corpus = LegalCorpus(root, pin)
    n = corpus.report['counts']['atoms']
    output.mkdir(parents=True)
    phases = dict(authenticate=time.perf_counter()-tick)
    try:
        tick = time.perf_counter(); hashes = np.empty(n, dtype='V32'); cursor = 0
        for batch in corpus.table.iter_batches(batch_size=8192, columns=['text_sha256', 'kind']):
            values = batch.to_pydict(); size = len(values['kind'])
            need(all(k == 'text' for k in values['kind']), 'REUSE_LAYOUT_REQUIRES_ABSENCE')
            raw = b''.join(bytes.fromhex(h) for h in values['text_sha256'])
            hashes[cursor:cursor+size] = np.frombuffer(raw, dtype='V32'); cursor += size
        need(cursor == n, 'REUSE_INPUT_COUNT')
        phases['address_pass'] = time.perf_counter()-tick; tick = time.perf_counter()
        unique, first, mapping, counts = np.unique(hashes, return_index=True,
                                                 return_inverse=True, return_counts=True)
        del hashes
        u = len(unique); retained = [None]*u; cursor = 0; text_bytes = 0
        phases['group_addresses'] = time.perf_counter()-tick; tick = time.perf_counter()
        for batch in corpus.table.iter_batches(batch_size=8192, columns=['text', 'text_sha256']):
            values = batch.to_pydict()
            for text, expected in zip(values['text'], values['text_sha256']):
                uid = int(mapping[cursor]); raw = text.encode('utf-8')
                need(unique[uid].tobytes().hex() == expected, 'REUSE_ADDRESS_BINDING')
                retained[uid] = bind(raw, expected, retained[uid])
                text_bytes += len(raw); cursor += 1
        need(cursor == n and all(raw is not None for raw in retained)
             and text_bytes == corpus.report['counts']['bytes'], 'REUSE_BYTE_COVERAGE')
        unique_bytes = sum(map(len, retained))
        phases['all_byte_bindings'] = time.perf_counter()-tick; tick = time.perf_counter()
        with pq.ParquetWriter(output/'unique.parquet', UNIQUE_SCHEMA, compression='zstd') as writer:
            for start_uid in range(0, u, GROUP):
                rows = [dict(index=i, text_sha256=unique[i].tobytes().hex(),
                    first_occurrence=int(first[i]), occurrences=int(counts[i]),
                    text=retained[i].decode('utf-8')) for i in range(start_uid, min(start_uid+GROUP, u))]
                writer.write_table(pa.Table.from_pylist(rows, schema=UNIQUE_SCHEMA), row_group_size=GROUP)
        del retained
        phases['unique_publication'] = time.perf_counter()-tick; tick = time.perf_counter()
        offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
        occurrences = np.argsort(mapping, kind='stable')
        documents = np.asarray([(d['atom_start'], d['atom_end']) for d in corpus.documents], dtype=np.int64)
        successors = successor_inputs(documents, n)
        need(len(successors) == corpus.report['counts']['transitions'], 'REUSE_TRANSITIONS')
        continuation = continuation_counts(mapping, successors)
        phases['occurrence_transition_index'] = time.perf_counter()-tick; tick = time.perf_counter()
        arrays = dict(mapping=mapping, hashes=unique, first=first, counts=counts, offsets=offsets,
                      occurrences=occurrences, documents=documents, successors=successors)
        for name, value in arrays.items():
            np.save(output/(name+'.npy'), value, allow_pickle=False)
        files = {name: dict(bytes=(output/name).stat().st_size, sha256=file_sha(output/name))
                 for name in ['unique.parquet']+[name+'.npy' for name in ARRAYS]}
        corpus.fresh()
        phases['array_publication_and_pins'] = time.perf_counter()-tick
        seconds = time.perf_counter()-start
        need(sum(phases.values())/seconds >= .9, 'REUSE_TIMING')
        report = dict(schema=SCHEMA, complete=True, errors=[], corpus=dict(path=str(root), pin=pin),
            counts=dict(occurrences=n, unique_inputs=u, reused_occurrences=n-u, documents=len(documents),
                successors=len(successors), text_bytes=text_bytes, unique_text_bytes=unique_bytes,
                maximum_multiplicity=int(counts.max()), duplicated_inputs=int((counts > 1).sum())),
            continuation=continuation, unique_order='lexicographic SHA256; every collision byte-checked',
            group_rows=GROUP, files=files, occurrences_merged=False, native_measurements=0, model_updates=0,
            training_ready=False, scientific_admission=False, splits_frozen=False,
            unknowns=['native window counts', 'canonical inference identity', 'near-duplicate case closure',
                      'learned association coverage', 'withheld predictor growth'],
            phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            code_sha256=file_sha(Path(__file__)), number_status='measured', acceptance='author-checked')
        result_pin = write_committed(output/'report.json', report)
        write_committed(output/'COMMITTED', dict(report_sha256=result_pin))
        print(json.dumps(dict(pin=result_pin, counts=report['counts'], continuation=continuation,
                              seconds=seconds)), flush=True)
    except BaseException as error:
        write_committed(output/'failure.json', dict(error=str(error), phases=phases))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); build(a.root, a.pin, a.output)

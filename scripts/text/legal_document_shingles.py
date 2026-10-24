"""Exact document shingles for leakage grouping; never model inputs or senses.

O(source bytes + sum H_d log H_d), O(vocabulary + largest document) RAM.
Each normalized five-word tuple is stored as five exact uint32 word IDs.
No shingle hashing, approximate candidate selection, pooling or document removal.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from kernel.capability_split import normalized
from text.constellation_corpus import canonical, file_sha, need
from text.constellation_input import stamp
from text.legal_stock import SOURCES

SCHEMA = 'legal-exact-document-shingles-v1'
POLICY = dict(normalization='casefold_whitespace_numeric_template', words='whitespace_split',
              width=5, minimum_words=20, jaccard_numerator=4, jaccard_denominator=5,
              identity='exact_word_ids_no_hashing', use='evaluation_grouping_only')


def bound_text(line, record, source, ordinal, offset):
    need(record['source'] == source and record['ordinal'] == ordinal
         and record['offset'] == offset and record['line_bytes'] == len(line)
         and record['line_sha256'] == hashlib.sha256(line).hexdigest(), 'SHINGLE_SOURCE_ADDRESS')
    text = json.loads(line)['text']
    need(hashlib.sha256(text.encode()).hexdigest() == record['text_sha256'], 'SHINGLE_SOURCE_TEXT')
    return text


def tuples(ids):
    if len(ids) < 5:
        return np.empty((0, 5), dtype='<u4')
    values = np.lib.stride_tricks.sliding_window_view(np.asarray(ids, dtype='<u4'), 5).copy()
    return np.unique(values.view('V20').reshape(-1)).view('<u4').reshape(-1, 5)


def run(source, pin, output):
    start = tick = time.perf_counter(); cpu = time.process_time()
    need(file_sha(source/'report.json') == pin, 'SHINGLE_SOURCE_REPORT')
    report = json.loads((source/'report.json').read_bytes())
    need(report['schema'] == 'legal-source-stock-v1' and report['complete']
         and json.loads((source/'COMMITTED').read_bytes()) == dict(report_sha256=pin),
         'SHINGLE_SOURCE_COMMIT')
    index = source/'records.jsonl'
    need(file_sha(index) == report['files']['records.jsonl']['sha256'], 'SHINGLE_SOURCE_INDEX')
    observed = {str(source/n): stamp(source/n) for n in ['report.json', 'COMMITTED', 'records.jsonl']
                + ['raw/'+name for _, name, _, _ in SOURCES]}
    output.mkdir(parents=True, exist_ok=False)
    write_committed(output/'before.json', observed)
    phases = dict(authenticate=time.perf_counter()-tick, source=0., normalize=0.,
                  word_ids=0., unique_tuples=0., persist=0.)
    vocabulary = {}; offsets = [0]; word_counts = []; sources = Counter(); errors = []
    words = total_bytes = raw_windows = 0
    try:
        with index.open('rb') as indexed, (output/'tuples.u32').open('xb') as packed, \
             (output/'documents.jsonl').open('xb') as docs:
            for domain, name, expected, _ in SOURCES:
                h = hashlib.sha256(); offset = count = 0
                with (source/'raw'/name).open('rb') as stream:
                    while True:
                        tick = time.perf_counter(); line = stream.readline()
                        if not line:
                            phases['source'] += time.perf_counter()-tick; break
                        record = json.loads(indexed.readline()); h.update(line)
                        text = bound_text(line, record, domain+':'+name, count, offset)
                        phases['source'] += time.perf_counter()-tick
                        offset += len(line); count += 1
                        tick = time.perf_counter(); tokens = normalized(text).split()
                        phases['normalize'] += time.perf_counter()-tick
                        tick = time.perf_counter(); ids = []
                        for token in tokens:
                            if token not in vocabulary:
                                need(len(vocabulary) < 2**32, 'SHINGLE_WORD_ID_WIDTH')
                                vocabulary[token] = len(vocabulary)
                            ids.append(vocabulary[token])
                        phases['word_ids'] += time.perf_counter()-tick
                        tick = time.perf_counter()
                        try:
                            values = tuples(ids)
                        except Exception as exc:
                            errors.append(dict(record=record['id'], error=repr(exc)))
                            values = np.empty((0, 5), dtype='<u4')
                        phases['unique_tuples'] += time.perf_counter()-tick
                        tick = time.perf_counter(); values.tofile(packed)
                        meta = dict(index=len(word_counts), record=record['id'], source=record['source'],
                                    ordinal=record['ordinal'], offset=record['offset'],
                                    line_bytes=record['line_bytes'], line_sha256=record['line_sha256'],
                                    text_sha256=record['text_sha256'], words=len(tokens),
                                    start=offsets[-1], end=offsets[-1]+len(values))
                        docs.write(canonical(meta)+b'\n'); offsets.append(meta['end'])
                        word_counts.append(len(tokens)); sources[domain] += 1
                        words += len(tokens); total_bytes += len(text.encode()); raw_windows += max(0, len(tokens)-4)
                        phases['persist'] += time.perf_counter()-tick
                        if len(word_counts) % 10000 == 0:
                            print(json.dumps(dict(documents=len(word_counts), words=words,
                                  memberships=offsets[-1], seconds=time.perf_counter()-start)), flush=True)
                need(count == expected and h.hexdigest() == report['files']['raw/'+name]['sha256'],
                     'SHINGLE_RAW_SOURCE_PIN')
            need(not indexed.read(1) and len(word_counts) == report['records'], 'SHINGLE_SOURCE_COUNT')
            for stream in (packed, docs): stream.flush(); os.fsync(stream.fileno())
        tick = time.perf_counter()
        with (output/'words.jsonl').open('xb') as stream:
            for word, number in vocabulary.items(): stream.write(canonical([number, word])+b'\n')
            stream.flush(); os.fsync(stream.fileno())
        for name, values, dtype in [('offsets', offsets, '<i8'), ('word-counts', word_counts, '<u4')]:
            with (output/(name+'.npy')).open('xb') as stream:
                np.save(stream, np.asarray(values, dtype=dtype), allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
        phases['persist'] += time.perf_counter()-tick; tick = time.perf_counter()
        need(all(stamp(Path(p)) == s for p, s in observed.items()), 'SHINGLE_SOURCE_DRIFT')
        names = ('tuples.u32', 'offsets.npy', 'word-counts.npy', 'words.jsonl', 'documents.jsonl', 'before.json')
        files = {n: dict(bytes=(output/n).stat().st_size, sha256=file_sha(output/n)) for n in names}
        phases['final_hashes'] = time.perf_counter()-tick
        seconds = time.perf_counter()-start
        need(sum(phases.values())/seconds >= .9, 'SHINGLE_TIMING_COVERAGE')
        counts = dict(documents=len(word_counts), words=words, vocabulary=len(vocabulary),
                      unique_per_document_memberships=offsets[-1], raw_windows=raw_windows,
                      text_bytes=total_bytes, eligible_documents=sum(w >= 20 for w in word_counts),
                      maximum_document_words=max(word_counts), sources=dict(sources))
        result = dict(schema=SCHEMA, complete=not errors, source=dict(path=str(source), pin=pin),
                      policy=POLICY, files=files, counts=counts, errors=errors,
                      global_unique_shingles=None, near_join_complete=False, splits_frozen=False,
                      model_updates=0, scientific_admission=False, training_ready=False,
                      acceptance='author-checked', number_status='measured', phases_seconds=phases,
                      seconds=seconds, cpu_seconds=time.process_time()-cpu,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      code_sha256=file_sha(Path(__file__)),
                      normalization_code_sha256=file_sha(Path(__file__).resolve().parents[1]/'kernel/capability_split.py'))
        receipt = write_committed(output/'report.json', result)
        need(not errors, 'SHINGLE_DOCUMENT_FAILURES')
        write_committed(output/'COMMITTED', dict(report_sha256=receipt))
        print(json.dumps(dict(pin=receipt, counts=counts, seconds=seconds)), flush=True)
    except BaseException as exc:
        write_committed(output/'failure.json', dict(error=repr(exc), phases_seconds=phases, document_errors=errors))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args(); run(a.source, a.pin, a.output)

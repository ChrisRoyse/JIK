"""Cold all-document reconstruction of exact leakage shingles from retained bytes."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need
from text.legal_stock import SOURCES
from text.legal_shingles_input import Shingles, check_document, header


def exact(ids, values):
    pack = struct.Struct('<5I').pack
    expected = {pack(*ids[j:j+5]) for j in range(max(0, len(ids)-4))}
    actual = values.view('V20').reshape(-1).tobytes()
    actual = {actual[j:j+20] for j in range(0, len(actual), 20)}
    need(len(actual) == len(values) and actual == expected, 'SHINGLE_EXACT_RECONSTRUCTION')


def refused(action):
    try:
        action()
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    raise AssertionError('SHINGLE_NEGATIVE_PASSED')


def run(root, pin, output):
    start = tick = time.perf_counter(); cpu = time.process_time()
    output.mkdir(parents=True, exist_ok=False); bank = Shingles(root, pin)
    r = bank.report; source = Path(r['source']['path'])
    need(file_sha(source/'report.json') == r['source']['pin'], 'SHINGLE_READBACK_SOURCE_PIN')
    stock = json.loads((source/'report.json').read_bytes())
    need(file_sha(source/'records.jsonl') == stock['files']['records.jsonl']['sha256'],
         'SHINGLE_READBACK_INDEX_PIN')
    words = {}; vocabulary = []
    with (root/'words.jsonl').open('rb') as stream:
        for line in stream:
            number, word = json.loads(line)
            need(number == len(vocabulary) and isinstance(word, str) and word not in words,
                 'SHINGLE_READBACK_VOCABULARY')
            words[word] = number; vocabulary.append(word)
    need(len(words) == r['counts']['vocabulary'], 'SHINGLE_READBACK_VOCABULARY_COUNT')
    phases = dict(authenticate_and_dictionary=time.perf_counter()-tick)
    tick = time.perf_counter(); document = total_words = total_bytes = windows = eligible = 0
    groups = Counter(); proof = None; seen_words = set(); largest = 0
    with (source/'records.jsonl').open('rb') as indexed:
        for domain, name, expected, _ in SOURCES:
            h = hashlib.sha256(); count = offset = 0
            with (source/'raw'/name).open('rb') as stream:
                for raw in stream:
                    h.update(raw); record = json.loads(indexed.readline()); meta = bank.documents[document]
                    text = json.loads(raw)['text']; original = text.encode()
                    need(record['id'] == meta['record'] and meta['source'] == domain+':'+name
                         and meta['ordinal'] == count and meta['offset'] == offset
                         and meta['line_bytes'] == len(raw)
                         and meta['line_sha256'] == hashlib.sha256(raw).hexdigest()
                         and meta['text_sha256'] == hashlib.sha256(original).hexdigest(),
                         'SHINGLE_READBACK_SOURCE_BINDING')
                    tokens = re.sub(r'\d+(?:[.,]\d+)*', '#', ' '.join(text.casefold().split())).split()
                    ids = [words[w] for w in tokens]; seen_words.update(ids)
                    lo, hi = map(int, bank.offsets[document:document+2]); values = bank.values[lo:hi]
                    need(meta['words'] == len(tokens), 'SHINGLE_READBACK_WORD_COUNT')
                    check_document(meta, (lo, hi), bank.counts[document], values, len(words))
                    exact(ids, values)
                    if proof is None and len(values) > 1:
                        proof = (meta, (lo, hi), ids, np.array(values, copy=True))
                    total_words += len(tokens); total_bytes += len(original); windows += max(0, len(tokens)-4)
                    eligible += len(tokens) >= 20; largest = max(largest, len(tokens)); groups[domain] += 1
                    document += 1; count += 1; offset += len(raw)
            need(count == expected and h.hexdigest() == stock['files']['raw/'+name]['sha256'],
                 'SHINGLE_READBACK_RAW_PIN')
        need(not indexed.read(1), 'SHINGLE_READBACK_TRAILING_INDEX')
    counts = dict(documents=document, words=total_words, vocabulary=len(seen_words),
                  unique_per_document_memberships=int(bank.offsets[-1]), raw_windows=windows,
                  text_bytes=total_bytes, eligible_documents=eligible,
                  maximum_document_words=largest, sources=dict(groups))
    need(counts == r['counts'], 'SHINGLE_READBACK_CENSUS')
    phases['all_original_document_reconstruction'] = time.perf_counter()-tick; tick = time.perf_counter()
    chosen = sorted(range(document), key=lambda i: bank.documents[i]['record'])[:17]
    consumer = []
    for i in chosen:
        meta = bank.documents[i]; values = bank.document(i, meta['record'])
        consumer.append(dict(index=i, record=meta['record'], tuples=len(values),
                             tuple_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    meta, offsets, ids, values = proof
    corrupt = np.array(values, copy=True); corrupt[0, 0] = (int(corrupt[0, 0])+1) % len(words)
    test = output/'negative-header'; test.mkdir()
    shutil.copy2(root/'report.json', test/'report.json')
    write_committed(test/'COMMITTED', dict(report_sha256='0'*64))
    changed = deepcopy(meta); changed['start'] += 1
    controls = dict(
        missing_tuple=refused(lambda: exact(ids, values[:-1])),
        changed_word=refused(lambda: exact(ids, corrupt)),
        repeated_tuple=refused(lambda: exact(ids, np.concatenate([values, values[:1]]))),
        changed_address=refused(lambda: check_document(changed, offsets, len(ids), values, len(words))),
        wrong_record=refused(lambda: bank.document(meta['index'], '0'*64)),
        wrong_report_pin=refused(lambda: header(root, '0'*64)),
        uncommitted=refused(lambda: header(test, pin)),
        implicit_training=refused(bank.training_authority))
    bank.fresh()
    write_committed(output/'consumer.json', consumer)
    phases['indexed_consumer_and_controls'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'SHINGLE_READBACK_TIMING')
    result = dict(schema='legal-exact-document-shingles-readback-v1', source=dict(path=str(root), pin=pin),
                  complete=True, counts=counts, all_original_documents_reconstructed=True,
                  all_dictionary_words_used=True, consumer_cases=len(consumer), controls=controls,
                  near_join_complete=False, splits_frozen=False, scientific_admission=False, model_updates=0,
                  acceptance='author-checked', number_status='measured', phases_seconds=phases,
                  seconds=seconds, cpu_seconds=time.process_time()-cpu,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  files={'consumer.json':file_sha(output/'consumer.json')}, code_sha256=file_sha(Path(__file__)))
    receipt = write_committed(output/'report.json', result)
    print(json.dumps(dict(pin=receipt, counts=counts, controls=controls, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args(); run(a.source, a.pin, a.output)

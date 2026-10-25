"""Source exposure in an immutable completed training journal; no model execution."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


def need(ok, code):
    if not ok:
        raise ValueError('MNLM_SOURCE_EXPOSURE_' + code)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, pin):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin, 'PIN')
    return json.loads(raw)


def save(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    return sha(path)


def checked(row, cursor, document, targets, parent, previous_model):
    body = {k: v for k, v in row.items() if k != 'hash'}
    need(set(row) == {'cursor', 'parent', 'hash', 'update'} and row['cursor'] == cursor
         and row['parent'] == parent and hashlib.sha256(encoded(body)).hexdigest() == row['hash'], 'CHAIN')
    u = row['update']
    need(u['step'] == cursor + 1 and u['document'] == document and u['targets'] == targets > 0, 'JOIN')
    need(previous_model is None or u['before'] == previous_model, 'MODEL_CHAIN')
    need(type(u['loss']) is float and math.isfinite(u['loss']), 'LOSS')
    stop = 0
    for tile in u['tiles']:
        need(tile['start'] == stop and type(tile['stop']) is int and stop < tile['stop'] <= targets, 'TILES')
        stop = tile['stop']
        need(type(tile['loss']) is float and math.isfinite(tile['loss']), 'LOSS')
    need(stop == targets and sum(t['loss'] for t in u['tiles']) == u['loss'], 'LOSS_SUM')
    return u


def negative(operation):
    try:
        operation()
    except ValueError as error:
        need(str(error).startswith('MNLM_SOURCE_EXPOSURE_'), 'NEGATIVE_REASON')
        return str(error)
    raise AssertionError('MNLM_SOURCE_EXPOSURE_NEGATIVE_ACCEPTED')


def aggregate(rows, first, last):
    n = len(rows); targets = sum(t for t, _ in rows)
    return dict(updates=n, targets=targets, first_step=first, last_step=last,
                document_mean_loss=math.fsum(v for _, v in rows)/n,
                target_weighted_loss=math.fsum(t*v for t, v in rows)/targets)


def run(a):
    began = time.perf_counter(); cpu = time.process_time()
    a.output.mkdir(exist_ok=False)
    boundary = read(a.seal/'boundary.json', a.pin)
    plan = boundary['binding']['source']['plan']['train_documents']
    need(len(plan) == len(set(plan)) == boundary['branches']['shared']['step'] == 91540, 'SCOPE')
    binding = boundary['binding']['source']['metadata']
    corpus = read(a.corpus/'report.json', binding['corpus'])
    metadata = a.corpus/'documents.jsonl'; entry = corpus['files']['documents.jsonl']
    need(entry['sha256'] == binding['documents'] and metadata.stat().st_size == entry['bytes'], 'METADATA')
    selected = set(plan); mapping = {}; digest = hashlib.sha256(); metadata_rows = 0
    with metadata.open('rb') as stream:
        for document, raw in enumerate(stream):
            digest.update(raw); value = json.loads(raw); metadata_rows += 1
            # The corpus reader addresses documents by physical row; ordinal is source-local.
            need(type(value['ordinal']) is int and value['ordinal'] >= 0 and isinstance(value['source_path'], str)
                 and value['source_path'], 'METADATA_ORDER')
            if document in selected:
                mapping[document] = (value['source_path'], value['atom_end']-value['atom_start']-1)
    need(digest.hexdigest() == entry['sha256'] and set(mapping) == selected, 'METADATA_PIN')
    counts = Counter(source for source, _ in mapping.values()); ranks = Counter()
    journal = a.seal/'shared/steps.jsonl'; expected_pin = boundary['files']['shared/steps.jsonl']
    journal_hash = hashlib.sha256(); rows_hash = hashlib.sha256(); groups = {}; quarters = {}; runs = []
    transitions = Counter(); parent = previous_model = None; first_record = None; total_targets = 0
    with journal.open('rb') as stream, (a.output/'rows.jsonl').open('xb') as out:
        for cursor, document in enumerate(plan):
            raw = stream.readline(); need(raw.endswith(b'\n'), 'JOURNAL_SHORT')
            journal_hash.update(raw); row = json.loads(raw)
            if parent is None: parent = row['parent']; first_record = deepcopy(row)
            source, targets = mapping[document]
            u = checked(row, cursor, document, targets, parent, previous_model)
            parent, previous_model = row['hash'], u['after']; step = cursor+1
            quarter = 4*ranks[source]//counts[source]; ranks[source] += 1; total_targets += targets
            for key, collection in ((source, groups), ((source, quarter), quarters)):
                state = collection.setdefault(key, dict(first=step, last=step, rows=[]))
                state['last'] = step; state['rows'].append((targets, u['loss']))
            if not runs or runs[-1]['source'] != source:
                if runs: transitions[(runs[-1]['source'], source)] += 1
                runs.append(dict(source=source, first_step=step, last_step=step, updates=0, targets=0))
            run = runs[-1]; run['last_step'] = step; run['updates'] += 1; run['targets'] += targets
            compact = dict(step=step, document=document, source=source, targets=targets,
                           source_training_ordinal=ranks[source]-1, quarter=quarter, loss=u['loss'], journal_hash=parent)
            raw = encoded(compact)+b'\n'; out.write(raw); rows_hash.update(raw)
        need(not stream.read(1) and journal_hash.hexdigest() == expected_pin, 'JOURNAL_PIN')
        need(total_targets == 6455636, 'TARGET_TOTAL'); out.flush(); os.fsync(out.fileno())
    controls = {}
    document = plan[0]; targets = mapping[document][1]
    for name, change in (
        ('wrong_document', lambda r: r['update'].__setitem__('document', plan[1])),
        ('wrong_targets', lambda r: r['update'].__setitem__('targets', targets+1)),
        ('reordered_step', lambda r: r.__setitem__('cursor', 1)),
        ('wrong_loss', lambda r: r['update'].__setitem__('loss', r['update']['loss']+1)),
        ('missing_tile', lambda r: r['update']['tiles'].pop()),
    ):
        bad = deepcopy(first_record); change(bad)
        bad['hash'] = hashlib.sha256(encoded({k:v for k,v in bad.items() if k != 'hash'})).hexdigest()
        controls[name] = negative(lambda: checked(bad,0,document,targets,first_record['parent'],None))
    controls['wrong_pin'] = negative(lambda: read(a.seal/'boundary.json','0'*64))
    controls['broken_model_chain'] = negative(lambda: checked(first_record,0,document,targets,first_record['parent'],'0'*64))
    stats = []
    for source, g in sorted(groups.items()):
        value = dict(source=source, **aggregate(g['rows'], g['first'], g['last']))
        value['update_fraction'] = value['updates']/len(plan)
        value['target_fraction'] = value['targets']/total_targets
        value['quarters'] = [dict(quarter=q, **aggregate(s['rows'],s['first'],s['last']))
                             for (name,q),s in sorted(quarters.items()) if name == source]
        stats.append(value)
    summary = dict(updates=len(plan), targets=total_targets, metadata_documents=metadata_rows,
                   sources=stats, runs=runs, transitions=[dict(left=l,right=r,count=n) for (l,r),n in sorted(transitions.items())],
                   training_order='frozen_document_plan', loss_scope='online_before_each_update_not_heldout_accuracy',
                   causal_order_effect_established=False, forgetting_established=False, architecture_winner=False)
    summary_pin = save(a.output/'summary.json', summary)
    files = {str(a.seal/'boundary.json'):a.pin, str(a.corpus/'report.json'):binding['corpus'],
             str(metadata):entry['sha256'], str(journal):expected_pin}
    need(all(sha(p) == h for p,h in files.items()), 'SOURCE_CHANGED')
    report = dict(schema='placement-source-exposure-v1', complete=True, inputs=files,
                  output_files={'rows.jsonl':rows_hash.hexdigest(),'summary.json':summary_pin},
                  source_bytes=sum(Path(p).stat().st_size for p in files), journal_bytes=journal.stat().st_size,
                  code={str(Path(__file__).resolve()):sha(__file__)}, negative_controls=controls,
                  seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  gpu_used=False, optimizer_updates=0, acceptance='author-checked', number_status='measured')
    save(a.output/'report.json', report)
    print(json.dumps(dict(report_pin=sha(a.output/'report.json'), summary=summary, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seal','corpus','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pin',required=True); args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists(): save(args.output/'failure.json',dict(error=repr(error)))
        raise

"""Capture retained legal source bytes into MNLM without modifying the Calyx store."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.training_scope import POLICY_SHA256, require_training_source
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from text.evaluation_multiscale import file_sha

EXPERIMENT = 'legal-source-growth-20261022'
ROOT = Path('/zfs/archive/calyx/freelaw-cuyahoga')
SOURCES = (
    ('cuyahoga-eighth-district', 'ingest/cap-cuyahoga.jsonl', 2422,
     '2b112b71cc47becb59c08e92a4bedbfc01e8f6de518dda0a6f62ca9f35f58093'),
    ('cuyahoga-eighth-district', 'ingest/ohctapp8-modern.jsonl', 16094,
     'bf90ef2174931d37f23d3fb229096aee87943afefbf180fe1df3fb6e2ddd2c50'),
    ('ohio-supreme-court', 'expansion-v5/ohio-sc-whole/ingest.jsonl', 96537, None))
AUXILIARY = ('ingest/ingest_manifest.json', 'ingest/dedup_dropped.jsonl',
             'expansion-v5/ohio-sc-whole/expansion_manifest.json',
             'expansion-v5/ohio-sc-whole/opinion_aliases.jsonl',
             'expansion-v5/ohio-sc-whole/dedup_dropped.jsonl')


def need(ok, code):
    if not ok:
        raise ValueError('MNLM_LEGAL_STOCK_' + code)


def identity(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, modified_ns=s.st_mtime_ns)


def record(row, source, ordinal, offset, raw):
    need(set(row) == {'text', 'metadata', 'anchors'}, 'ROW_SCHEMA')
    text = row['text']
    need(isinstance(text, str) and bool(text.strip()), 'EMPTY_TEXT')
    m = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    need(isinstance(m, dict) and isinstance(row['anchors'], list), 'METADATA')
    for name in ('opinion_id', 'cluster_id', 'docket_id', 'court_id', 'license', 'source_dataset'):
        need(isinstance(m.get(name), str) and bool(m[name]), 'MISSING_' + name.upper())
    data = text.encode()
    pin = hashlib.sha256(data).hexdigest()
    if 'ingest_text_sha256' in m:
        need(m['ingest_text_sha256'] == pin, 'INGEST_TEXT_HASH')
    return dict(id=digest(dict(source=source, ordinal=ordinal)), source=source, ordinal=ordinal,
        offset=offset, line_bytes=len(raw), line_sha256=hashlib.sha256(raw).hexdigest(),
        text_sha256=pin, normalized_sha256=digest(' '.join(text.casefold().split())),
        text_bytes=len(data), characters=len(text), metadata_sha256=digest(m),
        anchors_sha256=digest(row['anchors']),
        **{k:m.get(k) for k in ('opinion_id', 'cluster_id', 'docket_id', 'court_id',
            'docket_number', 'date_filed', 'retrieval_ts', 'license', 'source_dataset',
            'source_url', 'cuyahoga_signal')})


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    start = tick = time.perf_counter()
    paths = [n for _, n, _, _ in SOURCES] + list(AUXILIARY)
    before = {n:identity(ROOT/n) for n in paths}
    write_committed(output/'before.json', dict(root=str(ROOT), files=before,
        experiment=EXPERIMENT, policy_sha256=POLICY_SHA256))
    phases = dict(preflight=time.perf_counter()-tick)
    files, counts, groups = {}, {}, defaultdict(list)
    try:
        with (output/'records.jsonl').open('xb') as records:
            for source, name, expected, pin in SOURCES:
                require_training_source(source, experiment=EXPERIMENT)
                tick = time.perf_counter()
                destination = output/'raw'/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                h, offset, count, byte_count, chars, maximum = hashlib.sha256(), 0, 0, 0, 0, 0
                courts, signals, licences, dates = Counter(), Counter(), Counter(), Counter()
                with (ROOT/name).open('rb') as inp, destination.open('xb') as out:
                    for ordinal, raw in enumerate(inp):
                        out.write(raw); h.update(raw)
                        r = record(json.loads(raw), source + ':' + name, ordinal, offset, raw)
                        records.write(canonical(r)+b'\n')
                        for kind in ('text_sha256', 'normalized_sha256', 'opinion_id', 'cluster_id', 'docket_id'):
                            groups[kind+':'+r[kind]].append(r['id'])
                        offset += len(raw); count += 1; byte_count += r['text_bytes']; chars += r['characters']
                        maximum = max(maximum, r['characters'])
                        courts[r['court_id']] += 1; signals[str(r['cuyahoga_signal'])] += 1
                        licences[r['license']] += 1; dates['present' if r['date_filed'] else 'absent'] += 1
                    out.flush(); os.fsync(out.fileno())
                need(count == expected and (pin is None or pin == h.hexdigest()), 'SOURCE_PIN_OR_COUNT')
                need(identity(ROOT/name) == before[name], 'SOURCE_CHANGED')
                files['raw/'+name] = dict(bytes=offset, sha256=h.hexdigest(), source=str(ROOT/name))
                counts[name] = dict(records=count, text_bytes=byte_count, characters=chars,
                    max_characters=maximum, courts=dict(courts), signals=dict(signals),
                    licences=dict(licences), dates=dict(dates))
                phases['capture:'+name] = time.perf_counter()-tick
                print(json.dumps(dict(captured=name, records=count, seconds=phases['capture:'+name])), flush=True)
            records.flush(); os.fsync(records.fileno())
        tick = time.perf_counter()
        for name in AUXILIARY:
            dest = output/'raw'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with (ROOT/name).open('rb') as inp, dest.open('xb') as out:
                for raw in iter(lambda:inp.read(8 << 20), b''):
                    h.update(raw); out.write(raw)
                out.flush(); os.fsync(out.fileno())
            need(identity(ROOT/name) == before[name], 'AUXILIARY_CHANGED')
            files['raw/'+name] = dict(bytes=dest.stat().st_size, sha256=h.hexdigest(), source=str(ROOT/name))
        with (output/'shared-identities.jsonl').open('xb') as stream:
            for key, values in sorted(groups.items()):
                if len(values) > 1:
                    stream.write(canonical(dict(identity=key, records=values))+b'\n')
            stream.flush(); os.fsync(stream.fileno())
        phases['auxiliary_and_identity_index'] = time.perf_counter()-tick; tick = time.perf_counter()
        after = {n:identity(ROOT/n) for n in paths}
        need(before == after, 'FINAL_SOURCE_CHANGE')
        write_committed(output/'after.json', after)
        for name in ('before.json','after.json','records.jsonl','shared-identities.jsonl'):
            files[name] = dict(bytes=(output/name).stat().st_size, sha256=file_sha(output/name))
        phases['final_hashes'] = time.perf_counter()-tick
        seconds = time.perf_counter()-start
        need(sum(phases.values())/seconds >= .9, 'TIMING_COVERAGE')
        report = dict(schema='legal-source-stock-v1', complete=True, experiment=EXPERIMENT,
            source_root=str(ROOT), files=files, counts=counts, policy_sha256=POLICY_SHA256,
            records=sum(v['records'] for v in counts.values()),
            unique_texts=sum(k.startswith('text_sha256:') for k in groups),
            identity_counts=dict(Counter(k.split(':',1)[0] for k in groups)),
            original_sources_unchanged=True, full_records_retained=True,
            ancestry_closure_complete=False, splits_frozen=False, native_readings_reused=False,
            historical_source_exposure='Cuyahoga prior MNLM artifacts retained; encoder pretraining unknown',
            judicial_outcomes_are_ground_truth=False, training_ready=False, scientific_admission=False,
            model_updates=0, serving=False, acceptance='author-checked', number_status='measured',
            phases_seconds=phases, wall_seconds=seconds, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            code_sha256=file_sha(Path(__file__)))
        pin = write_committed(output/'report.json', report)
        write_committed(output/'COMMITTED', dict(report_sha256=pin))
        print(json.dumps(dict(report_sha256=pin, records=report['records'], unique_texts=report['unique_texts'],
                              seconds=seconds)), flush=True)
    except BaseException as error:
        write_committed(output/'failure.json', dict(error=str(error), phases=phases))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    capture(p.parse_args().output)

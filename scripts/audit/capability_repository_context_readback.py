"""Read the coding-context census against Git bytes and exercise corruption refusals."""
import argparse
import ast
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from oracle.prepare_swe import source_files
from senses.trajectory_population import read_pinned


def field_readback(raw, field):
    need(len(raw) == field['bytes'] and sha(raw) == field['sha256'], 'CONTEXT_READER_FILE')
    cursor, pieces = 0, []
    for part in field['parts']:
        lo, hi = part['start'], part['end']
        need(lo == cursor and lo < hi <= len(raw), 'CONTEXT_READER_SPAN')
        piece = raw[lo:hi]
        need(sha(piece) == part['sha256'], 'CONTEXT_READER_FRAGMENT')
        piece.decode('utf-8'); pieces.append(piece); cursor = hi
    need(cursor == len(raw) and b''.join(pieces) == raw, 'CONTEXT_READER_CONSERVATION')
    return pieces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('census', 'source-root', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--pin', required=True)
    a = p.parse_args(); start = time.perf_counter()
    need(not a.output.exists(), 'CONTEXT_READER_EXISTS')
    report = read_pinned(a.census / 'report.json', a.pin)
    values = {}
    for name, info in report['files'].items():
        raw = (a.census / name).read_bytes()
        need(dict(bytes=len(raw), sha256=sha(raw)) == info, 'CONTEXT_READER_REPORT_FILE')
        values[name] = json.loads(raw)
    rows = values['repositories.json']; unique, controls, total, occurrences = {}, [], 0, 0
    snapshot_bytes = 0; source_pins = {}
    for row in rows:
        source = read_pinned(Path(row['source']['path']), row['source']['sha256'])
        source_pins[row['source']['path']] = row['source']['sha256']
        need(source['src']['base_commit'] == row['base_commit'] and source['repo'] == row['repo'],
             'CONTEXT_READER_GIT_JOIN')
        files = source_files(a.source_root, source['src'])
        need(len(files) == row['snapshot_files'] and sum(map(len, files.values())) == row['snapshot_bytes'],
             'CONTEXT_READER_SNAPSHOT')
        snapshot_bytes += row['snapshot_bytes']
        for field in row['files']:
            raw = files[field['path']]; pieces = field_readback(raw, field)
            total += len(raw); occurrences += len(pieces)
            for piece in pieces: unique[sha(piece)] = piece
            if field['ast'] is not None:
                counts = Counter(type(n).__name__ for n in ast.walk(ast.parse(raw)))
                need({k: counts[k] for k in field['ast']} == field['ast'], 'CONTEXT_READER_AST')
            if not controls and len(pieces) > 1:
                def rejected(name, candidate, record):
                    try: field_readback(candidate, record)
                    except ValueError as exc:
                        controls.append(dict(control=name, error=str(exc), persisted_state_unchanged=True))
                    else: raise ValueError('CONTEXT_READER_CONTROL_ACCEPTED:'+name)
                rejected('changed_source_byte', bytes([raw[0] ^ 1])+raw[1:], field)
                broken = copy.deepcopy(field); broken['parts'] = broken['parts'][1:]
                rejected('dropped_first_fragment', raw, broken)
                broken = copy.deepcopy(field); broken['parts'][0]['sha256'] = '0'*64
                rejected('changed_fragment_address', raw, broken)
        need(dict(Counter(f['kind'] for f in row['files'])) == row['kinds'], 'CONTEXT_READER_KINDS')
    need(len(rows) == report['groups'] and sum(len(r['files']) for r in rows) == report['candidate_files']
         and total == report['candidate_bytes'] and occurrences == report['fragment_occurrences']
         and len(unique) == report['unique_fragments'] and sum(map(len, unique.values())) == report['unique_fragment_bytes']
         and snapshot_bytes == report['snapshot_bytes'] and len(controls) == 3, 'CONTEXT_READER_COUNTS')
    for path, pin in source_pins.items(): need(sha(Path(path).read_bytes()) == pin, 'CONTEXT_READER_SOURCE_UNCHANGED')
    need(sha((a.census / 'report.json').read_bytes()) == a.pin, 'CONTEXT_READER_UNCHANGED')
    result = dict(schema='capability-repository-context-readback-v1', source=dict(path=str(a.census), pin=a.pin),
        groups=len(rows), candidate_files=report['candidate_files'], candidate_bytes=total,
        unique_fragments=len(unique), controls=controls, shared_git_authentication='oracle.prepare_swe.source_files',
        independent_fragment_and_count_readback=True, number_status='measured', acceptance='author-checked',
        project_code_executed=False, native_measured=False, scientific_admission=False, learned_capability=False,
        elapsed=time.perf_counter()-start, producer_sha256=sha(Path(__file__).read_bytes()))
    pin = write_committed(a.output, result)
    print(json.dumps(dict(pin=pin, groups=len(rows), files=report['candidate_files'], controls=controls)))


if __name__ == '__main__': main()

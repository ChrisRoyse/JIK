"""Independent standard-library recount of persisted placement diagnostic scores."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


def need(ok, code):
    if not ok: raise ValueError(code)


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def pinned(path, pin):
    need(path.is_file() and not path.is_symlink() and sha(path) == pin, 'SCORE_FILE_PIN')
    return json.loads(path.read_bytes())


def avg(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def equal(a, b):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float): return isinstance(b, (float, int)) and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12
    return a == b


def require_summary(actual, expected):
    need(equal(actual, expected), 'SCORE_SUMMARY_MISMATCH')


def row_identity(r, selected, donor, offset):
    m = donor['metadata'][str(selected['document'])]
    need((r['document'], r['side'], r['component'], r['offset'], r['proposal'], r['source']) ==
         (selected['document'], selected['side'], selected['case_component'], offset, 'assembled', m['source']),
         'SCORE_ROW_IDENTITY')
    expected = donor['donors'][str(r['document'])][offset]
    need(r['donor'] == expected and (r['shuffled'] is None) == (expected == -1), 'SCORE_ROW_DONOR')


def coverage(rows, plan, donor):
    cursor = 0
    for selected in plan['evaluations']:
        n = selected['stop'] - selected['start'] - 1
        need(n == len(donor['donors'][str(selected['document'])]), 'SCORE_DOCUMENT_BOUNDS')
        for offset in range(n):
            need(cursor < len(rows), 'SCORE_ROW_COVERAGE')
            row_identity(rows[cursor], selected, donor, offset)
            cursor += 1
    need(cursor == len(rows), 'SCORE_ROW_COVERAGE')


def groups(rows):
    out = defaultdict(list)
    for r in rows: out['|'.join((r['side'], r['source'], r['proposal']))].append(r)
    return out


def summarize(rows, roster):
    out = {}
    for key, rs in groups(rows).items():
        levels = range(len(rs[0]['target'])); paired = [r for r in rs if r['shuffled'] is not None]
        out[key] = dict(rows=len(rs), documents=len({r['document'] for r in rs}),
            code_accuracy=[avg(r['predicted'][j] == r['target'][j] for r in rs) for j in levels],
            prefix_accuracy=[avg(r['predicted'][:j+1] == r['target'][:j+1] for r in rs) for j in levels],
            roundtrip_prefix=[avg(r['predicted'][:j+1] == r['measured_codes'][:j+1] for r in rs) for j in levels],
            ce=[avg(r['ce'][j] for r in rs) for j in levels], brier=[avg(r['brier'][j] for r in rs) for j in levels],
            conditional_energy=avg(r['conditional_energy']['loss'] for r in rs),
            conditional_frozen_baseline=avg(r['conditional_frozen_baseline']['loss'] for r in rs),
            paired_rows=len(paired), missing_donor_rows=len(rs)-len(paired),
            paired_energy_advantage=avg(r['shuffled']['energy']['loss']-r['conditional_energy']['loss'] for r in paired),
            paired_ce_advantage=[avg(r['shuffled']['ce'][j]-r['ce'][j] for r in paired) for j in levels] if paired else None,
            count_accuracy={n:avg(r['target_count'][n] == r['predicted_count'][n] for r in rs) for n in roster},
            baseline_accuracy={n:[avg(r['baselines'][n][j] == r['target'][j] for r in rs) for j in levels] for n in rs[0]['baselines']},
            baseline_available={n:[sum(r['baselines'][n][j] >= 0 for r in rs) for j in levels] for n in rs[0]['baselines']},
            native_distance={n:{a:avg(r['native_distance'][n][a] for r in rs) for a in ('model', 'copy_current', 'train_mean')} for n in roster})
    return out


def advantages(rows, roster):
    """Positive always favors the learned/correct-context arm; -1 codes stay misses."""
    out = {}
    for n in roster:
        for baseline in ('copy_current', 'train_mean'):
            out['native|' + n + '|' + baseline] = avg(r['native_distance'][n][baseline]-r['native_distance'][n]['model'] for r in rows)
        out['energy|' + n + '|frozen'] = avg(r['conditional_frozen_baseline']['lanes'][n]-r['conditional_energy']['lanes'][n] for r in rows)
        out['energy|' + n + '|shuffled'] = avg(r['shuffled']['energy']['lanes'][n]-r['conditional_energy']['lanes'][n] for r in rows if r['shuffled'] is not None)
    for j in range(len(rows[0]['target'])):
        for baseline in rows[0]['baselines']:
            out[f'prefix|{j+1}|{baseline}'] = avg(int(r['predicted'][:j+1] == r['target'][:j+1])-int(r['baselines'][baseline][:j+1] == r['target'][:j+1]) for r in rows)
        out[f'ce|{j+1}|shuffled'] = avg(r['shuffled']['ce'][j]-r['ce'][j] for r in rows if r['shuffled'] is not None)
    out['energy|joint_loss|frozen'] = avg(r['conditional_frozen_baseline']['loss']-r['conditional_energy']['loss'] for r in rows)
    out['energy|joint_loss|shuffled'] = avg(r['shuffled']['energy']['loss']-r['conditional_energy']['loss'] for r in rows if r['shuffled'] is not None)
    return out


def document_report(rows, roster):
    out = {}
    for key, rs in groups(rows).items():
        documents = defaultdict(list)
        for r in rs: documents[r['document']].append(r)
        ds = {str(d):dict(rows=len(v), paired_rows=sum(r['shuffled'] is not None for r in v), advantages=advantages(v, roster)) for d,v in documents.items()}
        scores = {}; atom_contrasts = advantages(rs, roster)
        for metric in next(iter(ds.values()))['advantages']:
            values = [v['advantages'][metric] for v in ds.values() if v['advantages'][metric] is not None]
            scores[metric] = dict(documents=len(values), missing=len(ds)-len(values),
                win=sum(v > 0 for v in values), tie=sum(v == 0 for v in values), loss=sum(v < 0 for v in values),
                mean_document_advantage=avg(values), atom_weighted_advantage=atom_contrasts[metric])
        out[key] = dict(documents=ds, contrasts=scores)
    return out


def controls(rows, plan, donor, summary, path, pin):
    out = {}
    def reject(name, fn, expected):
        try: fn()
        except ValueError as e:
            need(str(e) == expected, 'SCORE_CONTROL_WRONG_REFUSAL'); out[name] = str(e); return
        raise ValueError('SCORE_CORRUPTION_ACCEPTED:' + name)
    first = plan['evaluations'][0]
    for field, val in [('source', 'wrong-source'), ('side', 'train'), ('document', -1), ('offset', -1)]:
        bad = dict(rows[0]); bad[field] = val
        reject(field, lambda: row_identity(bad, first, donor, 0), 'SCORE_ROW_IDENTITY')
    bad = dict(rows[0]); bad['donor'] = -2
    reject('donor', lambda: row_identity(bad, first, donor, 0), 'SCORE_ROW_DONOR')
    reject('missing', lambda: coverage(rows[:-1], plan, donor), 'SCORE_ROW_COVERAGE')
    reject('duplicate', lambda: coverage(rows+[rows[-1]], plan, donor), 'SCORE_ROW_COVERAGE')
    bad = deepcopy(summary); bad[next(iter(bad))]['ce'][0] += .25
    reject('aggregate', lambda: require_summary(summary, bad), 'SCORE_SUMMARY_MISMATCH')
    reject('report_pin', lambda: pinned(path, '0'*64), 'SCORE_FILE_PIN')
    need(sha(path) == pin, 'SCORE_CONTROL_SOURCE_CHANGED')
    return out


def run(a):
    began = time.perf_counter(); a.output.mkdir(exist_ok=False)
    report = pinned(a.report, a.pin); census = pinned(a.census, a.census_pin)
    need(report['complete'] and not report['model_forward_replayed'] and not report['scientific_admission']
         and not report['model_promoted'] and report['binding'] == census['binding'], 'SCORE_AUTHORITY')
    base = Path(__file__).resolve().parents[1]
    need(all(sha(base/n) == h for n,h in report['code'].items()), 'SCORE_CODE_PIN')
    donor_path = Path(census['checkpoint']).parent.parent/'donors.json'
    donor = pinned(donor_path, census['donors_pin']); plan = report['binding']['source']['plan']; roster = report['roster']
    records = {}; summaries = {}; paired = {}; file_pins = {}
    for stage in ('initial', 'final'):
        path = a.report.parent/report['scores'][stage]['path']; pin = report['scores'][stage]['pin']
        need(sha(path) == pin, 'SCORE_FILE_PIN'); file_pins[str(path)] = pin
        with path.open() as f: rows = [json.loads(line) for line in f]
        coverage(rows, plan, donor); records[stage] = rows; summaries[stage] = summarize(rows, roster)
        require_summary(summaries[stage], report['summaries'][stage]); paired[stage] = document_report(rows, roster)
        need(sha(path) == pin, 'SCORE_FILE_CHANGED')
    need(len(records['initial']) == len(records['final']) == 24822, 'SCORE_TOTAL')
    invariant = ('document', 'offset', 'source', 'side', 'component', 'proposal', 'target', 'target_count', 'donor', 'baselines', 'conditional_frozen_baseline')
    for before, after in zip(records['initial'], records['final']):
        need(all(before[k] == after[k] for k in invariant) and all(before['native_distance'][n][v] == after['native_distance'][n][v]
             for n in roster for v in ('copy_current', 'train_mean')), 'SCORE_STAGE_INVARIANTS')
    checks = controls(records['final'], plan, donor, summaries['final'], a.report, a.pin)
    need(all(sha(Path(n)) == h for n,h in file_pins.items()) and sha(a.report) == a.pin and sha(a.census) == a.census_pin, 'SCORE_INPUT_CHANGED')
    out = dict(complete=True, schema='placement-branch-score-cold-v1', input_pin=a.pin, census_pin=a.census_pin,
        code={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}, score_pins=file_pins,
        summaries=summaries, document_contrasts=paired, controls=checks,
        rows=sum(map(len, records.values())), documents=len(plan['evaluations']), wall=time.perf_counter()-began,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        interpretation='Saved-output score/source diagnostic. Positive contrasts favor learned/correct context. Document wins are descriptive, not independent atom trials or significance tests.',
        gpu_forward_replayed=False, source_payload_revalidated_by=a.pin, scientific_admission=False, model_promoted=False, acceptance='author-checked')
    path = a.output/'report.json'
    with path.open('x') as f: json.dump(out, f, sort_keys=True, separators=(',', ':'), allow_nan=False); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(pin=sha(path), rows=out['rows'], wall=out['wall'], controls=len(checks))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('report', 'census', 'output'): p.add_argument('--'+n, type=Path, required=True)
    for n in ('pin', 'census-pin'): p.add_argument('--'+n, required=True)
    args = p.parse_args()
    try: run(args)
    except BaseException as e:
        if args.output.is_dir(): (args.output/'failure.json').write_text(json.dumps(dict(error=type(e).__name__, message=str(e))))
        raise

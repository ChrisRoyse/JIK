"""Independent source/denominator/statistic readback of the kernel learning assay."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from model.capability_kernel_learning import KernelLearningCorpus, pinned


ARMS = ('initial', 'correct', 'shuffled')


def read(path): return json.loads(Path(path).read_bytes())


def mean(values):
    need(values and all(math.isfinite(v) for v in values), 'EVALUATION_READER_FINITE')
    return math.fsum(values) / len(values)


def close(a, b):
    need(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), 'EVALUATION_READER_STATISTIC')


def comparison(values):
    return dict(better=sum(v < 0 for v in values), worse=sum(v > 0 for v in values),
                ties=sum(v == 0 for v in values), mean_delta=mean(values))


def source_bytes(corpus, key):
    task = corpus.tasks[key]; field = task['selections']['targets']['fields'][0]['id']
    record = corpus.sources[task['population']].population.fields[field]
    raw = record['text'].encode('utf-8')
    need(sha(raw) == record['text_sha256'], 'EVALUATION_READER_SOURCE')
    return raw


def audit(corpus, report, rows):
    need([r['task'] for r in rows] == list(corpus.query) and len(rows) == report['queries'],
         'EVALUATION_READER_DENOMINATOR')
    need(not any(report[k] for k in ('scientific_admission', 'learned_capability', 'serving',
             'sealed_capability_evaluation', 'generated_surface_evaluated'))
         and report['exposure'] == 'development_exposed_group_holdout', 'EVALUATION_READER_SCOPE')
    counts = defaultdict(lambda: Counter({i:1 for i in range(258)}))
    for key in corpus.train:
        counts[corpus.tasks[key]['kind']].update([*source_bytes(corpus, key), 257])
    families = defaultdict(list)
    for row in rows:
        task = corpus.tasks[row['task']]; raw = source_bytes(corpus, row['task'])
        need(row['family'] == task['family'] and row['operation'] == task['kind']
             and row['group'] == task['split_group'] and row['source_bytes'] == len(raw)
             and row['source_sha256'] == sha(raw), 'EVALUATION_READER_TARGET')
        need(set(row['arms']) == set(ARMS), 'EVALUATION_READER_ARMS')
        for arm in row['arms'].values():
            need(arm['scored_positions'] == len(raw)+1 and arm['region_correct'] ==
                 (arm['target_region'] == arm['predicted_region']), 'EVALUATION_READER_POSITIONS')
            need(set(arm['native_similarity']) == set(arm['reciprocal_loss']) ==
                 {s.name for s in corpus.specs}, 'EVALUATION_READER_INSTRUMENTS')
        need(len({v['target_region'] for v in row['arms'].values()}) == 1,
             'EVALUATION_READER_REGION_TARGET')
        c = counts[task['kind']]; total = sum(c.values())
        close(row['unigram_loss'], mean([-math.log(c[t]/total) for t in [*raw, 257]]))
        families[row['family']].append(row)
    need(set(families) == set(report['families']), 'EVALUATION_READER_FAMILIES')
    metrics = {}
    for family, selected in families.items():
        saved = report['families'][family]
        need(saved['tasks'] == len(selected) and saved['groups'] == len({r['group'] for r in selected}),
             'EVALUATION_READER_GROUPS')
        cell = dict(tasks=len(selected), groups=saved['groups'], surface={}, region_correct={},
                    per_instrument={})
        for arm in ARMS:
            values = [r['arms'][arm] for r in selected]
            cell['surface'][arm] = mean([v['surface_loss'] for v in values])
            cell['region_correct'][arm] = sum(v['region_correct'] for v in values)
            close(cell['surface'][arm], saved['arms'][arm]['surface_loss'])
            close(cell['region_correct'][arm]/len(selected), saved['arms'][arm]['region_accuracy'])
            for metric in ('native_similarity', 'reciprocal_loss'):
                for spec in corpus.specs:
                    close(mean([v[metric][spec.name] for v in values]), saved['arms'][arm][metric][spec.name])
        cell['surface']['train_unigram'] = mean([r['unigram_loss'] for r in selected])
        for baseline in ('initial', 'shuffled', 'train_unigram'):
            values = [r['arms']['correct']['surface_loss'] - (r['unigram_loss'] if baseline == 'train_unigram'
                      else r['arms'][baseline]['surface_loss']) for r in selected]
            actual = comparison(values); expected = saved['correct_vs_' + baseline]
            need(all(actual[k] == expected[k] for k in ('better', 'worse', 'ties')), 'EVALUATION_READER_PAIRED')
            close(actual['mean_delta'], expected['mean_delta'])
            cell['surface_vs_' + baseline] = actual
        for spec in corpus.specs:
            name = spec.name; per = {}
            for baseline in ('initial', 'shuffled', 'persistence'):
                # Negated similarity: all comparison deltas use lower-is-better.
                per['native_vs_' + baseline] = comparison([
                    (r['persistence_similarity'][name] if baseline == 'persistence'
                     else r['arms'][baseline]['native_similarity'][name]) -
                    r['arms']['correct']['native_similarity'][name] for r in selected])
            per['reciprocal_vs_shuffled'] = comparison([r['arms']['correct']['reciprocal_loss'][name]
                - r['arms']['shuffled']['reciprocal_loss'][name] for r in selected])
            cell['per_instrument'][name] = per
        cell['groups_detail'] = {}
        for group in sorted({r['group'] for r in selected}):
            group_rows = [r for r in selected if r['group'] == group]
            cell['groups_detail'][group] = dict(tasks=len(group_rows), correct_vs_shuffled=comparison([
                r['arms']['correct']['surface_loss'] - r['arms']['shuffled']['surface_loss'] for r in group_rows]))
        metrics[family] = cell
    return dict(families=metrics, full_surface_positions=sum(r['source_bytes']+1 for r in rows),
                target_bytes=sum(r['source_bytes'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--report-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'EVALUATION_READER_EXISTS')
    report = pinned(dict(path=str(a.root/'report.json'), pin=a.report_pin))
    need(sha(a.config.read_bytes()) == report['config_sha256'], 'EVALUATION_READER_CONFIG')
    raw = (a.root/'queries.jsonl').read_bytes()
    need(sha(raw) == report['query_records_sha256'], 'EVALUATION_READER_ROWS_PIN')
    corpus = KernelLearningCorpus(read(a.config)); rows = [json.loads(x) for x in raw.splitlines()]
    metrics = audit(corpus, report, rows)
    controls = []
    for name, code in [('removed_query', 'DENOMINATOR'), ('changed_target', 'TARGET'),
                       ('missing_eos_score', 'POSITIONS'), ('false_improvement', 'PAIRED'),
                       ('promoted_scope', 'SCOPE')]:
        r, rr = copy.deepcopy(report), copy.deepcopy(rows)
        if name == 'removed_query': rr.pop()
        if name == 'changed_target': rr[0]['source_sha256'] = '0'*64
        if name == 'missing_eos_score': rr[0]['arms']['correct']['scored_positions'] -= 1
        if name == 'false_improvement': r['families']['coding']['correct_vs_shuffled']['better'] += 1
        if name == 'promoted_scope': r['learned_capability'] = True
        try: audit(corpus, r, rr)
        except ValueError as exc:
            need('EVALUATION_READER_' + code in str(exc), 'EVALUATION_READER_CONTROL_CODE')
            controls.append(dict(name=name, refusal=str(exc)))
        else: raise AssertionError('EVALUATION_READER_CONTROL_ACCEPTED:' + name)
    result = dict(schema='capability-kernel-evaluation-readback-v1', source_report=a.report_pin,
        rows_sha256=sha(raw), producer_sha256=sha(Path(__file__).read_bytes()), controls=controls,
        queries=len(rows), **metrics, statistics_recomputed_without_producer_summary=True,
        source_target_bytes_and_train_only_baseline_recomputed=True, neural_forward_replayed=False,
        scientific_admission=False, learned_capability=False, serving=False,
        acceptance='author-checked', number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output, result), queries=len(rows),
                         positions=metrics['full_surface_positions'], controls=len(controls))), flush=True)


if __name__ == '__main__': main()

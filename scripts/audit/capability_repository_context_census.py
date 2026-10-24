"""Census retained base-commit API/test context without reading benchmark patches."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from claims.repository_cohort_population import fragments
from oracle.prepare_swe import source_files
from senses.trajectory_population import read_pinned


def read(path): return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'population', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--population-pin', required=True)
    args = parser.parse_args(); tick = start = time.perf_counter()
    need(not args.output.exists(), 'CONTEXT_CENSUS_EXISTS')
    pop = read_pinned(args.population / 'report.json', args.population_pin)
    need(pop['family'] == 'coding' and pop['complete'], 'CONTEXT_CENSUS_POPULATION')
    records = {}
    for name in ('tasks', 'fields'):
        raw = (args.population / (name+'.jsonl')).read_bytes()
        need(sha(raw) == pop['files'][name+'.jsonl']['sha256'], 'CONTEXT_CENSUS_SOURCE')
        records[name] = [json.loads(x) for x in raw.splitlines()]
    tasks = [t for t in records['tasks'] if t['kind'] == 'reproduce_recorded_patch']
    selected = {t['provenance']['task'] for t in tasks}
    config = read_pinned(args.config, pop['source']['config_sha256'])
    sources, bindings, loaded = {}, {}, {}
    def bound(path, pin):
        key = (str(path), pin)
        if key not in loaded: loaded[key] = read_pinned(Path(path), pin)
        bindings[str(path)] = pin
        return loaded[key]
    for spec in config['populations']:
        if spec['kind'] != 'repository_cohort_intake': continue
        native = spec['native']
        p = bound(Path(native['population']) / 'report.json', native['population_pin'])
        ob = bound(p['observations']['path'], p['observations']['pin'])
        witness = bound(ob['witness']['path'], ob['witness']['sha256'])
        catalog = bound(witness['sources']['path'], witness['sources']['sha256'])
        for item in catalog['cases']:
            if item['task'] not in selected or item['task'] in sources: continue
            source = bound(item['path'], item['sha256'])
            need(source['status'] == 'git_exact', 'CONTEXT_CENSUS_GIT_STATE')
            sources[item['task']] = (source, Path(witness['root']), item)
    need(set(sources) == selected, 'CONTEXT_CENSUS_TASK_COVERAGE')
    phases = dict(manifest_binding=time.perf_counter()-tick); tick = time.perf_counter()
    originals = {}
    for f in records['fields']:
        if f['role'] == 'original_source':
            s = f['source']['original_source']
            originals.setdefault(s['task'], {})[s['path']] = f
    rows, gaps, unique_fragments, source_cache = [], [], {}, {}
    for task, (source, root, item) in sorted(sources.items()):
        source_key = (source['repo'], source['src']['base_commit'])
        if source_key not in source_cache:
            files = source_files(root, source['src'])
            inventory = bound(Path(item['path']).parent / 'inventory.json', source['inventory_sha256'])
            need({n: dict(bytes=len(v), sha256=sha(v)) for n, v in sorted(files.items())} == inventory,
                 'CONTEXT_CENSUS_INVENTORY')
            source_cache[source_key] = files
        files = source_cache[source_key]
        for path, f in originals.get(task, {}).items():
            need(path in files and files[path] == f['text'].encode(), 'CONTEXT_CENSUS_ORIGINAL_JOIN')
        candidates = []
        for name, raw in sorted(files.items()):
            path = Path(name)
            public_test = path.suffix == '.py' and any(
                p in ('test', 'tests') or p.startswith('test_') for p in path.parts)
            kind = 'public_base_test' if public_test else 'python_source' if path.suffix == '.py' else (
                'documentation' if path.suffix in ('.md', '.rst') else None)
            if kind is None: continue
            try: text = raw.decode('utf-8')
            except UnicodeDecodeError:
                gaps.append(dict(task=task, path=name, code='NON_UTF8_BASE_FILE')); continue
            if not text.strip():
                gaps.append(dict(task=task, path=name, code='EMPTY_BASE_FILE')); continue
            structure = None
            if path.suffix == '.py':
                try:
                    tree = ast.parse(raw, filename=name)
                    counts = Counter(type(n).__name__ for n in ast.walk(tree))
                    structure = {k: counts[k] for k in ('FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Import', 'ImportFrom')}
                except (SyntaxError, ValueError) as exc:
                    gaps.append(dict(task=task, path=name, code='BASE_AST_UNAVAILABLE', error=type(exc).__name__))
            parts, cursor = [], 0
            for piece in fragments(raw):
                pin = sha(piece)
                need(pin not in unique_fragments or unique_fragments[pin] == piece, 'CONTEXT_CENSUS_COLLISION')
                unique_fragments[pin] = piece
                parts.append(dict(start=cursor, end=cursor+len(piece), sha256=pin)); cursor += len(piece)
            need(cursor == len(raw), 'CONTEXT_CENSUS_CONSERVATION')
            candidates.append(dict(path=name, kind=kind, bytes=len(raw), sha256=sha(raw), parts=parts,
                ast=structure, already_patch_original=name in originals.get(task, {}),
                source=dict(base_commit=source_key[1], repository=source_key[0], source_manifest=item)))
        rows.append(dict(task=task, source=item, base_commit=source_key[1], repo=source_key[0],
            snapshot_bytes=sum(map(len, files.values())), snapshot_files=len(files),
            licensing=source['licensing_files'], files=candidates,
            kinds=dict(Counter(c['kind'] for c in candidates)),
            existing_original_files=len(originals.get(task, {})),
            context_selection='inventory only; no training context selected',
            public_test_execution_observed=False))
    phases['git_authentication_and_census'] = time.perf_counter()-tick; tick = time.perf_counter()
    args.output.mkdir(parents=True)
    write_committed(args.output / 'sources.json', bindings)
    write_committed(args.output / 'repositories.json', rows)
    write_committed(args.output / 'gaps.json', gaps)
    phases['publication'] = time.perf_counter()-tick
    elapsed = time.perf_counter()-start
    report = dict(schema='capability-repository-context-census-v1',
        source_population=dict(path=str(args.population), pin=args.population_pin),
        config=dict(path=str(args.config), sha256=sha(args.config.read_bytes())),
        tasks=len(tasks), groups=len(rows), unique_snapshots=len(source_cache),
        snapshot_bytes=sum(sum(map(len, f.values())) for f in source_cache.values()),
        candidate_files=sum(len(r['files']) for r in rows),
        candidate_bytes=sum(f['bytes'] for r in rows for f in r['files']),
        unique_fragments=len(unique_fragments), unique_fragment_bytes=sum(map(len, unique_fragments.values())),
        fragment_occurrences=sum(len(f['parts']) for r in rows for f in r['files']),
        gaps=len(gaps), original_fields_matched=sum(r['existing_original_files'] for r in rows),
        all_groups_have_public_tests=all(r['kinds'].get('public_base_test', 0) for r in rows),
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in args.output.iterdir()},
        number_status='measured', acceptance='author-checked', benchmark_test_patches_read=False,
        model_inference=False, project_code_executed=False, scientific_admission=False,
        learned_capability=False, ready_for_training=False, native_window_count='unmeasured',
        producer_sha256=sha(Path(__file__).read_bytes()), phases_seconds=phases,
        elapsed=elapsed, phase_coverage=sum(phases.values())/elapsed,
        metric_mapping=['T1/spec27 source provenance', 'T3/spec27 capability input coverage'])
    need(report['phase_coverage'] >= .9, 'CONTEXT_CENSUS_TIMING')
    pin = write_committed(args.output / 'report.json', report)
    print(json.dumps(dict(pin=pin, **{k: report[k] for k in ('groups', 'candidate_files', 'candidate_bytes',
        'unique_fragments', 'fragment_occurrences', 'gaps', 'elapsed')})))


if __name__ == '__main__': main()

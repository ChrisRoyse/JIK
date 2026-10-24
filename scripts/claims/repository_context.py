"""Authenticated base-commit context and explicit static import paths; never executes code."""
import ast
from collections import defaultdict, deque
import json
from pathlib import Path, PurePosixPath

from claims.capability_population import need, sha
from oracle.prepare_swe import source_files
from senses.trajectory_population import read_pinned


POLICY = 'originals-direct-imports-reachable-base-tests-root-readme-v1'


def module_name(path):
    parts = list(PurePosixPath(path).with_suffix('').parts)
    if parts[0] in ('src', 'lib'): parts.pop(0)
    if parts and parts[-1] == '__init__': parts.pop()
    return '.'.join(parts)


def import_graph(files, catalog):
    modules, graph, gaps, declarations = defaultdict(set), {}, [], {}
    for name, item in catalog.items():
        if item['kind'] == 'python_source': modules[module_name(name)].add(name)
    for name, item in catalog.items():
        graph[name] = set()
        if not name.endswith('.py'): continue
        try: tree = ast.parse(files[name], filename=name)
        except (SyntaxError, ValueError) as exc:
            gaps.append(dict(path=name, code='BASE_IMPORT_PARSE_UNAVAILABLE', error=type(exc).__name__))
            continue
        package = module_name(name).split('.')
        if PurePosixPath(name).name != '__init__.py': package = package[:-1]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((alias.name, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                base = '.'.join(prefix + ([node.module] if node.module else []))
                imports.append((base, node.lineno))
                imports.extend(('.'.join(filter(None, [base, alias.name])), node.lineno)
                               for alias in node.names if alias.name != '*')
        for requested, line in imports:
            pieces = requested.split('.')
            while pieces and '.'.join(pieces) not in modules: pieces.pop()
            matches = modules.get('.'.join(pieces), set())
            if len(matches) > 1:
                gaps.append(dict(path=name, code='AMBIGUOUS_LOCAL_IMPORT', requested=requested,
                                 candidates=sorted(matches), line=line))
                continue
            for target in matches:
                if target != name:
                    graph[name].add(target)
                    declarations.setdefault((name, target), []).append(dict(line=line, requested=requested))
    return graph, declarations, gaps


def select_context(files, catalog, roots):
    need(roots and roots <= files.keys(), 'REPOSITORY_CONTEXT_ROOTS')
    graph, declarations, gaps = import_graph(files, catalog)
    reverse = defaultdict(set)
    for a, targets in graph.items():
        for b in targets: reverse[b].add(a)
    # One reverse walk yields an attested import path for each reachable test.
    paths = {r: [r] for r in sorted(roots)}
    queue = deque(sorted(roots))
    while queue:
        target = queue.popleft()
        for origin in sorted(reverse[target]):
            if origin not in paths:
                paths[origin] = [origin] + paths[target]
                queue.append(origin)
    selected = {}
    def add(path, reason):
        if path in catalog and path not in roots:
            selected.setdefault(path, []).append(reason)
    for root in sorted(roots):
        for dependency in sorted(graph.get(root, ())):
            add(dependency, dict(kind='direct_local_import', path=[root, dependency]))
    tests = [n for n, c in catalog.items() if c['kind'] == 'public_base_test']
    reached = [n for n in tests if n in paths]
    for name in reached:
        add(name, dict(kind='base_test_import_path', path=paths[name]))
    if not reached:
        gaps.append(dict(code='NO_STATIC_TEST_IMPORT_PATH', roots=sorted(roots)))
        for name in tests: add(name, dict(kind='same_commit_test_suite', path=[name]))
    for name, item in catalog.items():
        if item['kind'] == 'documentation' and len(PurePosixPath(name).parts) == 1 \
                and PurePosixPath(name).stem.lower() == 'readme':
            add(name, dict(kind='root_project_readme', path=[name]))
    edges = [dict(origin=a, target=b, declarations=declarations[(a, b)])
             for a in sorted(graph) for b in sorted(graph[a])]
    return dict(policy=POLICY, roots=sorted(roots), selected=selected, imports=edges, gaps=gaps,
                static_paths_establish_runtime_execution=False)


class RepositoryContext:
    """One authenticated snapshot and one parse per repository, reused across recorded attempts."""
    def __init__(self, root, pin, source_root, parent):
        self.root, self.pin = Path(root), pin
        self.report = read_pinned(self.root / 'report.json', pin)
        need(self.report['schema'] == 'capability-repository-context-census-v1'
             and self.report['source_population'] == dict(path=str(parent.root), pin=parent.pin),
             'REPOSITORY_CONTEXT_CENSUS')
        for name, spec in self.report['files'].items():
            raw = (self.root / name).read_bytes()
            need(dict(bytes=len(raw), sha256=sha(raw)) == spec, 'REPOSITORY_CONTEXT_CENSUS_FILE')
        self.rows = {r['task']: r for r in json.loads((self.root / 'repositories.json').read_bytes())}
        self.files, self.selections, self.catalogs = {}, {}, {}
        for task, row in sorted(self.rows.items()):
            source = read_pinned(Path(row['source']['path']), row['source']['sha256'])
            need(source['repo'] == row['repo'] and source['src']['base_commit'] == row['base_commit'],
                 'REPOSITORY_CONTEXT_COMMIT')
            files = source_files(Path(source_root), source['src'])
            catalog = {f['path']: f for f in row['files']}
            for name, item in catalog.items():
                need(name in files and sha(files[name]) == item['sha256']
                     and len(files[name]) == item['bytes'], 'REPOSITORY_CONTEXT_FILE')
            roots = {f['source']['original_source']['path'] for f in parent.fields.values()
                     if f['role'] == 'original_source' and f['source']['original_source']['task'] == task}
            self.files[task], self.catalogs[task] = files, catalog
            self.selections[task] = select_context(files, catalog, roots)

    def address(self, task, path):
        row, item = self.rows[task], self.catalogs[task][path]
        return dict(context_census=self.pin, task=task, repository=row['repo'],
                    base_commit=row['base_commit'], path=path, sha256=item['sha256'],
                    source_manifest=row['source'], source_kind=item['kind'])

    def clock(self, task):
        return dict(id='repository-base:'+self.rows[task]['base_commit'],
                    relation='base snapshot before current patch', wall_time=None,
                    ordering_evidence='retained task base_commit; no inferred wall clock')

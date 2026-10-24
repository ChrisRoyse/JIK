"""Acquire one original repository snapshot from a retained action/task contract."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.declared_deps import read as dependencies
from oracle.prepare_swe import source_files
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = 'MNLM_ACTION_REPOSITORY_SOURCE'
CAPS = dict(commit=1 << 20, tree=4 << 20, archive=16 << 20,
            files=5000, unpacked=64 << 20)


def select(root, contracts, pin, task):
    report = read_pinned(contracts, pin)
    require(report['schema'] == 'action-task-contracts-v1' and
            report['partition'] == 'development_inspection' and
            report['benchmark']['root'] == str(root), CODE, 'contract scope/root')
    selected = [c for c in report['contracts'] if c['task'] == task]
    require(len(selected) == 1, CODE, 'name a task with one inspected source attempt')
    c = selected[0]
    census_root = Path(report['census']['root'])
    census = read_pinned(census_root/'report.json', report['census']['pin'])
    chains = read_pinned(census_root/'chains.json', census['files']['chains.json']['sha256'])
    found = [x for x in chains if x['id'] == c['chain'] and x['task'] == task]
    require(len(found) == 1, CODE, 'chain identity')
    chain = found[0]
    shards = [s for s in census['source_scope']['shards'] if s['path'] == chain['shard']]
    require(len(shards) == 1 and shards[0]['sha256'] == chain['source_sha256'], CODE, 'source shard')
    shard = shards[0]
    benchmark = {k:v for k,v in report['benchmark'].items() if k != 'root'}
    benchmark['row'] = c['benchmark_row']
    attempt = {**shard, 'path': 'raw/swe-agent-trajectories/'+shard['path'], 'row':chain['row']}
    return c, dict(index=0, task=task, benchmark=benchmark, attempt=attempt,
                   partition='development_inspection', chain=c['chain'])


def acquire(root, contracts, pin, task, raw, output):
    started = time.perf_counter()
    root, raw, output = root.resolve(), raw.resolve(), output.resolve()
    require(raw.is_relative_to(root) and output.is_relative_to(root), CODE, 'MNLM data root required')
    c, case = select(root, contracts, pin, task)
    require(not raw.exists() and not output.exists(), CODE, 'fresh raw/output directories required')
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    token = subprocess.check_output(['gh','auth','token'], stderr=subprocess.DEVNULL).decode().strip()
    receipts, phases = [], {}

    def fetch(url, name, kind):
        tick = time.perf_counter()
        headers = {'User-Agent':'MNLM-bounded-action-oracle'}
        if url.startswith('https://api.github.com/'):
            headers['Authorization'] = 'Bearer '+token
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            data, status = r.read(CAPS[kind]+1), r.status
        require(len(data) <= CAPS[kind], CODE, 'source response above '+kind+' cap')
        path = raw/name
        with path.open('xb') as stream:
            stream.write(data)
        binding = dict(path=path.relative_to(root).as_posix(), bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest())
        receipts.append(dict(url=url, status=status, binding=binding,
                             seconds=time.perf_counter()-tick))
        return binding, data

    src = dict(task=task, base_commit=c['base_commit'])
    try:
        tick = time.perf_counter()
        base = 'https://api.github.com/repos/'+c['repo']+'/git/'
        src['commit'], data = fetch(base+'commits/'+c['base_commit'], 'commit.json', 'commit')
        commit = json.loads(data)
        require(commit['sha'] == c['base_commit'], CODE, 'commit identity')
        src['tree'], data = fetch(base+'trees/'+commit['tree']['sha']+'?recursive=1', 'tree.json', 'tree')
        tree = json.loads(data)
        require(tree['sha'] == commit['tree']['sha'] and tree['truncated'] is False,
                CODE, 'complete commit tree required')
        require(all(x['type'] in ('tree','blob') for x in tree['tree']), CODE, 'gitlink refused')
        blobs = [x for x in tree['tree'] if x['type'] == 'blob']
        require(blobs and len(blobs) <= CAPS['files'] and
                sum(x['size'] for x in blobs) <= CAPS['unpacked'] and
                all(x['mode'] in ('100644','100755') for x in blobs), CODE, 'source size/mode scope')
        src['archive'], _ = fetch('https://codeload.github.com/'+c['repo']+'/tar.gz/'+c['base_commit'],
                                  'source.tar.gz', 'archive')
        phases['acquisition'] = time.perf_counter()-tick
        tick = time.perf_counter()
        files = source_files(root, src)
        inventory = {n:dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                     for n,v in sorted(files.items())}
        inventory_pin = publish(output/'source-inventory.json', inventory)
        declared = dependencies(files)
        phases['source_check_and_declarations'] = time.perf_counter()-tick
        tick = time.perf_counter()
        controls = {}
        for name, edit in [('wrong_commit', lambda x:x.update(base_commit='0'*40)),
                           ('changed_archive_digest', lambda x:x['archive'].update(sha256='0'*64))]:
            changed = deepcopy(src)
            edit(changed)
            try:
                source_files(root, changed)
            except SenseError as exc:
                controls[name] = exc.envelope()
            else:
                raise SenseError(CODE, 'broken source accepted: '+name)
        # Independent persisted archive traversal: membership, bytes and Git SHA-1.
        import tarfile
        entries = {b['path']:b for b in blobs}
        seen = {}
        with tarfile.open(root/src['archive']['path'], 'r:gz') as tar:
            for member in tar:
                if member.isdir():
                    continue
                require(member.isfile(), CODE, 'regular archive member required')
                _, _, name = member.name.partition('/')
                require(name in entries and name not in seen, CODE, 'readback inventory')
                value = tar.extractfile(member).read()
                require(hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
                        == entries[name]['sha'], CODE, 'independent Git blob readback')
                seen[name] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        require(seen == read_pinned(output/'source-inventory.json', inventory_pin), CODE, 'readback differs')
        phases['readback_and_controls'] = time.perf_counter()-tick
        case['src'] = src
        wall = time.perf_counter()-started
        phases['other'] = wall-sum(phases.values())
        result = dict(schema='action-repository-source-v1', case=case,
            contract=dict(path=str(contracts), sha256=pin), source_license=c['source_license'],
            source_counts=dict(files=len(files), bytes=sum(map(len, files.values())),
                               modes=dict(Counter(b['mode'] for b in blobs))),
            declared=declared, receipts=receipts, caps=CAPS, phases=phases, wall_seconds=wall,
            inventory_sha256=inventory_pin, controls=controls, status='git_exact',
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            independent_archive_readback_equal=True, admitted=False, model_input=False,
            acceptance='author-checked only')
        result_pin = publish(output/'source.json', result)
        return dict(path=str(output/'source.json'), pin=result_pin, counts=result['source_counts'])
    except Exception as exc:
        publish(output/'failure.json', dict(schema='action-repository-source-failure-v1', task=task,
            receipts=receipts, src=src, caps=CAPS, exception=type(exc).__name__,
            error=exc.envelope() if isinstance(exc, SenseError) else 'source acquisition/readback failed',
            wall_seconds=time.perf_counter()-started, admitted=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','contracts','raw','output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('contracts-pin','task'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    print(json.dumps(acquire(a.root,a.contracts,a.contracts_pin,a.task,a.raw,a.output),sort_keys=True))

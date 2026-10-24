"""Acquire the frozen cohort's original Git snapshots; never execute project code."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle.action_repository_source import CAPS
from oracle.declared_deps import read as dependencies
from oracle.prepare_swe import source_files
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_COHORT_SOURCE'
CODE_FILES = ('oracle/repository_cohort_source.py','oracle/action_repository_source.py',
              'oracle/prepare_swe.py','oracle/declared_deps.py',
              'claims/repository_cohort_input.py','claims/repository_cohort.py',
              'senses/trajectory_population.py','senses/contract.py')


def acquire_one(root,case,raw,output,token):
    start = time.perf_counter()
    raw.mkdir();output.mkdir()
    receipts,phases = [],{}
    src = dict(task=case['task'],base_commit=case['base_commit'])

    def fetch(url,name,kind):
        tick = time.perf_counter()
        headers = {'User-Agent':'MNLM-bounded-repository-cohort'}
        if url.startswith('https://api.github.com/'):
            headers['Authorization'] = 'Bearer '+token
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
            value,status = response.read(CAPS[kind]+1),response.status
        require(len(value) <= CAPS[kind],CODE,'source response above '+kind+' cap')
        path = raw/name
        with path.open('xb') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
        binding = dict(path=path.relative_to(root).as_posix(),bytes=len(value),
                       sha256=hashlib.sha256(value).hexdigest())
        receipts.append(dict(url=url,status=status,binding=binding,seconds=time.perf_counter()-tick))
        return binding,value

    phase_name,tick = 'acquisition',time.perf_counter()
    try:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',case['repo']) is not None and
                re.fullmatch(r'[0-9a-f]{40}',case['base_commit']) is not None,CODE,'source identity')
        base = 'https://api.github.com/repos/'+case['repo']+'/git/'
        src['commit'],value = fetch(base+'commits/'+case['base_commit'],'commit.json','commit')
        commit = json.loads(value)
        require(commit['sha'] == case['base_commit'],CODE,'commit identity')
        src['tree'],value = fetch(base+'trees/'+commit['tree']['sha']+'?recursive=1','tree.json','tree')
        tree = json.loads(value)
        require(tree['sha'] == commit['tree']['sha'] and tree['truncated'] is False,
                CODE,'complete commit tree required')
        require(all(v['type'] in ('tree','blob') for v in tree['tree']),CODE,'gitlink refused')
        blobs = [v for v in tree['tree'] if v['type'] == 'blob']
        require(blobs and len(blobs) <= CAPS['files'] and
                sum(v['size'] for v in blobs) <= CAPS['unpacked'] and
                all(v['mode'] in ('100644','100755') for v in blobs),CODE,'source size/mode scope')
        src['archive'],_ = fetch('https://codeload.github.com/'+case['repo']+'/tar.gz/'+case['base_commit'],
                                 'source.tar.gz','archive')
        phases['acquisition'] = time.perf_counter()-tick
        phase_name,tick = 'git_check_inventory_and_declarations',time.perf_counter()
        files = source_files(root,src)
        inventory = {name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
                     for name,value in sorted(files.items())}
        inventory_pin = publish(output/'inventory.json',inventory)
        # Retain actual licensing files alongside the provider's declaration; no inferred licence grant.
        licensing = {n:v for n,v in inventory.items() if
                     Path(n).name.casefold().split('.')[0] in ('license','licence','copying','notice')}
        require(bool(licensing),CODE,'source licence artifact missing')
        declared = dependencies(files)
        phases['git_check_inventory_and_declarations'] = time.perf_counter()-tick
        status = 'git_exact'
        detail = dict(counts=dict(files=len(files),bytes=sum(map(len,files.values())),
                      modes=dict(Counter(v['mode'] for v in blobs))),
                      inventory_sha256=inventory_pin,licensing_files=licensing,declared=declared)
    except Exception as exc:
        phases[phase_name] = time.perf_counter()-tick
        status = 'unavailable'
        # Never serialize exception text from a network client or subprocess containing credentials.
        error = exc.envelope() if isinstance(exc,SenseError) else dict(
            code=CODE+'_FETCH_OR_PARSE',message='Source unavailable; inspect retained receipts.',
            exception=type(exc).__name__)
        if isinstance(exc,urllib.error.HTTPError):
            error['http_status'] = exc.code
        detail = dict(error=error)
    wall = time.perf_counter()-start
    phases['other'] = wall-sum(phases.values())
    result = dict(schema='repository-cohort-source-case-v1',index=case['index'],task=case['task'],
        repo=case['repo'],partition=case['partition'],source_license=case['source_license'],src=src,
        status=status,receipts=receipts,caps=CAPS,phases=phases,wall_seconds=wall,**detail,
        project_code_executed=False,outcomes_read=False,admitted=False,acceptance='author-checked')
    pin = publish(output/'source.json',result)
    return dict(index=case['index'],task=case['task'],partition=case['partition'],status=status,
                path=str(output/'source.json'),sha256=pin,wall_seconds=wall)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort','readback','raw','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('cohort-pin','readback-pin'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    start = time.perf_counter()
    root,raw,output = a.root.resolve(),a.raw.resolve(),a.output.resolve()
    require(raw.is_relative_to(root) and output.is_relative_to(root) and raw != output and
            not raw.exists() and not output.exists(),CODE,'fresh MNLM directories required')
    cohort = RepositoryCohort(a.cohort,a.cohort_pin)
    proof = read_pinned(a.readback,a.readback_pin)
    require(proof['schema'] == 'repository-cohort-readback-v1' and proof['cohort'] == a.cohort_pin and
            proof['results']['selected_tasks'] == len(cohort.cases) and len(proof['controls']) == 7,
            CODE,'cohort readback required')
    token = subprocess.check_output(['gh','auth','token'],stderr=subprocess.DEVNULL).decode().strip()
    raw.mkdir(parents=True);output.mkdir(parents=True)
    binding = time.perf_counter()-start
    results = []
    for case in cohort.cases:
        name = str(case['index']).zfill(2)
        result = acquire_one(root,case,raw/name,output/name,token)
        results.append(result)
        print(json.dumps(result,sort_keys=True),flush=True)
    wall = time.perf_counter()-start
    code_root = Path(__file__).resolve().parents[1]
    report = dict(schema='repository-cohort-sources-v1',root=str(root),raw=str(raw),cases=results,
        cohort=dict(path=str(a.cohort),sha256=a.cohort_pin),
        readback=dict(path=str(a.readback),sha256=a.readback_pin),caps=CAPS,
        code={n:hashlib.sha256((code_root/n).read_bytes()).hexdigest() for n in CODE_FILES},
        counts=dict(Counter(r['status'] for r in results)),wall_seconds=wall,
        phases=dict(binding=binding,cases=sum(v['wall_seconds'] for v in results),
                    other=wall-binding-sum(v['wall_seconds'] for v in results)),
        selection_changed=False,project_code_executed=False,outcomes_read=False,
        evaluation_status='reserved_pending_training_exposure_audit',admitted=False,
        acceptance='author-checked')
    pin = publish(output/'report.json',report)
    publish(output/'COMMITTED.json',dict(sha256=pin))
    print(json.dumps(dict(report=str(output/'report.json'),sha256=pin,counts=report['counts'],
                          wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    main()

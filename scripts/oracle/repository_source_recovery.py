"""Recover declared export-subst sources using the existing exact Git blob contract."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle.declared_deps import read as dependencies
from oracle.prepare_swe import source_files
from senses.contract import require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_SOURCE_RECOVERY'


def run(recipe,output):
    start = time.perf_counter()
    old = read_pinned(Path(recipe['report']),recipe['pin'])
    proof = read_pinned(Path(recipe['proof']),recipe['proof_pin'])
    require(old['schema'] == 'repository-cohort-sources-v1' and
            proof['source_report'] == recipe['pin'] and proof['counts'] == old['counts'],
            CODE,'original source readback required')
    root = Path(old['root'])
    require(output.resolve().is_relative_to(root) and not output.exists(),CODE,'fresh MNLM output')
    cohort = RepositoryCohort(old['cohort']['path'],old['cohort']['sha256'])
    recover = {v['index']:v for v in recipe['recover']}
    require(len(recover) == len(recipe['recover']) > 0 and
            set(recover) <= {c['index'] for c in cohort.cases
                            if c['partition'] in ('learn_1','learn_2')},CODE,'learning recovery scope')
    output.mkdir(parents=True)
    results = []
    for case,row in zip(cohort.cases,old['cases'],strict=True):
        previous = read_pinned(Path(row['path']),row['sha256'])
        result = copy.deepcopy(previous)
        target = output/str(case['index']).zfill(2)
        target.mkdir()
        if case['index'] in recover:
            tick = time.perf_counter()
            require(previous['status'] == 'unavailable' and
                    previous['error']['code'] == 'MNLM_EXECUTION_PREPARATION' and
                    previous['error']['message'] == 'source differs from pinned Git blob',
                    CODE,'only an original Git blob mismatch is recoverable')
            entry = recover[case['index']]
            receipt = read_pinned(Path(entry['receipt']),entry['pin'])
            binding = receipt['binding']
            response = read_pinned(root/binding['path'],binding['sha256'])
            require((root/binding['path']).stat().st_size == binding['bytes'] <= old['caps']['commit'] and
                    receipt['status'] == 200 and receipt['url'] ==
                    'https://api.github.com/repos/'+case['repo']+'/git/blobs/'+response['sha'],
                    CODE,'exact blob receipt identity')
            result['src']['exact_blobs'] = [dict(binding,name=entry['name'])]
            files = source_files(root,result['src'])
            inventory = {n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
                         for n,v in sorted(files.items())}
            licensing = {n:v for n,v in inventory.items() if
                         Path(n).name.casefold().split('.')[0] in ('license','licence','copying','notice')}
            require(bool(licensing),CODE,'source licence artifact missing')
            tree = json.loads((root/result['src']['tree']['path']).read_bytes())
            blobs = [v for v in tree['tree'] if v['type'] == 'blob']
            result.pop('error')
            result.update(status='git_exact',inventory_sha256=publish(target/'inventory.json',inventory),
                licensing_files=licensing,declared=dependencies(files),
                counts=dict(files=len(files),bytes=sum(map(len,files.values())),
                            modes=dict(Counter(v['mode'] for v in blobs))),
                recovery=dict(parent=row,receipt=dict(path=entry['receipt'],sha256=entry['pin']),
                              kind='pinned_git_blob_for_declared_export_subst'),
                receipts=previous['receipts']+[receipt])
            elapsed = time.perf_counter()-tick
            result.update(phases=dict(recovery=elapsed),wall_seconds=elapsed)
        elif previous['status'] == 'git_exact':
            raw = (Path(row['path']).parent/'inventory.json').read_bytes()
            require(hashlib.sha256(raw).hexdigest() == previous['inventory_sha256'],CODE,'old inventory')
            (target/'inventory.json').write_bytes(raw)
        pin = publish(target/'source.json',result)
        results.append(dict(row,path=str(target/'source.json'),sha256=pin,status=result['status'],
                            wall_seconds=result['wall_seconds']))
    wall = time.perf_counter()-start
    report = dict(old,cases=results,counts=dict(Counter(v['status'] for v in results)),
        recovery=dict(recipe=recipe,writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        wall_seconds=wall,phases=dict(recovery=wall))
    pin = publish(output/'report.json',report)
    publish(output/'COMMITTED.json',dict(sha256=pin))
    print(json.dumps(dict(pin=pin,counts=report['counts'],wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(json.loads(a.recipe.read_bytes()),a.output.resolve())

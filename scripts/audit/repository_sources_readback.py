"""Cold readback of commit/tree/archive bytes without the acquisition implementation."""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.repository_cohort_input import RepositoryCohort
from claims.process_conditions import write_committed
from senses.trajectory_population import read_pinned

CAPS = dict(commit=1 << 20,tree=4 << 20,archive=16 << 20,files=5000,unpacked=64 << 20)


def bound(root,entry,cap):
    name = entry['path']
    relative = PurePosixPath(name)
    check(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
          'COHORT_SOURCE_PATH')
    path = root/name
    check(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'COHORT_SOURCE_PATH')
    check(path.stat().st_size == entry['bytes'] <= cap,'COHORT_SOURCE_BYTES')
    value = path.read_bytes()
    check(sha(value) == entry['sha256'],'COHORT_SOURCE_DIGEST')
    return value


def source_read(root,case,result,inventory):
    check(all(result[k] == case[k] for k in ('index','task','repo','partition','source_license')) and
          result['src']['task'] == case['task'] and result['src']['base_commit'] == case['base_commit'],
          'COHORT_SOURCE_IDENTITY')
    check(result['caps'] == CAPS and result['project_code_executed'] is False and
          result['outcomes_read'] is False and result['admitted'] is False,'COHORT_SOURCE_SCOPE')
    src = result['src']
    commit = json.loads(bound(root,src['commit'],CAPS['commit']))
    tree = json.loads(bound(root,src['tree'],CAPS['tree']))
    check(commit['sha'] == case['base_commit'] and tree['sha'] == commit['tree']['sha'] and
          tree['truncated'] is False,'COHORT_SOURCE_COMMIT')
    check(all(v['type'] in ('blob','tree') for v in tree['tree']),'COHORT_SOURCE_ENTRY_TYPE')
    listed = [v for v in tree['tree'] if v['type'] == 'blob']
    blobs = {v['path']:v for v in listed}
    check(0 < len(blobs) == len(listed) <= CAPS['files'] and
          sum(v['size'] for v in listed) <= CAPS['unpacked'] and
          all(v['mode'] in ('100644','100755') for v in listed),'COHORT_SOURCE_TREE_CAP')
    archive = bound(root,src['archive'],CAPS['archive'])
    seen,total = {},0
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            prefix,separator,name = member.name.partition('/')
            check(member.isfile() and separator and prefix.endswith('-'+case['base_commit']) and
                  name in blobs and name not in seen,'COHORT_SOURCE_MEMBER')
            item = blobs[name]
            check(member.size == item['size'] and
                  bool(member.mode & 0o100) == (item['mode'] == '100755'),'COHORT_SOURCE_MODE_SIZE')
            total += member.size
            check(total <= CAPS['unpacked'],'COHORT_SOURCE_UNPACKED_CAP')
            value = tar.extractfile(member).read()
            check(len(value) == item['size'] and
                  hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == item['sha'],
                  'COHORT_SOURCE_GIT_BLOB')
            seen[name] = dict(bytes=len(value),sha256=sha(value))
    check(seen == inventory and set(seen) == set(blobs),'COHORT_SOURCE_INVENTORY')
    licensing = {n:v for n,v in seen.items() if
                 Path(n).name.casefold().split('.')[0] in ('license','licence','copying','notice')}
    check(licensing and licensing == result['licensing_files'],'COHORT_SOURCE_LICENSE_BYTES')
    counts = dict(files=len(seen),bytes=total,modes=dict(Counter(v['mode'] for v in listed)))
    check(counts == result['counts'],'COHORT_SOURCE_COUNTS')
    check([r['binding'] for r in result['receipts']] == [src[n] for n in ('commit','tree','archive')],
          'COHORT_SOURCE_RECEIPTS')
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    start = time.perf_counter()
    report = read_pinned(a.report,a.pin)
    check(report['schema'] == 'repository-cohort-sources-v1' and
          json.loads((a.report.parent/'COMMITTED.json').read_bytes()) == dict(sha256=a.pin),
          'COHORT_SOURCES_COMMITMENT')
    code_root = Path(__file__).resolve().parents[1]
    check(all(sha((code_root/n).read_bytes()) == v for n,v in report['code'].items()),'COHORT_SOURCES_CODE')
    cohort = RepositoryCohort(report['cohort']['path'],report['cohort']['sha256'])
    proof = read_pinned(Path(report['readback']['path']),report['readback']['sha256'])
    check(proof['cohort'] == cohort.pin and len(report['cases']) == len(cohort.cases) == 12,
          'COHORT_SOURCES_SELECTION')
    root = Path(report['root'])
    before = {str(a.report):sha(a.report.read_bytes())}
    summaries,first = [],None
    phases = dict(binding=time.perf_counter()-start)
    tick = time.perf_counter()
    for case,row in zip(cohort.cases,report['cases'],strict=True):
        path = a.report.parent/str(case['index']).zfill(2)/'source.json'
        check(str(path) == row['path'] and all(row[k] == case[k] for k in ('index','task','partition')),
              'COHORT_SOURCES_CASE_ORDER')
        result = read_pinned(path,row['sha256'])
        before[str(path)] = row['sha256']
        check(result['status'] == row['status'] and all(result[k] == case[k] for k in
              ('index','task','repo','partition','source_license')),'COHORT_SOURCES_CASE_BINDING')
        check(abs(sum(result['phases'].values())-result['wall_seconds']) < 1e-6,
              'COHORT_SOURCE_TIMING_SUM')
        if row['status'] == 'git_exact':
            inventory_path = path.parent/'inventory.json'
            inventory = read_pinned(inventory_path,result['inventory_sha256'])
            before[str(inventory_path)] = result['inventory_sha256']
            counts = source_read(root,case,result,inventory)
            if first is None:
                first = case,result,inventory
            summaries.append(dict(task=case['task'],partition=case['partition'],status='git_exact',**counts))
        else:
            check(row['status'] == 'unavailable' and result['error']['code'],'COHORT_SOURCE_FAILURE')
            for receipt in result['receipts']:
                entry = receipt['binding']
                bound(root,entry,CAPS['archive'])
            summaries.append(dict(task=case['task'],partition=case['partition'],status='unavailable',
                                  error=result['error']))
        for receipt in result['receipts']:
            before[str(root/receipt['binding']['path'])] = receipt['binding']['sha256']
    phases['source_readback'] = time.perf_counter()-tick
    tick = time.perf_counter()
    check(first is not None,'COHORT_NO_ORIGINAL_SOURCE')
    controls = []
    case,result,inventory = first
    for name,expected in (('wrong_commit','COHORT_SOURCE_IDENTITY'),
                          ('changed_archive_digest','COHORT_SOURCE_DIGEST'),
                          ('changed_partition','COHORT_SOURCE_IDENTITY'),
                          ('missing_inventory_member','COHORT_SOURCE_INVENTORY')):
        changed,listing = copy.deepcopy(result),copy.deepcopy(inventory)
        if name == 'wrong_commit':
            changed['src']['base_commit'] = '0'*40
        elif name == 'changed_archive_digest':
            changed['src']['archive']['sha256'] = '0'*64
        elif name == 'changed_partition':
            changed['partition'] = 'reserved_evaluation'
        else:
            listing.pop(next(iter(listing)))
        try:
            source_read(root,case,changed,listing)
        except ValueError as exc:
            check(expected in str(exc),'COHORT_SOURCE_CONTROL_ERROR')
        else:
            raise AssertionError('COHORT_SOURCE_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=expected))
    check(before == {n:sha(Path(n).read_bytes()) for n in before},'COHORT_SOURCE_READBACK_MUTATED')
    counts = dict(Counter(r['status'] for r in summaries))
    check(counts == report['counts'] and abs(sum(report['phases'].values())-report['wall_seconds']) < 1e-6,
          'COHORT_SOURCES_SUMMARY')
    phases['controls_and_final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    phases['other'] = wall-sum(phases.values())
    result = dict(schema='repository-sources-readback-v1',source_report=a.pin,cohort=cohort.pin,
        counts=counts,sources=summaries,files=before,controls=controls,wall_seconds=wall,phases=phases,
        reader_code=sha(Path(__file__).read_bytes()),
        source_files=sum(r.get('files',0) for r in summaries),
        source_bytes=sum(r.get('bytes',0) for r in summaries),
        evaluation_status='reserved_pending_training_exposure_audit',outcomes_read=False,
        learned_capability=False,acceptance='author-checked')
    pin = write_committed(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','sources')}|dict(sha256=pin)),flush=True)


if __name__ == '__main__':
    main()

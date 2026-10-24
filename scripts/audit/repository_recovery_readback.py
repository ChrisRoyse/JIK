"""Read original archive, exact blobs and recovered source independently of acquisition."""
import argparse
import base64
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
from audit.repository_sources_readback import CAPS,bound,source_read
from claims.codata_source import check,sha
from claims.repository_cohort_input import RepositoryCohort
from claims.process_conditions import write_committed
from oracle.prepare_swe import source_files
from senses.contract import SenseError
from senses.trajectory_population import read_pinned


def git_hash(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def exact_read(root,case,result,inventory):
    check(all(result[k] == case[k] for k in ('index','task','repo','partition','source_license')) and
          result['src']['task'] == case['task'] and result['src']['base_commit'] == case['base_commit'],
          'RECOVERY_IDENTITY')
    check(result['caps'] == CAPS and result['project_code_executed'] is False and
          result['outcomes_read'] is False and result['admitted'] is False,'RECOVERY_SCOPE')
    src = result['src']
    commit = json.loads(bound(root,src['commit'],CAPS['commit']))
    tree = json.loads(bound(root,src['tree'],CAPS['tree']))
    check(commit['sha'] == case['base_commit'] and tree['sha'] == commit['tree']['sha'] and
          tree['truncated'] is False,'RECOVERY_COMMIT')
    listed = [r for r in tree['tree'] if r['type'] == 'blob']
    blobs = {r['path']:r for r in listed}
    check(0 < len(blobs) == len(listed) <= CAPS['files'] and
          sum(r['size'] for r in listed) <= CAPS['unpacked'] and
          all(r['type'] in ('tree','blob') for r in tree['tree']) and
          all(r['mode'] in ('100644','100755') for r in listed),'RECOVERY_TREE')
    replacements = {}
    bindings = src.get('exact_blobs',[])
    check(bindings,'RECOVERY_BLOBS_MISSING')
    for entry in bindings:
        name = entry['name']
        check(name in blobs and name not in replacements,'RECOVERY_BLOB_IDENTITY')
        response = json.loads(bound(root,entry,CAPS['commit']))
        check(response['encoding'] == 'base64' and response['sha'] == blobs[name]['sha'],
              'RECOVERY_BLOB_IDENTITY')
        value = base64.b64decode(response['content'])
        check(len(value) == response['size'] == blobs[name]['size'] and
              git_hash(value) == blobs[name]['sha'],'RECOVERY_BLOB_BYTES')
        replacements[name] = value
    seen,transformed,total = {},set(),0
    with tarfile.open(fileobj=io.BytesIO(bound(root,src['archive'],CAPS['archive'])),mode='r:gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            prefix,separator,name = member.name.partition('/')
            check(member.isfile() and separator and prefix.endswith('-'+case['base_commit']) and
                  name in blobs and name not in seen and '..' not in PurePosixPath(name).parts,
                  'RECOVERY_ARCHIVE_MEMBER')
            check(bool(member.mode & 0o100) == (blobs[name]['mode'] == '100755'), 'RECOVERY_MODE')
            total += member.size
            check(total <= CAPS['unpacked'],'RECOVERY_ARCHIVE_SIZE')
            value = tar.extractfile(member).read()
            if name in replacements:
                check(git_hash(value) != blobs[name]['sha'],'RECOVERY_UNCHANGED_REPLACEMENT')
                transformed.add(name)
                value = replacements[name]
            check(len(value) == blobs[name]['size'] and git_hash(value) == blobs[name]['sha'],
                  'RECOVERY_GIT_BLOB')
            seen[name] = value
    check(set(seen) == set(blobs) and transformed == set(replacements),'RECOVERY_MEMBERSHIP')
    # This recovery deliberately supports only explicit root declarations; no inferred glob semantics.
    attributes = [line.split() for line in seen.get('.gitattributes',b'').decode().splitlines()]
    check(all([name,'export-subst'] in attributes for name in transformed),'RECOVERY_ATTRIBUTE')
    observed = {n:dict(bytes=len(v),sha256=sha(v)) for n,v in seen.items()}
    check(observed == inventory,'RECOVERY_INVENTORY')
    licensing = {n:v for n,v in observed.items() if
                 Path(n).name.casefold().split('.')[0] in ('license','licence','copying','notice')}
    check(licensing and licensing == result['licensing_files'],'RECOVERY_LICENSE')
    counts = dict(files=len(seen),bytes=sum(map(len,seen.values())),
                  modes=dict(Counter(r['mode'] for r in listed)))
    check(counts == result['counts'],'RECOVERY_COUNTS')
    expected = [src[k] for k in ('commit','tree','archive')]+[
        {k:v for k,v in b.items() if k != 'name'} for b in bindings]
    check([r['binding'] for r in result['receipts']] == expected,'RECOVERY_RECEIPTS')
    for entry,receipt in zip(bindings,result['receipts'][3:],strict=True):
        check(receipt['status'] == 200 and receipt['url'] == 'https://api.github.com/repos/'+
              case['repo']+'/git/blobs/'+blobs[entry['name']]['sha'],'RECOVERY_URL')
    return counts


def run(path,pin,output):
    start = time.perf_counter()
    report = read_pinned(path,pin)
    recipe = report['recovery']['recipe']
    old = read_pinned(Path(recipe['report']),recipe['pin'])
    prior_proof = read_pinned(Path(recipe['proof']),recipe['proof_pin'])
    check(prior_proof['source_report'] == recipe['pin'] and
          json.loads((path.parent/'COMMITTED.json').read_bytes()) == dict(sha256=pin),
          'RECOVERY_PARENT_PROOF')
    cohort = RepositoryCohort(old['cohort']['path'],old['cohort']['sha256'])
    check(report['cohort'] == old['cohort'] and len(report['cases']) == len(old['cases']) == 12,
          'RECOVERY_COHORT')
    root = Path(old['root'])
    before,rows,controls = {},[],[]
    changes = {v['index']:v for v in recipe['recover']}
    for case,row,parent in zip(cohort.cases,report['cases'],old['cases'],strict=True):
        check(row['index'] == parent['index'] == case['index'] and row['task'] == case['task'] and
              row['partition'] == case['partition'],'RECOVERY_ROW')
        result = read_pinned(Path(row['path']),row['sha256'])
        previous = read_pinned(Path(parent['path']),parent['sha256'])
        before[parent['path']] = parent['sha256']
        check(row['status'] == result['status'],'RECOVERY_STATUS')
        if case['index'] in changes:
            entry = changes[case['index']]
            receipt = read_pinned(Path(entry['receipt']),entry['pin'])
            check(case['partition'] in ('learn_1','learn_2') and previous['status'] == 'unavailable' and
                  result['recovery']['parent'] == parent and result['receipts'] == previous['receipts']+[receipt] and
                  {k:v for k,v in result['src'].items() if k != 'exact_blobs'} == previous['src'] and
                  result['src']['exact_blobs'] == [dict(receipt['binding'],name=entry['name'])],
                  'RECOVERY_LINEAGE')
            inventory = read_pinned(Path(row['path']).parent/'inventory.json',result['inventory_sha256'])
            counts = exact_read(root,case,result,inventory)
            for name,expected in (('missing_blob','RECOVERY_BLOBS_MISSING'),
                                  ('duplicate_blob','RECOVERY_BLOB_IDENTITY'),
                                  ('wrong_blob_digest','COHORT_SOURCE_DIGEST'),
                                  ('wrong_partition','RECOVERY_IDENTITY'),
                                  ('missing_member','RECOVERY_INVENTORY')):
                bad,listing = copy.deepcopy(result),copy.deepcopy(inventory)
                if name == 'missing_blob': bad['src'].pop('exact_blobs')
                elif name == 'duplicate_blob': bad['src']['exact_blobs'] *= 2
                elif name == 'wrong_blob_digest': bad['src']['exact_blobs'][0]['sha256'] = '0'*64
                elif name == 'wrong_partition': bad['partition'] = 'reserved_evaluation'
                else: listing.pop(next(iter(listing)))
                try: exact_read(root,case,bad,listing)
                except ValueError as exc: check(expected in str(exc),'RECOVERY_CONTROL_CODE')
                else: raise AssertionError('RECOVERY_CONTROL_ACCEPTED:'+name)
                controls.append(dict(name=name,code=expected))
            try: source_files(root,previous['src'])
            except SenseError as exc:
                check(exc.message == 'source differs from pinned Git blob','RECOVERY_ORIGINAL_FAILURE')
            else: raise AssertionError('RECOVERY_ORIGINAL_ACCEPTED')
            controls.append(dict(name='original_archive_still_refused',code='MNLM_EXECUTION_PREPARATION'))
        else:
            check(result == previous,'RECOVERY_UNCHANGED_SOURCE')
            if result['status'] == 'git_exact':
                inventory = read_pinned(Path(row['path']).parent/'inventory.json',result['inventory_sha256'])
                counts = source_read(root,case,result,inventory)
            else: counts = dict(error=result['error'])
        for r in result['receipts']:
            bound(root,r['binding'],CAPS['archive'])
            before[str(root/r['binding']['path'])] = r['binding']['sha256']
        rows.append(dict(task=case['task'],partition=case['partition'],status=result['status'],**counts))
    check(before == {n:sha(Path(n).read_bytes()) for n in before},'RECOVERY_MUTATED')
    counts = dict(Counter(r['status'] for r in rows))
    check(counts == report['counts'],'RECOVERY_SUMMARY')
    result = dict(schema='repository-sources-readback-v1',source_report=pin,cohort=cohort.pin,
        counts=counts,sources=rows,files=before,controls=controls,parent_proof=recipe['proof_pin'],
        wall_seconds=time.perf_counter()-start,reader_code=sha(Path(__file__).read_bytes()),
        admitted=False,acceptance='author-checked')
    value = write_committed(output,result)
    print(json.dumps(dict(pin=value,counts=counts,controls=len(controls))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.report,a.pin,a.output)

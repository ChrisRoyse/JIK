"""Pinned, training-only author release capture; original judgments stay private."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.connected_source_capture import capture,LIMITS
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

REPO='AbhilashaRavichander/CondaQA'
REVISION='bd4857f1f2819937f1505925ad8c809544e8e5ef'
FILES={'LICENSE':('LICENSE',11357,'261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64'),
       'README.md':('README.md',8421,'c171c44919a728d32c489959925210d1e54e3339'),
       'train.json':('data/condaqa_train.json',11054568,'451e97c26a6f0ad3b20d230ca541ceb87534b717')}
FLAGS=dict(training_ready=False,heldout_payloads_obtained=False,scientific_admission=False,served=False,
           source_judgments_are_world_facts=False,encoder_exposure_certified=False)

def check(ok,code):
    if not ok:raise ValueError('MNLM_CONDAQA_SOURCE_'+code)

def git_blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def inventory(payload):
    if payload.lstrip().startswith(b'['):records=json.loads(payload);form='json-array'
    else:records=[json.loads(line) for line in payload.splitlines() if line.strip()];form='json-lines'
    required={'sentence1','sentence2','label','QuestionID','PassageEditID','PassageID','SampleID','original passage','original sentence','original cue'}
    check(type(records)==list and records,'ROWS')
    check(all(type(r)==dict and required<=set(r) for r in records),'FIELDS')
    check(all(type(r['sentence1'])==type(r['sentence2'])==type(r['label'])==str for r in records),'SURFACE_TYPES')
    return dict(format=form,rows=len(records),field_sets=dict(Counter('|'.join(sorted(r)) for r in records)),
        labels=dict(sorted(Counter(r['label'] for r in records).items())),
        edit_ids=dict(sorted(Counter(str(r['PassageEditID']) for r in records).items())),
        unique_sample_ids=len({r['SampleID'] for r in records}),unique_passage_ids=len({r['PassageID'] for r in records}),
        blank_passages=sum(not r['sentence1'].strip() for r in records),blank_questions=sum(not r['sentence2'].strip() for r in records))

def build(output):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};receipts={};errors=[];inv=None
    try:
        receipts['provider-tree.json']=capture(f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1',output/'provider-tree.json',LIMITS['metadata_bytes'])
        tree=json.loads((output/'provider-tree.json').read_bytes());check(tree['sha']==REVISION and tree['truncated'] is False,'TREE_REVISION')
        by_path={v['path']:v for v in tree['tree']}
        for path,size,blob in FILES.values():
            entry=by_path[path];check(entry['type']=='blob' and entry['size']==size and entry['sha']==blob,'TREE_FILE_PIN')
        phases['provider_tree']=time.perf_counter()-start;tick=time.perf_counter()
        for name,(path,size,blob) in FILES.items():
            receipts[name]=capture(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}',output/name,size)
            payload=(output/name).read_bytes();check(len(payload)==size and git_blob(payload)==blob,'FILE_PIN')
        phases['exact_allowlisted_payloads']=time.perf_counter()-tick;tick=time.perf_counter()
        licence=(output/'LICENSE').read_text(encoding='utf-8');check('Apache License' in licence and 'Version 2.0' in licence,'LICENCE_CONTENT')
        readme=(output/'README.md').read_text(encoding='utf-8');check('condaqa_train.json' in readme and 'PassageEditID' in readme,'PROVIDER_SCHEMA')
        inv=inventory((output/'train.json').read_bytes());publish(output/'inventory.json',inv)
        phases['licence_and_training_inventory']=time.perf_counter()-tick
    except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(dict(error=str(exc),type=type(exc).__name__))
    tick=time.perf_counter();publish(output/'capture.json',dict(repository=REPO,revision=REVISION,receipts=receipts,
        selected_source_files={n:dict(path=p,bytes=s,git_blob_sha1=b) for n,(p,s,b) in FILES.items()},credentials_used=False))
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in output.iterdir() if p.is_file()}
    phases['publication_and_cold_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='condaqa-training-capture-v1',complete=not errors,errors=errors,source=REPO,revision=REVISION,
        files=files,flags=FLAGS,licence='Apache-2.0 provider release; retained exact licence and README',
        scope='source-relative human judgments over original and edited passages; no world-validity certification',
        code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(output/'report.json',report)
    print(json.dumps(dict(report_sha256=file_sha(output/'report.json'),complete=not errors,errors=errors,inventory=inv,wall_seconds=wall)))
    return 2 if errors else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(build(p.parse_args().output))

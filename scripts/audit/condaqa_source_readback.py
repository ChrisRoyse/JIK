"""Independent provider-blob, request-scope and training inventory readback."""
import hashlib
import json
from pathlib import Path

def need(ok,code):
    if not ok:raise ValueError('MNLM_CONDAQA_READBACK_'+code)

def read(root):
    digest=lambda b:hashlib.sha256(b).hexdigest()
    raw={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    names={'LICENSE','README.md','train.json','provider-tree.json','capture.json','inventory.json'}
    need(set(raw)==names|{'report.json'} and len(list(root.iterdir()))==7,'FILES')
    h=json.loads(raw['report.json']);need(h['schema']=='condaqa-training-capture-v1' and h['complete'] is True and h['errors']==[],'STATE')
    need(set(h['files'])==names and all(h['files'][n]==dict(bytes=len(raw[n]),sha256=digest(raw[n])) for n in names),'DIGEST')
    revision='bd4857f1f2819937f1505925ad8c809544e8e5ef';repo='AbhilashaRavichander/CondaQA'
    need(h['source']==repo and h['revision']==revision,'REVISION')
    need(h['flags']=={k:False for k in ('training_ready','heldout_payloads_obtained','scientific_admission','served','source_judgments_are_world_facts','encoder_exposure_certified')},'FLAGS')
    need(h['licence']=='Apache-2.0 provider release; retained exact licence and README' and
         h['scope']=='source-relative human judgments over original and edited passages; no world-validity certification','CLAIM_SCOPE')
    need(h['code_sha256']==digest((Path(__file__).resolve().parents[1]/'text/condaqa_source.py').read_bytes()),'CODE')
    need(h['wall_seconds']>0 and .9<=h['phase_coverage']<=1 and all(v>=0 for v in h['phases_seconds'].values())
        and abs(sum(h['phases_seconds'].values())/h['wall_seconds']-h['phase_coverage'])<1e-12,'COST')
    pins={'LICENSE':('LICENSE',11357,'261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64'),
          'README.md':('README.md',8421,'c171c44919a728d32c489959925210d1e54e3339'),
          'train.json':('data/condaqa_train.json',11054568,'451e97c26a6f0ad3b20d230ca541ceb87534b717')}
    tree=json.loads(raw['provider-tree.json']);need(tree['sha']==revision and tree['truncated'] is False,'TREE')
    for name,(path,size,blob) in pins.items():
        entries=[v for v in tree['tree'] if v['path']==path]
        need(len(entries)==1 and entries[0]['type']=='blob' and entries[0]['size']==size and entries[0]['sha']==blob,'TREE_PIN')
        need(len(raw[name])==size and hashlib.sha1(b'blob '+str(size).encode()+b'\0'+raw[name]).hexdigest()==blob,'BLOB_PIN')
    cap=json.loads(raw['capture.json'])
    need(cap['repository']==repo and cap['revision']==revision and cap['credentials_used'] is False,'CAPTURE')
    need(cap['selected_source_files']=={n:dict(path=p,bytes=s,git_blob_sha1=b) for n,(p,s,b) in pins.items()},'ALLOWLIST')
    urls={n:f'https://raw.githubusercontent.com/{repo}/{revision}/{p}' for n,(p,_,_) in pins.items()}
    urls['provider-tree.json']=f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'
    need(set(cap['receipts'])==set(urls),'REQUEST_SCOPE')
    for n,url in urls.items():
        receipt=cap['receipts'][n]
        need(receipt['url']==url and receipt['final_url']==url.split('?')[0],'URL')
        need(receipt['bytes']==len(raw[n]) and receipt['sha256']==digest(raw[n]) and receipt['content_length'] in (None,len(raw[n])),'RECEIPT')
    # The pinned release is JSONL: parse that grammar independently of producer dispatch.
    records=[json.loads(line) for line in raw['train.json'].splitlines() if line.strip()]
    fields={};labels={};edits={};samples=set();passages=set();blank_p=blank_q=0
    for r in records:
        key='|'.join(sorted(r));fields[key]=fields.get(key,0)+1
        labels[r['label']]=labels.get(r['label'],0)+1
        k=str(r['PassageEditID']);edits[k]=edits.get(k,0)+1
        samples.add(r['SampleID']);passages.add(r['PassageID']);blank_p+=not r['sentence1'].strip();blank_q+=not r['sentence2'].strip()
    expected=dict(format='json-lines',rows=len(records),field_sets=fields,labels=labels,edit_ids=edits,
        unique_sample_ids=len(samples),unique_passage_ids=len(passages),blank_passages=blank_p,blank_questions=blank_q)
    need(json.loads(raw['inventory.json'])==expected,'INVENTORY')
    need(len(records)==5832 and len(samples)==5832 and len(passages)==474,'SOURCE_COUNTS')
    return dict(report_sha256=digest(raw['report.json']),training_sha256=digest(raw['train.json']),rows=len(records),
        provider_blobs_reconstructed=True,only_train_payload_requested=True,inventory_reconstructed=True,
        source_judgments_are_world_facts=False,acceptance='author-checked')

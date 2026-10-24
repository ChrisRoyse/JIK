"""Resume a sealed document boundary without losing optimizer or ledger identity."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from model import native_document_training as steps
from model.native_population_training import fingerprint
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need
from claims.process_conditions import write_committed


def seal(source, output, step, expected_code):
    """Copy a fully written checkpoint and its exact committed ledger prefix.

    Callers choose a checkpoint already followed by a persisted progress receipt.
    Original files are never edited; before/after hashes reject concurrent change.
    """
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = source/f'checkpoint-{step}.pt'; before=file_sha(checkpoint)
    value=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(value['step']==step and value['binding']['code']==expected_code, 'PREDICTOR_RESUME_SOURCE_CODE')
    selection_path=source/'selection.json';selection_pin=file_sha(selection_path)
    plan=json.loads(selection_path.read_bytes())
    need(selection_pin==value['binding']['selection'], 'PREDICTOR_RESUME_SELECTION')
    chain=digest(value['binding']);rows=0;records=[]
    with (source/'steps.jsonl').open() as f:
        for cursor in range(step):
            record=json.loads(f.readline());body={k:v for k,v in record.items() if k!='hash'}
            need(record['hash']==digest(body) and record['parent']==chain and record['cursor']==cursor
                 and record['update']['step']==cursor+1 and record['update']['document']==plan['train_documents'][cursor],
                 'PREDICTOR_RESUME_SOURCE_LEDGER')
            chain=record['hash'];rows+=record['update']['targets'];records.append(record)
    need(chain==value['ledger'] and rows==value['rows']
         and fingerprint(value['model'])==records[-1]['update']['after']
         and fingerprint(value['optimizer'])==records[-1]['update']['optimizer'], 'PREDICTOR_RESUME_SOURCE_STATE')
    with (output/'steps.jsonl').open('x') as f:
        for record in records:f.write(canonical(record).decode()+'\n')
        f.flush()
        os.fsync(f.fileno())
    checkpoints={}
    for path in sorted(source.glob('checkpoint-*.pt')):
        number=int(path.stem.split('-')[-1])
        if number>step:continue
        pin=file_sha(path);saved=torch.load(path,map_location='cpu',weights_only=True)
        need(saved['step']==number and saved['binding']==value['binding'], 'PREDICTOR_RESUME_PREFIX_CHECKPOINT')
        dest=output/path.name
        with dest.open('xb') as f:f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
        need(file_sha(dest)==pin==file_sha(path),'PREDICTOR_RESUME_CHECKPOINT_COPY')
        checkpoints[str(number)]=dict(pin=pin,state=fingerprint(saved),rows=saved['rows'])
    need(file_sha(checkpoint)==before and file_sha(selection_path)==selection_pin,'PREDICTOR_RESUME_SOURCE_MOVED')
    files={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
    return write_committed(output/'report.json',dict(complete=True,binding=value['binding'],selection=plan,
        step=step,rows=rows,ledger_head=chain,checkpoints=checkpoints,files=files,source=str(source),
        acceptance='author-checked',number_status='measured',scientific_admission=False))


def restore(root, pin, output, model, optimizer, binding, plan, data, cache, policy):
    root=Path(root)
    need(file_sha(root/'report.json')==pin, 'PREDICTOR_RESUME_MANIFEST_PIN')
    r=json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['binding']==binding and r['selection']==plan,
         'PREDICTOR_RESUME_EXPERIMENT')
    for name, value in r['files'].items():
        need(file_sha(root/name)==value,'PREDICTOR_RESUME_PHYSICAL_PIN')
    chain=digest(binding); rows=0; windows={n:0 for n in data.roster}; records=[]
    with (root/'steps.jsonl').open() as source:
        for cursor,line in enumerate(source):
            record=json.loads(line); value={k:v for k,v in record.items() if k!='hash'}
            need(record['hash']==digest(value) and record['parent']==chain and record['cursor']==cursor,
                 'PREDICTOR_RESUME_LEDGER')
            number=plan['train_documents'][cursor];start,stop=map(int,data.documents[number])
            uids=data.mapping[start:stop].tolist();update=record['update']
            need(update['document']==number and update['step']==cursor+1 and update['atoms']==stop-start
                 and update['targets']==stop-start-1 and record['source_start']==start and record['source_stop']==stop
                 and record['uid_order']==fingerprint(torch.tensor(uids,dtype=torch.int64)),
                 'PREDICTOR_RESUME_SOURCE_ORDER')
            # Recount every native window of the imported prefix from authenticated offsets/cache.
            counts=np.asarray(cache.maps[1][uids],dtype=np.int64).sum(0)
            for n,count in zip(data.roster,counts):windows[n]+=int(count)
            chain=record['hash'];rows+=update['targets'];records.append(record)
    need(len(records)==r['step'] and rows==r['rows'] and chain==r['ledger_head'],'PREDICTOR_RESUME_CURSOR')
    need(records,'PREDICTOR_RESUME_EMPTY')
    checkpoints={}
    for key, expected in r['checkpoints'].items():
        path=root/f'checkpoint-{key}.pt'
        value=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(value)==expected['state'] and value['binding']==binding and value['step']==int(key),
             'PREDICTOR_RESUME_CHECKPOINT_STATE')
        dest=output/path.name
        if dest.exists():
            need(fingerprint(torch.load(dest,map_location='cpu',weights_only=True))==expected['state'],
                 'PREDICTOR_RESUME_EXISTING_CHECKPOINT')
        else:
            with dest.open('xb') as f:f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
            need(file_sha(dest)==expected['pin'],'PREDICTOR_RESUME_CHECKPOINT_COPY')
        # Pickle memoization may differ for an independently produced identical state.
        # Bind physical bytes and canonical tensor/configuration state separately.
        checkpoints[key]=dict(expected,pin=file_sha(dest))
    latest=root/f"checkpoint-{r['step']}.pt"
    saved=steps.load(latest,r['checkpoints'][str(r['step'])]['pin'],model,optimizer,binding,r['step'],rows,policy)
    need(saved['ledger']==chain and fingerprint(model.state_dict())==records[-1]['update']['after']
         and fingerprint(optimizer.state_dict())==records[-1]['update']['optimizer'],
         'PREDICTOR_RESUME_LAST_UPDATE')
    return dict(step=r['step'],rows=rows,ledger=chain,records=records,checkpoints=checkpoints,windows=windows,
                source_manifest=pin)

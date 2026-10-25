"""Immutable publication helpers for a resumed, complete placement comparison."""
import hashlib
import json
import os
from pathlib import Path

import torch
from audit import legal_grouped_placement_training as original
from model.legal_placement_resume import Census, recount
from model import placement_resume_tail as tail
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha, need


NEW = ('model/legal_placement_resume.py','model/placement_resume_publication.py',
       'audit/legal_placement_resume_training.py','model/placement_resume_tail.py')


def code():
    root=Path(__file__).resolve().parents[1]
    return {n:file_sha(root/n) for n in NEW}


def copy_file(source, destination, pin):
    """Bounded, independently owned bytes; no hardlink mutation of the source."""
    source,destination=Path(source),Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        while block:=src.read(1<<20):dst.write(block);h.update(block)
        dst.flush();os.fsync(dst.fileno())
    need(h.hexdigest()==pin==file_sha(destination)==file_sha(source),'PLACEMENT_PUBLICATION_COPY')


def evaluation(root, items, architecture, roster):
    """Read original evaluation order, checking each saved record's source identity."""
    rows=[];paths=[]
    for item in items:
        proposals=tuple(roster) if architecture=='whole_proposals' else ('assembled',)
        for lo in range(0,len(item['codes']),8):
            stop=min(lo+8,len(item['codes']))
            for proposal in proposals:
                name=f"document-{item['document']}-rows-{lo}-{stop}-{proposal}.pt";paths.append(name)
                values=torch.load(root/name,map_location='cpu',weights_only=True)
                need(len(values)==stop-lo,'PLACEMENT_PUBLICATION_EVALUATION_LENGTH')
                for offset,payload in enumerate(values,lo):
                    row=payload['record']
                    need((row['document'],row['offset'],row['proposal'])==(item['document'],offset,proposal)
                         and row['native_pin']==item['native_pin'] and row['label_pin']==item['label_pin']
                         and torch.equal(row['target'],item['codes'][offset]),'PLACEMENT_PUBLICATION_EVALUATION_SOURCE')
                    rows.append(row)
    need(set(paths)=={p.name for p in root.iterdir() if p.is_file()},'PLACEMENT_PUBLICATION_EVALUATION_FILES')
    return rows,paths


def validate_checkpoints(root, branch, saved_binding, census):
    """One ledger scan authenticates all checkpoint positions, rather than K scans."""
    checkpoints=branch['checkpoints'];step=branch['step']
    need('0' in checkpoints and str(step) in checkpoints,'PLACEMENT_PUBLICATION_CHECKPOINT_SET')
    initial=torch.load(root/'checkpoint-0.pt',map_location='cpu',weights_only=True)
    expected={0:(digest(saved_binding),fingerprint(initial['model']),fingerprint(initial['optimizer']))}
    with (root/'steps.jsonl').open('rb') as stream:
        for cursor,line in enumerate(stream,1):
            if str(cursor) in checkpoints:
                row=json.loads(line);expected[cursor]=(row['hash'],row['update']['after'],row['update']['optimizer'])
    need(set(expected)=={int(k) for k in checkpoints},'PLACEMENT_PUBLICATION_CHECKPOINT_CURSOR_SET')
    for cursor,ref in checkpoints.items():
        path=root/f'checkpoint-{cursor}.pt';need(file_sha(path)==ref['pin'],'PLACEMENT_PUBLICATION_CHECKPOINT_PIN')
        value=torch.load(path,map_location='cpu',weights_only=True);number=int(cursor)
        need(fingerprint(value)==ref['state'] and value['binding']==saved_binding and value['step']==number
             and value['rows']==census.counts(number)['rows']
             and value['noise_cursor']==saved_binding['policy']['noise_seed']+number,
             'PLACEMENT_PUBLICATION_CHECKPOINT_STATE')
        need((value['chain'],fingerprint(value['model']),fingerprint(value['optimizer']))==expected[number],
             'PLACEMENT_PUBLICATION_CHECKPOINT_LEDGER')
    final=torch.load(root/f'checkpoint-{step}.pt',map_location='cpu',weights_only=True)
    return initial,final,recount(root/'steps.jsonl',final,initial,census)


def boundary(root, pin, *, execution=None):
    need(file_sha(root/'boundary.json')==pin,'PLACEMENT_PUBLICATION_BOUNDARY_PIN')
    value=json.loads((root/'boundary.json').read_bytes())
    need(value['complete'] and value['kind']=='placement-resume-boundary-v1'
         and not value['training_complete'] and set(value['branches'])==set(original.runtime.original.ARCHITECTURES),
         'PLACEMENT_PUBLICATION_BOUNDARY_SCOPE')
    if execution is not None:need(value['execution']==execution,'PLACEMENT_PUBLICATION_EXECUTION')
    for name,h in value['files'].items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()),'PLACEMENT_PUBLICATION_FILE_SCOPE')
        need(file_sha(p)==h,'PLACEMENT_PUBLICATION_FILE_PIN')
    census=Census.load(torch.load(root/'census.pt',map_location='cpu',weights_only=True))
    need(census.pin==value['census'],'PLACEMENT_PUBLICATION_CENSUS')
    need(file_sha(root/'donors.json')==value['binding']['donors'],'PLACEMENT_PUBLICATION_DONOR_PIN')
    for name,b in value['branches'].items():
        if b.get('fresh'):
            need(b==dict(fresh=True,step=0),'PLACEMENT_PUBLICATION_FRESH_BRANCH')
        else:
            _,_,count=validate_checkpoints(root/name,b,dict(value['binding'],architecture=name),census)
            need(count==b['counts'],'PLACEMENT_PUBLICATION_PREFIX_COUNTS')
            if 'tail' in b:tail.validate(root/name,b['tail'],count,census,dict(value['binding'],architecture=name))
    return value,census


def import_branch(source, destination, manifest, name):
    branch=manifest['branches'][name]
    for rel,pin in manifest['files'].items():
        if rel in (name+'/steps.jsonl',name+'/retained-tail.jsonl',name+'/uncommitted-tail.bin') or rel.startswith(name+'/initial/') or (
            rel.startswith(name+'/checkpoint-') and rel.endswith('.pt')):
            copy_file(source/rel,destination/rel,pin)
    return branch

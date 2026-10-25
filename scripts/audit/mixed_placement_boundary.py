"""Build a real completed/partial/fresh continuation proof from saved experiments."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as original
from claims.process_conditions import write_committed
from model import completed_placement_branch as completed
from model import placement_resume_publication as publication
from model import placement_resume_tail as tail
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);root=a.output/'boundary';root.mkdir()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    old,census=publication.boundary(a.boundary,a.boundary_pin,execution=publication.code())
    reference=original.pinned(a.reference,a.reference_pin);original.physical(a.reference,reference)
    need(reference['complete'] and reference['binding']==old['binding'] and old['binding']['scope']=='small'
         and len(census.documents)==5 and census.counts(5)['rows']==66,'MIXED_BOUNDARY_SOURCE')
    source,bundle=original.source_bundle(a.root)
    need(reference['binding']['code']==original.code(source),'MIXED_BOUNDARY_CODE')
    data=original.load_data(argparse.Namespace(root=a.root,scope='small'),bundle)
    files={};branches={}
    def copy(src,rel,pin):
        publication.copy_file(src,root/rel,pin);files[rel]=pin
    for rel,pin in old['files'].items():
        if not any(rel.startswith(n+'/') for n in original.runtime.original.ARCHITECTURES):copy(a.boundary/rel,rel,pin)
    for name in original.runtime.original.ARCHITECTURES:
        if name=='whole_proposals':branches[name]=dict(fresh=True,step=0);continue
        if name=='shared':
            r=reference['results'][name]
            b=dict(step=r['updates'],checkpoints=deepcopy(r['checkpoints']),initial=r['initial'],initial_rows=r['initial_rows'])
            for rel,pin in reference['files'].items():
                if rel.startswith(name+'/') and (rel.endswith('.pt') or rel==name+'/steps.jsonl'):copy(a.reference/rel,rel,pin)
        else:
            b=deepcopy(old['branches'][name]);b.pop('tail',None)
            need(b['step']==2 and not b.get('fresh'),'MIXED_BOUNDARY_PARTIAL_CURSOR')
            for rel,pin in old['files'].items():
                if rel.startswith(name+'/') and (rel.endswith('.pt') or rel==name+'/steps.jsonl'):copy(a.boundary/rel,rel,pin)
        _,_,b['counts']=publication.validate_checkpoints(root/name,b,dict(old['binding'],architecture=name),census)
        if name=='shared':b['completed']=completed.describe(root,b,name,census,data['items'],data['labels'].roster)
        else:
            journal=a.reference/name/'steps.jsonl'
            b['tail']=tail.seal(journal,root/name,b['counts'],census,dict(old['binding'],architecture=name),journal.stat().st_size)
            for rel in ('retained-tail.jsonl','uncommitted-tail.bin'):files[name+'/'+rel]=file_sha(root/name/rel)
        branches[name]=b
    manifest=dict(old,branches=branches,files=files,owners=[],source_job=str(a.reference),
        mixed_completion_schema='completed-placement-branch-v1',creator=dict(source=file_sha(__file__),
        original=a.reference_pin,boundary=a.boundary_pin))
    pin=write_committed(root/'boundary.json',manifest);publication.boundary(root,pin,execution=publication.code())
    completed.validate(root,branches['shared'],'shared',census,data['items'],data['labels'].roster,files)
    negatives={}
    def reject(label,b,expected):
        try:completed.validate(root,b,'shared',census,data['items'],data['labels'].roster,files)
        except ValueError as e:need(expected in str(e),'MIXED_BOUNDARY_WRONG_REFUSAL');negatives[label]=str(e)
        else:raise ValueError('MIXED_BOUNDARY_ACCEPTED:'+label)
    for label,change,error in (
        ('incomplete_cursor',lambda b:b.update(step=4),'CURSOR'),
        ('changed_final_rows',lambda b:b['completed'].update(row_state='0'*64),'CHANGED'),
        ('changed_final_file',lambda b:b['completed']['files'].__setitem__(next(iter(b['completed']['files'])),'0'*64),'CHANGED'),
        ('incomplete_final_population',lambda b:b['completed'].update(rows=b['completed']['rows']-1),'CHANGED'),
        ('tail_on_completed_branch',lambda b:b.update(tail={'records':1}),'PENDING_TAIL')):
        bad=deepcopy(branches['shared']);change(bad);reject(label,bad,error)
    # Corrupt a private copy of a real final tensor file; the physical manifest
    # gate must refuse before interpreting a completion assertion.
    rel=next(iter(branches['shared']['completed']['files']));dest=a.output/'corrupt';(dest/rel).parent.mkdir(parents=True)
    raw=bytearray((root/rel).read_bytes());raw[-1]^=1;(dest/rel).write_bytes(raw)
    bad=deepcopy(manifest);bad['files']={rel:files[rel]};h=write_committed(dest/'boundary.json',bad)
    try:publication.boundary(dest,h)
    except ValueError as e:need('PUBLICATION_FILE_PIN' in str(e),'MIXED_BOUNDARY_WRONG_PHYSICAL_REFUSAL');negatives['physical_final_bytes']=str(e)
    else:raise ValueError('MIXED_BOUNDARY_ACCEPTED_PHYSICAL')
    data['close']();need(not torch.cuda.is_initialized(),'MIXED_BOUNDARY_UNEXPECTED_GPU')
    need(file_sha(a.reference/'report.json')==a.reference_pin and file_sha(a.boundary/'boundary.json')==a.boundary_pin,
         'MIXED_BOUNDARY_INPUT_CHANGED')
    report=dict(complete=True,boundary=dict(path=str(root),pin=pin),reference=a.reference_pin,
        input_boundary=a.boundary_pin,roles=['completed','partial','fresh'],controls=negatives,
        source=file_sha(__file__),device=manifest['binding']['device'],predictor_forwards=0,gpu_used=False,
        seconds=time.perf_counter()-began,files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()},
        acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),boundary=report['boundary'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','reference','boundary','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-pin',required=True);p.add_argument('--boundary-pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

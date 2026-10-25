"""Read source and sealed placement bytes independently, including the finite suffix."""
import argparse
from copy import deepcopy
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_successor_authority as authority
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model.native_population_training import fingerprint


def need(ok,code):
    if not ok:raise ValueError('SUCCESSOR_SEAL_READBACK_'+code)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def scope(value,cfg):
    need(value['owners']==list(cfg['owners'].values()) and len(value['owners'])==9,'OWNERS')
    need(value['owner_authority']['pin']==cfg['_pin'],'AUTHORITY')
    need(value['retirement_ready']==(value['quiescence'] is not None),'READINESS')
    for b in value['branches'].values():
        if b.get('fresh'):continue
        t=b['tail'];need(t['start']==b['counts'] and t['start']['step']==b['step']
                         and t['end']['step']==b['step']+t['records'],'TAIL_COUNTS')


def compare(source,expected):
    total=0;h=hashlib.sha256()
    while raw:=expected.read(1<<20):
        need(source.read(len(raw))==raw,'SOURCE_BYTES');h.update(raw);total+=len(raw)
    return dict(bytes=total,sha256=h.hexdigest())


def owners(cfg,stopped):
    result={}
    for name,ref in cfg['owners'].items():
        root=Path('/proc')/str(ref['pid']);stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        children=set()
        for p in (root/'task').iterdir():
            try:children.update(map(int,(p/'children').read_text().split()))
            except FileNotFoundError:pass
        need(stat[19]==ref['starttime'] and int(stat[1])==ref['parent'] and sorted(children)==ref['children']
             and hashlib.sha256((root/'cmdline').read_bytes()).hexdigest()==ref['command'],'KERNEL_IDENTITY')
        need(stat[0]=='T' if stopped else stat[0] in ('R','S','D'),'KERNEL_STATE')
        result[name]=dict(pid=ref['pid'],starttime=stat[19],state=stat[0],children=sorted(children))
    return result


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    manifest,census=publication.boundary(a.input,a.pin,execution=publication.code())
    ref=manifest['owner_authority'];cfg,_,_,fresh=authority.authenticate(Path(ref['path']),ref['pin'])
    cfg=dict(cfg,_pin=ref['pin']);scope(manifest,cfg);before=owners(cfg,manifest['retirement_ready'])
    need(manifest['source_job']==cfg['owners']['producer']['output'],'SOURCE_JOB')
    physical_bytes=0
    for path,pin in manifest['source_pins'].items():
        need(sha(path)==pin,'ORIGINAL_FILE');physical_bytes+=Path(path).stat().st_size
    job=Path(manifest['source_job']);results={}
    for name,b in manifest['branches'].items():
        if b.get('fresh'):continue
        root=a.input/name;cp0=torch.load(root/'checkpoint-0.pt',map_location='cpu',weights_only=True)
        chain=digest(cp0['binding']);model=fingerprint(cp0['model']);optimizer=fingerprint(cp0['optimizer']);count=0
        with (root/'steps.jsonl').open('rb') as prefix,(root/'retained-tail.jsonl').open('rb') as tail:
            for raw in itertools.chain(prefix,tail):
                row=json.loads(raw);need(raw.endswith(b'\n') and row['cursor']==count and row['parent']==chain
                    and row['hash']==digest({k:v for k,v in row.items() if k!='hash'}),'JOURNAL_CHAIN')
                u=row['update'];need(u['document']==census.documents[count] and u['step']==count+1
                    and u['targets']==int(census.targets[count]) and u['before']==model
                    and u['noise_seed']==cp0['binding']['policy']['noise_seed']+count,'JOURNAL_UPDATE')
                count+=1;chain,model,optimizer=row['hash'],u['after'],u['optimizer']
                if str(count) in b['checkpoints']:
                    cp=torch.load(root/f'checkpoint-{count}.pt',map_location='cpu',weights_only=True)
                    need(cp['chain']==chain and fingerprint(cp['model'])==model and fingerprint(cp['optimizer'])==optimizer,
                         'CHECKPOINT_JOURNAL')
        need(count==b['tail']['end']['step'] and chain==b['tail']['end']['chain']
             and model==b['tail']['end']['model'] and optimizer==b['tail']['end']['optimizer'],'TAIL_END')
        with (job/name/'steps.jsonl').open('rb') as original:
            parts={}
            for rel in ('steps.jsonl','retained-tail.jsonl','uncommitted-tail.bin'):
                with (root/rel).open('rb') as stored:parts[rel]=compare(original,stored)
            cut=sum(p['bytes'] for p in parts.values());need(cut==manifest['journal_cuts'][name]['bytes'],'SOURCE_CUT')
            if manifest['retirement_ready']:need(not original.read(1),'FROZEN_SOURCE_GREW')
        need(parts['steps.jsonl']['sha256']==b['counts']['prefix_sha256']
             and parts['retained-tail.jsonl']['sha256']==b['tail']['sha256']
             and parts['uncommitted-tail.bin']==b['tail']['torn'],'SOURCE_PARTS')
        results[name]=dict(checkpoint=b['step'],retained_records=b['tail']['records'],observed_step=count,
            chain=chain,model=model,optimizer=optimizer,source_bytes=cut,parts=parts)
    if manifest['retirement_ready']:
        q=manifest['quiescence'];need(sha(Path(q['path'])/'contract.json')==q['pin'],'QUIESCENCE_PIN')
        need(not (Path(q['path'])/'decision.json').exists(),'DECISION_ALREADY_MADE')
    after=owners(cfg,manifest['retirement_ready']);fresh()
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:need(code in str(error),'CONTROL_CODE');controls[name]=str(error)
        else:raise ValueError('SUCCESSOR_SEAL_READBACK_CONTROL_ACCEPTED:'+name)
    bad=deepcopy(manifest);bad['owners'].pop();reject('missing_owner',lambda:scope(bad,cfg),'OWNERS')
    bad=deepcopy(manifest);first=next(n for n,b in bad['branches'].items() if not b.get('fresh'))
    bad['branches'][first]['tail']['records']+=1;reject('wrong_tail_count',lambda:scope(bad,cfg),'TAIL_COUNTS')
    with (a.input/first/'steps.jsonl').open('rb') as stream:sample=stream.read(1024)
    changed=sample[:-1]+bytes([sample[-1]^1])
    reject('changed_source_bytes',lambda:compare(io.BytesIO(changed),io.BytesIO(sample)),'SOURCE_BYTES')
    need(not torch.cuda.is_initialized(),'UNEXPECTED_GPU')
    source=a.output/'source-reader.py';source.write_bytes(Path(__file__).read_bytes())
    report=dict(complete=True,boundary=a.pin,authority=ref['pin'],source=sha(__file__),
        all_checkpoint_bytes_exact=True,committed_prefix_exact=True,retained_tail_bytes_exact=True,owner_closure_complete=True,
        retirement_ready=manifest['retirement_ready'],quiescence=manifest['quiescence'],before=before,after=after,
        results=results,original_files=len(manifest['source_pins']),original_bytes=physical_bytes,controls=controls,
        seconds=time.perf_counter()-began,gpu_used=False,optimizer_updates=0,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),results=results,seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

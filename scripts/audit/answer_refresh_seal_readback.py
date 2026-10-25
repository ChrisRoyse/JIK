"""Independent journal, checkpoint and original-byte readback of a four-owner seal."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_refresh_contract as contract
from audit.legal_placement_successor_seal_readback import compare,digest,owners
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def inspect(root,job,name,b,census,cut):
    """Walk actual records once and compare prefix, suffix and torn bytes to source."""
    zero=torch.load(root/name/'checkpoint-0.pt',map_location='cpu',weights_only=True)
    chain=digest(zero['binding']);model=fingerprint(zero['model']);optimizer=fingerprint(zero['optimizer']);count=0
    with (root/name/'steps.jsonl').open('rb') as prefix,(root/name/'retained-tail.jsonl').open('rb') as tail:
        for raw in itertools.chain(prefix,tail):
            row=json.loads(raw)
            need(raw.endswith(b'\n') and row['cursor']==count and row['parent']==chain
                 and row['hash']==digest({k:v for k,v in row.items() if k!='hash'}),'ANSWER_REFRESH_BACK_JOURNAL')
            u=row['update'];need(u['document']==census.documents[count] and u['step']==count+1
                 and u['targets']==int(census.targets[count]) and u['before']==model
                 and u['noise_seed']==zero['binding']['policy']['noise_seed']+count,'ANSWER_REFRESH_BACK_UPDATE')
            count+=1;chain,model,optimizer=row['hash'],u['after'],u['optimizer']
            if str(count) in b['checkpoints']:
                cp=torch.load(root/name/f'checkpoint-{count}.pt',map_location='cpu',weights_only=True)
                need(cp['chain']==chain and fingerprint(cp['model'])==model and fingerprint(cp['optimizer'])==optimizer,
                     'ANSWER_REFRESH_BACK_CHECKPOINT')
    need(count==b['tail']['end']['step'] and chain==b['tail']['end']['chain']
         and model==b['tail']['end']['model'] and optimizer==b['tail']['end']['optimizer'],'ANSWER_REFRESH_BACK_END')
    with (job/name/'steps.jsonl').open('rb') as stream:
        parts={}
        for rel in ('steps.jsonl','retained-tail.jsonl','uncommitted-tail.bin'):
            with (root/name/rel).open('rb') as expected:parts[rel]=compare(stream,expected)
        need(sum(p['bytes'] for p in parts.values())==cut['bytes'] and not stream.read(1),'ANSWER_REFRESH_BACK_SOURCE_CUT')
    need(parts['steps.jsonl']['sha256']==b['counts']['prefix_sha256']
         and parts['retained-tail.jsonl']['sha256']==b['tail']['sha256']
         and parts['uncommitted-tail.bin']==b['tail']['torn'],'ANSWER_REFRESH_BACK_SOURCE_PARTS')
    final_bytes=0
    if 'completed' in b:
        need(b['step']==len(census.documents) and b['tail']['records']==0,'ANSWER_REFRESH_BACK_COMPLETED_CURSOR')
        for rel,pin in b['completed']['files'].items():
            need(file_sha(root/rel)==file_sha(job/rel)==pin,'ANSWER_REFRESH_BACK_COMPLETED_BYTES')
            final_bytes+=(root/rel).stat().st_size
    return dict(checkpoint=b['step'],observed_step=count,parts=parts,chain=chain,model=model,optimizer=optimizer,
                completed='completed' in b,retained_final_bytes=final_bytes)


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    m,census=publication.boundary(a.input,a.pin,execution=publication.code())
    ref=m['owner_authority'];cfg,_,_,fresh=contract.authenticate(Path(ref['path']),ref['pin'])
    need(m['owners']==list(cfg['owners'].values()) and len(m['owners'])==4 and m['retirement_ready']
         and m['source_job']==str(contract.SOURCE),'ANSWER_REFRESH_BACK_SCOPE')
    before=owners(cfg,True);physical=0
    for path,h in m['source_pins'].items():
        need(file_sha(path)==h,'ANSWER_REFRESH_BACK_ORIGINAL');physical+=Path(path).stat().st_size
    rows={n:inspect(a.input,contract.SOURCE,n,b,census,m['journal_cuts'][n]) for n,b in m['branches'].items() if not b.get('fresh')}
    q=m['quiescence'];need(file_sha(Path(q['path'])/'contract.json')==q['pin']
        and not (Path(q['path'])/'decision.json').exists(),'ANSWER_REFRESH_BACK_QUIESCENCE')
    after=owners(cfg,True);fresh();need(not torch.cuda.is_initialized(),'ANSWER_REFRESH_BACK_GPU')
    r=dict(complete=True,boundary=a.pin,authority=ref['pin'],source=file_sha(__file__),results=rows,
        all_checkpoint_bytes_exact=True,committed_prefix_exact=True,retained_tail_bytes_exact=True,
        completed_branch_bytes_exact=True,owner_closure_complete=True,retirement_ready=True,quiescence=q,
        before=before,after=after,original_bytes=physical,gpu_used=False,optimizer_updates=0,
        seconds=time.perf_counter()-began,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),seconds=r['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

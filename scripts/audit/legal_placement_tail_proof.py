"""Retain real post-checkpoint updates and execute original/resumed/cold comparisons."""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as original
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model import placement_resume_tail as tail
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need

OLD=('placement-driver-boundary-v1-20261024','970b310e13968124836e2c2af771bf705f5eca40bd21461f7c263dfedaecba23')
ORIGINAL=('placement-driver-original-v1-20261024','af334059da30b3a6cf1961b97c4c3b838fb00aeb8d2762a80f68c54a6202ce40')


def copy_boundary(source,out,manifest):
    out.mkdir();files={};scripts=Path(__file__).resolve().parents[1]
    for rel,pin in manifest['files'].items():
        p=source/rel
        if rel.startswith('source/') and rel[7:] in publication.NEW:p=scripts/rel[7:];pin=file_sha(p)
        publication.copy_file(p,out/rel,pin);files[rel]=pin
    for name,h in publication.code().items():
        rel='source/'+name
        if rel not in files:publication.copy_file(scripts/name,out/rel,h);files[rel]=h
    return files


def child(a,label,script,args,expected=0):
    out=a.output/label;argv=[sys.executable,'-I',str(Path(__file__).parent/script),*args,'--output',str(out)]
    with (a.output/(label+'.log')).open('xb') as log:
        result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        log.flush();os.fsync(log.fileno())
    need((result.returncode==0)==(expected==0),'TAIL_PROOF_CHILD_EXIT:'+label)
    return out


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    execution=publication.code();source_pins=dict(execution,**{'audit/legal_placement_tail_proof.py':file_sha(__file__)})
    for name,pin in source_pins.items():publication.copy_file(Path(__file__).resolve().parents[1]/name,a.output/'source'/name,pin)
    old,census=publication.boundary(a.root/OLD[0],OLD[1]);reference=original.pinned(a.root/ORIGINAL[0],ORIGINAL[1])
    original.physical(a.root/ORIGINAL[0],reference)
    need(old['binding']==reference['binding'] and reference['binding']['device']=='cpu','TAIL_PROOF_REFERENCE')
    dest=a.output/'boundary';files=copy_boundary(a.root/OLD[0],dest,old);branches=deepcopy(old['branches']);source_journals={}
    for name,b in branches.items():
        src=a.root/ORIGINAL[0]/name/'steps.jsonl';source_journals[name]=file_sha(src)
        b['tail']=tail.seal(src,dest/name,b['counts'],census,dict(old['binding'],architecture=name),src.stat().st_size)
        need(b['tail']['records']==3 and b['tail']['end']['step']==5,'TAIL_PROOF_SCOPE')
        for rel in ('retained-tail.jsonl','uncommitted-tail.bin'):files[name+'/'+rel]=file_sha(dest/name/rel)
    manifest=dict(old,execution=execution,files=files,branches=branches,tail_reference=ORIGINAL[1])
    pin=write_committed(dest/'boundary.json',manifest);publication.boundary(dest,pin,execution=execution)
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:need(code in str(error),'TAIL_PROOF_CONTROL_CODE');controls[name]=str(error)
        else:raise ValueError('TAIL_PROOF_CONTROL_ACCEPTED:'+name)
    b=branches['shared'];value=b['tail'];raw=(dest/'shared/retained-tail.jsonl').read_bytes();lines=raw.splitlines(keepends=True)
    binding=dict(old['binding'],architecture='shared')
    for label,change,code in (
        ('chain',lambda r:r.update(parent='0'*64),'CHAIN'),
        ('cursor',lambda r:r.update(cursor=r['cursor']+1),'CHAIN'),
        ('targets',lambda r:r['update'].update(targets=r['update']['targets']+1),'UPDATE'),
        ('model',lambda r:r['update'].update(before='0'*64),'UPDATE'),
        ('noise_cursor',lambda r:r['update'].update(noise_seed=r['update']['noise_seed']+1),'UPDATE')):
        row=json.loads(lines[0]);change(row);row['hash']=digest({k:v for k,v in row.items() if k!='hash'});bad=canonical(row)+b'\n'
        path=a.output/(label+'.jsonl');path.write_bytes(bad)
        reject(label,lambda bad=bad:tail.scan(io.BytesIO(bad),len(bad),b['counts'],census,binding),code)
    reject('wrong_replay_cursor',lambda:tail.Replay(dest/'shared/retained-tail.jsonl',value,3),'REPLAY_CURSOR')
    with tail.Replay(dest/'shared/retained-tail.jsonl',value,2) as replay:
        reject('early_replay_finish',replay.finish,'REPLAY_INCOMPLETE')
        reject('different_actual_update',lambda:replay.check(lines[0][:-2]+b'X\n',2),'REPLAY_MISMATCH')
    clipped=a.output/'clipped';clipped.mkdir();(clipped/'retained-tail.jsonl').write_bytes(raw[:-1]);(clipped/'uncommitted-tail.bin').touch()
    reject('truncated_committed_bytes',lambda:tail.validate(clipped,value,b['counts'],census,binding),'COMMITTED_BYTES')
    prefix=(dest/'shared/steps.jsonl').read_bytes();partial=lines[-1][:len(lines[-1])//2]
    torn_source=a.output/'torn-source.jsonl';torn_source.write_bytes(prefix+b''.join(lines[:-1])+partial)
    torn=a.output/'torn';torn.mkdir();v=tail.seal(torn_source,torn,b['counts'],census,binding,torn_source.stat().st_size)
    need(v['records']==2 and v['end']['step']==4 and (torn/'uncommitted-tail.bin').read_bytes()==partial,'TAIL_PROOF_TORN')
    write_committed(torn/'readback.json',dict(complete=True,tail=v,uncommitted_not_promoted=True))
    empty=a.output/'empty';empty.mkdir();source=a.output/'empty-source.jsonl';source.write_bytes(prefix)
    v=tail.seal(source,empty,b['counts'],census,binding,len(prefix))
    with tail.Replay(empty/'retained-tail.jsonl',v,2) as replay:need(replay.finish()['records']==0,'TAIL_PROOF_EMPTY')
    # A rehashed, internally consistent but numerically wrong witness must fail
    # the real driver before the first replayed update reaches its new ledger.
    corrupt=a.output/'corrupt-boundary';bad_manifest=deepcopy(manifest);bad_files=copy_boundary(dest,corrupt,manifest)
    altered=[];chain=b['counts']['chain']
    for i,line in enumerate(lines):
        row=json.loads(line)
        if i==0:row['update']['loss']+=1.0
        row['parent']=chain;row['hash']=digest({k:v for k,v in row.items() if k!='hash'});chain=row['hash'];altered.append(canonical(row)+b'\n')
    # Replace only the proof's private copied witness, preserving its original above.
    (corrupt/'shared/retained-tail.jsonl').write_bytes(b''.join(altered))
    with (corrupt/'shared/retained-tail.jsonl').open('rb') as stream:
        bad_manifest['branches']['shared']['tail']=tail.scan(stream,sum(map(len,altered)),b['counts'],census,binding)
    bad_files['shared/retained-tail.jsonl']=file_sha(corrupt/'shared/retained-tail.jsonl');bad_manifest['files']=bad_files
    bad_pin=write_committed(corrupt/'boundary.json',bad_manifest);publication.boundary(corrupt,bad_pin,execution=execution)
    base=['--root',str(a.root),'--scope','small','--cpu-proof']
    failed=child(a,'refused-replay','legal_placement_resume_training.py',base+['--resume',str(corrupt),'--resume-pin',bad_pin],expected=1)
    need('PLACEMENT_TAIL_REPLAY_MISMATCH' in (failed/'failure.json').read_text()
         and (failed/'shared/steps.jsonl').read_bytes()==prefix and not (failed/'shared/checkpoint-3.pt').exists()
         and not (failed/'report.json').exists(),'TAIL_PROOF_REFUSAL_BOUNDARY')
    controls['actual_wrong_update']='PLACEMENT_TAIL_REPLAY_MISMATCH; no new committed update'
    resume=base+['--resume',str(dest),'--resume-pin',pin]
    resumed=child(a,'resumed','legal_placement_resume_training.py',resume);resumed_pin=file_sha(resumed/'report.json')
    cold=child(a,'cold','legal_placement_resume_training.py',resume+['--previous',str(resumed),'--previous-pin',resumed_pin])
    cold_pin=file_sha(cold/'report.json')
    args=[]
    for name,path,h in (('original',a.root/ORIGINAL[0],ORIGINAL[1]),('candidate',resumed,resumed_pin),('cold',cold,cold_pin),('boundary',dest,pin)):
        args+=['--'+name,str(path),'--'+name+'-pin',h]
    matched=child(a,'match','legal_placement_resume_match.py',args);match=json.loads((matched/'report.json').read_bytes())
    child(a,'source-readback','legal_grouped_placement_readback.py',['--input',str(resumed),'--input-pin',resumed_pin])
    for path in (resumed,cold):
        report=json.loads((path/'report.json').read_bytes())
        need(set(report['retained_tail_replays'])==set(branches),'TAIL_PROOF_REPLAY_ROSTER')
        for name,receipt in report['retained_tail_replays'].items():
            need(receipt['complete'] and receipt['byte_exact'] and receipt['records']==3
                 and receipt['sha256']==branches[name]['tail']['sha256'],'TAIL_PROOF_REPLAY_RESULT')
    need(all(file_sha(a.root/ORIGINAL[0]/name/'steps.jsonl')==h for name,h in source_journals.items()),'TAIL_PROOF_SOURCE_CHANGED')
    need(publication.code()==execution and not torch.cuda.is_initialized(),'TAIL_PROOF_EXECUTION_CHANGED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,kind='placement-tail-proof-v1',execution=execution,source=file_sha(__file__),
        original=ORIGINAL[1],boundary=pin,resumed=resumed_pin,cold=cold_pin,match=file_sha(matched/'report.json'),
        exact_comparison=match['comparison'],evaluation_rows=match['evaluation_rows'],controls=controls,
        replayed_records_per_run=9,files=files,seconds=time.perf_counter()-began,gpu_used=False,
        production_promoted=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

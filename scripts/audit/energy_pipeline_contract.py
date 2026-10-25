"""Bind the current three owners and unchanged scientific programs to an energy trial."""
import json
from pathlib import Path

from audit import answer_refresh_contract as previous
from audit.surface_device_queue import identity, need, sha, stamp
from claims import process_handoff as process

ROOT=previous.ROOT; REPO=previous.REPO; answer=previous.answer
ANSWER=previous.ANSWER; DECODER=previous.DECODER
PARENT=dict(path=str(ROOT/'answer-refresh-authority-v4-20261025/authority.json'),
            pin='292b6bbf38e3cc5e6f6ed7d9e176b4777b1bf558f8e29a57e2f3cb021fea3ea7')
OLD_ROOT=ROOT/'answer-pipeline-refresh-v1-20261025'; SOURCE=OLD_ROOT/'placement/full'
OWNERS=(('coordinator',3643397,'412706163'),('placement',3643410,'412706479'),
        ('producer',3644105,'412732985'))
NEW=tuple('audit/'+n+'.py' for n in ('energy_pipeline_contract','energy_pipeline',
    'energy_placement_program','energy_pipeline_proof','energy_pipeline_seal',
    'energy_pipeline_seal_readback','energy_pipeline_seal_proof','energy_pipeline_retirement'))
ENERGY=('model/native_energy_batched.py','model/native_energy_forecaster.py','audit/energy_placement_qualification.py',
        'audit/energy_placement_readback.py','model/energy_placement_continuation.py',
        'audit/energy_placement_training.py','audit/energy_continuation_readback.py')
CPU=dict(
    update=dict(path=str(ROOT/'energy-placement-cpu-v1-20261025'),
                pin='24f2a64cdce988871e9a6ebd872c48f4cc486a5a72945332b86118de9006f937'),
    cold=dict(path=str(ROOT/'energy-placement-cold-v1-20261025'),
              pin='d596b5f9b4702f24c6ceda79ccd224f02187dd17f56d056e90139e3323e31aff'),
    continuation=dict(path=str(ROOT/'energy-continuation-match-v1-20261025'),
                      pin='6794170e9801bbb6d9d88c55bb20e5299d91202011238c503795b2ebc956b139'))
REFERENCE={name:dict(path=str(OLD_ROOT/'placement'/folder),pin=pin) for name,folder,pin in (
    ('qualification','resumed','8f2034330e8b2100cf5a16a44ea05af729d1daf6edb2467ad77d54c1a8a757b8'),
    ('qualification-cold','cold','ff783a63485afae5fc8ff6bd8c1e512311a0c8ab4a5d93093d57c72c66fe3e8b'),
    ('match','match','d28ab325567c851bf001740faf8dc33df35b17960fcb41966e2bb7b1e921002f'))}
CUDA_BOUNDARY=dict(path=str(OLD_ROOT/'placement/mixed-boundary/boundary'),
                   pin='e637d946575a56f282c4219afbefa358e908e75e1f6a379e0bfac091925ab00c')
read=previous.read
answer_complete=previous.answer_complete


def live(cfg):
    expected=list(cfg['owners'].values()); order=[cfg['owners'][n]['pid'] for n,_,_ in OWNERS]
    rows=process.capture(expected,order); by={v['pid']:v for v in expected}
    need(all(process.binding(v)==process.binding(by[v['pid']]) for v in rows),'ENERGY_PIPELINE_CLOSURE')
    return rows


def terminal(cfg):
    for row in cfg['owners'].values():
        now=identity(row['pid'])
        need(now in (None,row['starttime']),'ENERGY_PIPELINE_PID_REUSED')
        need(now is None,'ENERGY_PIPELINE_LIVE_PREDECESSOR')


def scope(cfg,old,d):
    need(cfg['schema']=='energy-answer-decoder-v1' and cfg['parent']==PARENT
         and cfg['answer']==ANSWER and cfg['decoder']==DECODER,'ENERGY_PIPELINE_PARENT')
    need(cfg['answer_stages']==list(answer.STAGES) and cfg['decoder_stages']==list(previous.decoder.STAGES),
         'ENERGY_PIPELINE_STAGES')
    need(cfg['counts']==old['counts'] and cfg['evaluation']==old['evaluation']
         and cfg['decoder_train']==d['train'] and cfg['decoder_updates']==d['updates'],'ENERGY_PIPELINE_POPULATION')
    need(cfg['execution']==dict(serial=True,automatic_retry=False,subset=None,length_limit=None,
         best_of_proposals=False,device='cuda',modes=['qualify','reference']),'ENERGY_PIPELINE_EXECUTION')
    need(set(cfg['owners'])=={n for n,_,_ in OWNERS} and cfg['stop_order']==[n for n,_,_ in OWNERS],
         'ENERGY_PIPELINE_OWNER_SCOPE')
    for n,pid,start in OWNERS:
        need((cfg['owners'][n]['pid'],cfg['owners'][n]['starttime'])==(pid,start),'ENERGY_PIPELINE_OWNER_IDENTITY')
    need(cfg['source_job']==str(SOURCE) and cfg['cpu']==CPU and cfg['reference']==REFERENCE
         and cfg['cuda_boundary']==CUDA_BOUNDARY and cfg['cuda_reference']==previous.CUDA_REFERENCE,
         'ENERGY_PIPELINE_INPUT_SCOPE')


def base():
    parent,old,d,fresh=previous.authenticate(Path(PARENT['path']),PARENT['pin'])
    code=dict(parent['code'])
    for n in (*NEW,*ENERGY):
        h=sha(REPO/'scripts'/n); need(n not in code or code[n]==h,'ENERGY_PIPELINE_CODE_CONFLICT'); code[n]=h
    return parent,old,d,code,fresh


def prepare():
    parent,old,d,code,fresh=base(); owners={}
    for n,pid,start in OWNERS:
        row=process.process(pid); need(row and row['starttime']==start,'ENERGY_PIPELINE_MISSING_OWNER'); owners[n]=row
    cfg=dict(schema='energy-answer-decoder-v1',parent=PARENT,answer=ANSWER,decoder=DECODER,code=code,
        owners=owners,stop_order=[n for n,_,_ in OWNERS],source_job=str(SOURCE),cpu=CPU,reference=REFERENCE,
        cuda_reference=previous.CUDA_REFERENCE,cuda_boundary=CUDA_BOUNDARY,
        answer_stages=list(answer.STAGES),decoder_stages=list(previous.decoder.STAGES),counts=old['counts'],
        evaluation=old['evaluation'],decoder_train=d['train'],decoder_updates=d['updates'],
        execution=dict(serial=True,automatic_retry=False,subset=None,length_limit=None,
                       best_of_proposals=False,device='cuda',modes=['qualify','reference']))
    scope(cfg,old,d); live(cfg); fresh(); return cfg


def authenticate(path,pin):
    cfg=read(dict(path=str(path),pin=pin)); parent,old,d,code,parent_fresh=base(); scope(cfg,old,d)
    need(cfg['code']==code,'ENERGY_PIPELINE_CODE_BINDING'); files={Path(path):pin}; reports={}
    for name,ref in {**CPU,**REFERENCE}.items():
        p=Path(ref['path']); r=read(ref,'report.json'); reports[name]=r
        need(r['complete'],'ENERGY_PIPELINE_INPUT_INCOMPLETE'); files[p/'report.json']=ref['pin']
        for rel,h in r['files'].items():files[p/rel]=h
    u,c,k=(reports[n] for n in ('update','cold','continuation'))
    need(u['device']==c['device']==k['device']=='cpu' and u['all_updates_exact'] and c['all_updates_exact']
         and c['source_pin']==CPU['update']['pin'] and c['restarts_exact'] and c['reference_replay_exact']
         and k['same_complete_comparison'] and k['candidate_class_at_all_boundaries']
         and k['implementation']=='energy','ENERGY_PIPELINE_CPU_PROOF')
    for r in (u,c):
        need(all(code.get(n)==h for n,h in r['code'].items()),'ENERGY_PIPELINE_CPU_CODE')
    need(all(code.get(n)==h for n,h in k['execution'].items()),'ENERGY_PIPELINE_CONTINUATION_CODE')
    first,cold,match=(reports[n] for n in ('qualification','qualification-cold','match'))
    need(first['binding']==cold['binding'] and first['binding']['device']=='cuda'
         and first['results_pin']==cold['results_pin']==match['results_pin']
         and cold['previous_pin']==match['candidate_pin']==REFERENCE['qualification']['pin']
         and match['cold_pin']==REFERENCE['qualification-cold']['pin'] and match['same_complete_comparison']
         and first['execution']['implementation']==cold['execution']['implementation']=='batched'
         and first['execution']['code']==cold['execution']['code']==match['execution'],
         'ENERGY_PIPELINE_REFERENCE_PROOF')
    need(all(code[n]==h for n,h in match['execution'].items()),'ENERGY_PIPELINE_REFERENCE_CODE')
    b=read(CUDA_BOUNDARY,'boundary.json'); need(b['complete'] and b['binding']==first['binding'],
                                               'ENERGY_PIPELINE_CUDA_BOUNDARY')
    files[Path(CUDA_BOUNDARY['path'])/'boundary.json']=CUDA_BOUNDARY['pin']
    for rel,h in b['files'].items():files[Path(CUDA_BOUNDARY['path'])/rel]=h
    for p,h in files.items():need(sha(p)==h,'ENERGY_PIPELINE_PHYSICAL_INPUT')
    stamps={p:stamp(p) for p in files}
    def fresh():
        parent_fresh()
        for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'ENERGY_PIPELINE_CODE_CHANGED')
        for p,s in stamps.items():need(stamp(p)==s,'ENERGY_PIPELINE_INPUT_CHANGED')
    fresh(); return cfg,old,d,fresh

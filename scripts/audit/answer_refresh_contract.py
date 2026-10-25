"""Keep both scientific programs intact while replacing their four current owners."""
import json
from pathlib import Path

from audit import legal_answer_successor_authority as answer
from audit import protected_corpus_queue_contract as decoder
from audit.surface_device_queue import identity,need,sha,stamp
from claims import process_handoff as process

ROOT=answer.ROOT;REPO=answer.REPO
ANSWER=decoder.PARENT
DECODER=dict(path=str(ROOT/'protected-corpus-queue-authority-v2-20261025/authority.json'),
             pin='1d9495cfcff59c6b6f3e2ab3eba89d58f3c7b8faa04bdc67cedcebcc10b40f41')
OLD_ROOT=ROOT/'legal-answer-successor-v1-20261024'
SOURCE=OLD_ROOT/'placement/full'
CUDA_REFERENCE=dict(path=str(OLD_ROOT/'placement/original'),
    pin='86a29a0c5ceaffa1de51406721320f4b5c6b0bb6388adceaa9f3766e4b60ffc4')
CUDA_BOUNDARY=dict(path=str(OLD_ROOT/'placement/boundary'),
    pin='bc917da9ccbc088718836706175b9c8622ea1df79a9fa85de5b0fe50632c2c15')
OWNERS=(('queue',3592252,'411605913'),('answer',3511919,'410085379'),
        ('placement',3511924,'410085679'),('producer',3513348,'410116925'))
NEW=('audit/answer_refresh_contract.py','audit/answer_pipeline_refresh.py',
     'audit/mixed_placement_program.py','audit/answer_refresh_proof.py',
     'audit/answer_refresh_seal.py','audit/answer_refresh_seal_readback.py',
     'audit/answer_refresh_retirement.py','audit/answer_refresh_seal_proof.py')
MIXED=('audit/mixed_placement_boundary.py','audit/mixed_placement_training.py',
       'audit/mixed_placement_match.py','model/completed_placement_branch.py',
       'model/batched_placement_continuation.py','model/native_batched_validation.py',
       'model/native_placement_batched_checks.py','audit/placement_validation_proof.py')
CPU={name:ROOT/f'mixed-placement-{name}-match-v1-20261025' for name in ('original','batched')}
VALIDATION=dict(path=str(ROOT/'placement-validation-cold-v4-20261025'),
                pin='8d7f827e6a186bfaa6c488b32f39a8af0d04a17effd3e99e6ef7bd6fb5fc143a')


def read(ref,filename=None):
    p=Path(ref['path'])/(filename or '')
    need(sha(p)==ref['pin'],'ANSWER_REFRESH_PIN');return json.loads(p.read_bytes())


def live(cfg):
    expected=list(cfg['owners'].values());order=[cfg['owners'][n]['pid'] for n,_,_ in OWNERS]
    rows=process.capture(expected,order);by={v['pid']:v for v in expected}
    need(all(process.binding(v)==process.binding(by[v['pid']]) for v in rows),'ANSWER_REFRESH_CLOSURE')
    return rows


def terminal(cfg):
    for row in cfg['owners'].values():
        now=identity(row['pid'])
        need(now in (None,row['starttime']),'ANSWER_REFRESH_PID_REUSED')
        need(now is None,'ANSWER_REFRESH_LIVE_PREDECESSOR')


def scope(cfg,old,d):
    need(cfg['schema']=='answer-and-decoder-refresh-v1' and cfg['answer']==ANSWER and cfg['decoder']==DECODER,
         'ANSWER_REFRESH_SOURCE')
    need(cfg['answer_stages']==list(answer.STAGES) and cfg['decoder_stages']==list(decoder.STAGES),
         'ANSWER_REFRESH_STAGES')
    need(cfg['counts']==old['counts'] and cfg['evaluation']==old['evaluation']
         and cfg['decoder_train']==d['train'] and cfg['decoder_updates']==d['updates'],'ANSWER_REFRESH_POPULATION')
    need(cfg['execution']==dict(serial=True,automatic_retry=False,subset=None,length_limit=None,
                               best_of_proposals=False,device='cuda'),'ANSWER_REFRESH_EXECUTION')
    need(set(cfg['owners'])=={n for n,_,_ in OWNERS} and cfg['stop_order']==[n for n,_,_ in OWNERS],
         'ANSWER_REFRESH_OWNER_SCOPE')
    for n,pid,start in OWNERS:
        need((cfg['owners'][n]['pid'],cfg['owners'][n]['starttime'])==(pid,start),'ANSWER_REFRESH_OWNER_IDENTITY')
    need(cfg['source_job']==str(SOURCE) and cfg['validation']==VALIDATION
         and cfg['cuda_reference']==CUDA_REFERENCE and cfg['cuda_boundary']==CUDA_BOUNDARY,'ANSWER_REFRESH_PLACEMENT_SOURCE')


def base():
    old,_,_,af=answer.authenticate(Path(ANSWER['path']),ANSWER['pin'])
    d,_,df=decoder.authenticate(Path(DECODER['path']),DECODER['pin'])
    code=dict(old['code'])
    for n,h in d['code'].items():
        need(n not in code or code[n]==h,'ANSWER_REFRESH_CODE_CONFLICT');code[n]=h
    for n in (*NEW,*MIXED):
        h=sha(REPO/'scripts'/n);need(n not in code or code[n]==h,'ANSWER_REFRESH_CODE_CONFLICT');code[n]=h
    return old,d,code,af,df


def prepare():
    old,d,code,af,df=base();owners={}
    for n,pid,start in OWNERS:
        row=process.process(pid);need(row and row['starttime']==start,'ANSWER_REFRESH_MISSING_OWNER');owners[n]=row
    cpu={n:dict(path=str(p),pin=sha(p/'report.json')) for n,p in CPU.items()}
    cfg=dict(schema='answer-and-decoder-refresh-v1',answer=ANSWER,decoder=DECODER,code=code,
        owners=owners,stop_order=[n for n,_,_ in OWNERS],source_job=str(SOURCE),cpu=cpu,validation=VALIDATION,
        cuda_reference=CUDA_REFERENCE,cuda_boundary=CUDA_BOUNDARY,
        answer_stages=list(answer.STAGES),decoder_stages=list(decoder.STAGES),counts=old['counts'],
        evaluation=old['evaluation'],decoder_train=d['train'],decoder_updates=d['updates'],
        execution=dict(serial=True,automatic_retry=False,subset=None,length_limit=None,best_of_proposals=False,device='cuda'))
    scope(cfg,old,d);live(cfg);af();df();return cfg


def authenticate(path,pin):
    cfg=read(dict(path=str(path),pin=pin));old,d,code,af,df=base();scope(cfg,old,d)
    need(cfg['code']==code,'ANSWER_REFRESH_CODE_BINDING')
    files={Path(path):pin};execution={n:code[n] for n in (
        'model/legal_placement_resume.py','model/placement_resume_publication.py','audit/legal_placement_resume_training.py',
        'model/placement_resume_tail.py','audit/mixed_placement_training.py','audit/mixed_placement_match.py',
        'model/completed_placement_branch.py','model/batched_placement_continuation.py',
        'model/native_batched_validation.py','model/native_placement_batched_checks.py')}
    for name,p in CPU.items():
        ref=cfg['cpu'][name];need(Path(ref['path'])==p,'ANSWER_REFRESH_CPU_PATH');r=read(ref,'report.json')
        need(r['complete'] and r['same_complete_comparison'] and r['device']=='cpu'
             and r['mixed_branch_roles']==['completed','partial','fresh'] and r['execution']==execution,
             'ANSWER_REFRESH_CPU_SCOPE')
        files[p/'report.json']=ref['pin']
        for rel,h in r['files'].items():files[p/rel]=h
        producer=ROOT/f'mixed-placement-{name}-cpu-v1-20261025/report.json'
        v=read(dict(path=str(producer),pin=r['candidate_pin']))
        need(v['execution']['implementation']==name and v['execution']['code']==execution,'ANSWER_REFRESH_CPU_IMPLEMENTATION')
        files[producer]=r['candidate_pin']
    q=read(VALIDATION,'report.json')
    need(q['complete'] and q['cold_readback_exact'] and q['all_updates_exact'] and q['binding']['device']=='cpu',
         'ANSWER_REFRESH_VALIDATION_SCOPE')
    files[Path(VALIDATION['path'])/'report.json']=VALIDATION['pin']
    for ref,filename in ((CUDA_REFERENCE,'report.json'),(CUDA_BOUNDARY,'boundary.json')):
        r=read(ref,filename)
        need(r['complete'] and r['binding']['scope']=='small' and r['binding']['device']=='cuda','ANSWER_REFRESH_CUDA_REFERENCE')
        files[Path(ref['path'])/filename]=ref['pin']
    for p,h in files.items():need(sha(p)==h,'ANSWER_REFRESH_PHYSICAL_INPUT')
    stamps={p:stamp(p) for p in files}
    def fresh():
        af();df()
        for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'ANSWER_REFRESH_CODE_CHANGED')
        for p,s in stamps.items():need(stamp(p)==s,'ANSWER_REFRESH_INPUT_CHANGED')
    fresh();return cfg,old,d,fresh


def answer_complete(root,outputs):
    need(tuple(outputs)==answer.STAGES,'ANSWER_REFRESH_COMPLETION_STAGES')
    values={n:decoder.stage(outputs[n],n,root) for n in answer.STAGES}
    t,b,e=(values[n] for n in ('training','training-readback','evaluation'))
    need(t['full_population_trained'] and t['gpu_used'] and t['mode']=='full'
         and t['optimizer_updates']==b['optimizer_updates']==3*91540 and b['full_population_trained']
         and b['training']['pin']==outputs['training']['pin'],'ANSWER_REFRESH_COMPLETION_TRAINING')
    need(e['full_population_trained'] and e['full_population_evaluated'] and e['scored']['cases']==23513,
         'ANSWER_REFRESH_COMPLETION_EVALUATION')
    for n,ref in outputs.items():
        need(json.loads((root/f'completed-{n}.json').read_bytes())==ref,'ANSWER_REFRESH_COMPLETION_RECEIPT')
    return {n:r['pin'] for n,r in outputs.items()}

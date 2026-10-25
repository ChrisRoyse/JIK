"""Tensor-free authority and scope gates for the paired decoder successor."""
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import task_transfer_contract as prior
from audit.surface_device_queue import check_owner,identity,need,sha,stamp,write

P=prior.ROOT;REPO=prior.REPO;read=prior.read
LOCK=REPO/'.lease/paired-surface-queue'
WAIT=P/'task-transfer-successor-queue-v1-20261025'
AUTH=dict(path=str(P/'task-transfer-successor-authority-v2-20261025/authority.json'),
          pin='0bddaaeabebda27aa8724829bf1e0914cdb5d15cf5911fdec45a0d73c678c0fb')
SOURCE=dict(path=str(P/'paired-corpus-task-scoring-v2-20261025/report.json'),
            pin='e732f40b075c304751972f6684d0516a1ca397a9e9c2c8db96ce13a0d0d024af')
NEW=('model/paired_surface_scoring.py','audit/paired_surface_scoring.py','audit/paired_surface_readback.py',
     'model/corpus_protected_decoder.py','model/task_surface_scoring.py','audit/task_surface_scoring.py',
     'audit/task_surface_readback.py','audit/paired_surface_contract.py','audit/paired_surface_queue.py',
     'audit/paired_surface_proof.py')
STAGES=('qualification','qualification-readback','full','full-readback')
POLICY=dict(device='cuda:0',query_cases=170,qualification=[0,1,3],all_proposals=True,
            teacher_forced=True,free_generation=False,automatic_retry=False,reference_length=False,
            retention_owner='ChrisRoyse/MNLM#29',wait='coordinator_exit_and_complete_report')


def worker_scope(m,*,cuda,qualification):
    need(not qualification or cuda,'PAIRED_SCOPE_QUALIFICATION_DEVICE')
    need(m['device']==('cuda:0' if cuda else 'cpu') and m['full_training']==cuda,'PAIRED_SCOPE_DEVICE')
    need(m['ordinals']==([0,1,3] if qualification or not cuda else None),'PAIRED_SCOPE_POPULATION')


def owner_fd(fd):
    if fd is None:return
    actual=os.fstat(fd);expected=LOCK.stat()
    need((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'PAIRED_OWNER_FD')
    with LOCK.open('a+b') as other:
        try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        raise ValueError('PAIRED_OWNER_UNLOCKED')


def base():
    cfg,_,fresh=prior.authenticate(**AUTH);fresh();source=read(**SOURCE);code=dict(cfg['code'])
    for name in NEW:code[name]=sha(REPO/'scripts'/name)
    return cfg,source,code


def prepare(output):
    cfg,source,code=base();launch=WAIT/'launch.json';v=read(launch,sha(launch))
    need(v['authority']==AUTH['pin'] and v['driver_pin']==cfg['code']['audit/task_transfer_queue.py'],
         'PAIRED_PARENT_LAUNCH')
    wait=dict(path=str(WAIT),pid=v['pid'],starttime=str(v['starttime']),launch_pin=sha(launch))
    need(wait['pid']==3724744 and wait['starttime']=='414620484','PAIRED_PARENT_IDENTITY')
    check_owner(wait);need(identity(wait['pid'])==wait['starttime'],'PAIRED_PREPARE_LIVE')
    out=dict(schema='paired-surface-successor-v1',parent=AUTH,source=SOURCE,wait=wait,code=code,
             policy=POLICY,stages=list(STAGES))
    output.mkdir(exist_ok=False);write(output/'authority.json',out);return sha(output/'authority.json')


def authenticate(path,pin):
    cfg=read(path,pin);parent,source,code=base()
    need(set(cfg)=={'schema','parent','source','wait','code','policy','stages'}
         and cfg['schema']=='paired-surface-successor-v1' and cfg['parent']==AUTH and cfg['source']==SOURCE,
         'PAIRED_AUTHORITY')
    need(cfg['policy']==POLICY and cfg['stages']==list(STAGES),'PAIRED_POLICY')
    need(cfg['code']==code,'PAIRED_CODE_BINDING');wait=cfg['wait']
    need(wait['path']==str(WAIT) and wait['pid']==3724744 and wait['starttime']=='414620484','PAIRED_WAIT')
    check_owner(wait);launch=read(WAIT/'launch.json',wait['launch_pin'])
    need(launch['authority']==AUTH['pin'] and launch['driver_pin']==code['audit/task_transfer_queue.py'],
         'PAIRED_PARENT_LAUNCH')
    files={Path(path):pin,Path(AUTH['path']):AUTH['pin'],Path(SOURCE['path']):SOURCE['pin'],
           WAIT/'launch.json':wait['launch_pin']};stamps={p:stamp(p) for p in files}
    def fresh():
        for p,s in stamps.items():need(stamp(p)==s,'PAIRED_AUTHORITY_CHANGED')
        for name,h in code.items():need(sha(REPO/'scripts'/name)==h,'PAIRED_CODE_CHANGED')
        check_owner(wait)
    fresh();return cfg,source,fresh


def complete(r,cfg):
    need(r['complete'] and r['authority']==AUTH['pin'] and r['driver_pin']==cfg['code']['audit/task_transfer_queue.py']
         and r['query_cases']==170 and r['full_population_transfer_measured']
         and r['qualification']['all_events_exact'] and r['qualification']['cases']==3,
         'PAIRED_PARENT_INCOMPLETE')
    need([v['label'] for v in r['outputs']]==list(prior.STAGES),'PAIRED_PARENT_STAGES')


def terminal(cfg):
    check_owner(cfg['wait']);need(identity(cfg['wait']['pid']) is None,'PAIRED_PREDECESSOR_LIVE')
    need(not (WAIT/'failure.json').exists() and (WAIT/'report.json').is_file(),'PAIRED_PARENT_FAILED')
    parent_pin=sha(WAIT/'report.json');r=read(WAIT/'report.json',parent_pin);complete(r,cfg)
    old=read(**AUTH);prior.terminal(old)
    outputs={v['label']:v for v in r['outputs']}
    for label,ref in outputs.items():
        need(Path(ref['path'])==WAIT/label/'report.json','PAIRED_PARENT_OUTPUT_PATH')
        need(read(ref['path'],ref['pin'])['complete'],'PAIRED_PARENT_OUTPUT_INCOMPLETE')
    pairs={}
    for label,request_name,request_pin in (('qualification','request-qualification.json',r['qualification_request']),
                                           ('full','request-full.json',r['request'])):
        request=dict(path=str(WAIT/request_name),pin=request_pin);m=read(**request)
        prior.request_scope(m,'cuda:0');worker_scope(m,cuda=True,qualification=label=='qualification')
        ref=outputs[label];g=read(ref['path'],ref['pin'])
        need(g['request']==request_pin and g['full_training'] and g['gpu_used']
             and g['full_query']==(label=='full') and g['intervention']==dict(kind='trained_runtime_policy'),
             'PAIRED_PARENT_GENERATION')
        pairs[label]=dict(request=request,generated=dict(path=ref['path'],pin=ref['pin']))
    e=read(prior.PARENT/'report.json',sha(prior.PARENT/'report.json'))
    corpus=dict(plan=read(**SOURCE)['corpus']['plan'],plan_pin=read(**SOURCE)['corpus']['plan_pin'])
    for name,label in (('run','full'),('readback','full-readback')):
        ref=e['decoder_outputs'][label]
        need(Path(ref['path'])==prior.PARENT/('decoder-'+label),'PAIRED_CORPUS_PATH')
        v=read(Path(ref['path'])/'report.json',ref['pin'])
        need(v['complete'] and v['full_training'] and v['device']=='cuda:0' and v['plan']==corpus['plan_pin'],
             'PAIRED_CORPUS_SCOPE')
        corpus[name]=ref['path'];corpus[name+'_pin']=ref['pin']
    return dict(parent=parent_pin,pairs=pairs,corpus=corpus)


def check_scores(r,cold,*,cuda,qualification):
    count=3 if not cuda or qualification else 170
    need(r['complete'] and cold['complete'] and r['gpu_used']==cold['gpu_used']==cuda
         and r['qualification']==cold['qualification']==qualification
         and r['full_query']==cold['full_query']==(cuda and not qualification)
         and r['evaluated_cases']==cold['evaluated_cases']==count,'PAIRED_SCORE_SCOPE')
    need(cold['all_scored_logits_exact'] and cold['all_reference_denominators_recounted']
         and cold['all_context_and_decoder_contrasts_exact'] and cold['weights_unchanged']
         and cold['positions_per_decoder']>0 and cold['matched_atoms']>0 and len(cold['controls'])==7,
         'PAIRED_SCORE_QUALIFICATION')


def arguments(pair,corpus):
    args=['--request',pair['request']['path'],'--pin',pair['request']['pin'],
          '--generated',pair['generated']['path'],'--generated-pin',pair['generated']['pin']]
    for name in ('plan','run','readback'):args+=['--corpus-'+name,corpus[name],'--corpus-'+name+'-pin',corpus[name+'_pin']]
    return args

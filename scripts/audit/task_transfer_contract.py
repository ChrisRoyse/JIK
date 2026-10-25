"""Immutable successor scope; final CUDA requests come from the completed parent."""
from copy import deepcopy
import json
from pathlib import Path

from audit.surface_device_queue import check_owner,identity,need,sha,stamp,write

REPO=Path(__file__).resolve().parents[2];ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
PARENT=ROOT/'energy-reference-pipeline-v1-20261025'
PARENT_AUTH=dict(path=str(ROOT/'energy-pipeline-authority-v2-20261025/authority.json'),
                 pin='2c716a506004068fbe5fe550439446b8f413234326b6f18acd66ce002755702c')
SOURCE=dict(path=str(ROOT/'task-answer-transfer-authority-v4-20261025/request.json'),
            pin='fd19663d36a3e7f8a22bb3392671846c58d2b2d6488a640c88ba8f3edc541de3')
PROOF=dict(path=str(ROOT/'task-answer-transfer-readback-v1-20261025/report.json'),
           pin='45990874c5c160aaaa0a794df6e62d534b9cce35b0f32f7f115f44e8445c246f')
NEW=tuple('audit/'+n+'.py' for n in ('task_transfer_contract','task_transfer_device','task_transfer_queue','task_transfer_proof'))
STAGES=('inspect','qualification','qualification-cold','full','score')
POLICY=dict(device='cuda:0',query_cases=170,all_proposals=True,automatic_retry=False,
            reference_length=False,retention_owner='ChrisRoyse/MNLM#29',wait='coordinator_exit_and_complete_report')


def read(path,pin):
    need(sha(path)==pin,'TASK_DEVICE_PIN');return json.loads(Path(path).read_bytes())


def base():
    parent=read(**PARENT_AUTH);source=read(**SOURCE);proof=read(**PROOF)
    need(proof['complete'] and proof['request']==SOURCE['pin'] and proof['all_cold_event_states_exact']
         and proof['all_reference_metrics_exact'] and len(proof['controls'])==7,'TASK_DEVICE_PRIOR')
    need(source['plan']['summary']==dict(query=170,with_donor=66,families=dict(coding=16,conversation=100,reasoning=54))
         and source['ordinals']==[0,1,3],'TASK_DEVICE_SOURCE_SCOPE')
    code=dict(parent['code'])
    for n,h in source['code'].items():
        need(n not in code or code[n]==h,'TASK_DEVICE_CODE_CONFLICT');code[n]=h
    for n in NEW:code[n]=sha(REPO/'scripts'/n)
    for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'TASK_DEVICE_CODE')
    return parent,source,code


def prepare(output):
    parent,source,code=base();launch=PARENT/'launch.json';r=json.loads(launch.read_bytes())
    wait=dict(path=str(PARENT),launch_pin=sha(launch),pid=r['pid'],starttime=str(r['starttime']))
    need(wait['pid']==3699411 and wait['starttime']=='414043764' and r['authority']==PARENT_AUTH['pin']
         and r['driver_pin']==parent['code']['audit/energy_pipeline.py'],'TASK_DEVICE_PARENT')
    check_owner(wait);need(identity(wait['pid'])==wait['starttime'],'TASK_DEVICE_PREPARE_OWNER_LIVE')
    cfg=dict(schema='task-transfer-successor-v1',parent=PARENT_AUTH,source=SOURCE,proof=PROOF,wait=wait,
             code=code,policy=POLICY,stages=list(STAGES),qualification=source['ordinals'])
    output.mkdir(exist_ok=False);write(output/'authority.json',cfg);return sha(output/'authority.json')


def authenticate(path,pin):
    cfg=read(path,pin);parent,source,code=base()
    need(set(cfg)=={'schema','parent','source','proof','wait','code','policy','stages','qualification'}
         and cfg['schema']=='task-transfer-successor-v1' and cfg['parent']==PARENT_AUTH
         and cfg['source']==SOURCE and cfg['proof']==PROOF,'TASK_DEVICE_AUTHORITY')
    need(cfg['policy']==POLICY and cfg['stages']==list(STAGES) and cfg['qualification']==source['ordinals'],
         'TASK_DEVICE_SCOPE')
    need(cfg['code']==code,'TASK_DEVICE_CODE_BINDING')
    wait=cfg['wait'];need(wait['path']==str(PARENT) and wait['pid']==3699411 and wait['starttime']=='414043764',
                          'TASK_DEVICE_WAIT')
    check_owner(wait);launch=read(PARENT/'launch.json',wait['launch_pin'])
    need(launch['authority']==PARENT_AUTH['pin'] and launch['driver_pin']==parent['code']['audit/energy_pipeline.py'],
         'TASK_DEVICE_PARENT_BINDING')
    files={Path(path):pin,Path(PARENT_AUTH['path']):PARENT_AUTH['pin'],Path(SOURCE['path']):SOURCE['pin'],
           Path(PROOF['path']):PROOF['pin'],PARENT/'launch.json':wait['launch_pin']}
    stamps={p:stamp(p) for p in files}
    def fresh():
        for p,s in stamps.items():need(stamp(p)==s,'TASK_DEVICE_AUTHORITY_CHANGED')
        for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'TASK_DEVICE_CODE_CHANGED')
        check_owner(wait)
    fresh();return cfg,source,fresh


def terminal(cfg):
    check_owner(cfg['wait']);need(identity(cfg['wait']['pid']) is None,'TASK_DEVICE_PREDECESSOR_LIVE')
    need(not (PARENT/'failure.json').exists() and (PARENT/'report.json').is_file(),'TASK_DEVICE_PARENT_FAILED')
    parent_pin=sha(PARENT/'report.json');r=read(PARENT/'report.json',parent_pin)
    need(r['complete'] and r['authority']==PARENT_AUTH['pin'] and r['driver_pin']==cfg['code']['audit/energy_pipeline.py']
         and r['full_population_trained'] and r['full_population_evaluated'] and r['protected_decoder_complete'],
         'TASK_DEVICE_PARENT_INCOMPLETE')
    ref=r['outputs']['evaluation'];need(Path(ref['path'])==PARENT/'evaluation','TASK_DEVICE_EVALUATION_PATH')
    e=read(Path(ref['path'])/'report.json',ref['pin'])
    need(e['complete'] and e['mode']=='cuda' and e['full_population_trained'] and e['full_population_evaluated']
         and e['qualification']['all_events_exact'] and e['qualification']['gpu_used'],'TASK_DEVICE_EVALUATION')
    full=[x for x in e['outputs'] if x['label']=='full'];need(len(full)==1,'TASK_DEVICE_FULL_EXPORT')
    ref=full[0];need(Path(ref['path'])==PARENT/'evaluation/full/worker/report.json','TASK_DEVICE_FULL_PATH')
    worker=read(ref['path'],ref['pin'])
    need(worker['complete'] and worker['mode']=='cuda' and worker['device']=='cuda:0' and worker['gpu_used']
         and worker['memory_policy']['intervention']==dict(kind='trained_runtime_policy'),'TASK_DEVICE_FULL_WORKER')
    path=PARENT/'evaluation/full/request.json';m=read(path,worker['request'])
    return m,dict(parent=parent_pin,evaluation=r['outputs']['evaluation']['pin'],worker=ref['pin'],
                  exported=dict(path=str(path),pin=worker['request']))


def request(cfg,authority_pin,source,*,full,qualification=False):
    if full:m,handoff=terminal(cfg)
    else:m=source;handoff=None
    value={k:deepcopy(m[k]) for k in ('runtime','checkpoints','intervention')}
    value.update(schema='task-answer-device-request-v1',code=cfg['code'],pack=source['pack'],plan=source['plan'],
        plan_pin=source['plan_pin'],ordinals=cfg['qualification'] if qualification or not full else None,
        authority=dict(source['authority'],successor=authority_pin,handoff=handoff,code=None,
            scope='original_tasks_development_transfer' if full else 'CPU_device_handoff_mechanics'),
        device='cuda:0' if full else 'cpu',full_training=full)
    # Canonical fingerprint is added by the worker's tensor-free request seal convention.
    value['authority']['code']=sha_json(cfg['code'])
    return value


def sha_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def request_scope(m,device):
    need(set(m)=={'runtime','checkpoints','intervention','schema','code','pack','plan','plan_pin','ordinals',
                 'authority','device','full_training'} and m['schema']=='task-answer-device-request-v1',
         'TASK_DEVICE_REQUEST_SCHEMA')
    need(m['device']==device and device in ('cpu','cuda:0') and m['full_training']==(device=='cuda:0'),
         'TASK_DEVICE_REQUEST_DEVICE')
    source=read(**SOURCE)
    need(m['pack']==source['pack'] and m['plan']==source['plan'] and m['plan_pin']==source['plan_pin']
         and m['ordinals'] in (None,source['ordinals']) and (device=='cuda:0' or m['ordinals']==source['ordinals']),
         'TASK_DEVICE_REQUEST_SCOPE')
    need(m['authority']['code']==sha_json(m['code']),'TASK_DEVICE_REQUEST_CODE')
    for n,h in m['code'].items():need(sha(REPO/'scripts'/n)==h,'TASK_DEVICE_REQUEST_SOURCE')
    if device=='cuda:0':
        need(m['authority']['handoff'] is not None and m['intervention']==dict(kind='trained_runtime_policy'),
             'TASK_DEVICE_FULL_POLICY')
        for row in m['checkpoints'].values():
            v=row['metadata'];b=v['binding'];c=v['counts']
            need(b['full_population'] and b['documents']==v['cursor']==c['documents']==c['ends']==91540
                 and c['atoms']==6547176 and c['answers']==6368511 and c['decisions']==6460051,
                 'TASK_DEVICE_FULL_CHECKPOINT')
    else:
        need(m['checkpoints']==source['checkpoints'] and m['runtime']==source['runtime']
             and m['intervention']==source['intervention'] and m['authority']['handoff'] is None,'TASK_DEVICE_PROOF_SOURCE')

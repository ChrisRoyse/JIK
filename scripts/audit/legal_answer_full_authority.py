"""Pinned handoff from real CUDA qualification to the unchanged full-source runner."""
import json
from pathlib import Path
import sys

from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
WORKER=REPO/'scripts/audit/legal_answer_source_training.py'
RECOVERY=ROOT/'legal-answer-recovery-queue-v1-20261024'
AUTHORITY=dict(path=str(ROOT/'legal-answer-recovery-authority-v3-20261024/authority.json'),
               pin='6ef5df736cde13791252f3d35b440ef03197af4a2f70f497d8013918c6e9a549')
NEW=('audit/legal_answer_full_authority.py','audit/legal_answer_full_queue.py')
ARCHES=('shared','own_slot','whole_proposals')


def read(path,pin):
    need(sha(path)==pin,'ANSWER_FULL_PIN');return json.loads(Path(path).read_bytes())


def owner(path):
    p=Path(path)/'launch.json';v=json.loads(p.read_bytes())
    return dict(path=str(path),pid=v['pid'],starttime=str(v['starttime']),launch_pin=sha(p))


def counts(value):
    need(set(value)==set(ARCHES),'ANSWER_FULL_ARCHITECTURES')
    for v in value.values():
        need(v['documents']==v['ends']==91540 and v['atoms']==6547176
             and v['answers']==6368511 and v['decisions']==6460051,'ANSWER_FULL_COUNTS')


def stages(outputs,expected):
    need(len(outputs)==len(expected) and {r['label'] for r in outputs}==set(expected),
         'ANSWER_FULL_STAGES')
    return {r['label']:r for r in outputs}


def ready(ref):
    check_owner(ref);need(identity(ref['pid']) is None,'ANSWER_FULL_PREDECESSOR_LIVE')


def prepare(output):
    parent=read(AUTHORITY['path'],AUTHORITY['pin']);ref=parent['queues']['source']['authority']
    source=read(ref['path'],ref['pin']);inspection=read(Path(source['inspection']['path'])/'report.json',source['inspection']['pin'])
    counts(inspection['source']['counts']);wait=owner(RECOVERY);check_owner(wait)
    launch=read(RECOVERY/'launch.json',wait['launch_pin'])
    need(launch['authority']==AUTHORITY['pin'],'ANSWER_FULL_PARENT_AUTHORITY')
    code=dict(parent['code'])
    for n,h in inspection['code'].items():
        need(n not in code or code[n]==h,'ANSWER_FULL_CODE_CONFLICT');code[n]=h
    code.update({n:sha(REPO/'scripts'/n) for n in NEW})
    value=dict(schema='complete-answer-full-queue-v1',wait=wait,recovery_authority=AUTHORITY,
        source_authority=ref,inspection=source['inspection'],code=code,counts=inspection['source']['counts'],
        training=dict(mode='full',architectures=list(ARCHES),document_updates=3*91540,
                      subset=None,length_limit=None,automatic_retry=False))
    write(output/'authority.json',value);return sha(output/'authority.json')


def authenticate(path,pin):
    cfg=read(path,pin)
    need(cfg['schema']=='complete-answer-full-queue-v1' and cfg['recovery_authority']==AUTHORITY,
         'ANSWER_FULL_AUTHORITY')
    parent=read(AUTHORITY['path'],AUTHORITY['pin']);ref=cfg['source_authority']
    need(ref==parent['queues']['source']['authority'],'ANSWER_FULL_SOURCE_AUTHORITY')
    prior=read(ref['path'],ref['pin']);i=cfg['inspection'];p=Path(i['path'])/'report.json'
    inspection=read(p,i['pin']);counts(cfg['counts'])
    need(i==prior['inspection'] and inspection['complete'] and inspection['mode']=='inspect'
         and inspection['optimizer_updates']==0 and not inspection['gpu_used']
         and inspection['source']['counts']==cfg['counts'],'ANSWER_FULL_INSPECTION')
    need(cfg['training']==dict(mode='full',architectures=list(ARCHES),document_updates=3*91540,
         subset=None,length_limit=None,automatic_retry=False),'ANSWER_FULL_TRAINING_SCOPE')
    expected=dict(parent['code'])
    for n,h in inspection['code'].items():
        need(n not in expected or expected[n]==h,'ANSWER_FULL_CODE_CONFLICT');expected[n]=h
    expected.update({n:sha(REPO/'scripts'/n) for n in NEW})
    need(cfg['code']==expected,'ANSWER_FULL_CODE_BINDING');check_owner(cfg['wait'])
    launch=read(Path(cfg['wait']['path'])/'launch.json',cfg['wait']['launch_pin'])
    need(launch['authority']==AUTHORITY['pin'],'ANSWER_FULL_PARENT_AUTHORITY')
    files={Path(path):pin,Path(AUTHORITY['path']):AUTHORITY['pin'],Path(ref['path']):ref['pin'],p:i['pin'],
           Path(cfg['wait']['path'])/'launch.json':cfg['wait']['launch_pin']}
    files.update({Path(i['path'])/n:h for n,h in inspection['files'].items()})
    for p,h in files.items():need(sha(p)==h,'ANSWER_FULL_PROOF_BYTES')
    snapshots={p:stamp(p) for p in files}
    def fresh():
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h,'ANSWER_FULL_CODE_CHANGED')
        for p,s in snapshots.items():need(stamp(p)==s,'ANSWER_FULL_AUTHORITY_CHANGED')
    fresh();return cfg,inspection,fresh


def planned_arguments(cfg,output):
    """Exact fixed paths; pins are late-bound only from the real terminal receipts."""
    root=Path(cfg['wait']['path'])
    return dict(worker=str(WORKER),mode='full',root=str(ROOT),output=str(output/'training'),
        inspection=cfg['inspection'],qualification_path=str(root/'epoch'),
        proof_path=str(root/'source/proof'),cold_path=str(root/'source/proof-cold'))


def predecessor(cfg,output):
    ready(cfg['wait']);root=Path(cfg['wait']['path']);pin=sha(root/'report.json');r=read(root/'report.json',pin)
    need(r['complete'] and r['mode']=='run' and r['authority']==cfg['recovery_authority']['pin']
         and r['driver_pin']==cfg['code']['audit/legal_answer_recovery_queue.py']
         and r['capacity_qualified'] and not r['full_population_trained'],'ANSWER_FULL_PREDECESSOR_SCOPE')
    outputs=stages(r['outputs'],('tile','epoch','source'))
    need(all(Path(v['path'])==root/n for n,v in outputs.items()),'ANSWER_FULL_PREDECESSOR_PATHS')
    source=outputs['source'];s=read(Path(source['path'])/'report.json',source['pin'])
    need(s['complete'] and s['capacity_qualified'] and not s['full_population_trained']
         and s['predecessor_pin']==outputs['epoch']['pin'],'ANSWER_FULL_CAPACITY_CHAIN')
    pair=stages(s['outputs'],('proof','proof-cold'));plan=planned_arguments(cfg,output)
    need(pair['proof']['path']==plan['proof_path'] and pair['proof-cold']['path']==plan['cold_path'],
         'ANSWER_FULL_CAPACITY_PATHS')
    # Imports/loaders run only after the existing CUDA owner has terminated.
    from audit.legal_answer_source_training import qualification,capacity
    qualification(Path(plan['qualification_path']),outputs['epoch']['pin'],cfg['code'])
    a=read(Path(pair['proof']['path'])/'report.json',pair['proof']['pin'])
    inspection=read(Path(cfg['inspection']['path'])/'report.json',cfg['inspection']['pin'])
    need(a['source']==inspection['source'],'ANSWER_FULL_CAPACITY_SOURCE')
    capacity(Path(pair['proof']['path']),pair['proof']['pin'],Path(pair['proof-cold']['path']),
             pair['proof-cold']['pin'],a['source'],a['binding'])
    args=['--mode','full','--root',str(ROOT),'--output',plan['output']]
    for name,ref in (('inspection',cfg['inspection']),('qualification',outputs['epoch']),
                     ('proof',pair['proof']),('cold',pair['proof-cold'])):
        args+=['--'+name,ref['path'],'--'+name+'-pin',ref['pin']]
    return dict(parent=pin,capacity=source['pin'],arguments=args,source=a['source'],binding=a['binding'])


def completed(path,cfg,handoff):
    pin=sha(path/'report.json');r=read(path/'report.json',pin)
    need(r['complete'] and r['mode']=='full' and r['gpu_used'] and r['full_population_trained']
         and r['optimizer_updates']==3*91540 and r['source']==handoff['source']
         and r['binding']==handoff['binding'] and set(r['results'])==set(ARCHES),'ANSWER_FULL_RESULT_SCOPE')
    from model.native_population_training import fingerprint
    import torch
    last={}
    for arch in ARCHES:
        v=r['results'][arch]['full'];result=v['result'];need(result['complete'] and result['cursor']==91540
            and result['counts']==cfg['counts'][arch],'ANSWER_FULL_RESULT_COUNTS')
        run=path/arch/'full/run';commits=result['commits']
        need(len(commits)==(91540+255)//256 and commits[-1]['stop']==91540,'ANSWER_FULL_COMMIT_COVERAGE')
        for index,ref in enumerate(commits,1):
            need(ref['path']==f'commit-{index:06d}.json','ANSWER_FULL_COMMIT_ORDER')
            c=read(run/ref['path'],ref['pin']);need(c['stop']==ref['stop'],'ANSWER_FULL_COMMIT_STOP')
        ref=c['checkpoint'];p=run/ref['path'];need(p.resolve().is_relative_to(run.resolve()),'ANSWER_FULL_CHECKPOINT_PATH')
        need(sha(p)==ref['pin'],'ANSWER_FULL_CHECKPOINT_PIN');state=torch.load(p,map_location='cpu',weights_only=True)
        need(fingerprint(state)==ref['state']==v['final_state'] and state['cursor']==91540
             and state['counts']==cfg['counts'][arch] and state['chain']==result['chain']
             and fingerprint(state['model'])==result['model'] and fingerprint(state['optimizer'])==result['optimizer'],
             'ANSWER_FULL_CHECKPOINT_STATE')
        last[arch]=dict(path=str(p),pin=ref['pin'],state=ref['state'],counts=state['counts']);del state
    return dict(path=str(path),pin=pin,checkpoints=last)

"""Immutable authority and terminal handoff for complete-answer evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from audit import legal_answer_full_authority as training
from audit.surface_device_queue import check_owner,identity,need,sha,stamp,write

ROOT=training.ROOT;REPO=training.REPO
FULL=ROOT/'legal-answer-full-queue-v1-20261024'
FULL_AUTH=dict(path=str(ROOT/'legal-answer-full-authority-v2-20261024/authority.json'),
              pin='837ee0afb5c602de7c8636b9b9c0deadb2b04344d054d260c9c3af8680344328')
PROOFS=(('complete-answer-device-proof-v1-20261024','320a29e0512f48271dbb1ac29b13feb752e25c8d2098371aaad4e927e352e3f0'),
        ('complete-answer-device-cold-v1-20261024','72690a2086ef00025885221cf1145ac8390cc7dbead124661c6fb91976274fd8'))
READBACK=('phase141-readback.json','5acc365e418e6e63d6555d7585a076ebd697e527594b21a919ab304fc8d5291e')
SCORING=(('legal-answer-cohort-scores-proof-v2-20261024','ec1b30d9de654550eea33e07c6ea178341afd020c07bb3b8d70883bcbb7f2995'),
         ('legal-answer-cohort-scores-cold-v2-20261024','59a5916e6fcbb4f8d5b2f4ee0cd56a0ecca04ec220a2915b4a34774e90ea65d3'))
NEW=('audit/legal_answer_evaluation_handoff.py','audit/legal_answer_evaluation_pipeline.py',
     'audit/legal_answer_evaluation_queue.py')
DOCUMENTS=(19295,20107,22924,30520)
PLAN=dict(architectures=list(training.ARCHES),cases=23513,trajectories=564312,qualification_cases=4,
          qualification_trajectories=192,scoring_scope='all',reference_atoms=1595325,
          memory_policy='trained_runtime_policy',automatic_retry=False,best_of_proposals=False)


def base():
    cfg,inspection,fresh=training.authenticate(FULL_AUTH['path'],FULL_AUTH['pin'])
    reports=[training.read(ROOT/d/'report.json',h) for d,h in PROOFS]
    a,b=reports;back=training.read(ROOT/READBACK[0],READBACK[1])
    need(a['complete'] and b['complete'] and b['previous_pin']==PROOFS[0][1] and b['cold_reproduced']
         and a['events']==b['events']==1704 and a['trajectories']==b['trajectories']==96
         and a['code']==b['code'] and a['controls']==b['controls'],'ANSWER_EVALUATION_DEVICE_PROOF')
    need(back['complete'] and back['producer']==PROOFS[0][1] and back['cold']==PROOFS[1][1]
         and back['source_events_exact'] and back['tensor_bytes_exact'] and back['negative_controls']==9,
         'ANSWER_EVALUATION_DEVICE_READBACK')
    scores=[training.read(ROOT/d/'report.json',h) for d,h in SCORING]
    need(all(r['complete'] and r['all_metrics_exact'] and r['summary_exact'] for r in scores)
         and scores[1]['previous_pin']==SCORING[0][1] and scores[0]['controls']==scores[1]['controls'],
         'ANSWER_EVALUATION_SCORE_PROOF')
    code=dict(cfg['code'])
    for r in (*reports,*scores):
        for n,h in r['code'].items():
            need(n not in code or code[n]==h,'ANSWER_EVALUATION_CODE_CONFLICT');code[n]=h
    code.update({n:sha(REPO/'scripts'/n) for n in NEW})
    template_ref=dict(path=str(ROOT/PROOFS[0][0]/'request.json'),pin=a['files']['request.json'])
    template=training.read(template_ref['path'],template_ref['pin'])
    files={ROOT/d/'report.json':h for d,h in (*PROOFS,*SCORING)}
    files[ROOT/READBACK[0]]=READBACK[1];files[Path(template_ref['path'])]=template_ref['pin']
    for (d,_),r in zip(PROOFS,reports):files.update({ROOT/d/n:h for n,h in r['files'].items()})
    for p,h in files.items():need(sha(p)==h,'ANSWER_EVALUATION_PROOF_BYTES')
    snapshots={p:stamp(p) for p in files}
    def check():
        fresh()
        for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'ANSWER_EVALUATION_CODE_CHANGED')
        for p,s in snapshots.items():need(stamp(p)==s,'ANSWER_EVALUATION_PROOF_CHANGED')
    check();return cfg,inspection,code,template_ref,template,check


def prepare(output):
    _,inspection,code,template_ref,_,fresh=base();wait=training.owner(FULL);check_owner(wait)
    launch=training.read(FULL/'launch.json',wait['launch_pin'])
    need(launch['authority']==FULL_AUTH['pin'],'ANSWER_EVALUATION_TRAINING_AUTHORITY')
    write(output/'authority.json',dict(schema='complete-answer-evaluation-queue-v1',wait=wait,
        training_authority=FULL_AUTH,code=code,template=template_ref,plan=PLAN,counts=inspection['source']['counts']))
    fresh()


def authenticate(path,pin):
    cfg=training.read(path,pin);parent,inspection,code,ref,template,fresh=base()
    need(set(cfg)=={'schema','wait','training_authority','code','template','plan','counts'}
         and cfg['schema']=='complete-answer-evaluation-queue-v1' and cfg['training_authority']==FULL_AUTH
         and cfg['template']==ref,'ANSWER_EVALUATION_AUTHORITY')
    need(cfg['plan']==PLAN and cfg['counts']==inspection['source']['counts'],'ANSWER_EVALUATION_SCOPE')
    need(cfg['code']==code,'ANSWER_EVALUATION_CODE_BINDING');check_owner(cfg['wait'])
    need(Path(cfg['wait']['path'])==FULL,'ANSWER_EVALUATION_OWNER_PATH')
    launch=training.read(FULL/'launch.json',cfg['wait']['launch_pin'])
    need(launch['authority']==FULL_AUTH['pin'],'ANSWER_EVALUATION_TRAINING_AUTHORITY')
    original=stamp(Path(path))
    def check():
        fresh();need(stamp(Path(path))==original,'ANSWER_EVALUATION_AUTHORITY_CHANGED');check_owner(cfg['wait'])
    check();return cfg,parent,template,check


def terminal(cfg,parent):
    training.ready(cfg['wait']);root=Path(cfg['wait']['path']);pin=sha(root/'report.json')
    r=training.read(root/'report.json',pin)
    need(r['complete'] and r['mode']=='run' and r['authority']==FULL_AUTH['pin']
         and r['driver_pin']==cfg['code']['audit/legal_answer_full_queue.py'] and r['full_population_trained']
         and r['optimizer_updates']==3*91540,'ANSWER_EVALUATION_TERMINAL_SCOPE')
    handoff=json.loads((root/'handoff.json').read_bytes());actual=r['training'];path=root/'training'
    need(actual['path']==str(path),'ANSWER_EVALUATION_TRAINING_PATH')
    t=training.read(path/'report.json',actual['pin'])
    inspection=training.read(Path(parent['inspection']['path'])/'report.json',parent['inspection']['pin'])
    need(t['complete'] and t['mode']=='full' and t['gpu_used'] and t['full_population_trained']
         and t['optimizer_updates']==3*91540 and t['source']==handoff['source']==inspection['source']
         and t['binding']==handoff['binding'] and set(t['results'])==set(training.ARCHES),
         'ANSWER_EVALUATION_TRAINING_SCOPE')
    for n,h in t['files'].items():need(sha(path/n)==h,'ANSWER_EVALUATION_TRAINING_BYTES')
    finals={}
    for arch in training.ARCHES:
        v=t['results'][arch]['full'];result=v['result'];run=path/arch/'full/run'
        need(result['complete'] and result['cursor']==91540 and result['counts']==cfg['counts'][arch]
             and len(result['commits'])==(91540+255)//256,'ANSWER_EVALUATION_COMMIT_COVERAGE')
        for i,ref in enumerate(result['commits'],1):
            need(ref['path']==f'commit-{i:06d}.json' and ref['stop']==min(i*256,91540),
                 'ANSWER_EVALUATION_COMMIT_ORDER')
            c=training.read(run/ref['path'],ref['pin']);need(c['stop']==ref['stop'],'ANSWER_EVALUATION_COMMIT_STOP')
        cp=c['checkpoint'];p=run/cp['path']
        need(p.resolve().is_relative_to(run.resolve()) and cp['state']==v['final_state'],
             'ANSWER_EVALUATION_CHECKPOINT_PATH')
        expected=dict(path=str(p),pin=cp['pin'],state=cp['state'],counts=result['counts'])
        need(actual['checkpoints'][arch]==expected,'ANSWER_EVALUATION_TERMINAL_CHECKPOINTS')
        finals[arch]=dict(model=result['model'],optimizer=result['optimizer'],chain=result['chain'],counts=result['counts'])
    return dict(queue=pin,training=actual,finals=finals)


def exported(cfg,template,refs,training_ref,*,device,full,finals=None):
    """Read real checkpoint payloads once; never rewrite their device or metadata."""
    import torch
    from audit.complete_answer_device_cohort import validate
    from audit.complete_answer_evaluation import artifact
    from model.complete_answer_evaluation import METADATA
    from model.native_population_training import fingerprint
    need(set(refs)==set(training.ARCHES),'ANSWER_EVALUATION_EXPORT_ARCHITECTURES')
    need(not full or finals is not None and set(finals)==set(training.ARCHES),'ANSWER_EVALUATION_FINAL_STATES')
    m=deepcopy(template);m['code']=cfg['code'];m['checkpoints']={}
    original_source=template['checkpoints']['shared']['metadata']['binding']['source']
    for name,ref in refs.items():
        narrow={k:ref[k] for k in ('path','pin','state')};state=artifact(narrow)
        need(set(state)=={*METADATA,'model','optimizer'} and state['binding']['source']==original_source,
             'ANSWER_EVALUATION_EXPORT_SOURCE')
        if full:
            final=finals[name]
            need(fingerprint(state['model'])==final['model'] and fingerprint(state['optimizer'])==final['optimizer']
                 and state['chain']==final['chain'] and state['counts']==final['counts'],
                 'ANSWER_EVALUATION_FINAL_CHECKPOINT')
        m['checkpoints'][name]=dict(reference=narrow,metadata={k:state[k] for k in METADATA});del state
    m['intervention']=dict(kind='trained_runtime_policy') if device=='cuda:0' else dict(kind='CPU_memory_pressure_mechanics',cache_bytes=65536)
    m['ordinals']=None if full else template['ordinals']
    need(full==(device=='cuda:0'),'ANSWER_EVALUATION_EXPORT_MODE')
    m['authority']={k:template['authority'][k] for k in ('source','evaluation_plan','cold_plan','public_pack')}
    m['authority'].update(training=training_ref,code=fingerprint(cfg['code']),
        scope='complete_document_continuation_development_exposed' if full else 'CPU_pipeline_mechanics_exposed_cases',
        semantic_validity_established=False,capacity_matched=False)
    runtime=artifact(m['runtime']);validate(m,runtime,device)
    need(not torch.cuda.is_initialized(),'ANSWER_EVALUATION_EXPORT_CUDA_INITIALIZED')
    return m


def planned(output):
    return dict(pipeline=str(REPO/'scripts/audit/legal_answer_evaluation_pipeline.py'),device='cuda:0',
        stages=['export','inspect','qualification','qualification-cold','qualification-readback','full','score'],
        output=str(output/'evaluation'),full_cases=23513,full_trajectories=564312,
        original_memory_policy=True,private_references_after_generation=True)

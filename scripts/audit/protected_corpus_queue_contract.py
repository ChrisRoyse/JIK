"""Pure-metadata authority for the protected decoder's serial GPU program."""
import fcntl
import json
import os
from pathlib import Path

from audit.surface_device_queue import check_owner,identity,need,sha,stamp,write

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
REPO=Path(__file__).resolve().parents[2]
WAIT_ROOT=ROOT/'legal-answer-successor-v1-20261024'
PARENT=dict(path=str(ROOT/'answer-successor-authority-v6-20261024/authority.json'),
            pin='e3a391c71c7a007d345a22a2c5ceb3052b8dec9df45a6bfdf6ab049d3a43400b')
RECEIPTS={
 'plan':('protected-corpus-plan-v1-20261025','67f40647604d678b15993bc2070f1993c4e72083db6914af91d77bb66fe1519f'),
 'cpu':('protected-corpus-cpu-v3-20261025','cdb60b0c720265dcfdb914cd8d548d9027fae631c372c2ac21434b661103d5a8'),
 'cold':('protected-corpus-cold-v3-20261025','30e3f88d76d565f6a40ee03311201dabc7b17391eb0a9476340819251b214d50'),
 'selection':('protected-corpus-document-census-v1-20261025','edc1114463134e19f8ca92c92571cd82afcc30ea2553b355e794845c7deb3bb3'),
 'documents':('protected-corpus-document-producer-v1-20261025','83bba15a64139623f8b1ca3eb68176d83011a28846c8e88773f98354d70a9194'),
 'readback':('protected-corpus-document-readback-v1-20261025','ec2c98293e8c91e0e7d076ba5a0c67dddebea4fd7338594ce2f4c255479fee37')}
NEW=('audit/protected_corpus_queue_contract.py','audit/protected_corpus_queue.py','audit/protected_corpus_queue_proof.py')
STAGES=('gpu-proof','gpu-cold','gpu-documents','gpu-document-readback','full','full-readback')
PREDECESSOR_STAGES=('prior-readback','placement','tile','epoch','source','capacity','training','training-readback','evaluation')
TRAIN=dict(atoms=6547176,bytes=669898601,prefix_rows=45357686)
LOCK=REPO/'.lease/protected-corpus-queue'


def persist(path,value):
    write(path,value);fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return sha(path)


def read(path,pin):
    path=Path(path);need(path.is_file() and sha(path)==pin,'PROTECTED_QUEUE_PIN')
    return json.loads(path.read_bytes())


def process_current(owner):
    value=identity(owner['pid'])
    need(value in (owner['starttime'],None),'PROTECTED_QUEUE_PID_REUSED');return value


def lock():
    f=LOCK.open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();raise ValueError('PROTECTED_QUEUE_BUSY')
    return f


def proof_scope(reports):
    p,r,c,s,d,b=(reports[n] for n in RECEIPTS)
    need(all(v['complete'] for v in reports.values()),'PROTECTED_QUEUE_INCOMPLETE_PROOF')
    need(p['train']==TRAIN and p['updates']==51150 and p['parameters']==78325042,'PROTECTED_QUEUE_POPULATION')
    need(not r['full_training'] and r['device']=='cpu' and r['steps']==3 and r['cursor']==384
         and r['plan']==RECEIPTS['plan'][1],'PROTECTED_QUEUE_CPU_SCOPE')
    need(c['input_pin']==RECEIPTS['cpu'][1] and c['updates']==3 and c['train_atoms']==384
         and all(c[k] for k in ('all_update_state_scores_exact','midpoint_resume_exact','initial_parent_exact','all_evaluation_exact')),
         'PROTECTED_QUEUE_COLD_PROOF')
    need(c['runtime']==r['runtime']==d['runtime']==b['runtime'] and r['runtime']['device']=='cpu'
         and c['implementation']==r['implementation'] and d['implementation']==b['implementation'],
         'PROTECTED_QUEUE_PROOF_RUNTIME')
    need(s['no_forward'] and s['plan']==RECEIPTS['plan'][1] and s['source_census']==TRAIN
         and [(side,v['document'],v['atoms'],v['bytes'],v['positions']) for side,v in s['selected'].items()]
         ==[('development',49017,2,109,111),('query',50318,2,113,115)],'PROTECTED_QUEUE_DOCUMENT_COHORT')
    need(d['mode']=='produce' and b['mode']=='read' and d['input_pin']==b['input_pin']==RECEIPTS['cpu'][1]
         and d['selection']==b['selection']==RECEIPTS['selection'][1] and b['all_final_document_evaluation_bytes_exact']
         and not b['full_training'] and not b['full_gradient_replay'],'PROTECTED_QUEUE_DOCUMENT_PROOF')
    need(b['ledger']['rows']==3 and b['ledger']['source_occurrences']==b['ledger']['native_occurrences_replayed']==384
         and sum(x['positions'] for x in b['evaluation']['totals'].values())==226
         and len(b['controls'])==10,'PROTECTED_QUEUE_READBACK_SCOPE')


def build():
    refs={n:dict(path=str(ROOT/d),pin=h) for n,(d,h) in RECEIPTS.items()}
    reports={n:read(Path(ref['path'])/'report.json',ref['pin']) for n,ref in refs.items()};proof_scope(reports)
    parent=read(PARENT['path'],PARENT['pin']);launch=json.loads((WAIT_ROOT/'launch.json').read_bytes())
    need(launch['authority']==PARENT['pin'] and launch['driver_pin']==parent['code']['audit/legal_answer_successor.py'],
         'PROTECTED_QUEUE_PARENT_LAUNCH')
    owner=dict(path=str(WAIT_ROOT),pid=launch['pid'],starttime=str(launch['starttime']),launch_pin=sha(WAIT_ROOT/'launch.json'))
    need((owner['pid'],owner['starttime'])==(3511919,'410085379'),'PROTECTED_QUEUE_PARENT_IDENTITY')
    check_owner(owner)
    code=dict(parent['code'])
    for n,h in reports['readback']['implementation'].items():
        need(n not in code or code[n]==h,'PROTECTED_QUEUE_CODE_CONFLICT');code[n]=h
    for n in NEW:code[n]=sha(REPO/'scripts'/n)
    return dict(schema='protected-corpus-queue-v1',proofs=refs,wait=owner,parent=PARENT,code=code,
                worker_code=reports['cpu']['implementation'],reader_code=reports['readback']['implementation'],
                stages=list(STAGES),predecessor_stages=list(PREDECESSOR_STAGES),train=TRAIN,updates=51150,
                execution=dict(device='cuda:0',serial=True,automatic_retry=False,subset=None,epochs=1))


def authenticate(path,pin):
    cfg=read(path,pin);expected=build();need(cfg==expected,'PROTECTED_QUEUE_AUTHORITY_BINDING')
    files={Path(path):pin,Path(PARENT['path']):PARENT['pin'],WAIT_ROOT/'launch.json':cfg['wait']['launch_pin']}
    reports={}
    for label,ref in cfg['proofs'].items():
        p=Path(ref['path'])/'report.json';files[p]=ref['pin'];reports[label]=read(p,ref['pin'])
    for label in ('plan','cpu'):
        root=Path(cfg['proofs'][label]['path'])
        for n,ref in reports[label]['files'].items():files[root/n]=ref['sha256']
    p=reports['plan'];files[Path(p['train_index']['path'])]=p['train_index']['pin']
    for ref in reports['cpu']['checkpoints']:
        files[Path(ref['root'])/'manifest.json']=ref['pin'];files[Path(ref['origin']['path'])]=ref['origin']['pin']
    for ref in reports['documents']['evaluation']['refs']:files[Path(ref['path'])]=ref['pin']
    for p,h in files.items():need(sha(p)==h,'PROTECTED_QUEUE_PHYSICAL_INPUT')
    snapshots={p:stamp(p) for p in files}
    def fresh():
        for p,s in snapshots.items():need(stamp(p)==s,'PROTECTED_QUEUE_INPUT_CHANGED')
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h,'PROTECTED_QUEUE_CODE_CHANGED')
        check_owner(cfg['wait'])
    fresh();return cfg,reports,fresh


def stage(ref,label,root):
    need(isinstance(ref,dict) and ref.get('label')==label and ref.get('report')=='report.json'
         and Path(ref.get('path','')).resolve()==(Path(root)/label).resolve(),'PROTECTED_QUEUE_STAGE_PATH')
    value=read(Path(ref['path'])/'report.json',ref['pin'])
    need(value.get('complete') is True,'PROTECTED_QUEUE_STAGE_FAILED');return value


def completion(value,cfg):
    launch=read(WAIT_ROOT/'launch.json',cfg['wait']['launch_pin'])
    need(value.get('complete') is True and value.get('authority')==PARENT['pin']
         and value.get('driver_pin')==launch['driver_pin'] and value.get('full_population_trained') is True
         and value.get('full_population_evaluated') is True,'PROTECTED_QUEUE_PREDECESSOR_INCOMPLETE')
    need(isinstance(value.get('outputs'),dict) and set(value['outputs'])==set(PREDECESSOR_STAGES),
         'PROTECTED_QUEUE_PREDECESSOR_STAGES')
    values={}
    for label in PREDECESSOR_STAGES:
        ref=value['outputs'][label];values[label]=stage(ref,label,WAIT_ROOT)
        path=WAIT_ROOT/f'completed-{label}.json'
        need(path.is_file() and json.loads(path.read_bytes())==ref,'PROTECTED_QUEUE_PREDECESSOR_RECEIPT')
    train=values['training'];back=values['training-readback'];evaluation=values['evaluation']
    need(train['full_population_trained'] and train['gpu_used'] and train['mode']=='full'
         and train['optimizer_updates']==back['optimizer_updates']==3*91540 and back['full_population_trained']
         and back['training']['pin']==value['outputs']['training']['pin'],'PROTECTED_QUEUE_PREDECESSOR_TRAINING')
    need(evaluation['full_population_trained'] and evaluation['full_population_evaluated']
         and evaluation['scored']['cases']==23513,'PROTECTED_QUEUE_PREDECESSOR_EVALUATION')
    return {n:value['outputs'][n]['pin'] for n in PREDECESSOR_STAGES}


def predecessor(cfg):
    check_owner(cfg['wait']);need(process_current(cfg['wait']) is None,'PROTECTED_QUEUE_LIVE_PREDECESSOR')
    path=WAIT_ROOT/'report.json';need(path.is_file(),'PROTECTED_QUEUE_PREDECESSOR_MISSING')
    pin=sha(path);value=read(path,pin);stages=completion(value,cfg)
    return dict(path=str(WAIT_ROOT),pin=pin,stages=stages)


def arguments(label,cfg,outputs):
    plan=cfg['proofs']['plan'];common=['--root',str(ROOT),'--plan',plan['path'],'--plan-pin',plan['pin']]
    def reference(name,ref):return ['--'+name,ref['path'],'--'+name+'-pin',ref['pin']]
    if label in ('gpu-proof','gpu-cold','full'):
        mode={'gpu-proof':'proof','gpu-cold':'cold','full':'full'}[label];args=['--mode',mode,*common]
        if label=='gpu-cold':args+=reference('input',outputs['gpu-proof'])
        if label=='full':args+=reference('qualification',outputs['gpu-cold'])
        return 'protected_corpus_program.py',args
    args=['--mode','produce' if label=='gpu-documents' else 'read',*common]
    args+=reference('input',outputs['full' if label=='full-readback' else 'gpu-proof'])
    if label!='full-readback':args+=reference('selection',cfg['proofs']['selection'])
    if label=='gpu-document-readback':args+=reference('evaluation',outputs['gpu-documents'])
    return 'protected_corpus_readback.py',args


def child_scope(label,r,cfg,outputs):
    need(r.get('complete') is True and r.get('device')=='cuda:0' and r.get('plan')==cfg['proofs']['plan']['pin'],
         'PROTECTED_QUEUE_CHILD_SCOPE')
    code=cfg['worker_code'] if label in ('gpu-proof','gpu-cold','full') else cfg['reader_code']
    need(r['implementation']==code and r['runtime']['device']=='cuda:0','PROTECTED_QUEUE_CHILD_IMPLEMENTATION')
    if label=='gpu-proof':
        need(r['steps']==3 and r['cursor']==384 and not r['full_training'],'PROTECTED_QUEUE_GPU_PROOF')
    elif label=='gpu-cold':
        need(r['input_pin']==outputs['gpu-proof']['pin'] and r['updates']==3 and r['train_atoms']==384
             and all(r[k] for k in ('all_update_state_scores_exact','midpoint_resume_exact','initial_parent_exact','all_evaluation_exact')),
             'PROTECTED_QUEUE_GPU_COLD')
    elif label=='full':
        need(r['full_training'] and r['steps']==51150 and r['cursor']==TRAIN['atoms'] and r['resume'] is None
             and len(r['full_evaluation'])==200,'PROTECTED_QUEUE_FULL_TRAINING')
    else:
        expected=outputs['full' if label=='full-readback' else 'gpu-proof']
        need(r['input_pin']==expected['pin'],'PROTECTED_QUEUE_CHILD_INPUT')
        if label=='gpu-documents':
            need(r['mode']=='produce' and r['selection']==cfg['proofs']['selection']['pin']
                 and sum(v['positions'] for v in r['evaluation']['refs'])==226,'PROTECTED_QUEUE_GPU_DOCUMENTS')
        else:
            need(r['mode']=='read' and r['all_final_document_evaluation_bytes_exact']
                 and r['full_training']==(label=='full-readback'),'PROTECTED_QUEUE_GPU_READBACK')
            totals=r['evaluation']['totals']
            need(set(totals)=={'development','query'} and all(v['documents']==(100 if label=='full-readback' else 1)
                 for v in totals.values()),'PROTECTED_QUEUE_READBACK_DOCUMENTS')
            need(r['ledger']['source_occurrences']==(TRAIN['atoms'] if label=='full-readback' else 384)
                 and r['ledger']['rows']==(51150 if label=='full-readback' else 3),'PROTECTED_QUEUE_READBACK_COVERAGE')
    if label!='gpu-proof':
        proof=read(Path(outputs['gpu-proof']['path'])/'report.json',outputs['gpu-proof']['pin'])
        need(r['runtime']==proof['runtime'],'PROTECTED_QUEUE_GPU_RUNTIME')

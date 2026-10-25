"""Case-bound reconciliation of the original inversion-reader refusal."""
import json
from pathlib import Path

from audit import complete_answer_recovery as original
from audit import complete_answer_schedule as schedule
from audit import legal_tile_gpu_queue as tile
from audit import legal_answer_epoch_queue as epoch
from audit import legal_answer_source_queue as source
from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29');REPO=Path(__file__).resolve().parents[2]
MODULES={'tile':tile,'epoch':epoch,'source':source}
QUEUES={
 'tile':('legal-tile-gpu-authority-v1-20261024','760197c7dddeed287f3487614ca1b6ab7867ecc223cc97424e70d7469e781940',
         'legal-tile-gpu-queue-v1-20261024','legal_tile_gpu_queue.py'),
 'epoch':('legal-answer-epoch-cuda-authority-v1-20261024','efe7d2d6ab4da098d30c5334500a2dacfb3de8173ec163ec5211d06302d200aa',
          'legal-answer-epoch-cuda-queue-v1-20261024','legal_answer_epoch_queue.py'),
 'source':('legal-answer-source-authority-v1-20261024','f9e00eec7d62619d79a4f1919cf501c50b2cb5e453592828fc6dba75ecac46db',
           'legal-answer-source-cuda-queue-v1-20261024','legal_answer_source_queue.py')}
RECEIPTS={
 'full':('phase115-complete-answer-recovery-v1-20261024/inversion-full','1fe1850ea1cedfe834259073f8ba71b56b969808499492bc7aaf83a98dc2b125'),
 'cold':('inversion-canonical-repair-v2-20261024/readback','2680402106463fbfe867f307946cb85f0e94400a129706f6c89ada471f7b1b60'),
 'effect':('legal-inversion-effect-v2-20261024','05c6dddbe3bdb3998f42ceaecad36d66e17299c6b2ac337d9bdd1bbd9091dcc6'),
 'readback':('legal-inversion-final-readback-v1-20261024','02e63b0d17f059a83dd43541a8590d9842a627c825ca12e2a33a252c53018582')}
NEW=('audit/legal_answer_recovery_authority.py','audit/legal_answer_recovery_queue.py')


def read(path,pin):
    need(sha(path)==pin,'ANSWER_RECOVERY_PIN');return json.loads(Path(path).read_bytes())


def owner(path):
    launch=json.loads((path/'launch.json').read_bytes())
    return dict(path=str(path),pid=launch['pid'],starttime=str(launch['starttime']),launch_pin=sha(path/'launch.json'))


def ready(ref):
    check_owner(ref);need(identity(ref['pid']) is None,'ANSWER_RECOVERY_PREDECESSOR_LIVE')


def correction_scope(records):
    f,c,e,b=(records[n] for n in ('full','cold','effect','readback'))
    need(f['complete'] and f['full_training'] and f['steps']==51150 and f['cursor']==6547176
         and f['binding']['device']=='cuda:0','ANSWER_RECOVERY_ORIGINAL_TRAINING')
    need(c['complete'] and c['input_pin']==RECEIPTS['full'][1] and c['plan']==f['plan']
         and c['device']=='cuda:0' and c['every_evaluation_byte_exact'] and c['full_ledger_coverage']
         and c['comparison_tolerance']==0. and c['cold_optimizer_updates']==0
         and not c['all_update_replay_exact'],'ANSWER_RECOVERY_CORRECTION_SCOPE')
    need(set(c['controls'])=={'held_fit','changed_surface','missing_instrument','missing_byte_tail',
                             'changed_denominator','changed_sum','changed_byte_score'},'ANSWER_RECOVERY_CONTROLS')
    need(e['complete'] and e['input_pin']==RECEIPTS['full'][1] and e['readback_pin']==RECEIPTS['cold'][1]
         and e['steps']==51150 and e['checkpoints']==200 and e['evaluation_atom_records']==27700
         and e['paired_byte_positions']==6151148 and not e['full_optimizer_updates_replayed'],
         'ANSWER_RECOVERY_EFFECT_SCOPE')
    need(b['complete'] and b['input_pin']==RECEIPTS['full'][1] and b['cold_pin']==RECEIPTS['cold'][1]
         and b['effect_pin']==RECEIPTS['effect'][1] and b['repair_parent_resumed']
         and b['original_failure_preserved'] and b['source_strata_and_denominators_recounted']
         and not b['training_restarted'] and not b['full_update_cold_replay'],'ANSWER_RECOVERY_FINAL_READBACK')


def labels(values):
    expected=set(schedule.ORDER)-{'inversion-cold'}
    need(len(values)==len(set(values))==len(expected) and set(values)==expected,'ANSWER_RECOVERY_MISSING_OR_EXTRA_WORK')


def errors(values,expected):
    need(values=={'inversion-cold':expected},'ANSWER_RECOVERY_ADDITIONAL_FAILURE')


def waiter_failure(name,value,predecessor):
    if name=='tile':
        expected=dict(error='ValueError',detail='TILE_GPU_PREDECESSOR_FAILED')
    else:
        expected=dict(error='FileNotFoundError',detail=f"[Errno 2] No such file or directory: '{Path(predecessor)/'report.json'}'")
    need(value==expected,'ANSWER_RECOVERY_WAITER_FAILURE')


def payload(root,name,entry):
    path=root/name
    need(path.resolve().is_relative_to(root.resolve()),'ANSWER_RECOVERY_FILE_SCOPE')
    if isinstance(entry,dict):
        need(set(entry)=={'bytes','sha256'},'ANSWER_RECOVERY_FILE_SCHEMA')
        need(path.stat().st_size==entry['bytes'],'ANSWER_RECOVERY_OUTPUT_SIZE');pin=entry['sha256']
    else:
        need(isinstance(entry,str),'ANSWER_RECOVERY_FILE_SCHEMA');pin=entry
    need(sha(path)==pin,'ANSWER_RECOVERY_OUTPUT_BYTES');return pin


def prepare(output):
    queues={};code={};parent=None
    for name,(directory,pin,run,script) in QUEUES.items():
        # Capture immutable bindings here; authenticate their payloads once below.
        path=ROOT/directory/'authority.json';cfg=read(path,pin)
        queues[name]=dict(authority=dict(path=str(path),pin=pin),owner=owner(ROOT/run),script=script)
        if name=='tile':parent=cfg['wait']
        for n,h in cfg['code'].items():
            need(n not in code or code[n]==h,'ANSWER_RECOVERY_CONFLICTING_CODE');code[n]=h
    for n in NEW:code[n]=sha(REPO/'scripts'/n)
    p=Path(parent['path']);launch=json.loads((p/'launch.json').read_bytes())
    cfg=dict(schema='answer-reader-recovery-v1',parent=parent,queues=queues,code=code,
        original_authority=dict(path=str(ROOT/'phase115-recovery-authority-v1-20261024/authority.json'),pin=launch['authority']),
        plan_pin=sha(p/'plan.json'),original_failure_pin=sha(p/'inversion-cold/failure.json'),
        error_pin=sha(p/'inversion-cold-failure.json'),exit_pin=sha(p/'inversion-cold-exit.json'),
        correction={n:dict(path=str(ROOT/d),pin=h) for n,(d,h) in RECEIPTS.items()})
    write(output/'authority.json',cfg);return sha(output/'authority.json')


def authenticate(path,pin):
    cfg=read(path,pin);need(cfg['schema']=='answer-reader-recovery-v1','ANSWER_RECOVERY_SCHEMA')
    parent=Path(cfg['parent']['path']);check_owner(cfg['parent'])
    _,sources,old_fresh=original.authenticate(Path(cfg['original_authority']['path']),cfg['original_authority']['pin'])
    plan=read(parent/'plan.json',cfg['plan_pin']);need(plan==schedule.plan(sources),'ANSWER_RECOVERY_PLAN_CHANGED')
    failure=read(parent/'inversion-cold/failure.json',cfg['original_failure_pin'])
    need(failure=={'error':"ValueError('MNLM_TEXT_CORPUS_INVERSION_READBACK_DENOMINATOR')"},'ANSWER_RECOVERY_FAILURE_KIND')
    error=read(parent/'inversion-cold-failure.json',cfg['error_pin'])
    need(error==dict(code='RECOVERY_WORKER_REFUSED',error="ValueError('RECOVERY_WORKER_FAILED:inversion-cold')"),
         'ANSWER_RECOVERY_PARENT_ERROR')
    need(read(parent/'inversion-cold-exit.json',cfg['exit_pin'])['exit_code']==1,'ANSWER_RECOVERY_EXIT')
    files={Path(path):pin,parent/'plan.json':cfg['plan_pin'],parent/'launch.json':cfg['parent']['launch_pin'],
           parent/'inversion-cold/failure.json':cfg['original_failure_pin'],
           parent/'inversion-cold-failure.json':cfg['error_pin'],parent/'inversion-cold-exit.json':cfg['exit_pin']}
    records={};guards=[old_fresh]
    for name,(directory,h) in RECEIPTS.items():
        ref=cfg['correction'][name];need(ref==dict(path=str(ROOT/directory),pin=h),'ANSWER_RECOVERY_CORRECTION_CHANGED')
        records[name]=read(Path(ref['path'])/'report.json',h);files[Path(ref['path'])/'report.json']=h
    correction_scope(records)
    c=records['cold'];reader=REPO/'scripts/audit/legal_surface_inversion_canonical_readback.py'
    need(sha(reader)==c['readback_code'],'ANSWER_RECOVERY_READER_CODE');files[reader]=c['readback_code']
    # The final independent readback lists the exact original and corrected bytes.
    for row in records['readback']['members'].values():files[Path(row['path'])]=row['sha256']
    for name,(directory,h,run,script) in QUEUES.items():
        q=cfg['queues'][name];need(q['authority']==dict(path=str(ROOT/directory/'authority.json'),pin=h)
             and q['owner']==owner(ROOT/run) and q['script']==script,'ANSWER_RECOVERY_QUEUED_WORK_CHANGED')
        _,guard=MODULES[name].authenticate(Path(q['authority']['path']),h);guards.append(guard)
        check_owner(q['owner']);files[Path(q['owner']['path'])/'launch.json']=q['owner']['launch_pin']
    for f,h in files.items():need(sha(f)==h,'ANSWER_RECOVERY_PHYSICAL_BYTES')
    stamps={f:stamp(f) for f in files}
    def fresh():
        for guard in guards:guard()
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h,'ANSWER_RECOVERY_CODE_CHANGED')
        for f,s in stamps.items():need(stamp(f)==s,'ANSWER_RECOVERY_EVIDENCE_CHANGED')
    fresh();return cfg,sources,records,error,fresh


def reconcile(cfg,sources,error):
    for ref in [cfg['parent'],*[q['owner'] for q in cfg['queues'].values()]]:ready(ref)
    parent=Path(cfg['parent']['path']);report=json.loads((parent/'report.json').read_bytes())
    launch=json.loads((parent/'launch.json').read_bytes())
    need(report['complete'] is False and report['authority']==launch['authority']
         and report['driver_pin']==launch['driver_pin'],'ANSWER_RECOVERY_PARENT_TERMINAL')
    errors(report['errors'],error);labels([r['label'] for r in report['outputs']])
    for ref in report['outputs']:
        need(Path(ref['path'])==parent/ref['label'],'ANSWER_RECOVERY_OUTPUT_PATH')
        schedule.output_check(ref['label'],ref,sources)
    checked={}
    def audit(ref):
        root=Path(ref['path']);path=root/'report.json';r=read(path,ref['pin'])
        need(r['complete'],'ANSWER_RECOVERY_CHILD_INCOMPLETE')
        if str(path) in checked:need(checked[str(path)]==ref['pin'],'ANSWER_RECOVERY_CHILD_CONFLICT');return
        checked[str(path)]=ref['pin']
        for n,h in r.get('files',{}).items():checked[str(root/n)]=payload(root,n,h)
        for child in r.get('outputs',[]):audit(child)
    for ref in report['outputs']:audit(ref)
    failures={};previous=parent
    for name in ('tile','epoch','source'):
        q=cfg['queues'][name]
        root=Path(q['owner']['path']);need(not (root/'report.json').exists(),'ANSWER_RECOVERY_ALREADY_EXECUTED')
        f=json.loads((root/'failure.json').read_bytes());events=[json.loads(v) for v in (root/'events.jsonl').read_text().splitlines()]
        need(not any(v['stage']=='running' for v in events),'ANSWER_RECOVERY_PARTIAL_GPU_WORK')
        waiter_failure(name,f,previous)
        failures[name]=dict(path=str(root/'failure.json'),pin=sha(root/'failure.json'),failure=f)
        previous=root
    corrected={r['label']:r for r in report['outputs']}
    corrected['inversion-cold']=dict(label='inversion-cold',**cfg['correction']['cold'])
    return dict(original_report_pin=sha(parent/'report.json'),original_errors=report['errors'],
        outputs=[corrected[n] for n in schedule.ORDER],physical_outputs=checked,waiter_failures=failures,
        original_reports_unchanged=True,full_optimizer_trajectory_cold_replayed=False)

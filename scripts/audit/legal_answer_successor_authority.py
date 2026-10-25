"""Versioned scientific work contract for the complete placement successor roster."""
import json
from pathlib import Path

from audit import legal_answer_evaluation_handoff as evaluation
from audit import legal_placement_handoff_census as census
from audit.surface_device_queue import identity, need, sha, stamp, write
from claims import process_handoff as process

ROOT=evaluation.ROOT;REPO=evaluation.REPO
CENSUS=('placement-handoff-census-v2-20261024','de11e6e7943ee6fdebff7bed9ae24ca2589a778f5eb64c369bfb8996ecf44e4a')
RUNTIME={
 'runtime':dict(path=str(ROOT/'phase115-runtime-recovery-v1-20261024/gpu'),
                pin='0aaeea252e07d64caedbd427249b3707815b8d98ede2a8cdad09ce9d7c115d3a'),
 'runtime-cold':dict(path=str(ROOT/'phase115-runtime-recovery-v1-20261024/gpu-cold'),
                     pin='a57442c469c7b5582166f9444733b0f5276f45247826780be29c8ca2ce38725f')}
STAGES=('prior-readback','placement','tile','epoch','source','capacity','training','training-readback','evaluation')
PLACEMENT=('original','boundary','resumed','cold','match','source-readback','full','full-readback')
NEW=('audit/legal_answer_successor_authority.py','audit/legal_answer_successor.py',
     'audit/legal_answer_successor_worker.py','audit/legal_answer_successor_evaluation.py',
     'audit/legal_answer_successor_proof.py','audit/legal_answer_successor_readback.py')
EXTRA=('audit/legal_grouped_placement_training.py','audit/legal_grouped_placement_readback.py',
       'audit/legal_placement_resume_training.py','audit/legal_placement_resume_match.py',
       'model/legal_placement_resume.py','model/placement_resume_publication.py',
       'model/placement_resume_tail.py',
       'audit/legal_placement_successor_seal.py','audit/legal_placement_successor_seal_readback.py',
       'audit/legal_answer_retirement.py','audit/legal_answer_retirement_readback.py',
       'claims/process_retirement.py',
       'audit/legal_placement_handoff_census.py','claims/process_handoff.py')
WAITERS={'tile':(3260214,'405094809'),'epoch':(3281519,'405492744'),'source':(3295694,'405818358')}
STOP_ORDER=('evaluation','full','recovery','source_waiter','epoch_waiter','tile_waiter','driver','grouped','producer')


def read(path,pin):
    need(sha(path)==pin,'SUCCESSOR_PIN');return json.loads(Path(path).read_bytes())


def scope(cfg):
    need(cfg['schema']=='complete-answer-successor-v1','SUCCESSOR_SCHEMA')
    need(cfg['stages']==list(STAGES) and cfg['placement_stages']==list(PLACEMENT),'SUCCESSOR_STAGES')
    need(cfg['remaining_work']==list(census.REMAINING),'SUCCESSOR_WORK')
    evaluation.training.counts(cfg['counts'])
    need(cfg['evaluation']==evaluation.PLAN,'SUCCESSOR_EVALUATION')
    need(cfg['execution']==dict(device='cuda',serial=True,automatic_retry=False,subset=None,
                               length_limit=None,best_of_proposals=False),'SUCCESSOR_EXECUTION')
    expected={n:(r[0],r[1]) for n,r in census.ROLES.items()}
    expected.update({n+'_waiter':r for n,r in WAITERS.items()})
    need(set(cfg['owners'])==set(expected) and cfg['stop_order']==list(STOP_ORDER),'SUCCESSOR_OWNER_SCOPE')
    for name,row in cfg['owners'].items():
        pid,start=expected[name]
        need(row['pid']==pid and row['starttime']==start,'SUCCESSOR_OWNER_IDENTITY')


def owners(old,queues):
    values={n:dict(v) for n,v in old['owners'].items()};refs={}
    for name,(pid,start) in WAITERS.items():
        q=queues[name];root=Path(q['owner']['path']);script=q['script']
        launch=root/'launch.json';v=json.loads(launch.read_bytes())
        need(v['pid']==pid and str(v['starttime'])==start,'SUCCESSOR_WAITER_LAUNCH')
        row=process.process(pid)
        if row is not None:
            need(row['starttime']==start,'SUCCESSOR_WAITER_REUSED')
            argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
            need(sum(Path(a).name==script for a in argv)==1 and '--output' in argv
                 and Path(argv[argv.index('--output')+1])==root,'SUCCESSOR_WAITER_COMMAND')
        else:raise ValueError('SUCCESSOR_WAITER_MISSING_AT_PREPARE')
        values[name+'_waiter']=dict(row,script=script,output=str(root))
        for f in ('launch.json','events.jsonl'):
            path=root/f;refs[str(path)]=sha(path)
    ordered=process.capture(list(values.values()),[values[n]['pid'] for n in STOP_ORDER])
    need(all(r['state'] in ('R','S','D') for r in ordered),'SUCCESSOR_ALREADY_STOPPED')
    # Refresh mutable state while retaining all immutable command/child bindings.
    by_pid={r['pid']:r for r in ordered}
    for row in values.values():row.update(by_pid[row['pid']])
    return values,refs


def prepare(output):
    old=read(ROOT/CENSUS[0]/'report.json',CENSUS[1]);census.scope(old)
    parent,inspection,code,template_ref,_,fresh=evaluation.base()
    parent_queues=read(evaluation.training.AUTHORITY['path'],evaluation.training.AUTHORITY['pin'])['queues']
    queues={};roster,waiter_refs=owners(old,parent_queues)
    for label,q in parent_queues.items():
        path=Path(q['authority']['path']);pin=q['authority']['pin'];script=q['script'];value=read(path,pin)
        queues[label]=dict(authority=dict(path=str(path),pin=pin),script=script)
        for name,h in value['code'].items():
            need(name not in code or code[name]==h,'SUCCESSOR_CODE_CONFLICT');code[name]=h
    grouped=Path(old['owners']['grouped']['output'])/'qualification/report.json'
    g=json.loads(grouped.read_bytes())
    need(g['complete'] and g['binding']['device']=='cuda','SUCCESSOR_GROUPED_PROOF')
    for name,h in g['binding']['code'].items():
        need(name not in code or code[name]==h,'SUCCESSOR_CODE_CONFLICT');code[name]=h
    for name in (*EXTRA,*NEW):
        h=sha(REPO/'scripts'/name)
        need(name not in code or code[name]==h,'SUCCESSOR_CODE_CONFLICT');code[name]=h
    value=dict(schema='complete-answer-successor-v1',census=dict(path=str(ROOT/CENSUS[0]),pin=CENSUS[1]),
        stages=list(STAGES),placement_stages=list(PLACEMENT),remaining_work=list(census.REMAINING),
        owners=roster,stop_order=list(STOP_ORDER),waiter_refs=waiter_refs,
        counts=inspection['source']['counts'],inspection=parent['inspection'],
        template=template_ref,evaluation=evaluation.PLAN,queues=queues,runtime=RUNTIME,code=code,
        placement_proof=dict(path=str(grouped),pin=sha(grouped)),
        execution=dict(device='cuda',serial=True,automatic_retry=False,subset=None,length_limit=None,best_of_proposals=False))
    scope(value);fresh();write(output/'authority.json',value);return sha(output/'authority.json')


def authenticate(path,pin):
    cfg=read(path,pin);scope(cfg)
    old=read(Path(cfg['census']['path'])/'report.json',cfg['census']['pin']);census.scope(old)
    need(cfg['census']==dict(path=str(ROOT/CENSUS[0]),pin=CENSUS[1]),'SUCCESSOR_CENSUS')
    for name,row in old['owners'].items():
        need(process.binding(cfg['owners'][name])==process.binding(row),'SUCCESSOR_CENSUS_OWNER')
    parent,inspection,code,ref,template,base_fresh=evaluation.base()
    parent_queues=read(evaluation.training.AUTHORITY['path'],evaluation.training.AUTHORITY['pin'])['queues']
    need(cfg['counts']==inspection['source']['counts'] and cfg['inspection']==parent['inspection']
         and cfg['template']==ref and cfg['runtime']==RUNTIME,'SUCCESSOR_SOURCE')
    files={Path(path):pin,Path(cfg['census']['path'])/'report.json':cfg['census']['pin']}
    files.update({Path(p):v['pin'] for p,v in old['sources'].items()})
    expected_waiter_files=set()
    for name,(pid,start) in WAITERS.items():
        q=parent_queues[name];root=Path(q['owner']['path']);script=q['script'];row=cfg['owners'][name+'_waiter']
        need(row['script']==script and row['output']==str(root) and row['children']==[],
             'SUCCESSOR_WAITER_SCOPE')
        for f in ('launch.json','events.jsonl'):
            p=root/f;expected_waiter_files.add(str(p));files[p]=cfg['waiter_refs'][str(p)]
        launch=read(root/'launch.json',files[root/'launch.json'])
        need(launch['pid']==pid and str(launch['starttime'])==start,'SUCCESSOR_WAITER_LAUNCH')
    need(set(cfg['waiter_refs'])==expected_waiter_files,'SUCCESSOR_WAITER_FILES')
    expected_queues={}
    for label,parent_queue in parent_queues.items():
        p=Path(parent_queue['authority']['path']);h=parent_queue['authority']['pin'];script=parent_queue['script']
        q=read(p,h);files[p]=h
        expected_queues[label]=dict(authority=dict(path=str(p),pin=h),script=script)
        for name,v in q['code'].items():
            need(name not in code or code[name]==v,'SUCCESSOR_CODE_CONFLICT');code[name]=v
    need(cfg['queues']==expected_queues,'SUCCESSOR_QUEUES')
    gref=cfg['placement_proof'];expected=Path(old['owners']['grouped']['output'])/'qualification/report.json'
    need(Path(gref['path'])==expected,'SUCCESSOR_GROUPED_PATH');g=read(expected,gref['pin']);files[expected]=gref['pin']
    for name,h in g['binding']['code'].items():
        need(name not in code or code[name]==h,'SUCCESSOR_CODE_CONFLICT');code[name]=h
    for name in (*EXTRA,*NEW):
        h=sha(REPO/'scripts'/name)
        need(name not in code or code[name]==h,'SUCCESSOR_CODE_CONFLICT');code[name]=h
    need(cfg['code']==code,'SUCCESSOR_CODE_BINDING')
    for p,h in files.items():need(sha(p)==h,'SUCCESSOR_PHYSICAL_INPUT')
    stamps={p:stamp(p) for p in files}
    def fresh():
        base_fresh()
        for n,h in code.items():need(sha(REPO/'scripts'/n)==h,'SUCCESSOR_CODE_CHANGED')
        for p,s in stamps.items():need(stamp(p)==s,'SUCCESSOR_INPUT_CHANGED')
    fresh();return cfg,old,template,fresh


def terminal_owners(cfg):
    # Includes the grouped controller and all three original qualification waiters.
    for row in cfg['owners'].values():
        current=process.process(row['pid'])
        need(current is None or current['starttime']==row['starttime'],'SUCCESSOR_PID_REUSED')
        need(current is None,'SUCCESSOR_LIVE_PREDECESSOR')


def sealed(cfg,authority_pin,b,back):
    m=read(Path(b['path'])/'boundary.json',b['pin'])
    need(m['complete'] and m['kind']=='placement-resume-boundary-v1' and not m['training_complete']
         and m['binding']['scope']=='full' and m['binding']['device']=='cuda'
         and m.get('retirement_ready') is True and m.get('quiescence')
         and m['owner_authority']['pin']==authority_pin
         and m['source_job']==cfg['owners']['producer']['output'],'SUCCESSOR_BOUNDARY_SCOPE')
    expected=sorted((x['pid'],x['starttime']) for x in cfg['owners'].values())
    need(sorted((x['pid'],x['starttime']) for x in m['owners'])==expected,'SUCCESSOR_BOUNDARY_OWNERS')
    v=read(Path(back['path']),back['pin'])
    need(v['complete'] and v['boundary']==b['pin'] and v['all_checkpoint_bytes_exact']
         and v['committed_prefix_exact'] and v['owner_closure_complete'] and v['retained_tail_bytes_exact']
         and v['retirement_ready'] and v['authority']==authority_pin and v['quiescence']==m['quiescence']
         and v['source']==cfg['code']['audit/legal_placement_successor_seal_readback.py'],
         'SUCCESSOR_RETIREMENT_READBACK')
    return m,v


def handoff(cfg,authority_pin,path,pin):
    terminal_owners(cfg)  # Always before reading a claimed retirement receipt.
    r=read(path,pin)
    need(r['complete'] and r['kind']=='coordinated-placement-retirement-v1'
         and r['authority']==authority_pin and r['census']==cfg['census']['pin']
         and r['owners']==cfg['owners'] and r['all_terminal'] and not r['original_complete'],
         'SUCCESSOR_RETIREMENT_SCOPE')
    m,_=sealed(cfg,authority_pin,r['boundary'],r['readback'])
    from claims import process_retirement as retirement
    q=m['quiescence'];root=Path(q['path']);contract=read(root/'contract.json',q['pin'])
    need(contract['sources']==retirement.sources() and contract['schema']=='process-retirement-v1'
         and contract['order']==[cfg['owners'][n]['pid'] for n in cfg['stop_order']]
         and [process.binding(x) for x in contract['owners']]==
             [process.binding(cfg['owners'][n]) for n in cfg['stop_order']], 'SUCCESSOR_RETIREMENT_CONTRACT')
    decision=retirement.decision(root,q['pin'])
    need(decision is not None and r['decision']==sha(root/'decision.json')
         and decision['receipts']==dict(boundary=dict(path=str(Path(r['boundary']['path'])/'boundary.json'),pin=r['boundary']['pin']),
                                       readback=r['readback']), 'SUCCESSOR_RETIREMENT_DECISION')
    return r


def owner(path):return evaluation.training.owner(Path(path))


def reference(path,name='report.json'):return dict(path=str(path),pin=sha(Path(path)/name))

"""Exercise successor bindings against actual live owners and real saved answer cases."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import legal_answer_successor_authority as authority
from audit.legal_answer_successor import Children,paired_args
from audit.surface_device_queue import need,sha,write
from claims import process_handoff as process


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False)
    for name in authority.NEW:
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((authority.REPO/'scripts'/name).read_bytes())
    cfg,old,_,fresh=authority.authenticate(a.authority,a.pin)
    expected=list(cfg['owners'].values());order=[cfg['owners'][n]['pid'] for n in cfg['stop_order']]
    before=process.capture(expected,order);write(a.output/'before.json',before)
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:
            need(code in str(error),'SUCCESSOR_PROOF_WRONG_REFUSAL');controls[name]=str(error)
        else:raise ValueError('SUCCESSOR_PROOF_CONTROL_ACCEPTED:'+name)
    for name,change,code in (
        ('missing_stage',lambda x:x['stages'].pop(),'SUCCESSOR_STAGES'),
        ('wrong_order',lambda x:x['stages'].reverse(),'SUCCESSOR_STAGES'),
        ('missing_evaluation',lambda x:x['remaining_work'].pop(),'SUCCESSOR_WORK'),
        ('fewer_documents',lambda x:x['counts']['shared'].update(documents=91539),'ANSWER_FULL_COUNTS'),
        ('subset',lambda x:x['execution'].update(subset=5),'SUCCESSOR_EXECUTION'),
        ('six_owners_only',lambda x:x['owners'].pop('tile_waiter'),'SUCCESSOR_OWNER_SCOPE')):
        bad=deepcopy(cfg);change(bad);reject(name,lambda bad=bad:authority.scope(bad),code)
    reject('old_six_owner_closure',lambda:authority.scope(dict(cfg,owners=old['owners'])),'SUCCESSOR_OWNER_SCOPE')
    reject('wrong_pin',lambda:authority.authenticate(a.authority,'0'*64),'SUCCESSOR_PIN')
    reject('live_handoff',lambda:authority.handoff(cfg,a.pin,a.output/'absent.json','0'*64),'SUCCESSOR_LIVE_PREDECESSOR')
    bad=deepcopy(cfg);bad['code'][authority.NEW[0]]='0'*64;p=a.output/'wrong-code.json';write(p,bad)
    reject('changed_code',lambda:authority.authenticate(p,sha(p)),'SUCCESSOR_CODE_BINDING')
    # Execute the actual launch path. A missing handoff/proof must not hide the earlier live-owner refusal.
    dest=a.output/'refused-live-run';command=[sys.executable,'-I',str(authority.REPO/'scripts/audit/legal_answer_successor.py'),
        '--mode','run','--authority',str(a.authority),'--pin',a.pin,'--handoff',str(a.output/'absent.json'),
        '--handoff-pin','0'*64,'--proof',str(a.output/'absent-proof'),'--proof-pin','0'*64,'--output',str(dest)]
    with (a.output/'refused-live-run.log').open('xb') as log:
        result=subprocess.run(command,cwd=authority.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        log.flush();os.fsync(log.fileno())
    need(result.returncode!=0 and 'SUCCESSOR_LIVE_PREDECESSOR' in (a.output/'refused-live-run.log').read_text()
         and not dest.exists(),'SUCCESSOR_PROOF_LIVE_LAUNCH')
    controls['actual_live_launch']='SUCCESSOR_LIVE_PREDECESSOR'
    children=Children(a.output,fresh);prepared={}
    for name,q in cfg['queues'].items():
        oldq=authority.read(q['authority']['path'],q['authority']['pin']);wait=oldq['wait']
        need((wait['pid'],wait['starttime']) in {(v['pid'],v['starttime']) for v in cfg['owners'].values()},
             'SUCCESSOR_PROOF_UNDECLARED_DEPENDENCY')
        args=['--mode','prepare','--root',str(authority.ROOT),'--wait-root',wait['path'],
              '--wait-pid',str(wait['pid']),'--wait-start',wait['starttime']]
        if name!='tile':args+=paired_args('inspection',oldq['inspection'])
        ref=children.run(name+'-prepare',q['script'],args,a.output/(name+'-prepare'),'authority.json')
        need(authority.read(Path(ref['path'])/'authority.json',ref['pin'])==oldq,'SUCCESSOR_PROOF_QUEUE_CHANGED')
        prepared[name]=ref
    empty=a.output/'empty-inputs.json';write(empty,{})
    common=['--authority',str(a.authority),'--pin',a.pin]
    prior=children.run('prior-readback','legal_answer_successor_worker.py',
        ['--stage','prior-readback',*common,'--inputs',str(empty),'--inputs-pin',sha(empty)],a.output/'prior-readback')
    # The production numerical pipeline runs on the original four real cases, CPU mechanics only.
    candidate=children.run('pipeline','legal_answer_successor_evaluation.py',['--cpu-proof',*common],a.output/'pipeline')
    value=authority.read(Path(candidate['path'])/'report.json',candidate['pin'])
    original=authority.read(a.original/'report.json',a.original_pin)
    need(original['mode']==value['mode']=='cpu-proof' and original['qualification']==value['qualification']
         and original['scored']['summary']==value['scored']['summary'],'SUCCESSOR_PROOF_NUMERICAL_PIPELINE')
    need(not value['full_population_evaluated'] and not value['full_population_trained']
         and value['optimizer_updates']==0 and not value['qualification']['gpu_used'],'SUCCESSOR_PROOF_SCOPE')
    after=process.capture(expected,order);write(a.output/'after.json',after)
    need([process.binding(v) for v in before]==[process.binding(v) for v in after]
         and all(v['state'] in ('R','S','D') for v in after),'SUCCESSOR_PROOF_OWNER_CHANGED')
    need('torch' not in sys.modules,'SUCCESSOR_PROOF_PARENT_MODEL_LOADED');fresh()
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
    files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()}
    write(a.output/'report.json',dict(complete=True,mode='successor-wiring-proof',authority=a.pin,code=cfg['code'],
        controls=controls,live_launch_refused=True,prepared=prepared,prior=prior,pipeline=candidate,
        original=dict(path=str(a.original),pin=a.original_pin),same_qualification=True,same_scores=True,
        owner_count=len(after),gpu_residents=gpu,gpu_used=False,optimizer_updates=0,signals_sent=0,
        torch_loaded=False,retirement_performed=False,files=files,seconds=time.perf_counter()-began,
        acceptance='author-checked',number_status='measured'))
    print(json.dumps(dict(pin=sha(a.output/'report.json'),controls=len(controls),seconds=time.perf_counter()-began)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','original','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','original-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise

"""Four-owner retirement with the already qualified durable decision/recovery primitive."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import answer_refresh_contract as contract
from audit.legal_answer_retirement import child,FIXED
from audit.surface_device_queue import need,sha
from claims import process_handoff as process
from claims import process_retirement as retirement


def qualifications(cfg,pin,proof,proof_pin):
    p=contract.read(dict(path=str(proof/'report.json'),pin=proof_pin))
    need(p['complete'] and p.get('cold_readback_exact') is True and p.get('previous')
         and p['authority']==pin and p['code']==cfg['code'] and p['live_launch_refused']
         and p['seal_branch_proof']['complete'] and p['seal_branch_proof']['roles']==['completed','partial','fresh']
         and p['answer_stages']==cfg['answer_stages'] and p['decoder_stages']==cfg['decoder_stages'],
         'ANSWER_REFRESH_RETIREMENT_PROOF')
    for rel,h in p['files'].items():need(sha(proof/rel)==h,'ANSWER_REFRESH_RETIREMENT_PROOF_BYTES')
    checked={}
    for key in ('retirement','retirement_readback'):
        path,h=FIXED[key];v=contract.read(dict(path=str(contract.ROOT/path),pin=h))
        need(v['complete'],'ANSWER_REFRESH_RETIREMENT_CRASH_PROOF')
        checked[key]=v
    rp,rb=checked['retirement'],checked['retirement_readback']
    need(rp['coordinator_boundaries']==45 and rp['recovery_boundaries']==42
         and not rp['original_jobs_signalled'] and rp['successors_started']==0
         and all(sha(p)==h for p,h in rp['sources'].items()),'ANSWER_REFRESH_RETIREMENT_PRIMITIVE')
    need(rb['proof']==FIXED['retirement'][1] and rb['fixture_identities_terminal']
         and rb['source']==sha(contract.REPO/'scripts/audit/process_retirement_readback.py'),
         'ANSWER_REFRESH_RETIREMENT_PRIMITIVE_BACK')
    return p


def sealed(cfg,pin,b,back):
    m=contract.read(b,'boundary.json');r=contract.read(back)
    need(m['complete'] and m['kind']=='placement-resume-boundary-v1' and m['retirement_ready']
         and m['binding']['scope']=='full' and m['binding']['device']=='cuda' and not m['training_complete']
         and m['owners']==list(cfg['owners'].values()) and m['source_job']==cfg['source_job']
         and m['owner_authority']['pin']==pin,'ANSWER_REFRESH_RETIREMENT_SEAL')
    need(r['complete'] and r['boundary']==b['pin'] and r['authority']==pin and r['retirement_ready']
         and r['quiescence']==m['quiescence'] and r['source']==cfg['code']['audit/answer_refresh_seal_readback.py']
         and all(r[k] for k in ('all_checkpoint_bytes_exact','committed_prefix_exact','retained_tail_bytes_exact',
                                'completed_branch_bytes_exact','owner_closure_complete')),'ANSWER_REFRESH_RETIREMENT_BACK')
    return m


def handoff(cfg,pin,path,h):
    contract.terminal(cfg);r=contract.read(dict(path=str(path),pin=h))
    need(r['complete'] and r['kind']=='answer-decoder-retirement-v1' and r['authority']==pin
         and r['owners']==cfg['owners'] and r['all_terminal'] and not r['original_complete'],'ANSWER_REFRESH_HANDOFF_SCOPE')
    m=sealed(cfg,pin,r['boundary'],r['readback']);q=m['quiescence'];root=Path(q['path'])
    c=contract.read(dict(path=str(root/'contract.json'),pin=q['pin']))
    need(c['sources']==retirement.sources() and c['order']==[cfg['owners'][n]['pid'] for n in cfg['stop_order']]
         and [process.binding(v) for v in c['owners']]==[process.binding(cfg['owners'][n]) for n in cfg['stop_order']],
         'ANSWER_REFRESH_HANDOFF_CONTRACT')
    decision=retirement.decision(root,q['pin'])
    need(decision and r['decision']==sha(root/'decision.json') and decision['receipts']==dict(
        boundary=dict(path=str(Path(r['boundary']['path'])/'boundary.json'),pin=r['boundary']['pin']),readback=r['readback']),
        'ANSWER_REFRESH_HANDOFF_DECISION');return r


def finish(root,cfg,pin):
    transition=root/'transition';decision=retirement.decision(transition,sha(transition/'contract.json'))
    need(decision is not None,'ANSWER_REFRESH_RETIREMENT_NO_DECISION');contract.terminal(cfg)
    b=decision['receipts']['boundary'];back=decision['receipts']['readback']
    value=dict(complete=True,kind='answer-decoder-retirement-v1',authority=pin,owners=cfg['owners'],
        all_terminal=True,original_complete=False,boundary=dict(path=str(Path(b['path']).parent),pin=b['pin']),
        readback=back,decision=sha(transition/'decision.json'),successors_started=0,acceptance='author-checked')
    path=root/'handoff.json'
    if path.exists():need(contract.read(dict(path=str(path),pin=sha(path)))==value,'ANSWER_REFRESH_HANDOFF_CHANGED')
    else:process.write(path,value)
    handoff(cfg,pin,path,sha(path));return dict(path=str(path),pin=sha(path))


def run(a):
    cfg,_,_,fresh=contract.authenticate(a.authority,a.pin);qualifications(cfg,a.pin,a.proof,a.proof_pin)
    rows=contract.live(cfg);a.output.mkdir(exist_ok=False)
    process.write(a.output/'inputs.json',dict(authority=dict(path=str(a.authority),pin=a.pin),
        proof=dict(path=str(a.proof),pin=a.proof_pin)))
    process.write(a.output/'before.json',rows)
    turn=retirement.Transition(a.output/'transition',list(cfg['owners'].values()),
        [cfg['owners'][n]['pid'] for n in cfg['stop_order']],contract.REPO/'.lease/legal-answer-retirement')
    try:
        turn.pause();fresh()
        b=child(a.output,'seal','answer_refresh_seal.py',['--authority',str(a.authority),'--pin',a.pin,
            '--transition',str(turn.root),'--transition-pin',turn.pin],fresh);bp=sha(b/'boundary.json')
        back=child(a.output,'seal-readback','answer_refresh_seal_readback.py',['--input',str(b),'--pin',bp],fresh)
        bref=dict(path=str(b),pin=bp);rref=dict(path=str(back/'report.json'),pin=sha(back/'report.json'))
        m=sealed(cfg,a.pin,bref,rref);need(m['quiescence']==dict(path=str(turn.root),pin=turn.pin),'ANSWER_REFRESH_TRANSITION')
        turn.quiescent();fresh();qualifications(cfg,a.pin,a.proof,a.proof_pin)
        turn.commit(dict(boundary=dict(path=str(b/'boundary.json'),pin=bp),readback=rref))
        process.write(a.output/'resolution.json',turn.resolve())
    except BaseException as e:
        try:process.write(a.output/'failure.json',dict(error=repr(e)))
        finally:process.write(a.output/'failure-resolution.json',turn.resolve())
        raise
    finally:turn.close()
    ref=finish(a.output,cfg,a.pin);fresh();process.write(a.output/'report.json',dict(complete=True,handoff=ref,
         successors_started=0,original_complete=False,acceptance='author-checked'));print(json.dumps(ref),flush=True)


def recover(a):
    cfg,_,_,fresh=contract.authenticate(a.authority,a.pin)
    saved=json.loads((a.input/'inputs.json').read_bytes())
    need(saved['authority']==dict(path=str(a.authority),pin=a.pin),'ANSWER_REFRESH_RECOVERY_AUTHORITY')
    result=retirement.recover(a.input/'transition',sha(a.input/'transition/contract.json'),a.output);fresh()
    if result['all_terminal']:finish(a.input,cfg,a.pin)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('run','recover'),required=True)
    for n in ('authority','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--proof',type=Path);p.add_argument('--proof-pin')
    p.add_argument('--input',type=Path);a=p.parse_args();{'run':run,'recover':recover}[a.mode](a)

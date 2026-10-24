"""Real native source-memory/identity-loop integration, with cold neural replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lens'))
from audit.fsv_identity_loop import configure
from lens.lenscommon import take_lease, run_cli
from model.loop_input import LoopInputs
from model.loop_session import LoopSession, sha, digest, check
from senses.trajectory_population import publish


def source(args):
    resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    take_lease('28','evidence-bound-identity-loop')
    configure()
    inputs=LoopInputs(args.population,args.native,args.source)
    retained=json.loads((args.base.parent/'plan.json').read_bytes())
    check(args.base.name=='initial' and retained['source_id']==inputs.source_id,'AUDIT_RETAINED_BASE')
    kwargs=dict(base=args.base,inputs=inputs,source=args.source,checkpoint=retained['initial'],
                standard=retained['standard_id'],device='cuda')
    return resident,inputs,kwargs


def independent_chain(path, expected_head):
    # Independent SQL/byte traversal; no runtime transition methods.
    db=sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)
    rows=db.execute('SELECT seq,previous,body,hidden,hash FROM events ORDER BY seq').fetchall()
    head='0'*64;advances=0
    for i,(seq,previous,raw,hidden,pin) in enumerate(rows):
        body=json.loads(raw)
        encoded=json.dumps(dict(seq=seq,previous=previous,body=body),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        check(seq==i and previous==head and hashlib.sha256(encoded).hexdigest()==pin,'AUDIT_CHAIN')
        if body['kind']=='advance':
            check(hidden is not None and hashlib.sha256(hidden).hexdigest()==body['hidden_sha256'],'AUDIT_HIDDEN')
            r=body['request'];check(r['ledger_head']==head,'AUDIT_EVENT_BINDING')
            if r['phase']==3:
                receipt=r['receipt']
                check(receipt['episode']==r['episode'] and receipt['world_validity']=='unassessed'
                      and receipt['scope']=='pinned_artifact_contents_only','AUDIT_ECHO_SCOPE')
            advances+=1
        else:check(hidden is None,'AUDIT_EXTRA_STATE')
        head=pin
    db.close();check(head==expected_head,'AUDIT_HEAD')
    return dict(events=len(rows),neural_transitions=advances,head=head)


def main(args):
    started=time.perf_counter();check(not args.output.exists(),'AUDIT_EXISTS')
    resident,inputs,kw=source(args);loaded=time.perf_counter()
    args.output.mkdir(parents=True);(args.output/'gpu-before.txt').write_text(resident,encoding='utf-8')
    path=args.output/'session.sqlite3';session=LoopSession(path,create=True,**kw)
    controls=[];boundaries=[]
    def refuse(name,fn,code):
        before=session.snapshot()
        try:fn()
        except (ValueError,RuntimeError) as e:check(code in str(e),'AUDIT_WRONG_ERROR_'+name)
        else:raise ValueError('MNLM_LOOP_SESSION_AUDIT_ACCEPTED_'+name)
        after=session.snapshot();check(before==after,'AUDIT_REFUSAL_MUTATED')
        controls.append(dict(name=name,code=code,unchanged_head=before['head']))
    units={c:{f:inputs.lookup[(c,f)] for f in ('echo','response','not_chosen')} for c in ('C1','C2')}
    episodes={c:{f:session.checker.episode(u) for f,u in fs.items()} for c,fs in units.items()}
    before=session.snapshot()
    for fields in episodes.values():
        for ep in fields.values():
            session.apply_memory(dict(op='intake',episode=ep))
    key=episodes['C1']['echo']['id']
    child=session.checker.episode(units['C1']['echo'],'interpretation',[key])
    session.apply_memory(dict(op='intake',episode=child))
    refuse('unchecked_native_source',lambda:session.advance('development:C1',units['C1']['echo']),'UNCHECKED_SOURCE')
    for fields in episodes.values():
        for ep in fields.values():
            event=session.apply_memory(dict(op='check_source',episode=ep['id']))
            session.apply_memory(dict(op='promote',episode=ep['id'],receipt=event['event']['result']['id']))
    refuse('profile_interpretation_is_not_world_truth',lambda:session.apply_memory(dict(op='check_source',episode=child['id'])),'NO_APPLICABLE_GROUNDING_CHECKER')
    boundaries.append(dict(name='source_to_memory',before=before,after=session.snapshot()))
    for phase,field in enumerate(('echo','echo','not_chosen')):
        for c in units:session.advance('development:'+c,units[c][field])
    pending=session.snapshot();publish(args.output/'pending.json',pending)
    session.close()
    session=LoopSession(path,expected_contract=pending['contract'],expected_head=pending['head'],**kw)
    check(session.snapshot()==pending,'AUDIT_PENDING_RESTART')
    refuse('missing_echo_receipt',lambda:session.advance('development:C1',units['C1']['response']),'ECHO_RECEIPT')
    wrong=session.memory.state[episodes['C2']['response']['id']]['receipt']['id']
    refuse('other_context_source_receipt',lambda:session.advance('development:C1',units['C1']['response'],receipt=wrong),'ECHO_RECEIPT')
    refuse('generated_echo_receipt',lambda:session.advance('development:C1',units['C1']['response'],receipt='0'*64),'ECHO_RECEIPT')
    for c in units:
        rid=session.memory.state[episodes[c]['response']['id']]['receipt']['id']
        session.advance('development:'+c,units[c]['response'],receipt=rid)
        session.advance('development:'+c,units[c]['response'])
    before=session.snapshot()
    session.apply_memory(dict(op='invalidate',episode=key,reason='engineering evidence withdrawal: reassess this source interpretation'))
    after=session.snapshot()
    check(after['contexts']['development:C1']['needs_review'] and not after['contexts']['development:C2']['needs_review'],'AUDIT_CONTEXT_SCOPE')
    check(after['contexts']['development:C2']==before['contexts']['development:C2'],'AUDIT_OTHER_CONTEXT_CHANGED')
    check(after['memory']['states'][child['id']]['review']=='needs_review','AUDIT_DEPENDENT_REVIEW')
    refuse('stale_context_after_withdrawal',lambda:session.advance('development:C1',units['C1']['response']),'CONTEXT_NEEDS_REVIEW')
    old_receipt=before['memory']['states'][key]['receipt']['id']
    refuse('stale_receipt_promotion',lambda:session.apply_memory(dict(op='promote',episode=key,receipt=old_receipt)),'NO_MATCHING_CHECKED_RECEIPT')
    event=session.apply_memory(dict(op='check_source',episode=key))
    session.apply_memory(dict(op='promote',episode=key,receipt=event['event']['result']['id']))
    refuse('recheck_cannot_clear_neural_review',lambda:session.advance('development:C1',units['C1']['echo']),'CONTEXT_NEEDS_REVIEW')
    refuse('unaffected_context_reset',lambda:session.reset('development:C2',reason='not dirty'),'RESET_REQUIRES_REVIEW')
    boundaries.append(dict(name='evidence_withdrawal',before=before,after=after))
    session.reset('development:C1',reason='discard unsupported state and reconsider rechecked source')
    for phase,field in enumerate(('echo','echo','not_chosen','response','response')):
        rid=session.memory.state[episodes['C1'][field]['id']]['receipt']['id'] if phase==3 else None
        session.advance('development:C1',units['C1'][field],receipt=rid)
    final=session.snapshot();check(final['memory']['grounded_world_claims']==[],'AUDIT_WORLD_SCOPE')
    check(final['contexts']['development:C2']==before['contexts']['development:C2'],'AUDIT_REPAIR_LEAK')
    check(all(s['sequence']==5 and not s['needs_review'] for s in final['contexts'].values()),'AUDIT_CYCLES')
    session.model.check_reference();session.close();executed=time.perf_counter()
    def reopen(p=path,head=final['head'],contract=final['contract'],neural=False):
        return LoopSession(p,expected_contract=contract,expected_head=head,neural_replay=neural,**kw)
    def bad_open(name,p,code,**kwargs):
        try:s=reopen(p,**kwargs)
        except (ValueError,RuntimeError) as e:check(code in str(e),'AUDIT_BAD_RESTART_ERROR')
        else:s.close();raise ValueError('MNLM_LOOP_SESSION_AUDIT_ACCEPTED_'+name)
        controls.append(dict(name=name,code=code))
    bad_open('retained_head_required',path,'HEAD_PIN',head='0'*64)
    bad_open('foreign_contract',path,'CONTRACT',contract='0'*64)
    shadow=args.output/'corrupt.sqlite3';shutil.copyfile(path,shadow)
    db=sqlite3.connect(shadow)
    row=db.execute('SELECT seq,hidden FROM events WHERE hidden IS NOT NULL LIMIT 1').fetchone()
    mutated=bytearray(row[1]);mutated[-1]^=1
    db.execute('UPDATE events SET hidden=? WHERE seq=?',(bytes(mutated),row[0]));db.commit();db.close()
    bad_open('corrupted_context_bytes',shadow,'CONTEXT_BYTES')
    shutil.copyfile(path,shadow)
    db=sqlite3.connect(shadow);db.execute('DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)');db.commit();db.close()
    bad_open('truncated_history',shadow,'HEAD_PIN')
    shutil.copyfile(path,shadow)
    db=sqlite3.connect(shadow);db.execute('BEGIN IMMEDIATE');db.execute('DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)');db.close()
    s=reopen(shadow);check(s.snapshot()==final,'AUDIT_UNCOMMITTED_WRITE');s.close()
    controls.append(dict(name='uncommitted_transaction_preserves_state',passed=True))
    s=reopen();rival=reopen()
    s.apply_memory(dict(op='retire',episode=child['id']));winning=s.snapshot()
    try:rival.apply_memory(dict(op='retire',episode=child['id']))
    except RuntimeError as e:check('WRITER_CONFLICT' in str(e),'AUDIT_CONFLICT_CODE')
    else:raise ValueError('MNLM_LOOP_SESSION_AUDIT_COMPETING_WRITE')
    try:rival.snapshot()
    except RuntimeError as e:check('RELOAD_REQUIRED' in str(e),'AUDIT_POISON_CODE')
    else:raise ValueError('MNLM_LOOP_SESSION_AUDIT_STALE_RAM_VISIBLE')
    controls.append(dict(name='competing_writer_poisoned',passed=True))
    rival.close();s.close()
    boundaries.append(dict(name='durable_commit_and_restart',before=pending,after=winning))
    chain=independent_chain(path,winning['head'])
    check(chain['neural_transitions']==15,'AUDIT_TRANSITION_COUNT')
    s=reopen(head=winning['head'],neural=True)
    check(s.snapshot()==winning and s.neural_replayed==15,'AUDIT_FULL_REPLAY');s.close()
    checked=time.perf_counter()
    plan=dict(contract=winning['contract'],head=winning['head'],checkpoint=kw['checkpoint'],standard=kw['standard'],snapshot=winning)
    publish(args.output/'plan.json',plan);publish(args.output/'boundaries.json',boundaries)
    wall=time.perf_counter()-started
    publish(args.output/'report.json',dict(complete=True,acceptance='author-checked',measured=True,
        source=inputs.source_readback,native=inputs.window_readback,chain=chain,controls=controls,
        audit_code_sha256=sha(Path(__file__).read_bytes()),
        evidence_root=str(args.output),prototype_only=True,served=False,truth_discrimination_established=False,
        behavioural_compliance_established=False,scientific_admission=False,meaning_asserted=False,
        source_scope='pinned_artifact_contents_only',world_truth='unassessed',historical_reconstruction=False,
        files={n:sha((args.output/n).read_bytes()) for n in ('plan.json','boundaries.json','pending.json','session.sqlite3')},
        phases_seconds=dict(source_load=loaded-started,execution=executed-loaded,readback_and_controls=checked-executed,publication=wall-(checked-started)),wall_seconds=wall))
    print(json.dumps(dict(complete=True,events=chain['events'],transitions=15,controls=len(controls),wall_seconds=wall)),flush=True)


def cold(args):
    started=time.perf_counter();resident,inputs,kw=source(args)
    plan=json.loads((args.output/'plan.json').read_bytes())
    report=json.loads((args.output/'report.json').read_bytes())
    for n,pin in report['files'].items():check(sha((args.output/n).read_bytes())==pin,'AUDIT_FILE_PIN')
    check(plan['checkpoint']==kw['checkpoint'] and plan['standard']==kw['standard'],'AUDIT_BASE_PIN')
    chain=independent_chain(args.output/'session.sqlite3',plan['head'])
    s=LoopSession(args.output/'session.sqlite3',expected_contract=plan['contract'],expected_head=plan['head'],neural_replay=True,**kw)
    check(s.snapshot()==plan['snapshot'] and s.neural_replayed==15,'AUDIT_COLD_REPLAY');s.close()
    publish(args.output/'cold-readback.json',dict(complete=True,acceptance='author-checked',all_transitions_exact=True,
        neural_transitions=15,chain=chain,source=inputs.source_readback,native=inputs.window_readback,
        report_sha256=sha((args.output/'report.json').read_bytes()),wall_seconds=time.perf_counter()-started,gpu_before=resident))
    print(json.dumps(dict(complete=True,neural_transitions=15,all_transitions_exact=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','source','base','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cold',action='store_true');a=p.parse_args()
    raise SystemExit(run_cli(lambda:cold(a) if a.cold else main(a)))

"""Run and cold-replay exact retention around actual repository-learning updates."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_cohort_open import authenticate,open_memory
from audit.repository_relation_learning import retained
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli,take_lease
from model.loop_training_checkpoint import fingerprint,tensor_fingerprint
from model.repository_relation_data import RelationData
from model import repository_relation_checkpoint as parent
from model import repository_retention_curvature_checkpoint as checkpoint
from model.repository_retention_curvature import Retention,POLICY,compare,measurements,update
from model.repository_retention import update as fixed_update
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('learn','cold'),required=True)
    args=p.parse_args();started=time.perf_counter()
    c=read(args.config);root=Path(c['root']);previous=Path(c['previous'])
    prior=read_pinned(previous/'result.json',c['previous_pin'])
    config=read(previous/'config.json');source=Path(config['source'])
    proof=read_pinned(Path(config['proof']),config['proof_pin'])
    check(sha((source/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((source/'snapshot-final.json').read_bytes())==proof['snapshot'] and proof['cold_replay_equal'],
          'RETENTION_SOURCE_PIN')
    for name,pin in proof['immutable_after'].items():
        check(sha(Path(name).read_bytes())==pin,'RETENTION_ANCESTOR_PIN')
    recipes=authenticate(read(source/'config.json'))
    if args.stage=='learn':
        check(not root.exists(),'RETENTION_OUTPUT_EXISTS');root.mkdir()
        write_committed(root/'config.json',c)
    else:check(read(root/'config.json')==c,'RETENTION_CONFIG_PIN')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-retention-'+args.stage)
    (root/('gpu-'+args.stage+'.txt')).open('x').write(gpu)
    configure();authenticated=time.perf_counter()
    session,sessions=open_memory(recipes,source,read(source/'snapshot-final.json'))
    snapshot=digest(session.snapshot());restored=time.perf_counter()
    print(dict(stage=args.stage,restored=session.journal.count),flush=True)
    data=RelationData(session,config['native'],config['native_proof'],config['native_proof_pin'])
    model,optimizer,_=parent.load(previous/'learn_1-final',data,prior['checkpoints']['learn_1-final'])
    retention=Retention(data,model,prior['stages']['learn_1']['after'],
        dict(root=str(previous),result=c['previous_pin'],checkpoint=prior['checkpoints']['learn_1-final']))
    old=retained(session,model)
    check(old==prior['parent_retained'],'RETENTION_LEGACY_PARENT')
    controls=[]
    def reject(name,code,fn):
        try:fn()
        except (ValueError,SenseError) as exc:check(code in str(exc),'RETENTION_CONTROL_CODE:'+str(exc))
        else:raise AssertionError('RETENTION_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    missing=dict(retention.ceiling);missing.pop(next(iter(missing)))
    reject('missing_roster','RETENTION_SCORE_ROSTER',lambda:compare(retention.ceiling,missing))
    saved_contract=retention.contract
    retention.contract=copy.deepcopy(saved_contract)
    retention.contract['ceiling'][next(iter(retention.ceiling))]=-1.
    reject('forged_ceiling','RETENTION_CONTRACT_CHANGED',retention.checked)
    retention.contract=saved_contract
    bad,_,_=parent.load(previous/'learn_2-final',data,prior['checkpoints']['learn_2-final'])
    bad_scores=measurements(data.evaluate(bad,'learn_1'))
    bad_regressions=compare(retention.ceiling,bad_scores)
    check(bad_regressions,'RETENTION_NEGATIVE_CANDIDATE_NOT_DETECTED')
    reject('real_regressing_candidate','RETENTION_EXISTING_REGRESSION',lambda:retention.scores(bad))
    del bad
    fixed=read_pinned(Path(c['fixed_root'])/'result.json',c['fixed_pin'])
    fixed_replay=fixed_update(retention,model,optimizer)
    for k in ('accepted','code','before','after','before_scores','after_scores','trials','projection'):
        check(fixed_replay[k]==fixed['attempts'][0][k],'RETENTION_FIXED_REJECTION_REPLAY:'+k)
    check(not fixed_replay['accepted'],'RETENTION_FIXED_REJECTION_EXPECTED')
    write_committed(root/('fixed-rejection-'+args.stage+'.json'),fixed_replay)
    materialized=time.perf_counter()
    phases=dict(authentication=authenticated-started,lineage_restore=restored-authenticated,
                materialization_controls=materialized-restored)
    cursor=time.perf_counter()
    if args.stage=='learn':
        result=dict(contract=retention.contract,code=checkpoint.code_pins(),controls=controls,
            previous_regressions=bad_regressions,checkpoints={},attempts=[],initial_parameters=
            tensor_fingerprint(dict(model.named_parameters())),legacy=old)
        result['before']=data.evaluate(model,'learn_2')
        pin=checkpoint.save(root/'initial',model,optimizer,retention,8,0,retention.contract['predecessor'])
        result['checkpoints']['initial']=pin
        # Exercise rejection after a real optimizer proposal, never a fake loss or fake native input.
        control=update(retention,model,optimizer,force_reject=True)
        check(not control['accepted'] and control['before']==control['after'] and
              control['proposal_fingerprint']!=control['before'],'RETENTION_FORCED_ROLLBACK')
        result['forced_rejection']=control
        clock=8
        for attempt in range(1,POLICY['attempts']+1):
            row=update(retention,model,optimizer)
            if row['accepted']:clock+=1
            row['attempt']=attempt;row['optimizer_step']=clock
            result['attempts'].append(row)
            check(retained(session,model)==old,'RETENTION_LEGACY_DRIFT')
            name=f'attempt-{attempt:02}'
            pin=checkpoint.save(root/name,model,optimizer,retention,clock,attempt,pin)
            result['checkpoints'][name]=pin
            write_committed(root/(name+'.json'),row)
            print(dict(attempt=attempt,accepted=row['accepted'],projection=row['projection']['status'],
                       trials=len(row['trials']),seconds=row['wall_seconds']),flush=True)
            if not row['accepted']:break
        result['after']=data.evaluate(model,'learn_2')
        result['retained_after']=data.evaluate(model,'learn_1')
        result['withheld_action']=data.evaluate(model,'learn_2',intervention=True)
        result['final_parameters']=tensor_fingerprint(dict(model.named_parameters()))
        result['final_fingerprint']=fingerprint(model,optimizer)
        result['changed_parameters']=[n for n,v in result['final_parameters'].items()
                                      if v!=result['initial_parameters'][n]]
        check(all(n.startswith('query._repository.') for n in result['changed_parameters']),
              'RETENTION_PARAMETER_SCOPE')
        rollback,rollback_opt,_=checkpoint.load(root/'initial',retention,result['checkpoints']['initial'])
        check(data.evaluate(rollback,'learn_2')==result['before'] and retained(session,rollback)==old,
              'RETENTION_ROLLBACK_REPLAY')
        reject('wrong_checkpoint_pin','RETENTION_CHECKPOINT_PIN',lambda:checkpoint.load(root/'initial',retention,'0'*64))
        write_committed(root/'result.json',result)
        phases['learning_controls_persistence']=time.perf_counter()-cursor
    else:
        expected=read(root/'result.json')
        check(expected['contract']==retention.contract and expected['code']==checkpoint.code_pins(),
              'RETENTION_COLD_CONTRACT')
        # Resume after the first attempt; if refused, reproduce that refusal from the initial state.
        first=expected['attempts'][0]
        resume='attempt-01' if first['accepted'] else 'initial'
        model,optimizer,m=checkpoint.load(root/resume,retention,expected['checkpoints'][resume])
        rows=expected['attempts'][1:] if first['accepted'] else expected['attempts']
        for expected_row in rows:
            actual=update(retention,model,optimizer)
            for key in ('accepted','code','before','after','before_scores','after_scores','trials','projection'):
                check(actual[key]==expected_row[key],'RETENTION_RESUMED_UPDATE:'+key)
        check(fingerprint(model,optimizer)==expected['final_fingerprint'] and
              data.evaluate(model,'learn_2')==expected['after'] and
              data.evaluate(model,'learn_1')==expected['retained_after'] and retained(session,model)==old,
              'RETENTION_COLD_FINAL')
        final_name=f"attempt-{len(expected['attempts']):02}"
        loaded,loaded_opt,_=checkpoint.load(root/final_name,retention,expected['checkpoints'][final_name])
        check(fingerprint(loaded,loaded_opt)==expected['final_fingerprint'],'RETENTION_FINAL_LOAD')
        rollback,_,_=checkpoint.load(root/'initial',retention,expected['checkpoints']['initial'])
        check(data.evaluate(rollback,'learn_2')==expected['before'] and retained(session,rollback)==old,
              'RETENTION_COLD_ROLLBACK')
        result=dict(resumed_exact=True,final_load_equal=True,rollback_equal=True,legacy_equal=True)
        phases['restart_controls']=time.perf_counter()-cursor
    data.fresh()
    check(digest(session.snapshot())==snapshot and sha((source/'experience.sqlite3').read_bytes())==proof['journal'],
          'RETENTION_SOURCE_CHANGED')
    for s in sessions:s.close()
    ended=time.perf_counter();phases['final_source_readback']=ended-started-sum(phases.values())
    report=dict(stage=args.stage,wall_seconds=ended-started,phases=phases,controls=controls,
        source_unchanged=True,serving=False,scientific_admission=False,heldout=False,
        useful_new_capability_demonstrated=False,acceptance='author-checked',number_status='measured',
        result=sha((root/'result.json').read_bytes()),cold=result if args.stage=='cold' else None,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
    write_committed(root/(args.stage+'-readback.json'),report)
    print(report,flush=True)


if __name__=='__main__':run_cli(main)

"""One predeclared continued-learning run with source-bound optimizer replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lens'))
import torch
from audit.fsv_identity_loop import configure
from lens.lenscommon import run_cli,take_lease
from model.loop_input import LoopInputs
from model.loop_session import LoopSession
from model.loop_checkpoint import load as model_load
from model.loop_learning import LearningInputs,check,decide
from model import loop_training_checkpoint as checkpoint
from senses.contract import digest
from senses.trajectory_population import publish


def sha(path):return checkpoint.sha(path.read_bytes())


def setup(args):
    resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    take_lease('28','connected-intrinsic-learning');configure()
    check(sha(args.plan)==args.plan_sha256,'PREDECLARED_PLAN_PIN')
    plan=json.loads(args.plan.read_bytes())
    inputs=LoopInputs(args.population,args.native,args.source)
    retained=json.loads((args.base.parent/'plan.json').read_bytes())
    check(args.base.name=='initial' and retained['source_id']==inputs.source_id,'BASE_PIN')
    kw=dict(base=args.base,inputs=inputs,source=args.source,checkpoint=retained['initial'],standard=retained['standard_id'],device='cuda')
    return resident,plan,inputs,kw


def next_fingerprint(model,opt,loss):
    return dict(**checkpoint.fingerprint(model,opt),loss=loss,
                cpu_rng=digest(checkpoint.tensor_fingerprint(torch.get_rng_state())),
                cuda_rng=digest(checkpoint.tensor_fingerprint(torch.cuda.get_rng_state(model.initial.device))))


def check_gate(assessment):
    # Independent persisted-case readback, separate from the runtime decide().
    old={r['record']:r for r in assessment['baseline']}
    new={r['record']:r for r in assessment['candidate']}
    check(set(old)==set(new) and len(old)==10,'GATE_COVERAGE')
    violations=[];gains=[]
    for key,a in old.items():
        b=new[key]
        if a['role']=='retention' and any(y>x for x,y in zip(a['loss'],b['loss'])):violations.append(key)
        if any(x>0>=y for x,y in zip(a['margin'],b['margin'])):violations.append(key)
        if a['role']=='acquisition':gains.extend(y-x for x,y in zip(a['loss'],b['loss']))
    import math
    allowed=not violations and math.fsum(gains)/len(gains)<0
    check(assessment['decision']['eligible']==allowed and assessment['selected']==assessment['candidate_pin' if allowed else 'baseline_pin'],'GATE_SELECTION')
    return dict(eligible=allowed,regressed_records=sorted(set(violations)),acquisition_mean_delta=math.fsum(gains)/len(gains))


def main(args):
    started=time.perf_counter();check(not args.output.exists(),'OUTPUT_EXISTS')
    resident,plan,inputs,kw=setup(args);args.output.mkdir(parents=True)
    (args.output/'gpu-before.txt').write_text(resident,encoding='utf-8')
    session=LoopSession(args.output/'session.sqlite3',create=True,**kw)
    for c in sorted(plan['roles']):
        for field in ('echo','response','not_chosen'):
            ep=session.checker.episode(inputs.lookup[(c,field)])
            session.apply_memory(dict(op='intake',episode=ep))
            event=session.apply_memory(dict(op='check_source',episode=ep['id']))
            session.apply_memory(dict(op='promote',episode=ep['id'],receipt=event['event']['result']['id']))
    source_snapshot=session.snapshot();publish(args.output/'source-snapshot.json',source_snapshot)
    check(len(session.memory.episodes)==66 and session.count==198,'SOURCE_COUNTS')
    model,_,_=model_load(args.base,expected_checkpoint=kw['checkpoint'],expected_standard=kw['standard'],device='cuda')
    data=LearningInputs(session,model,plan);prepared=time.perf_counter()
    opt=checkpoint.optimizer_for(model,plan);initial=checkpoint.save(args.output/'initial',model,opt,data,step=0)
    history=[];initial_eval=data.evaluate(model)
    for step in range(plan['base_steps']):history.append(dict(step=step+1,role='old_train',loss=data.step(model,opt,'old_train')))
    model.check_reference();baseline=checkpoint.save(args.output/'baseline',model,opt,data,step=64,parent=initial)
    old_model,old_opt,old_manifest=checkpoint.load(args.output/'baseline',data,expected=baseline)
    check(old_manifest['step']==64,'BASELINE_CURSOR')
    trained_base=time.perf_counter()
    # Continue the persisted baseline's optimizer. Preserve the first next step
    # for a separate fresh-process replay; never recreate a fresh AdamW here.
    value=data.step(old_model,old_opt,'continuation')
    expected=next_fingerprint(old_model,old_opt,value);publish(args.output/'expected-next-step.json',expected)
    history.append(dict(step=65,role='continuation',loss=value))
    for step in range(65,128):history.append(dict(step=step+1,role='continuation',loss=data.step(old_model,old_opt,'continuation')))
    candidate=checkpoint.save(args.output/'candidate',old_model,old_opt,data,step=128,parent=baseline)
    base_model,_,base_manifest=checkpoint.load(args.output/'baseline',data,expected=baseline)
    candidate_model,_,candidate_manifest=checkpoint.load(args.output/'candidate',data,expected=candidate)
    fitted=time.perf_counter()
    manifests=dict(baseline_manifest=base_manifest,candidate_manifest=candidate_manifest)
    assessment=session.assess_learning(data,base_model,candidate_model,baseline_pin=baseline,candidate_pin=candidate,**manifests)
    measured=check_gate(assessment)
    publish(args.output/'assessment.json',assessment);publish(args.output/'history.json',history)
    publish(args.output/'initial-evaluation.json',initial_eval)
    assessed=time.perf_counter();controls=[]
    def refuse(name,fn,code):
        before=checkpoint.fingerprint(candidate_model,old_opt)
        try:fn()
        except (ValueError,RuntimeError) as e:check(code in str(e),'WRONG_ERROR_'+name)
        else:raise ValueError('MNLM_LOOP_LEARNING_ACCEPTED_'+name)
        check(checkpoint.fingerprint(candidate_model,old_opt)==before,'REFUSAL_MUTATED')
        controls.append(dict(name=name,code=code))
    refuse('held_out_gradient',lambda:data.step(candidate_model,old_opt,'retention'),'EVALUATION_GRADIENT')
    refused_head=session.head
    refuse('swapped_checkpoint_pins',lambda:session.assess_learning(data,base_model,candidate_model,
            baseline_pin=candidate,candidate_pin=baseline,**manifests),'LEARNING_CHECKPOINT')
    refuse('swapped_model_objects',lambda:session.assess_learning(data,candidate_model,base_model,
            baseline_pin=baseline,candidate_pin=candidate,**manifests),'LEARNING_CHECKPOINT')
    check(session.head==refused_head,'REFUSED_SELECTION_MUTATED')
    refuse('foreign_training_pin',lambda:checkpoint.load(args.output/'baseline',data,expected='0'*64),'TRAINING_SNAPSHOT_PIN')
    wrong=copy.copy(data);wrong.plan={**plan,'learning_rate':.002}
    refuse('changed_recipe',lambda:checkpoint.load(args.output/'baseline',wrong,expected=baseline),'TRAINING_SNAPSHOT_CONTRACT')
    refuse('overwrite_candidate',lambda:checkpoint.save(args.output/'candidate',candidate_model,old_opt,data,step=128),'TRAINING_SNAPSHOT_EXISTS')
    fresh=checkpoint.optimizer_for(base_model,plan)
    refuse('fresh_optimizer_cannot_resume',lambda:checkpoint.validate_optimizer(base_model,fresh,plan,64),'OPTIMIZER_CURSOR')
    bad_lookup=inputs.lookup[('C1','response')];inputs.lookup[('C1','response')]=inputs.lookup[('C2','response')]
    refuse('swapped_source_field',lambda:LearningInputs(session,model,plan),'SOURCE_FIELD_JOIN')
    inputs.lookup[('C1','response')]=bad_lookup
    no_gain=decide(plan,assessment['baseline'],assessment['baseline'],len(model.specs))
    check(not no_gain['eligible'] and 'NO_ACQUISITION_GAIN' in no_gain['reasons'],'UNCHANGED_PROMOTION')
    controls.append(dict(name='unchanged_model_cannot_claim_acquisition',passed=True))
    shadow=args.output/'corrupt-copy';shutil.copytree(args.output/'baseline',shadow)
    raw=(shadow/'training.pt').read_bytes();bad=bytearray(raw);bad[-1]^=1;(shadow/'training.pt').write_bytes(bad)
    refuse('optimizer_corruption',lambda:checkpoint.load(shadow,data,expected=baseline),'OPTIMIZER_PIN')
    (shadow/'training.pt').write_bytes(raw);marker=(shadow/'COMMITTED').read_bytes();(shadow/'COMMITTED').unlink()
    refuse('interrupted_snapshot',lambda:checkpoint.load(shadow,data,expected=baseline),'TRAINING_SNAPSHOT_INCOMPLETE')
    (shadow/'COMMITTED').write_bytes(marker)
    # Preserve the authoritative ledger; withdraw evidence only in an owned copy.
    snapshot=session.snapshot();session.close();shadow_db=args.output/'withdrawal.sqlite3';shutil.copyfile(args.output/'session.sqlite3',shadow_db)
    restored=LoopSession(shadow_db,expected_contract=snapshot['contract'],expected_head=snapshot['head'],**kw)
    stale=LearningInputs(restored,model,plan)
    key=restored.checker.episode(inputs.lookup[('C1','response')])['id']
    restored.apply_memory(dict(op='invalidate',episode=key,reason='FSV source withdrawal before candidate publication'))
    refuse('withdrawn_training_snapshot',lambda:checkpoint.load(args.output/'baseline',stale,expected=baseline),'SOURCE_HEAD_CHANGED')
    refuse('withdrawn_candidate_assessment',lambda:restored.assess_learning(stale,base_model,candidate_model,baseline_pin=baseline,candidate_pin=candidate,**manifests),'SOURCE_HEAD_CHANGED')
    restored.close()
    s=LoopSession(args.output/'session.sqlite3',expected_contract=snapshot['contract'],expected_head=snapshot['head'],**kw)
    check(s.snapshot()==snapshot and s.selected_learning==assessment['selected'],'SELECTED_REPLAY')
    data2=LearningInputs(s,model,plan);m,o,_=checkpoint.load(args.output/'baseline',data2,expected=baseline)
    check(next_fingerprint(m,o,data2.step(m,o,'continuation'))==expected,'NEXT_STEP_REPLAY')
    s.close();checked=time.perf_counter()
    recipe=dict(initial=initial,baseline=baseline,candidate=candidate,selected=assessment['selected'],source=snapshot,
                plan_sha256=args.plan_sha256,binding=data.binding,base_checkpoint=kw['checkpoint'],standard=kw['standard'])
    publish(args.output/'run.json',recipe)
    wall=time.perf_counter()-started
    report=dict(complete=True,acceptance='author-checked',source=inputs.source_readback,native=inputs.window_readback,
        measured=dict(source_episodes=66,source_events=198,ledger_events=snapshot['events'],train_steps=128,controls=len(controls)),
        decision=measured,controls=controls,prototype_only=True,production_admission=False,behavioural_compliance_established=False,
        world_truth_established=False,served=False,meaning_asserted=False,
        phase_seconds=dict(prepare=prepared-started,baseline=trained_base-prepared,continuation=fitted-trained_base,
                           assessment=assessed-fitted,controls_and_replay=checked-assessed,publication=wall-(checked-started)),wall_seconds=wall,
        files={n:sha(args.output/n) for n in ('run.json','source-snapshot.json','assessment.json','history.json','expected-next-step.json','initial-evaluation.json','session.sqlite3')})
    publish(args.output/'report.json',report)
    print(json.dumps(dict(complete=True,decision=measured,controls=len(controls),wall_seconds=wall)),flush=True)


def cold(args):
    started=time.perf_counter();resident,plan,inputs,kw=setup(args)
    run=json.loads((args.output/'run.json').read_bytes());rep=json.loads((args.output/'report.json').read_bytes())
    for n,pin in rep['files'].items():check(sha(args.output/n)==pin,'COLD_FILE_PIN')
    check(run['plan_sha256']==args.plan_sha256 and run['base_checkpoint']==kw['checkpoint'] and run['standard']==kw['standard'],'COLD_RECIPE')
    session=LoopSession(args.output/'session.sqlite3',expected_contract=run['source']['contract'],expected_head=run['source']['head'],**kw)
    data=LearningInputs(session,session.model,plan)
    baseline,opt,_=checkpoint.load(args.output/'baseline',data,expected=run['baseline'])
    value=data.step(baseline,opt,'continuation')
    check(next_fingerprint(baseline,opt,value)==json.loads((args.output/'expected-next-step.json').read_bytes()),'COLD_NEXT_STEP')
    baseline,_,_=checkpoint.load(args.output/'baseline',data,expected=run['baseline'])
    candidate,_,_=checkpoint.load(args.output/'candidate',data,expected=run['candidate'])
    a=json.loads((args.output/'assessment.json').read_bytes())
    check(data.evaluate(baseline)==a['baseline'] and data.evaluate(candidate)==a['candidate'],'COLD_ASSESSMENT')
    decision=check_gate(a);check(session.selected_learning==a['selected']==run['selected'],'COLD_SELECTION')
    session.close()
    publish(args.output/'cold-readback.json',dict(complete=True,acceptance='author-checked',next_step_exact=True,
        optimizer_and_rng_exact=True,assessment_exact=True,decision=decision,report_sha256=sha(args.output/'report.json'),
        wall_seconds=time.perf_counter()-started,gpu_before=resident))
    print(json.dumps(dict(complete=True,next_step_exact=True,assessment_exact=True,decision=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','source','base','plan','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--cold',action='store_true');a=p.parse_args()
    raise SystemExit(run_cli(lambda:cold(a) if a.cold else main(a)))

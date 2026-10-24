"""Physical complete-answer target consumption and scoped candidate assessment."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import complete_supervision_source as source
from audit.complete_answer_baselines import reject
from claims.episodic_memory import digest as row_digest
from claims.process_conditions import write_committed
from model.capability_supervision import PURPOSES
from model.complete_answer_supervision import arithmetic
from text.constellation_corpus import file_sha,need


def controls(s):
    m=s['model'];train=m.view.source.train;query=m.view.source.query;out={}
    failed=next(k for k in train if m.rows[k]['original_evidence']['outcome']==0)
    passed=next(k for k in train if m.rows[k]['original_evidence']['outcome']==1)
    out['failed_as_success']=reject(lambda:m.training_target(failed,purpose='successful_patch'),'SUPERVISION_FAILED_ACTION')
    out['query_fit']=reject(lambda:m.training_target(query[0],purpose='recorded_continuation'),'COMPLETE_SUPERVISION_QUERY_FIT')
    out['wrong_purpose']=reject(lambda:m.decide(train[0],'world_truth'),'COMPLETE_SUPERVISION_PURPOSE')
    decision,target=m.training_target(passed,purpose='successful_patch');need(decision['allowed'] and target['atoms'],'SUPERVISION_PASS_CONTROL')
    out['passing_complete_target']=dict(task=passed,atoms=len(target['atoms']))
    d,t=m.training_target(failed,purpose='recorded_continuation');need(d['allowed'] and t['atoms'],'SUPERVISION_FAILURE_RETENTION')
    out['failed_record_retained']=dict(task=failed,atoms=len(t['atoms']))
    raw=m.text[passed];wrong=m.assess(failed,raw)
    need(wrong['execution'] is None and wrong['requires_new_execution'] and not wrong['accepted'],'SUPERVISION_DONOR_TRANSFER')
    out['donor_success_does_not_transfer']=True
    original=deepcopy(m.rows[failed]);m.rows[failed]['original_evidence']['outcome']=1
    out['mutated_outcome']=reject(lambda:m.evidence(failed),'COMPLETE_SUPERVISION_MUTATED');m.rows[failed]=original
    original_raw=m.text[failed];m.text[failed]+=b'!'
    out['mutated_target']=reject(lambda:m.evidence(failed),'COMPLETE_SUPERVISION_MUTATED');m.text[failed]=original_raw
    original_binding=m.single.corpus.tasks[failed]['binding'];m.single.corpus.tasks[failed]['binding']='changed'
    out['changed_original_binding']=reject(lambda:m.decide(failed,'recorded_continuation'),'COMPLETE_SUPERVISION_ORIGINAL_CHANGED')
    m.single.corpus.tasks[failed]['binding']=original_binding
    e=m.evidence(failed);e['original_evidence']['outcome']=1
    need(m.evidence(failed)['original_evidence']['outcome']==0,'SUPERVISION_CALLER_MUTATION');out['caller_owned_evidence']=True
    invalid=m.assess(failed,b'\xff');need(not invalid['utf8_valid'] and not invalid['accepted'],'SUPERVISION_INVALID_UTF8');out['invalid_utf8']=invalid
    probes={'wrong_equality':'<<2+2=5>>','unsupported_call':'<<f(2)=2>>','partial_delimiter':'<<2+2=4>> <<3+3=6',
            'no_equations':'A fluent but unchecked statement.','true_but_irrelevant':'<<2+2=4>>'}
    out['arithmetic_probes']={k:arithmetic(v) for k,v in probes.items()}
    need(all(not out['arithmetic_probes'][k]['all_recognized_equalities_pass'] for k in probes if k!='true_but_irrelevant')
         and out['arithmetic_probes']['true_but_irrelevant']['all_recognized_equalities_pass'], 'SUPERVISION_ARITHMETIC_CONTROLS')
    reasoning=next(k for k in query if m.rows[k]['operation']=='solve_recorded_problem')
    weak=m.assess(reasoning,probes['true_but_irrelevant'].encode())
    need(not weak['accepted'] and not weak['source_agreement'] and not weak['problem_interpretation_checked'],'SUPERVISION_SCOPE_INFLATION')
    out['arithmetic_is_not_answer_correctness']=True
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};s=source.load(a.root);m=s['model']
    phases['original_source_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    write_committed(a.output/'before.json',dict(code=s['code'],inputs=s['inputs'],baseline=source.BASE))
    evidence={k:m.evidence(k) for k in m.rows};write_committed(a.output/'evidence.json',evidence)
    consumed=[]
    for key in m.view.source.train:
        decision,target=m.training_target(key,purpose='recorded_continuation')
        need([v['measurement'] for v in target['atoms']]==evidence[key]['atoms'],'SUPERVISION_WHOLE_TARGET')
        consumed.append(dict(task=key,decision=decision,target=row_digest(target),atoms=len(target['atoms'])))
    decisions=[m.decide(k,p,fitting=k in m.view.source.train) for k in m.rows for p in PURPOSES]
    write_committed(a.output/'consumed.json',consumed);write_committed(a.output/'decisions.json',decisions)
    phases['purpose_targets']=time.perf_counter()-tick;tick=time.perf_counter();rows=[]
    for key in m.rows:
        rows.append(dict(task=key,family=m.rows[key]['family'],split=m.rows[key]['split'],arm='observed_source_control',assessment=m.assess(key,m.text[key])))
    for row in s['predictions']:
        for method,pred in row['predictions'].items():
            assessment=m.assess(row['task'],pred['answer']['text'].encode()) if pred['status']=='candidate' else None
            rows.append(dict(task=row['task'],family=row['family'],split='query',arm=method,status=pred['status'],assessment=assessment))
    write_committed(a.output/'assessments.json',rows)
    phases['complete_answer_assessment']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(s);s['fresh']();phases['controls_final_readback']=time.perf_counter()-tick
    elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'SUPERVISION_PHASES')
    counts=Counter(e['split']+':'+str(e['original_evidence']['outcome']) for e in evidence.values() if e['family']=='coding')
    value=dict(complete=True,schema='complete-answer-supervision-v1',code=s['code'],inputs=s['inputs'],
        files={p.name:file_sha(p) for p in a.output.iterdir()},execution_populations=len(m.single.populations),
        tasks=len(evidence),train_consumed=len(consumed),train_atoms=sum(r['atoms'] for r in consumed),decisions=len(decisions),
        patch_outcomes=dict(counts),summary=source.summary(rows),controls=negative,phases=phases,seconds=elapsed,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,new_executions=0,
        training_performed=False,existing_queued_objective_changed=False,scientific_admission=False,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',value),seconds=elapsed,patch_outcomes=dict(counts))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

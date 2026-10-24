"""Cold full-answer scope reconstruction with a separate exact arithmetic evaluator."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import complete_supervision_source as source
from audit.arithmetic_readback import receipt
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need,digest as byte_digest


def check_math(text,saved):
    receipt(text,saved['records']);count=len(re.findall(r'<<([^<>]+)>>',text))
    coverage=text.count('<<')==text.count('>>')==count
    need(saved['delimiter_coverage']==coverage and saved['all_recognized_equalities_pass']==
         (count>0 and coverage and all(r['status']=='pass' for r in saved['records']))
         and saved['scope']=='annotated_equalities_only' and not saved['problem_interpretation_checked'],'SUPERVISION_COLD_MATH')


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    need(file_sha(a.input/'report.json')==a.pin,'SUPERVISION_COLD_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['schema']=='complete-answer-supervision-v1','SUPERVISION_COLD_REPORT')
    for n,h in r['files'].items():need(file_sha(a.input/n)==h,'SUPERVISION_COLD_PAYLOAD')
    s=source.load(a.root);m=s['model'];need(s['code']==r['code'] and s['inputs']==r['inputs'],'SUPERVISION_COLD_SOURCE')
    def read(n):return json.loads((a.input/n).read_bytes())
    evidence=read('evidence.json');need(set(evidence)==set(m.view.rows),'SUPERVISION_COLD_ROSTER');outcomes={};atoms=0
    for key,e in evidence.items():
        pop,task,split=m.view.source.selected(key);target=m.view.rows[key]['answer'];lo,hi=target['reference_span']
        raw=pop.fields[target['source_field']]['text'].encode()[lo:hi]
        need(e['answer_sha256']==byte_digest(raw) and e['reference_span']==[lo,hi] and e['end']==target['end']
             and e['atoms']==[v['measurement'] for v in target['atoms']] and e['split']==split,'SUPERVISION_COLD_COMPLETE_SPAN')
        check_math(raw.decode(),e['arithmetic']);atoms+=len(e['atoms']);outcomes[key]=None
        if pop.family=='coding':
            original=pop.parent.tasks[task['provenance']['parent_task']];p=original['provenance']
            witnesses=[p['failure'],p['repair']] if original['kind']=='contrast_failure_with_supplied_repair' else [p]
            joins=[]
            for witness in witnesses:
                population=m.single.populations[witness['population']]
                entries=json.loads((population.root/'anchors.json').read_bytes())['entries'];entry=entries[witness['condition']]
                need(entry==witness['execution_evidence'] and entry['outcome']==witness['observed_outcome']
                     and entry['tests']['independently_executed'],'SUPERVISION_COLD_EXECUTION')
                joins.append(dict(population=witness['population'],condition=witness['condition'],receipt=digest(entry),outcome=entry['outcome']))
            need(entry['action_sha256']==byte_digest(raw) and e['original_evidence']['observations']==joins
                 and e['original_evidence']['outcome']==entry['outcome'],'SUPERVISION_COLD_PATCH_BINDING')
            outcomes[key]=entry['outcome']
    phases['source_spans_execution']=time.perf_counter()-tick;tick=time.perf_counter()
    decisions=read('decisions.json');need(len(decisions)==len(evidence)*5,'SUPERVISION_COLD_DECISION_COUNT')
    for d in decisions:
        e=evidence[d['task']];purpose=d['purpose'];scope=e['original_evidence']['scope']
        expected={'recorded_continuation':True,'successful_patch':scope=='retained_execution_test_scope' and outcomes[d['task']]==1,
            'annotated_arithmetic':scope=='annotated_equalities_only' and e['arithmetic']['all_recognized_equalities_pass'],
            'source_answer':scope=='question_author_answer_only','grounded_correct_response':False}[purpose]
        need(d['allowed']==expected and d['fitting']==(e['split']=='train') and d['answer_sha256']==e['answer_sha256']
             and d['evidence']==digest(e) and d['atom_count']==len(e['atoms']),'SUPERVISION_COLD_PURPOSE')
    consumed=read('consumed.json');need({r['task'] for r in consumed}==set(m.view.source.train),'SUPERVISION_COLD_TRAIN_ROSTER')
    for row in consumed:
        target=m.view.rows[row['task']]['answer'];need(row['target']==digest(target) and row['atoms']==len(target['atoms']),
                                                    'SUPERVISION_COLD_CONSUMED_TARGET')
    predictions={r['task']:r for r in s['predictions']};rows=read('assessments.json');need(len(rows)==489+170*4,'SUPERVISION_COLD_DENOMINATOR')
    seen=set();assessed=0
    for row in rows:
        key,arm=row['task'],row['arm'];need((key,arm) not in seen,'SUPERVISION_COLD_DUPLICATE');seen.add((key,arm))
        if arm=='observed_source_control':raw=m.text[key]
        else:
            proposal=predictions[key]['predictions'][arm]
            if proposal['status']!='candidate':need(row['assessment'] is None,'SUPERVISION_COLD_UNAVAILABLE');continue
            raw=proposal['answer']['text'].encode()
        a0=row['assessment'];e=evidence[key];match=raw==m.text[key];scope=e['original_evidence']['scope']
        need(a0['source_agreement']==match and a0['answer_sha256']==byte_digest(raw) and a0['evidence']==digest(e)
             and a0['utf8_valid'] and not any(a0[n] for n in ('accepted','serving','scientific_admission','world_correctness_established',
                 'problem_interpretation_checked','response_appropriateness_checked','donor_execution_transfers')),'SUPERVISION_COLD_ASSESSMENT')
        check_math(raw.decode(),a0['arithmetic'])
        if match and scope=='retained_execution_test_scope':
            need(a0['execution']==dict(outcome=outcomes[key],observations=e['original_evidence']['observations'],
                status='matches_recorded_passing_patch' if outcomes[key]==1 else 'matches_recorded_failed_patch',
                scope='this_recorded_task_action_and_pinned_runtime_only',new_execution=False),'SUPERVISION_COLD_EXECUTION_RESULT')
        else:need(a0['execution'] is None,'SUPERVISION_COLD_DONOR_TRANSFER')
        need(a0['requires_new_execution']==(scope=='retained_execution_test_scope' and not match),'SUPERVISION_COLD_REQUIRED_EXECUTION');assessed+=1
    need(source.summary(rows)==r['summary'],'SUPERVISION_COLD_SUMMARY')
    phases['decisions_and_assessments']=time.perf_counter()-tick;tick=time.perf_counter();s['fresh']()
    phases['final_source_readback']=time.perf_counter()-tick;elapsed=time.perf_counter()-began
    need(.9<=sum(phases.values())/elapsed<=1.001,'SUPERVISION_COLD_PHASES')
    value=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),code=s['code'],tasks=len(evidence),atoms=atoms,
        decisions=len(decisions),assessment_rows=len(rows),available_assessments=assessed,all_purpose_and_scope_checks_exact=True,
        patch_outcomes=dict(Counter(evidence[k]['split']+':'+str(v) for k,v in outcomes.items() if v is not None)),
        phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_executions=0,gpu_used=False,scientific_admission=False,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',value),seconds=elapsed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

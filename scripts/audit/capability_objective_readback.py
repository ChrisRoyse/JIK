"""Independent outcome joins, arithmetic evaluation, direct lexical ranking and target readback."""
import argparse
from collections import Counter
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.arithmetic_readback import receipt, value
from audit.capability_conditioning_readback import refused
from audit.capability_objective_audit import load
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.capability_kernel_learning import pinned
from model.capability_supervision import CapabilitySupervision, PURPOSES


def source_evidence(corpus,supervision,row):
    key=row['task'];t=corpus.tasks[key];pop=corpus.sources[t['population']].population;task=pop.tasks[key]
    target=pop.fields[task['targets'][0]];scope='attested_source_only';outcome=None;observations=[]
    if t['family']=='coding':
        parent=pop.parent.tasks[task['provenance']['parent_task']];p=parent['provenance']
        records=[p] if parent['kind']=='reproduce_recorded_patch' else [p['failure'],p['repair']]
        for record in records:
            observation=supervision.populations[record['population']].observations.entries[record['condition']]
            need(observation==record['execution_evidence'] and observation['outcome']==record['observed_outcome'],'OBJECTIVE_READER_EXECUTION')
            observations.append(dict(population=record['population'],condition=record['condition'],receipt=digest(observation),outcome=observation['outcome']))
        need(target['source']['condition']==observations[-1]['condition'] and row['parent_kind']==parent['kind'],'OBJECTIVE_READER_CONDITION')
        outcome=observations[-1]['outcome'];scope='retained_execution_test_scope'
    elif target['role']=='source_answer':scope='question_author_answer_only'
    elif t['kind'] in ('continue_recorded_steps','solve_recorded_problem'):
        scope='annotated_equalities_only';receipt(target['text'],task['checks'])
        need(row['equalities']==dict(Counter(x['status'] for x in task['checks'])),'OBJECTIVE_READER_EQUALITIES')
    expected=dict(population=pop.pin,binding=t['binding'],family=t['family'],operation=t['kind'],
        split='train' if key in corpus.train else 'query',group=t['split_group'],target=target['id'],target_sha256=sha(target['text'].encode()),
        scope=scope,outcome=outcome,observations=observations,checks=task['checks'],scientific_admission=False,world_correctness_established=False)
    need(all(row[n]==v for n,v in expected.items()),'OBJECTIVE_READER_EVIDENCE')
    allowed={'recorded_continuation':True,'successful_patch':scope=='retained_execution_test_scope' and outcome==1,
        'annotated_arithmetic':scope=='annotated_equalities_only' and bool(task['checks']) and all(x['status']=='pass' for x in task['checks']),
        'source_answer':scope=='question_author_answer_only','grounded_correct_response':False}
    for purpose,admit in allowed.items():
        decision=row['decisions'][purpose]
        need(decision==dict(purpose=purpose,allowed=admit,refusal=None if admit else 'SUPERVISION_FAILED_ACTION'
                           if purpose=='successful_patch' and outcome==0 else 'SUPERVISION_SCOPE_UNESTABLISHED',
            evidence=digest({k:v for k,v in row.items() if k!='decisions'}),scope=scope,target=target['id'],task=key,
            split=expected['split'],fitting=key in corpus.train),'OBJECTIVE_READER_DECISION')


def lexical(text):return Counter(re.findall(r'\w+|[^\w\s]',text.casefold()))


def context(corpus,key):
    task=corpus.tasks[key];pop=corpus.sources[task['population']].population
    fields=[pop.fields[f['id']] for f in task['selections']['context']['fields']]
    return fields,lexical('\n'.join(f['text'] for f in fields))


def direct_baselines(corpus,labels):
    bank={};indices={}
    for key in corpus.train:
        task=corpus.tasks[key];pop=corpus.sources[task['population']].population
        bank[key]=dict(terms=context(corpus,key)[1],text=pop.fields[task['selections']['targets']['fields'][0]['id']]['text'])
    for kind in {corpus.tasks[k]['kind'] for k in corpus.train}:
        keys=sorted(k for k in bank if corpus.tasks[k]['kind']==kind);df=Counter()
        for k in keys:df.update(bank[k]['terms'].keys())
        frequencies=Counter(sha(bank[k]['text'].encode()) for k in keys);modal=min(frequencies,key=lambda h:(-frequencies[h],h))
        indices[kind]=dict(keys=keys,df=df,average=sum(sum(bank[k]['terms'].values()) for k in keys)/len(keys),
            modal=min(k for k in keys if sha(bank[k]['text'].encode())==modal))
    result={}
    for key in corpus.query:
        fields,tokens=context(corpus,key);task=corpus.tasks[key];index=indices[task['kind']];keys=index['keys'];scores={}
        for donor in keys:
            need(corpus.tasks[donor]['split_group']!=task['split_group'],'OBJECTIVE_READER_SPLIT')
            counts=bank[donor]['terms'];length=sum(counts.values());contributions=[]
            for token in sorted(tokens):
                frequency=counts[token]
                if frequency:
                    idf=math.log1p((len(keys)-index['df'][token]+.5)/(index['df'][token]+.5))
                    contributions.append(idf*frequency*2.2/(frequency+1.2*(.25+.75*length/index['average'])))
            scores[donor]=math.fsum(contributions)
        best=min(scores,key=lambda k:(-scores[k],k));choices={}
        for name,donor in (('modal',index['modal']),('bm25',best)):
            choices[name]=dict(text=bank[donor]['text'],donor=donor,region=labels[donor]['recognition']['label'],
                score=scores[donor] if name=='bm25' else None)
        compatible=[f for f in fields if f['role'] in ('recorded_patch','recorded_step','source_answer','attested_speaker_act')]
        if compatible:
            chosen=sorted(compatible,key=lambda f:(Fraction(f['clock']['end']),f['id']))[-1] if all(f['clock'] and 'end' in f['clock'] for f in compatible) else compatible[-1]
            choices['persistence']=dict(text=chosen['text'],field=chosen['id'],region=None)
        else:choices['persistence']=None
        equations=[m[1] for f in fields if f['role']=='recorded_step' for m in re.finditer(r'<<([^<>]+)>>',f['text'])]
        arithmetic=None
        if task['kind']=='continue_recorded_steps' and equations:
            try:
                sides=equations[-1].split('=');need(len(sides)==2,'OBJECTIVE_READER_ARITHMETIC')
                a,b=map(value,sides)
                if a==b:arithmetic=dict(text='#### '+str(b),region=None)
            except (ValueError,ZeroDivisionError,RecursionError):pass
        choices['last_arithmetic']=arithmetic;result[key]=choices
    return result


def check_scores(corpus,labels,rows,predictions,summary):
    need([r['task'] for r in rows]==list(corpus.query),'OBJECTIVE_READER_POPULATION')
    for row in rows:
        key=row['task'];task=corpus.tasks[key];pop=corpus.sources[task['population']].population
        truth=pop.fields[task['selections']['targets']['fields'][0]['id']]['text'];label=labels[key]['recognition']['label']
        need(row['target_sha256']==sha(truth.encode()) and row['region_label']==label
             and row['operation']==task['kind'] and row['family']==task['family'] and row['group']==task['split_group']
             and row['parent_region']==labels[key]['recognition']['selected'],'OBJECTIVE_READER_TARGET')
        for name,prediction in predictions[key].items():
            saved=row['arms'][name];available=prediction is not None
            need(saved['status']==('candidate' if available else 'unavailable') and saved['exact']==(available and prediction['text']==truth)
                 and saved['text_sha256']==(sha(prediction['text'].encode()) if available else None),'OBJECTIVE_READER_SCORE')
            region=prediction.get('region') if available else None
            need(saved['region_match']==(None if region is None else region==label),'OBJECTIVE_READER_SCORE')
            if available:
                for n,v in prediction.items():
                    if n!='text':need(saved[n]==v,'OBJECTIVE_READER_RANK')
    for operation,cells in summary.items():
        selected=[r for r in rows if r['operation']==operation]
        for name in ('modal','bm25','persistence','last_arithmetic'):
            expected=dict(tasks=len(selected),available=sum(r['arms'][name]['status']=='candidate' for r in selected),
                exact=sum(r['arms'][name]['exact'] for r in selected),regions=sum(r['arms'][name]['region_match'] is True for r in selected),
                region_defined=sum(r['arms'][name]['region_match'] is not None for r in selected))
            need(cells[name]==expected,'OBJECTIVE_READER_SUMMARY')
        need(cells['parent_region_hits']==sum(r['parent_region']==r['region_label'] for r in selected),'OBJECTIVE_READER_SUMMARY')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'OBJECTIVE_READER_EXISTS');ref=dict(path=str(a.report),pin=a.pin);report=pinned(ref);root=a.report.parent
    for name,pin in report['source']['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'OBJECTIVE_READER_CODE')
    for name,pin in report['files'].items():need(sha((root/name).read_bytes())==pin,'OBJECTIVE_READER_FILE')
    corpus,labels=load(report['source']);supervision=CapabilitySupervision(corpus)
    evidence=[json.loads(x) for x in (root/'evidence.jsonl').read_bytes().splitlines()]
    need([r['task'] for r in evidence]==[*corpus.train,*corpus.query] and report['train']==list(corpus.train)
         and report['query']==list(corpus.query),'OBJECTIVE_READER_EVIDENCE_ROSTER')
    for row in evidence:source_evidence(corpus,supervision,row)
    for split,operations in report['summary']['evidence'].items():
        for kind,saved in operations.items():
            rows=[r for r in evidence if r['split']==split and r['operation']==kind]
            expected=dict(tasks=len(rows),groups=len({r['group'] for r in rows}),outcomes=dict(Counter(str(r['outcome']) for r in rows)),
                intents={p:sum(r['decisions'][p]['allowed'] for r in rows) for p in PURPOSES})
            need(expected==saved,'OBJECTIVE_READER_SUMMARY')
    predictions=direct_baselines(corpus,labels);rows=[json.loads(x) for x in (root/'baselines.jsonl').read_bytes().splitlines()]
    check_scores(corpus,labels,rows,predictions,report['summary']['baselines'])
    boundaries=json.loads((root/'native-targets.json').read_bytes());arrays=np.load(root/'native-targets.npz',allow_pickle=False)
    for boundary in boundaries:
        key=boundary['task'];task=corpus.tasks[key];target=task['selections']['targets']['fields'][0]['id'];pop=task['population']
        expected=next(r for r in evidence if r['task']==key)
        need(boundary['decision']==expected['decisions'][boundary['purpose']] and boundary['source_sha256']==expected['target_sha256'],'OBJECTIVE_READER_NATIVE_SCOPE')
        for spec in corpus.specs:
            stored=corpus.snapshot['capability_memory']['fields'][pop+':'+target]['slots'][spec.name]
            tensor=arrays[boundary['native'][spec.name]]
            need(stored['instrument']==spec.identity and list(tensor.shape)==stored['shape']
                 and sha(tensor.astype('<f4').tobytes())==stored['sha256'],'OBJECTIVE_READER_NATIVE')
    controls=[]
    failed=next(r for r in evidence if r['outcome']==0);bad=copy.deepcopy(failed);bad['outcome']=1
    controls.append(refused('OBJECTIVE_READER_EVIDENCE',lambda:source_evidence(corpus,supervision,bad)))
    bad=copy.deepcopy(failed);bad['decisions']['successful_patch']['allowed']=True
    controls.append(refused('OBJECTIVE_READER_DECISION',lambda:source_evidence(corpus,supervision,bad)))
    for mode,code in (('missing','POPULATION'),('target','TARGET'),('score','SCORE')):
        damaged=copy.deepcopy(rows)
        if mode=='missing':damaged.pop()
        elif mode=='target':damaged[0]['target_sha256']='0'*64
        else:damaged[0]['arms']['bm25']['exact']=not damaged[0]['arms']['bm25']['exact']
        controls.append(refused('OBJECTIVE_READER_'+code,lambda:check_scores(corpus,labels,damaged,predictions,report['summary']['baselines'])))
    need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'OBJECTIVE_READER_JOURNAL')
    result=dict(schema='capability-objective-readback-v1',source=ref,reader_sha256=sha(Path(__file__).read_bytes()),
        source_tasks=len(evidence),purpose_decisions=len(evidence)*len(PURPOSES),baseline_queries=len(rows),baseline_predictions=len(rows)*4,
        native_boundaries=len(boundaries),native_slots=sum(len(b['native']) for b in boundaries),controls=controls,
        acceptance='author-checked',number_status='measured',scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result),report=result)),flush=True)


if __name__=='__main__':main()

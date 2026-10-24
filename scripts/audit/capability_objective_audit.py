"""Evidence-scoped target admission and input-only explicit baseline measurements."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.capability_conditioning_readback import refused
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from model.capability_kernel_learning import KernelLearningCorpus, pinned
from model.capability_baselines import ExplicitBaselines, inputs
from model.capability_supervision import CapabilitySupervision, PURPOSES
from model.contracts import ModelConfig


def load(source):
    config=pinned(source['config']);diagnostic=pinned(source['diagnostic'])
    parent=pinned(diagnostic['plan']['source']);need(parent['source']['config']==source['config'],'OBJECTIVE_CONFIG')
    raw=(Path(source['diagnostic']['path']).parent/'tasks.jsonl').read_bytes()
    need(sha(raw)==diagnostic['tasks_sha256'],'OBJECTIVE_LABELS')
    labels={r['task']:r for r in map(json.loads,raw.splitlines())};corpus=KernelLearningCorpus(config)
    need(set(labels)==set(corpus.train)|set(corpus.query),'OBJECTIVE_LABEL_ROSTER')
    return corpus,labels


def summarize(evidence,baselines):
    result={}
    for split in ('train','query'):
        result[split]={}
        for kind in sorted({r['operation'] for r in evidence}):
            rows=[r for r in evidence if r['split']==split and r['operation']==kind]
            result[split][kind]=dict(tasks=len(rows),groups=len({r['group'] for r in rows}),
                outcomes=dict(Counter(str(r['outcome']) for r in rows)),
                intents={p:sum(r['decisions'][p]['allowed'] for r in rows) for p in PURPOSES})
    comparisons={}
    for kind in sorted({r['operation'] for r in baselines}):
        rows=[r for r in baselines if r['operation']==kind];comparisons[kind]={}
        for name in ('modal','bm25','persistence','last_arithmetic'):
            comparisons[kind][name]=dict(tasks=len(rows),available=sum(r['arms'][name]['status']=='candidate' for r in rows),
                exact=sum(r['arms'][name]['exact'] for r in rows),regions=sum(r['arms'][name]['region_match'] is True for r in rows),
                region_defined=sum(r['arms'][name]['region_match'] is not None for r in rows))
        comparisons[kind]['parent_region_hits']=sum(r['parent_region']==r['region_label'] for r in rows)
    return dict(evidence=result,baselines=comparisons)


def publish_rows(path,rows):
    with path.open('x') as out:
        for row in rows:out.write(json.dumps(row,sort_keys=True)+'\n')
        out.flush();os.fsync(out.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','diagnostic'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'OBJECTIVE_EXISTS');a.output.mkdir()
    start=tick=time.perf_counter();phases={}
    source={n:dict(path=str(getattr(a,n)),pin=getattr(a,n+'_pin')) for n in ('config','diagnostic')}
    source['code']={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in (
        'model/capability_supervision.py','model/capability_baselines.py','audit/capability_objective_audit.py')}
    try:
        corpus,labels=load(source);supervision=CapabilitySupervision(corpus)
        phases['source_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
        rows=[]
        for key in (*corpus.train,*corpus.query):
            evidence=supervision.evidence(key)
            decisions={p:supervision.decide(key,p,fitting=key in corpus.train) for p in PURPOSES}
            rows.append(dict(evidence,decisions=decisions))
        publish_rows(a.output/'evidence.jsonl',rows)
        phases['evidence_decisions']=time.perf_counter()-tick;tick=time.perf_counter()
        baseline=ExplicitBaselines(corpus,labels);records=[];input_bytes=0
        for key in corpus.query:
            context=inputs(corpus,key);input_bytes+=sum(len(f['text'].encode()) for f in context['fields'])
            predictions=baseline.select(context);task=corpus.tasks[key];pop=corpus.sources[task['population']].population
            truth=pop.fields[task['selections']['targets']['fields'][0]['id']]['text'];old=labels[key]
            arms={}
            for name,prediction in predictions.items():
                available=prediction['status']=='candidate';region=prediction.get('region')
                arms[name]=dict({k:v for k,v in prediction.items() if k!='text'},
                    text_sha256=sha(prediction['text'].encode()) if available else None,
                    exact=available and prediction['text']==truth,region_match=(region==old['recognition']['label']) if region is not None else None)
            records.append(dict(task=key,operation=task['kind'],family=task['family'],group=task['split_group'],
                target_sha256=sha(truth.encode()),region_label=old['recognition']['label'],parent_region=old['recognition']['selected'],arms=arms))
        publish_rows(a.output/'baselines.jsonl',records)
        phases['fit_and_query_baselines']=time.perf_counter()-tick;tick=time.perf_counter()
        native={};boundaries=[]
        for purpose in PURPOSES[:-1]:
            key=next(r['task'] for r in rows if r['split']=='train' and r['decisions'][purpose]['allowed'])
            decision,raw,slots=supervision.training_target(key,purpose,ModelConfig(),'cpu');names={}
            for name,slot in slots.items():
                cell=str(len(boundaries))+'__'+name;native[cell]=slot.values[0,~slot.padding[0]].numpy();names[name]=cell
            boundaries.append(dict(task=key,purpose=purpose,decision=decision,source_sha256=sha(raw),native=names))
        np.savez(a.output/'native-targets.npz',**native);write_committed(a.output/'native-targets.json',boundaries)
        failed=next(r for r in rows if r['split']=='train' and r['outcome']==0)
        controls=[refused('SUPERVISION_FAILED_ACTION',lambda:supervision.training_target(failed['task'],'successful_patch',ModelConfig(),'cpu')),
            refused('SUPERVISION_FIT_QUERY',lambda:supervision.training_target(corpus.query[0],'recorded_continuation',ModelConfig(),'cpu')),
            refused('SUPERVISION_SCOPE_UNESTABLISHED',lambda:supervision.training_target(corpus.train[0],'grounded_correct_response',ModelConfig(),'cpu'))]
        witness=failed['observations'][-1];address=witness['population'],witness['condition'];saved=supervision.executions.pop(address)
        try:controls.append(refused('SUPERVISION_MISSING_EXECUTION',lambda:supervision.evidence(failed['task'])))
        finally:supervision.executions[address]=saved
        key=corpus.query[0];expected=baseline.select(inputs(corpus,key));selection=corpus.tasks[key]['selections']['targets']
        corpus.tasks[key]['selections']['targets']={'fields':[{'id':'forbidden-query-target'}]}
        try:need(baseline.select(inputs(corpus,key))==expected,'OBJECTIVE_QUERY_TARGET_LEAK')
        finally:corpus.tasks[key]['selections']['targets']=selection
        controls.append(dict(query_target_removed_prediction_unchanged=True))
        phases['native_boundary_and_controls']=time.perf_counter()-tick;tick=time.perf_counter()
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'OBJECTIVE_JOURNAL')
        files={n:sha((a.output/n).read_bytes()) for n in ('evidence.jsonl','baselines.jsonl','native-targets.npz','native-targets.json')}
        report=dict(schema='capability-objective-audit-v1',source=source,files=files,train=list(corpus.train),query=list(corpus.query),
            summary=summarize(rows,records),controls=controls,posting_visits=baseline.visits,query_context_bytes=input_bytes,
            fitting='TRAIN-only; no model fit',query_exposure='development_exposed_group_holdout',number_status='measured',
            acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
        phases['final_readback']=time.perf_counter()-tick;wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'OBJECTIVE_PHASE_COVERAGE');report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),summary=report['summary'])),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':main()

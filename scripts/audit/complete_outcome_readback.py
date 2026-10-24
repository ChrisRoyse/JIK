"""Cold native-slice reconstruction, complete optimizer replay and independent metric recount."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import complete_outcome_source as source
from claims.process_conditions import write_committed
from model.complete_answer_outcome import OutcomeInputs,make,update,predict,validate,ARMS
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def recount(value,logits,saved):
    for split,lanes in logits.items():
        keys=[k for k,r in value['rows'].items() if r['split']==split];y=np.array([value['rows'][k]['outcome'] for k in keys])
        values={n:x.numpy().astype('float64') for n,x in lanes.items()}
        values['combined']=np.stack(list(values.values()),axis=1).mean(axis=1)
        for name,x in values.items():
            p=1/(1+np.exp(-x));b=(p-y)**2;loss=np.logaddexp(0,x)-x*y;correct=(x>0)==y.astype(bool)
            row=saved[split][name]
            need(row['n']==len(keys) and row['correct']==int(correct.sum()) and abs(row['brier']-b.mean())<1e-12
                 and abs(row['log_loss']-loss.mean())<1e-12,'OUTCOME_COLD_METRICS')
            expected_groups=sorted({value['rows'][k]['group'] for k in keys})
            need(sorted(row['groups'])==expected_groups,'OUTCOME_COLD_GROUPS')
            for g in expected_groups:
                ids=[i for i,k in enumerate(keys) if value['rows'][k]['group']==g];r=row['groups'][g]
                need(r['n']==len(ids) and r['correct']==int(correct[ids].sum()) and abs(r['brier']-b[ids].mean())<1e-12
                     and abs(r['log_loss']-loss[ids].mean())<1e-12,'OUTCOME_COLD_GROUP_METRICS')
            need(abs(row['equal_group_brier']-np.mean([r['brier'] for r in row['groups'].values()]))<1e-12,
                 'OUTCOME_COLD_GROUP_MEAN')


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'OUTCOME_COLD_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['schema']=='complete-answer-outcome-probe-v1','OUTCOME_COLD_REPORT')
    for n,h in r['files'].items():need(file_sha(a.input/n)==h,'OUTCOME_COLD_PAYLOAD')
    s=source.load(a.root);v=torch.load(a.input/'inputs.pt',map_location='cpu',weights_only=True)
    need(s['code']==r['code'] and s['inputs']==r['inputs'] and fingerprint(v)==fingerprint(s['value']), 'OUTCOME_COLD_SOURCE')
    data=OutcomeInputs(v['rows'],v['fields'],v['roster']);banks={};windows=0
    # Independent offset slicing, without the producer's field assembler.
    for field,record in v['field_records'].items():
        for name,slot in record['slots'].items():
            parts=[]
            for part in slot['description']['parts']:
                origin=part['origin'];key=origin,name
                if key not in banks:
                    with np.load(Path(v['roots'][origin])/name/'slots.npz',allow_pickle=False) as z:
                        banks[key]=(z['vectors'].copy(),z['offsets'].copy())
                vectors,offsets=banks[key];i=part['row'];lo,hi=map(int,offsets[i:i+2])
                need([lo,hi]==part['source_window_span'],'OUTCOME_COLD_NATIVE_SPAN');parts.append(vectors[lo:hi])
            joined=np.concatenate(parts);need(np.array_equal(joined,v['fields'][field][name].numpy()),'OUTCOME_COLD_NATIVE_VALUES')
            windows+=len(joined)
    del banks
    decisions=json.loads((a.root/source.REF[0]/'decisions.json').read_bytes());lookup={(d['task'],d['purpose']):d for d in decisions}
    for key,route in v['route'].items():
        need(route['generation']==lookup[key,route['generation']['purpose']],'OUTCOME_COLD_ROUTE')
        if route['scope']=='retained_execution_test_scope':
            need(route['generation']['allowed']==(route['outcome']==1),'OUTCOME_COLD_FAILED_IMITATION')
    need(r['counts']['native_bytes']==sum(x.numel()*x.element_size() for f in v['fields'].values() for x in f.values())
         and r['counts']['windows']==windows and r['counts']['routes']==len(v['route'])
         and r['counts']['unique_fields']==len(v['fields']) and r['counts']['excluded_prior_fields']==len(v['excluded'])
         and r['counts']['train']==len(data.train) and r['counts']['query']==len(data.query), 'OUTCOME_COLD_COUNTS')
    phases['source_native_routes']=time.perf_counter()-tick;tick=time.perf_counter();checks=[]
    need({(x['seed'],x['arm']) for x in r['runs']}=={(seed,arm) for seed in range(3) for arm in ARMS}
         and len(r['runs'])==12,'OUTCOME_COLD_RUN_COVERAGE')
    for result in r['runs']:
        seed,arm=result['seed'],result['arm'];path=a.input/(str(seed)+'-'+arm)
        def load(n):return torch.load(path/n,map_location='cpu',weights_only=True)
        trace=load('trace.pt');model,opt=make(v['dimensions'],seed)
        need(fingerprint(model.state_dict())==fingerprint(load('initial.pt'))==result['initial'],'OUTCOME_COLD_INITIAL')
        before=predict(model,data,arm);need(fingerprint(before)==fingerprint(trace['before']),'OUTCOME_COLD_BEFORE')
        recount(v,before,result['before']);need(len(trace['ledger'])==20,'OUTCOME_COLD_STEPS')
        for step in range(20):
            row=update(model,opt,data,arm,step);need(fingerprint(row)==fingerprint(trace['ledger'][step]),'OUTCOME_COLD_UPDATE')
            if step==9:
                saved=load('mid.pt');need(saved['step']==10 and fingerprint(saved['model'])==fingerprint(model.state_dict())
                    and fingerprint(saved['optimizer'])==fingerprint(opt.state_dict()),'OUTCOME_COLD_MID')
                # Reconstruct modules and reload both weights and moments before continuing.
                model,opt=make(v['dimensions'],seed);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
                validate(model,opt,10)
        final=load('final.pt');need(final['step']==20 and fingerprint(model.state_dict())==fingerprint(final['model'])==result['final']
            and fingerprint(opt.state_dict())==fingerprint(final['optimizer']),'OUTCOME_COLD_FINAL')
        after=predict(model,data,arm);need(fingerprint(after)==fingerprint(trace['after']),'OUTCOME_COLD_AFTER')
        recount(v,after,result['after']);checks.append(dict(seed=seed,arm=arm,updates=20,exact=True,resumed_moments=True))
        print(json.dumps(checks[-1]),flush=True)
    phases['all_optimizer_replay_and_metrics']=time.perf_counter()-tick;tick=time.perf_counter()
    train=[row['outcome'] for row in v['rows'].values() if row['split']=='train'];rate=sum(train)/len(train)
    for split,arms in r['baselines'].items():
        rows=[row for row in v['rows'].values() if row['split']==split]
        for name,p in (('always_failure',0.),('train_base_rate',rate)):
            y=np.array([row['outcome'] for row in rows]);out=arms[name]
            need(out['probability']==p and out['n']==len(rows) and out['correct']==int(((p>0.5)==y.astype(bool)).sum())
                 and abs(out['brier']-np.mean((y-p)**2))<1e-12,'OUTCOME_COLD_BASELINE')
            need(set(out['groups'])=={row['group'] for row in rows},'OUTCOME_COLD_BASELINE_GROUPS')
            for g,cell in out['groups'].items():
                group=np.array([row['outcome'] for row in rows if row['group']==g])
                need(cell['n']==len(group) and abs(cell['brier']-np.mean((group-p)**2))<1e-12,'OUTCOME_COLD_BASELINE_GROUP')
    data.fresh(full=True);s['fresh']();phases['baselines_final_sources']=time.perf_counter()-tick
    elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'OUTCOME_COLD_PHASES')
    report=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),code=s['code'],checks=checks,
        all_updates_exact=True,updates=240,native_windows=windows,route_decisions=len(v['route']),
        all_metrics_independently_recounted=True,phases=phases,seconds=elapsed,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,new_executions=0,
        scientific_admission=False,serving=False,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=elapsed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

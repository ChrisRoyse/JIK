"""Cold replay and independent source/loss recount for masked answer training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import conditioned_answer as producer
from audit import legal_answer_sequence_readback as original
from claims.process_conditions import write_committed
from model.capability_answer_targets import AnswerTargets
from model.native_conditioned_answer import document_item, update
from model.native_population_training import fingerprint
from text.constellation_corpus import digest,file_sha,need


def recount_schedule(rows,plan):
    report=json.loads((Path(plan['path'])/'report.json').read_bytes())
    source=AnswerTargets(report['source']['path'],report['source']['pin'])
    need({r['task'] for r in rows}==set(source.rows),'CONDITIONED_READBACK_TASKS')
    counts=dict(tasks=0,contexts=0,targets=0,decisions=0,ends=0)
    for saved in rows:
        key=saved['task'];pop,task,side=source.source.selected(key);row=source.rows[key]
        request=saved['request'];private=saved['private'];context=task['context'];atoms=row['answer']['atoms']
        need(saved['split']==side and saved['family']==pop.family and saved['group']==task['split_group'],
             'CONDITIONED_READBACK_PARTITION')
        need(set(request)=={'operation','fields','position_kind'} and request['operation']==task['kind']
             and request['position_kind']=='request_field_layout_not_event_chronology', 'CONDITIONED_READBACK_PUBLIC_SCOPE')
        need(request['fields']==[dict(field=f,input=pop.fields[f]['text_sha256'],role=pop.fields[f]['role'],
                                     clock=pop.fields[f]['clock']) for f in context], 'CONDITIONED_READBACK_CONTEXT')
        need(private['targets']==[a['measurement'] for a in atoms] and private['end']==row['answer']['end'],
             'CONDITIONED_READBACK_TARGETS')
        expected=[]
        for i in range(len(atoms)):
            expected.append(dict(position=len(context)-1+i,answer_index=i,end=False))
        expected.append(dict(position=len(context)+len(atoms)-1,answer_index=None,end=True))
        need(private['decisions']==expected,'CONDITIONED_READBACK_DECISIONS')
        counts['tasks']+=1;counts['contexts']+=len(context);counts['targets']+=len(atoms)
        counts['decisions']+=len(expected);counts['ends']+=1
    source.source.fresh();return counts


def recount_losses(record,original_item):
    c=2;a=len(original_item['frames'])-c;maximum=0.
    need(record['context_positions']==c and record['answer_targets']==a and record['decisions']==a+1
         and record['end_labels']==1 and record['prompt_transition_labels']==0
         and len(record['records'])==a+1,'CONDITIONED_READBACK_LOSS_COVERAGE')
    for i,row in enumerate(record['records']):
        need(row['position']==c-1+i and row['end']==(i==a) and row['answer_index']==(i if i<a else None),
             'CONDITIONED_READBACK_LOSS_POSITION')
        losses=[]
        for value in row['stop']['proposals'].values():
            z=value['logits'].numpy().astype(np.float64)[0];z=z-z.max()
            loss=float(np.log(np.exp(z).sum())-z[int(i==a)]);losses.append(loss)
            maximum=max(maximum,abs(loss-float(value['loss'])))
        stop=sum(losses)/len(losses)
        maximum=max(maximum,abs(stop-float(row['stop']['loss'])))
        expected=stop/(a+1)+(row['continuation']/a if i<a else 0.)
        need((row['continuation'] is None)==(i==a),'CONDITIONED_READBACK_FINAL_LABEL')
        maximum=max(maximum,abs(expected-row['weighted_loss']))
    need(maximum<=1e-10,'CONDITIONED_READBACK_LOSS');return maximum


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter()
    report=producer.sequence.runtime.pinned(a.input/'report.json',a.pin)
    for n,h in report['files'].items():need(file_sha(a.input/n)==h,'CONDITIONED_READBACK_FILE')
    rows=json.loads((a.input/'schedules.json').read_bytes());counts=recount_schedule(rows,report['plan'])
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source,bundle,data=producer.sequence.prepare(producer.ROOT);need(producer.code(source)==report['code'],'CONDITIONED_READBACK_CODE')
    target=data['labels'].model;fitting=[i for i in data['items'] if i['side']=='train']
    lookup={i['document']:i for i in data['items']};policy=report['policy'];summary={}
    results=json.loads((a.input/'results.json').read_bytes());phases=dict(authentication=time.perf_counter()-tick)
    for architecture in producer.sequence.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();dest=a.input/architecture
        need(producer.compatibility(bundle,target,architecture,fitting[0])==results[architecture]['compatibility'],
             'CONDITIONED_READBACK_COMPATIBILITY')
        model,optimizer=producer.sequence.make(bundle,target,architecture)
        initial=torch.load(dest/'initial.pt',weights_only=True,map_location='cpu')
        need(fingerprint(initial)==fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict())),
             'CONDITIONED_READBACK_INITIAL')
        max_loss=0.
        for step,item in enumerate(fitting):
            saved=torch.load(dest/f'update-{step+1}.pt',weights_only=True,map_location='cpu')
            max_loss=max(max_loss,recount_losses(saved['record'],item))
            record,gradients=update(model,optimizer,document_item(item,2),step,policy,target=target)
            need(fingerprint(saved)==fingerprint(dict(record=record,gradients=gradients,
                 model=model.state_dict(),optimizer=optimizer.state_dict())),'CONDITIONED_READBACK_UPDATE')
        phases[architecture+'_cold_update']=time.perf_counter()-tick;tick=time.perf_counter()
        payload=torch.load(dest/'evaluation.pt',weights_only=True,map_location='cpu');metric_error=0.;events=0
        for saved in payload:
            row=saved['row'];item=lookup[row['document']];contexts,donor=producer.sequence.contexts(data['items'],item)
            need(row['donor_document']==donor,'CONDITIONED_READBACK_DONOR');prefix=contexts[row['mode']]
            ref=original.reference(model,target,prefix,row['proposal'],row['seed'],row['observation_decisions'])
            need(fingerprint(ref)==fingerprint(saved['events']),'CONDITIONED_READBACK_ROLLOUT')
            metric_error=max(metric_error,original.recount(row,ref,item,prefix,model.roster));events+=len(ref)
        need(fingerprint((model.state_dict(),optimizer.state_dict()))==results[architecture]['final'],
             'CONDITIONED_READBACK_FINAL')
        summary[architecture]=dict(updates=len(fitting),trajectories=len(payload),events=events,
            cold_updates_exact=True,independent_rollouts_exact=True,max_loss_error=max_loss,metric_error=metric_error)
        phases[architecture+'_rollout']=time.perf_counter()-tick
        print(json.dumps(dict(architecture=architecture,**summary[architecture])),flush=True)
    tick=time.perf_counter();data['close']();need(fingerprint(target.state_dict())==report['frozen_target'],'CONDITIONED_READBACK_FROZEN')
    phases['close']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'CONDITIONED_READBACK_PHASE_COVERAGE')
    pin=write_committed(a.output/'report.json',dict(complete=True,input_pin=a.pin,code=report['code'],counts=counts,
        summary=summary,phases=phases,seconds=seconds,all_source_schedules_exact=True,all_cold_updates_exact=True,
        all_independent_rollouts_exact=True,capability_native_fit=False,gpu_qualified=False,
        learned_capability=False,scientific_admission=False,acceptance='author-checked'))
    print(json.dumps(dict(pin=pin,counts=counts,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise

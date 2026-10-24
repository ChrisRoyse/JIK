"""Cold optimizer replay and full-history reference rollout for answer sequences."""
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
from audit import legal_answer_sequence as producer
from claims.process_conditions import write_committed
from model.native_answer_sequence import update_answer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


@torch.no_grad()
def reference(model,target,prefix,proposal,seed,decisions):
    # Deliberately encode all six histories. The production whole-proposal path
    # encodes only its source branch; equality checks that optimization too.
    stream=model.forecaster.predictor.stream();state=None;offset=0;out=[];generated=0
    for frame in prefix:
        hidden,state=stream.step((frame,),context='independent-answer',offset=offset,state=state,detach_history=True)
        offset+=1
    rng=torch.Generator().manual_seed(seed)
    for _ in range(decisions):
        head,context=model.forecaster.contexts(hidden)[proposal]
        logits=torch.stack([model.ending[proposal][name](context[:,i]) for i,name in enumerate(model.roster)]).mean(0).cpu()
        if int(logits.argmax(-1))==1:
            out.append(dict(kind='end',proposal=proposal,generated=generated,stop_logits=logits.clone(),validity='unassessed'));break
        forecast=head.forecast(context);placement=model.forecaster.placement[proposal]
        noise=torch.randn(1,1,placement.noise_width,generator=rng)
        samples=placement(context,forecast['codes'],forecast['counts'],noise,target=target)[0]
        frame={name:value[0].detach() for name,value in samples.items()}
        hidden,state=stream.step((frame,),context='independent-answer',offset=offset,state=state,detach_history=True);offset+=1
        out.append(dict(kind='constellation',proposal=proposal,index=generated,slots=cpu(frame),codes=cpu(forecast['codes']),
                        counts=cpu(forecast['counts']),stop_logits=logits.clone(),feedback_index=offset-1,
                        source='predicted',validity='unassessed'));generated+=1
    return out


def recount(row,events,item,prefix,roster):
    future=item['frames'][2:];m=row['metrics'];generated=[e for e in events if e['kind']=='constellation']
    ended=events[-1]['kind']=='end';covered=min(len(future),len(generated))
    need(m['expected_atoms']==len(future) and m['generated_atoms']==len(generated) and m['ended']==ended
         and m['ended_at_reference_boundary']==(ended and len(generated)==len(future))
         and m['censored']==(not ended) and m['missing']==len(future)-covered
         and m['extra']==max(0,len(generated)-len(future)), 'ANSWER_READBACK_DENOMINATOR')
    maximum=0.
    for i,event in enumerate(generated):
        need(event['index']==i and event['feedback_index']==2+i and event['source']=='predicted'
             and set(event['slots'])==set(roster) and event['validity']=='unassessed','ANSWER_READBACK_SEQUENCE')
    for name in roster:
        for i,f in enumerate(future):
            for label,x in [('per_slot',generated[i]['slots'][name] if i<covered else None),('copy_current',prefix[-1][name])]:
                if x is None:value=-1.
                else:
                    a=x.numpy().astype(np.float64);b=f[name].numpy().astype(np.float64)
                    a/=np.sqrt(np.square(a).sum(1))[:,None];b/=np.sqrt(np.square(b).sum(1))[:,None]
                    scores=np.clip(np.einsum('ik,jk->ij',a,b,optimize=False),-1,1)
                    value=min(sum(max(v) for v in scores)/len(scores),sum(max(v) for v in scores.T)/scores.shape[1])
                error=abs(value-m[label][name][i]);maximum=max(maximum,error)
                need(error<=1e-12,'ANSWER_READBACK_NATIVE_METRIC')
    return maximum


def stop_recount(record):
    maximum=0.;weighted=0.;decisions=record['decisions'];total=record['transitions']
    need(decisions==total+1 and record['end_labels']==1 and len(record['records'])==decisions,'ANSWER_READBACK_TRAIN_COVERAGE')
    for i,row in enumerate(record['records']):
        need(row['index']==i and row['end']==(i==total),'ANSWER_READBACK_END_SUPERVISION')
        losses=[]
        for value in row['stop']['proposals'].values():
            logits=value['logits'].numpy().astype(np.float64)[0];z=logits-logits.max()
            loss=float(np.log(np.exp(z).sum())-z[int(row['end'])]);losses.append(loss)
            maximum=max(maximum,abs(loss-float(value['loss'])))
        stop=sum(losses)/len(losses);maximum=max(maximum,abs(stop-float(row['stop']['loss'])))
        total_loss=stop/decisions+(row['continuation']/total if i<total else 0.)
        maximum=max(maximum,abs(total_loss-row['weighted_loss']));weighted+=total_loss
    need(maximum<=1e-10,'ANSWER_READBACK_STOP_LOSS')
    return maximum,weighted


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    report=producer.runtime.pinned(a.input/'report.json',a.input_pin)
    for n,pin in report['files'].items():need(file_sha(a.input/n)==pin,'ANSWER_READBACK_FILE')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source,bundle,data=producer.prepare(a.root);need(report['code']==producer.code(source),'ANSWER_READBACK_CODE')
    target=data['labels'].model;items=data['items'];fitting=[i for i in items if i['side']=='train']
    lookup={i['document']:i for i in items};policy=dict(bundle['policy'],device='cpu')
    results=json.loads((a.input/'results.json').read_bytes());summary={};phases={'authentication':time.perf_counter()-tick}
    for architecture in producer.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();directory=a.input/architecture;model,optimizer=producer.make(bundle,target,architecture)
        initial=torch.load(directory/'initial.pt',weights_only=True,map_location='cpu')
        need(fingerprint(initial)==fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict())), 'ANSWER_READBACK_INITIAL')
        loss_error=0.
        for step,item in enumerate(fitting):
            saved=torch.load(directory/f'update-{step+1}.pt',weights_only=True,map_location='cpu')
            error,_=stop_recount(saved['record']);loss_error=max(loss_error,error)
            record,gradients=update_answer(model,optimizer,item,step,policy,target=target)
            need(fingerprint(dict(record=record,gradients=gradients,model=model.state_dict(),optimizer=optimizer.state_dict()))
                 ==fingerprint(saved),'ANSWER_READBACK_COLD_UPDATE')
        phases[architecture+'_cold_training']=time.perf_counter()-tick;tick=time.perf_counter()
        payload=torch.load(directory/'evaluation.pt',weights_only=True,map_location='cpu');metric_error=0.;rows=0;events_count=0
        for saved in payload:
            row=saved['row'];item=lookup[row['document']];prefixes,donor=producer.contexts(items,item)
            prefix=prefixes[row['mode']];need(row['donor_document']==donor,'ANSWER_READBACK_DONOR')
            fresh=producer.observe(model,target,prefix,row['proposal'],row['seed'],row['observation_decisions'])
            independent=reference(model,target,prefix,row['proposal'],row['seed'],row['observation_decisions'])
            need(fingerprint(fresh)==fingerprint(independent)==fingerprint(saved['events']),'ANSWER_READBACK_FEEDBACK_EXACT')
            metric_error=max(metric_error,recount(row,independent,item,prefix,model.roster));rows+=1;events_count+=len(independent)
        need([v['row'] for v in payload]==results[architecture]['rows'],'ANSWER_READBACK_RESULT_ROWS')
        summary[architecture]=dict(updates=len(fitting),trajectories=rows,events=events_count,stop_loss_error=loss_error,
                                   native_metric_error=metric_error,cold_state_exact=True,independent_full_history_exact=True)
        phases[architecture+'_rollout']=time.perf_counter()-tick
        print(json.dumps(dict(architecture=architecture,**summary[architecture])),flush=True)
    tick=time.perf_counter();data['close']();need(fingerprint(target.state_dict())==report['frozen_target'],'ANSWER_READBACK_FROZEN')
    phases['close']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'ANSWER_READBACK_PHASE_COVERAGE')
    pin=write_committed(a.output/'report.json',dict(complete=True,input_pin=a.input_pin,code=report['code'],summary=summary,
        phases=phases,seconds=seconds,all_cold_updates_exact=True,all_independent_rollouts_exact=True,
        target_semantics=report['target_semantics'],scientific_admission=False,learned_capability=False,gpu_qualified=False,
        acceptance='author-checked',number_status='measured'))
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

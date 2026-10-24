"""Compare autonomous continuation with supplied true history on the same checkpoints."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_sequence as producer
from audit import legal_answer_sequence_readback as recount
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


@torch.no_grad()
def teacher(model,target,item,proposal,seed):
    stream=model.forecaster.predictor.stream();state=None;events=[];ending=[]
    rng=torch.Generator().manual_seed(seed)
    for i,frame in enumerate(item['frames']):
        hidden,state=stream.step((frame,),context='teacher-reference',offset=i,state=state,detach_history=True)
        if i<1:continue
        head,context=model.forecaster.contexts(hidden)[proposal]
        stop=model.stop_logits(context,proposal).detach().cpu();ending.append(dict(after_atom=i,logits=stop,end_label=i+1==len(item['frames'])))
        if i+1==len(item['frames']):break
        forecast=head.forecast(context);placement=model.forecaster.placement[proposal]
        noise=torch.randn(1,1,placement.noise_width,generator=rng)
        sample=placement(context,forecast['codes'],forecast['counts'],noise,target=target)[0]
        events.append(dict(kind='constellation',proposal=proposal,index=i-1,feedback_index=i+1,
                           slots=cpu({n:x[0] for n,x in sample.items()}),codes=cpu(forecast['codes']),counts=cpu(forecast['counts']),
                           stop_logits=stop,source='predicted',validity='unassessed'))
    return dict(events=events,ending=ending,history='true_source_history_at_each_step',autonomous=False)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    r=producer.runtime.pinned(a.input/'report.json',a.input_pin)
    for n,pin in r['files'].items():need(file_sha(a.input/n)==pin,'ANSWER_CONTEXT_INPUT_BYTES')
    cold=producer.runtime.pinned(a.readback/'report.json',a.readback_pin)
    need(cold['complete'] and cold['input_pin']==a.input_pin and cold['all_independent_rollouts_exact'],'ANSWER_CONTEXT_COLD_REQUIRED')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source,bundle,data=producer.prepare(a.root);need(r['code']==producer.code(source),'ANSWER_CONTEXT_CODE')
    target=data['labels'].model;lookup={i['document']:i for i in data['items']};results={};phases={'authentication':time.perf_counter()-tick}
    for architecture in producer.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();model,optimizer=producer.make(bundle,target,architecture)
        saved=torch.load(a.input/architecture/'update-5.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        producer.validate_optimizer(model,optimizer,r['policy'],5)
        immutable=fingerprint((model.state_dict(),optimizer.state_dict(),target.state_dict()))
        original=torch.load(a.input/architecture/'evaluation.pt',map_location='cpu',weights_only=True)
        selected=[v for v in original if v['row']['mode']=='correct'];payload=[];rows=[];max_error=0.
        for entry in selected:
            row=entry['row'];item=lookup[row['document']];value=teacher(model,target,item,row['proposal'],row['seed'])
            metric=producer.metrics(value['events'],item['frames'][2:],model.roster,item['frames'][1])
            descriptor=dict(row,metrics=metric,mode='teacher_forced_diagnostic')
            max_error=max(max_error,recount.recount(descriptor,value['events'],item,tuple(item['frames'][:2]),model.roster))
            positions=len(item['frames'])-2
            paired={n:dict(teacher=sum(metric['per_slot'][n])/positions,
                           autonomous=sum(row['metrics']['per_slot'][n])/positions,
                           copy_current=sum(metric['copy_current'][n])/positions)
                    for n in model.roster}
            order=next(v for v in original if v['row']['document']==row['document'] and v['row']['proposal']==row['proposal'] and v['row']['mode']=='reversed')
            rows.append(dict(document=row['document'],proposal=row['proposal'],side=row['side'],component=row['component'],positions=positions,
                              per_slot=paired,reversing_prefix_changed_output=fingerprint(entry['events'])!=fingerprint(order['events']),
                              teacher_end_hits=sum(bool(v['logits'].argmax(-1).item()) for v in value['ending'] if v['end_label']),
                              teacher_nonend_false_stops=sum(bool(v['logits'].argmax(-1).item()) for v in value['ending'] if not v['end_label'])))
            payload.append(dict(row=descriptor,**value))
        ref=producer.runtime.persist(a.output/(architecture+'.pt'),payload)
        stored=torch.load(a.output/(architecture+'.pt'),map_location='cpu',weights_only=True)
        for entry in stored:
            row=entry['row'];item=lookup[row['document']]
            max_error=max(max_error,recount.recount(row,entry['events'],item,tuple(item['frames'][:2]),model.roster))
            fresh=teacher(model,target,item,row['proposal'],row['seed'])
            need(fingerprint(fresh)==fingerprint({k:entry[k] for k in fresh}),'ANSWER_CONTEXT_REPLAY')
        need(fingerprint((model.state_dict(),optimizer.state_dict(),target.state_dict()))==immutable,'ANSWER_CONTEXT_MUTATION')
        # Controlled model perturbations exercise end/refusal handling. They are
        # not learned stopping results and never replace the measured failures.
        example=lookup[selected[0]['row']['document']];proposal=model.proposals[0]
        forced=deepcopy(model)
        with torch.no_grad():
            for head in forced.ending[proposal].values():
                head.weight.zero_();head.bias.copy_(torch.tensor([0.,1.]))
        end=producer.observe(forced,target,tuple(example['frames'][:2]),proposal,1,2)
        ending_metric=producer.metrics(end,example['frames'][2:],model.roster,example['frames'][1])
        need(len(end)==1 and end[0]['kind']=='end' and ending_metric['missing']==len(example['frames'])-2
             and not ending_metric['ended_at_reference_boundary'],'ANSWER_CONTEXT_EARLY_END_CONTROL')
        cramped=deepcopy(model);cramped.forecaster.predictor.cache_bytes=1
        try:producer.observe(cramped,target,tuple(example['frames'][:2]),proposal,1,2)
        except (ValueError,RuntimeError) as error:
            refusal=str(error)
            expected=dict(shared='PREFIX_STREAM_CACHE_MEMORY',own_slot='BRANCH_CACHE_ALLOWANCE',
                          whole_proposals='PREFIX_STREAM_MEMORY_POLICY')[architecture]
            need(refusal.endswith(expected),'ANSWER_CONTEXT_MEMORY_REASON')
        else:raise ValueError('ANSWER_CONTEXT_MEMORY_ACCEPTED')
        need(fingerprint((model.state_dict(),optimizer.state_dict(),target.state_dict()))==immutable,'ANSWER_CONTEXT_CONTROL_MUTATION')
        results[architecture]=dict(payload=ref,rows=rows,metric_error=max_error,replayed_exact=True,
            controls=dict(controlled_premature_end=ending_metric,memory_refusal=refusal,
                          controlled_end_is_not_learned_success=True,real_model_unchanged=True))
        phases[architecture]=time.perf_counter()-tick
        print(json.dumps(dict(architecture=architecture,paired_trajectories=len(rows),order_changes=sum(v['reversing_prefix_changed_output'] for v in rows))),flush=True)
    tick=time.perf_counter();data['close']();result_pin=write_committed(a.output/'results.json',results)
    pins=dict(r['code'],**{'audit/legal_answer_context_diagnostic.py':file_sha(Path(__file__))})
    for name,pin in pins.items():
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(p)==pin,'ANSWER_CONTEXT_SOURCE_COPY')
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'ANSWER_CONTEXT_PHASES')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    pin=write_committed(a.output/'report.json',dict(complete=True,input_pin=a.input_pin,readback_pin=a.readback_pin,
        code=pins,files=files,results_pin=result_pin,phases=phases,seconds=seconds,additional_updates=0,
        target_semantics=r['target_semantics'],teacher_has_true_history=True,teacher_is_not_a_generated_answer=True,
        scientific_admission=False,learned_capability=False,acceptance='author-checked',number_status='measured'))
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--readback',type=Path,required=True);p.add_argument('--readback-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

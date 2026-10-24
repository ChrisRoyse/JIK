"""Independent region geometry, placement and surface readback for the frozen diagnostic."""
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import independent_score, refused
from audit.capability_region_diagnostic import load, CONDITIONS
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict


def compare(values):
    return dict(mean=math.fsum(values)/len(values),better=sum(x<0 for x in values),
                worse=sum(x>0 for x in values),ties=sum(x==0 for x in values))


def validate(corpus, rows, report, prior):
    expected = [('train',k) for k in corpus.train]+[('query',k) for k in corpus.query]
    need([(r['split'],r['task']) for r in rows] == expected, 'REGION_READER_POPULATION')
    for row in rows:
        task=corpus.tasks[row['task']]; field=task['selections']['targets']['fields'][0]['id']
        raw=corpus.sources[task['population']].population.fields[field]['text'].encode()
        need(row['family']==task['family'] and row['group']==task['split_group']
             and row['source_sha256']==sha(raw) and row['source_bytes']==len(raw), 'REGION_READER_TARGET')
        observed=row['recognition']; assignments=[]; support=None
        need(set(observed['instruments'])=={s.name for s in corpus.specs}, 'REGION_READER_ROSTER')
        for spec in corpus.specs:
            cell=observed['instruments'][spec.name]
            a=torch.tensor(cell['assignments'],device='cuda'); logits=torch.tensor(cell['logits'],device='cuda')
            need(a.shape==logits.shape==(len(corpus.regions),) and bool(torch.isfinite(a).all())
                 and bool(torch.isfinite(logits).all()), 'REGION_READER_ARRAY')
            assignments.append(a); logp=logits.log_softmax(-1); p=logits.softmax(-1)
            support=logp if support is None else support+logp
            label=observed['label']
            need(0<=label<len(corpus.regions), 'REGION_READER_LABEL')
            expected_metrics=dict(target_probability=float(p[label]),entropy_nats=float(-(p*logp).sum()),
                target_rank=1+int((logits>logits[label]).sum()),logit_span=float(logits.max()-logits.min()))
            # Reproduce the declared CUDA float32 arithmetic, including ties.
            for name,value in expected_metrics.items():
                need(value==cell[name], 'REGION_READER_DISTRIBUTION')
        joint=torch.stack(assignments).mean(0); ordered=joint.sort(descending=True).values
        need(int(joint.argmax())==label and int(support.argmax())==observed['selected']
             and int((joint==joint[label]).sum())==observed['target_ties']
             and float(ordered[0]-ordered[1])==observed['target_margin'], 'REGION_READER_LABEL')
        if row['split']=='query':
            need(set(row['scores'])==set(CONDITIONS), 'REGION_READER_CONDITIONS')
            for name,cell in row['scores'].items():
                need(cell['positions']==len(raw)+1 and math.isfinite(cell['loss']), 'REGION_READER_DENOMINATOR')
            need(all(row['scores'][n]==prior[row['task']]['scores'][n] for n in ('predicted','target')), 'REGION_READER_PRIOR')
    cells=0
    for split in ('train','query'):
        for family in ('coding','conversation','reasoning'):
            selected=[r for r in rows if r['split']==split and r['family']==family]; cell=report['families'][split][family]
            need(cell['tasks']==len(selected) and cell['groups']==len({r['group'] for r in selected})
                 and cell['region_hits']==sum(r['recognition']['label']==r['recognition']['selected'] for r in selected)
                 and cell['selected_regions']==dict(Counter(str(r['recognition']['selected']) for r in selected))
                 and cell['target_regions']==dict(Counter(str(r['recognition']['label']) for r in selected)), 'REGION_READER_SUMMARY')
            for name,metrics in cell['instruments'].items():
                for metric in ('target_probability','entropy_nats','target_rank','logit_span'):
                    need(metrics[metric]==math.fsum(r['recognition']['instruments'][name][metric] for r in selected)/len(selected),
                         'REGION_READER_SUMMARY'); cells+=1
                if split=='query':
                    for condition,value in metrics['coverage_cosine'].items():
                        need(value==math.fsum(r['lanes'][name]['conditions'][condition]['coverage_cosine'] for r in selected)/len(selected),
                             'REGION_READER_SUMMARY'); cells+=1
            if split=='query':
                for condition in CONDITIONS:
                    need(cell['surface'][condition]==dict(mean_loss=math.fsum(r['scores'][condition]['loss'] for r in selected)/len(selected),
                        vs_predicted=compare([r['scores'][condition]['loss']-r['scores']['predicted']['loss'] for r in selected])),
                        'REGION_READER_SUMMARY'); cells+=1
    return cells


def independent_conditions(world, output, target, label):
    result=dict(predicted=output['predicted_constellation'].slots,target=target)
    for stem,index in (('selected',int(next(iter(output['world_owner'].values()))[0])),('oracle',label)):
        bases={}; placed={}
        for i,spec in enumerate(world.specs):
            book=getattr(world,f'world_values_{i}')[index:index+1]; pad=getattr(world,f'world_padding_{i}')[index:index+1]
            width=int((~pad).sum()); book,pad=book[:,:width],pad[:,:width]; base=F.normalize(book,dim=-1)
            template=result['predicted'][spec.name]
            bases[spec.name]=replace(template,values=base,padding=pad)
            state=output['placement_hidden'][:,i,None,:]+world.core.projectors[spec.name](book)
            residual=world.place[spec.name](state).tanh()
            residual=.1*residual/torch.linalg.vector_norm(residual,dim=-1,keepdim=True).clamp_min(1.)
            values=F.normalize(base+residual,dim=-1).masked_fill(pad[:,:,None],0.)
            placed[spec.name]=replace(template,values=values,padding=pad)
        result[stem+'_base']=bases
        if stem=='oracle': result['oracle_placed']=placed
        else:
            need(fingerprint({n:x.values for n,x in placed.items()})==fingerprint({n:x.values for n,x in result['predicted'].items()}),
                 'REGION_READER_PLACEMENT')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pin',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    need(not a.output.exists(),'REGION_READER_EXISTS')
    report=pinned(dict(path=str(a.root/'report.json'),pin=a.pin)); plan=report['plan']
    need(plan==json.loads((a.root/'plan.json').read_bytes()) and
         plan['code_sha256']==sha(Path(__file__).with_name('capability_region_diagnostic.py').read_bytes()),'REGION_READER_CODE')
    raw=(a.root/'tasks.jsonl').read_bytes(); need(sha(raw)==report['tasks_sha256'],'REGION_READER_ROWS')
    rows=[json.loads(x) for x in raw.splitlines()]
    gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-region-readback'); configure()
    model,corpus,parent,prior=load(plan['source']); world=model['world']; before=fingerprint(model.state_dict())
    need(before==report['model_identity'] and plan['train']==list(corpus.train) and plan['query']==list(corpus.query)
         and plan['regions']==[list(r) for r in corpus.regions], 'REGION_READER_PLAN')
    cells=validate(corpus,rows,report,prior); controls=[]
    for mutation,code in (('missing','POPULATION'),('target','TARGET'),('label','LABEL'),('score','SUMMARY')):
        damaged=copy.deepcopy(rows)
        if mutation=='missing':damaged.pop()
        elif mutation=='target':damaged[0]['source_sha256']='0'*64
        elif mutation=='label':damaged[0]['recognition']['selected']=-1
        else:next(r for r in damaged if r['split']=='query')['scores']['oracle_base']['loss']+=1
        controls.append(refused('REGION_READER_'+code,lambda:validate(corpus,damaged,report,prior)))
    cases=[]
    with torch.no_grad():
        for split in ('train','query'):
            for family in ('coding','conversation','reasoning'):
                row=min((r for r in rows if r['split']==split and r['family']==family),key=lambda r:r['task'])
                key=row['task']; native=corpus.batch(key,'context',world.config,'cuda')
                raw,target=corpus.target(key,world.config,'cuda',fitting=split=='train')
                output=predict(world,model['request'],native,key=key,operation=corpus.tasks[key]['kind'])
                assignments=[]
                for i,spec in enumerate(world.specs):
                    truth=target[spec.name]; x=F.normalize(truth.values[0,~truth.padding[0]],dim=-1); values=[]
                    for book,pad in zip(getattr(world,f'world_values_{i}'),getattr(world,f'world_padding_{i}')):
                        matrix=x@F.normalize(book[~pad],dim=-1).T
                        values.append((matrix.amax(0).mean()+matrix.amax(1).mean())/2)
                    assignment=torch.stack(values); assignments.append(assignment)
                    saved=row['recognition']['instruments'][spec.name]
                    need(assignment.cpu().tolist()==saved['assignments'] and output['regions'][spec.name][0].cpu().tolist()==saved['logits'],
                         'REGION_READER_GEOMETRY')
                label=int(torch.stack(assignments).mean(0).argmax())
                need(label==row['recognition']['label'],'REGION_READER_LABEL')
                if split=='query':
                    conditions=independent_conditions(world,output,target,label)
                    for name,slots in conditions.items():
                        saved=row['scores'][name]
                        need(saved['condition_sha256']==fingerprint({n:dict(identity=x.identity,values=x.values,
                            padding=x.padding,absent=x.absent) for n,x in slots.items()}), 'REGION_READER_CONDITION')
                        actual=independent_score(model['decoder'],raw,slots,skipped=False,size=saved['physical_batch'])
                        need(actual==saved['loss'],'REGION_READER_SURFACE')
                        for lane,truth in target.items():
                            x=slots[lane]; matrix=F.normalize(x.values[0,~x.padding[0]],dim=-1)@F.normalize(truth.values[0,~truth.padding[0]],dim=-1).T
                            similarity=float((matrix.amax(0).mean()+matrix.amax(1).mean())/2)
                            need(similarity==row['lanes'][lane]['conditions'][name]['coverage_cosine'],'REGION_READER_LANE')
                    controls.append(refused('CONSTELLATION_SUPERVISION_TRAINING_ONLY',lambda:world.supervised_placement(output,
                        torch.tensor([label],device='cuda'),allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),event_ids=[key])))
                cases.append(dict(split=split,family=family,task=key,all_region_assignments_and_logits_exact=True,
                                  independent_surface_scores=len(CONDITIONS) if split=='query' else 0))
    need(fingerprint(model.state_dict())==before,'REGION_READER_DRIFT')
    result=dict(schema='capability-region-readback-v3',source=dict(path=str(a.root/'report.json'),pin=a.pin),rows=len(rows),
        aggregate_cells=cells,cold_cases=cases,controls=controls,weights_unchanged=True,gpu_before=gpu,
        model_updates=0,scientific_admission=False,learned_capability=False,serving=False,generated_surface_evaluated=False,
        number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output,result))),flush=True)


if __name__=='__main__':run_cli(main)

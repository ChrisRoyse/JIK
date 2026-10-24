"""Cold optimizer replay and complete paired evaluation readback for renderer learning."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_conditioning import fingerprint, select_donors
from audit.capability_conditioning_readback import refused, independent_score
from audit.capability_renderer_learning import restore, frozen
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, pinned, predict
from model.stream_surface import whole_surface_loss


def read(path):return json.loads(Path(path).read_bytes())


def compare(values):
    return dict(mean_delta=math.fsum(values)/len(values),better=sum(x<0 for x in values),
                worse=sum(x>0 for x in values),ties=sum(x==0 for x in values))


def statistics(corpus, rows, reports, prior):
    donors=select_donors(corpus)
    for arm, records in rows.items():
        need([r['task'] for r in records]==list(corpus.query),'RENDERER_READER_POPULATION')
        for row in records:
            task=corpus.tasks[row['task']]; field=task['selections']['targets']['fields'][0]['id']
            raw=corpus.sources[task['population']].population.fields[field]['text'].encode()
            need(row['source_sha256']==sha(raw) and row['source_bytes']==len(raw) and row['family']==task['family']
                 and row['group']==task['split_group'] and row['donor']==donors[row['task']], 'RENDERER_READER_TARGET')
            need(set(row['scores'])=={'predicted','target','wrong_target'},'RENDERER_READER_CONDITIONS')
            for name,value in row['scores'].items():
                need(value['positions']==len(raw)+1 and math.isfinite(value['loss']),'RENDERER_READER_POSITIONS')
                need(value['condition_sha256']==prior[row['task']]['scores'][name]['condition_sha256']
                     and row['parent'][name]==prior[row['task']]['scores'][name]['loss'],'RENDERER_READER_CONDITION')
        for family in {r['family'] for r in records}:
            selected=[r for r in records if r['family']==family]
            cell=reports[arm]['families'][family][arm]
            need(cell['tasks']==len(selected) and cell['groups']==len({r['group'] for r in selected}),'RENDERER_READER_GROUPS')
            for name in selected[0]['scores']:
                values=[r['scores'][name]['loss'] for r in selected]
                expected=cell['conditions'][name]
                paired=compare([r['scores'][name]['loss']-r['scores']['predicted']['loss'] for r in selected])
                need(math.fsum(values)/len(values)==expected['mean_loss']
                     and paired['mean_delta']==expected['mean_delta_from_predicted']
                     and all(paired[k]==expected[k] for k in ('better','worse','ties')),'RENDERER_READER_SUMMARY')
    result={}
    for family in ('coding','conversation','reasoning'):
        correct=[r for r in rows['correct'] if r['family']==family]
        shuffled=[r for r in rows['shuffled'] if r['family']==family]
        need([r['task'] for r in correct]==[r['task'] for r in shuffled],'RENDERER_READER_PAIRED')
        result[family]=dict(tasks=len(correct),groups=len({r['group'] for r in correct}),conditions={})
        for name in ('predicted','target','wrong_target'):
            result[family]['conditions'][name]=dict(
                parent_mean=math.fsum(r['parent'][name] for r in correct)/len(correct),
                correct_mean=math.fsum(r['scores'][name]['loss'] for r in correct)/len(correct),
                shuffled_mean=math.fsum(r['scores'][name]['loss'] for r in shuffled)/len(shuffled),
                correct_vs_parent=compare([r['scores'][name]['loss']-r['parent'][name] for r in correct]),
                correct_vs_shuffled=compare([r['scores'][name]['loss']-s['scores'][name]['loss'] for r,s in zip(correct,shuffled)]))
        result[family]['target_vs_wrong']={arm:compare([r['scores']['target']['loss']-r['scores']['wrong_target']['loss'] for r in selected])
                                         for arm,selected in (('correct',correct),('shuffled',shuffled))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arm in ('correct','shuffled'):
        p.add_argument('--'+arm,type=Path,required=True); p.add_argument('--'+arm+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    need(not a.output.exists(),'RENDERER_READER_EXISTS')
    reports={arm:pinned(dict(path=str(getattr(a,arm)/'report.json'),pin=getattr(a,arm+'_pin'))) for arm in ('correct','shuffled')}
    source=reports['correct']['source']; need(source==reports['shuffled']['source'],'RENDERER_READER_SOURCE')
    for name,pin in source['code'].items():
        need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'RENDERER_READER_CODE')
    config=pinned(source['config']); config['_path']=source['config']['path']; corpus=KernelLearningCorpus(config)
    previous=pinned(source['conditioning']); pp=Path(source['conditioning']['path']).parent/'queries.jsonl'
    need(sha(pp.read_bytes())==previous['query_records_sha256'],'RENDERER_READER_PRIOR')
    prior={r['task']:r for r in map(json.loads,pp.read_bytes().splitlines()) if r['arm']=='correct'}
    rows={}; controls=[]; cases=[]; query_cases=[]
    gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-renderer-readback'); configure()
    for arm,report in reports.items():
        root=getattr(a,arm); plan=read(root/'plan.json'); raw=(root/'steps.jsonl').read_bytes()
        need(sha(raw)==report['steps_sha256'],'RENDERER_READER_LEDGER')
        steps=[json.loads(x) for x in raw.splitlines()]
        need(len(steps)==len(corpus.train)==report['new_updates'],'RENDERER_READER_COUNT')
        chain=sha((root/'initial.pt').read_bytes()); current=report['initial_parameters']
        for index,step in enumerate(steps):
            payload={k:v for k,v in step.items() if k!='sha256'}
            key=corpus.train[index]; condition=key if arm=='correct' else plan['shuffled'][key]
            need(step['index']==index and step['task']==key and step['condition']==condition and step['previous']==chain
                 and step['before']==current and digest(payload)==step['sha256'],'RENDERER_READER_CHAIN')
            chain,current=step['sha256'],step['after']
        need(current==report['final_parameters'] and sha((root/'final.pt').read_bytes())==report['final_checkpoint'],'RENDERER_READER_FINAL')
        model,optimizer,parent,parent_saved=restore(config,corpus,source['parent'])
        need(fingerprint(optimizer.state_dict())==report['initial_optimizer'],'RENDERER_READER_CARRIED_MOMENTS')
        initial_frozen=frozen(model,optimizer)
        for index in plan['boundaries']:
            saved=torch.load(root/('initial.pt' if index==0 else f'before-{index}.pt'),map_location='cpu',weights_only=True)
            expected=torch.load(root/f'after-{index}.pt',map_location='cpu',weights_only=True)
            need(saved['source']==expected['source']==source and saved['updates']==report['parent_updates']+index
                 and expected['updates']==saved['updates']+1,'RENDERER_READER_CHECKPOINT_BINDING')
            model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
            need(fingerprint(optimizer.state_dict())==saved['optimizer_hash'] and frozen(model,optimizer)==initial_frozen,'RENDERER_READER_BEFORE')
            row=steps[index]; optimizer.zero_grad(set_to_none=True)
            raw,_=corpus.target(row['task'],model['world'].config,'cuda',fitting=True)
            _,native=corpus.target(row['condition'],model['world'].config,'cuda',fitting=True)
            surface=whole_surface_loss(model['decoder'],raw,native,backward=True)
            need(surface==row['surface'],'RENDERER_READER_FORWARD')
            for branch,evidence in row['gradients'].items():
                gradients={n:p.grad for n,p in model.named_parameters() if p.grad is not None
                    and next((s.name for s in model['world'].specs if '.'+s.name+'.' in n),'shared')==branch}
                need(fingerprint(gradients)==evidence['sha256'],'RENDERER_READER_GRADIENT')
            optimizer.step()
            need(fingerprint(model.state_dict())==fingerprint(expected['model'])
                 and fingerprint(optimizer.state_dict())==expected['optimizer_hash']
                 and frozen(model,optimizer)==initial_frozen,'RENDERER_READER_AFTER')
            cases.append(dict(arm=arm,index=index,family=row['family'],forward_gradients_parameters_moments_exact=True))
            if len(cases)==1:
                damaged=copy.deepcopy(saved['optimizer']); next(iter(damaged['state'].values()))['exp_avg'].reshape(-1)[0]+=1
                controls.append(refused('RENDERER_READER_MOMENT',lambda:need(fingerprint(damaged)==saved['optimizer_hash'],'RENDERER_READER_MOMENT')))
                controls.append(refused('LEARNING_TARGET_SPLIT',lambda:corpus.target(corpus.query[0],model['world'].config,'cuda',fitting=True)))
        need(frozen(model,optimizer)==report['frozen_identity'],'RENDERER_READER_FROZEN')
        raw=(root/'queries.jsonl').read_bytes(); need(sha(raw)==report['queries_sha256'],'RENDERER_READER_QUERIES')
        rows[arm]=[json.loads(x) for x in raw.splitlines()]
        # The final replayed update is the published final state. Rebuild each
        # family's first query through ordinary block execution, not evaluate().
        with torch.no_grad():
            for family in ('coding','conversation','reasoning'):
                key=min(k for k in corpus.query if corpus.tasks[k]['family']==family)
                row=next(r for r in rows[arm] if r['task']==key); world=model['world']
                raw,target=corpus.target(key,world.config,'cuda',fitting=False)
                _,wrong=corpus.target(row['donor'],world.config,'cuda',fitting=True)
                native=corpus.batch(key,'context',world.config,'cuda')
                prediction=predict(world,model['request'],native,key=key,operation=corpus.tasks[key]['kind'])
                conditions=dict(predicted=prediction['predicted_constellation'].slots,target=target,wrong_target=wrong)
                for name,slots in conditions.items():
                    value=independent_score(model['decoder'],raw,slots,skipped=False,size=row['scores'][name]['physical_batch'])
                    need(value==row['scores'][name]['loss'],'RENDERER_READER_QUERY_FORWARD')
                query_cases.append(dict(arm=arm,family=family,task=key,conditions=3,all_losses_exact=True))
    result=statistics(corpus,rows,reports,prior)
    for name,code in [('missing','POPULATION'),('target','TARGET'),('score','SUMMARY')]:
        damaged=copy.deepcopy(rows)
        if name=='missing':damaged['correct'].pop()
        elif name=='target':damaged['correct'][0]['source_sha256']='0'*64
        else:damaged['correct'][0]['scores']['target']['loss']+=1
        controls.append(refused('RENDERER_READER_'+code,lambda:statistics(corpus,damaged,reports,prior)))
    output=dict(schema='capability-renderer-readback-v2',sources={arm:getattr(a,arm+'_pin') for arm in reports},
        cases=cases,query_cases=query_cases,controls=controls,families=result,all_queries=2*len(corpus.query),all_steps=2*len(corpus.train),
        original_optimizer_moments_carried=True,frozen_parameters_and_moments_unchanged=True,
        producer_sha256=sha(Path(__file__).read_bytes()),gpu_before=gpu,number_status='measured',acceptance='author-checked',
        scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,output),cases=len(cases),controls=len(controls),families=result)),flush=True)


if __name__=='__main__':run_cli(main)

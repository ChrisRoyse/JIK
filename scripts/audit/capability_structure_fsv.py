"""Real source structure -> typed context -> predictor, with unchanged-parent and perturbation evidence."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_conditioning import fingerprint, score
from audit.capability_conditioning_readback import refused
from audit.capability_order_diagnostic import snapshot, difference
from audit.capability_recognition_learning import prepare, frozen
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import predict
from model.capability_structure_input import StructuredContexts, field_features, window_features
from model.capability_structure import attach_context, predict_structured


def selected(contexts):
    descriptors={k:json.loads(raw) for k,raw in contexts.descriptors.items()}; keys=[]
    for operation in sorted({d['operation'] for d in descriptors.values()}):
        choices=[(k,d) for k,d in descriptors.items() if d['operation']==operation]
        key=min(choices,key=lambda item:(-len(item[1]['fields']),item[0]))[0]
        keys.append(key)
    key=max(descriptors,key=lambda k:sum(f['readings'][s.name]['windows'] for f in descriptors[k]['fields'] for s in contexts.corpus.specs))
    if key not in keys:keys.append(key)
    return keys


def feature_record(descriptor):
    return dict(field=field_features(descriptor).tolist(),windows={n:[window_features(f,n).tolist() for f in descriptor['fields']]
        for n in descriptor['fields'][0]['readings']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','readback'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'STRUCTURE_FSV_EXISTS');a.output.mkdir()
    source={n:dict(path=str(getattr(a,n)),pin=getattr(a,n+'_pin')) for n in ('diagnostic','readback')}
    source['code']={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in
        ('model/capability_structure_input.py','model/capability_structure.py','audit/capability_structure_fsv.py')}
    start=tick=time.perf_counter();phases={}
    try:
        gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-structure-fsv');configure()
        model,optimizer,corpus,previous,parent,saved=prepare(source['diagnostic'],source['readback'])
        parent_identity=fingerprint(model.state_dict());old_optimizer=fingerprint(optimizer.state_dict())
        extension=attach_context(model,optimizer,corpus.config['seed']);protected=frozen(model,optimizer)
        initial=copy.deepcopy(extension.state_dict());torch.save(initial,a.output/'adapter-initial.pt')
        phases['restore_extend']=time.perf_counter()-tick;tick=time.perf_counter();contexts=StructuredContexts(corpus);census=contexts.census()
        free,total=torch.cuda.mem_get_info();reserve=model['world'].config.max_native_elements*4
        need(census['native_bytes']+reserve<free,'STRUCTURE_MEMORY_ADMISSION')
        keys=selected(contexts)
        with (a.output/'descriptors.jsonl').open('x') as out,(a.output/'features.jsonl').open('x') as features:
            for key,raw in contexts.descriptors.items():
                out.write(raw.decode()+'\n');features.write(json.dumps(dict(task=key,features=feature_record(json.loads(raw))),sort_keys=True)+'\n')
        plan=dict(source=source,catalogue=contexts.identity,census=census,selected=keys,memory=dict(free=free,total=total,reserve=reserve),
            initial_parent=parent_identity,initial_optimizer=old_optimizer,new_parameters=sum(p.numel() for p in extension.parameters()))
        write_committed(a.output/'plan.json',plan);phases['compile']=time.perf_counter()-tick;tick=time.perf_counter()
        records=[];outputs={}
        with torch.no_grad():
            for key in keys:
                task=corpus.tasks[key];item=contexts.get(key,model['world'].config,'cuda');raw,_=corpus.target(key,model['world'].config,'cuda',fitting=key in corpus.train)
                baseline=predict(model['world'],model['request'],item.native,key=key,operation=task['kind'])
                changed=predict_structured(model,item,task['kind']);states=dict(parent=snapshot(baseline),initial=snapshot(changed))
                need(fingerprint(states['parent'])==fingerprint(states['initial']),'STRUCTURE_INITIAL_PARENT')
                # A disclosed diagnostic intervention, never fitted or promoted: make the new output maps identities.
                for layer in extension.out.values():layer.weight.copy_(torch.eye(layer.out_features,device='cuda'))
                cases={mode:predict_structured(model,item,task['kind'],mode=mode) for mode in ('boundaries','structured','mismatched')}
                for mode,out in cases.items():states[mode]=snapshot(out)
                extension.load_state_dict(initial,strict=True);outputs[key]=states
                records.append(dict(task=key,operation=task['kind'],family=task['family'],split='train' if key in corpus.train else 'query',
                    output_hashes={n:fingerprint(v) for n,v in states.items()},
                    differences={n:difference(states['parent'],v) for n,v in states.items() if n!='parent'},
                    residual_hashes={n:fingerprint(v['structural_context']) for n,v in cases.items()},
                    scores={n:score(model['decoder'],raw,out['predicted_constellation'].slots) for n,out in dict(parent=baseline,initial=changed,**cases).items()}))
        torch.save(outputs,a.output/'outputs.pt')
        with (a.output/'tasks.jsonl').open('x') as out:
            for row in records:out.write(json.dumps(row,sort_keys=True)+'\n')
        phases['physical_paths']=time.perf_counter()-tick;tick=time.perf_counter();controls=[]
        key=keys[0]
        for mode in ('missing','role','clock','target'):
            value=json.loads(contexts.descriptors[key])
            if mode=='missing':value['fields'].pop()
            elif mode=='role':value['fields'][0]['role']='source_answer'
            elif mode=='clock':value['fields'][0]['clock']={'start':'0','end':'1','id':'fabricated'}
            else:value['fields'][0]['id']=corpus.tasks[key]['selections']['targets']['fields'][0]['id']
            controls.append(refused('STRUCTURE_DESCRIPTOR_BINDING',lambda:contexts.get(key,model['world'].config,'cuda',descriptor=canonical(value))))
        item=copy.deepcopy(contexts.get(key,model['world'].config,'cuda'));item.field_metadata[0,0]+=1
        controls.append(refused('STRUCTURE_METADATA_BINDING',lambda:item.validate(corpus.specs,model['world'].config)))
        need(frozen(model,optimizer)==protected,'STRUCTURE_PARENT_DRIFT')
        need(fingerprint({n:v for n,v in model.state_dict().items() if not n.startswith('context.')})==parent_identity,'STRUCTURE_PARENT_PARAMETERS')
        parent_opt=optimizer.state_dict();parent_opt['param_groups']=parent_opt['param_groups'][:-1]
        need(fingerprint(parent_opt)==old_optimizer,'STRUCTURE_PARENT_OPTIMIZER')
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'STRUCTURE_JOURNAL')
        model['world'].check_reference();phases['readback_controls']=time.perf_counter()-tick
        report=dict(schema='capability-structure-fsv-v1',plan=plan,controls=controls,
            files={n:sha((a.output/n).read_bytes()) for n in ('descriptors.jsonl','features.jsonl','tasks.jsonl','outputs.pt','adapter-initial.pt')},
            protected_parent=protected,gpu_before=gpu,model_updates=0,parent_parameters_and_optimizer_exact=True,
            diagnostic_identity_output_maps_only=True,scientific_admission=False,learned_capability=False,serving=False,
            number_status='measured',acceptance='author-checked',phases=phases,seconds=time.perf_counter()-start)
        need(sum(phases.values())/report['seconds']>=.9,'STRUCTURE_PHASES')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),census=census,selected=keys)),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':run_cli(main)

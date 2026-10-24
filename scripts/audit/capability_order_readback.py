"""Cold source census, ordered-prefix, permutation and decoder readback."""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import independent_score, refused
from audit.capability_region_diagnostic import load
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.capability_sequence import source_prefix, predict_sequence


def census_readback(corpus, ref):
    census=pinned(ref); root=Path(ref['path']).parent; raw=(root/'tasks.jsonl').read_bytes()
    need(sha(raw)==census['tasks_sha256'],'ORDER_READER_CENSUS_PIN')
    rows=[json.loads(x) for x in raw.splitlines()]
    diagnostic=pinned(census['source']); raw=(Path(census['source']['path']).parent/'tasks.jsonl').read_bytes()
    need(sha(raw)==diagnostic['tasks_sha256'],'ORDER_READER_LABEL_PIN')
    prior={r['task']:r for r in map(json.loads,raw.splitlines())}
    need([(r['split'],r['task']) for r in rows]==[('train',k) for k in corpus.train]+[('query',k) for k in corpus.query],
         'ORDER_READER_CENSUS_ROSTER')
    groups={mode:defaultdict(list) for mode in ('ordered','multiset')}
    for row in rows:
        key=row['task']; task=corpus.tasks[key]; population=corpus.sources[task['population']].population
        signatures={mode:{} for mode in groups}; fields=[]
        for item in task['selections']['context']['fields']:
            field=population.fields[item['id']]
            fields.append({**{n:field[n] for n in ('id','role','clock','source')},
                'windows':{s.name:len(corpus.field(task['population'],field['id'],s.name)) for s in corpus.specs}})
        need(fields==row['fields'] and row['operation']==task['kind'] and row['family']==task['family']
             and row['group']==task['split_group'] and row['source_provenance']==population.tasks[key]['provenance'],
             'ORDER_READER_FIELDS')
        need(row['target_sha256']==prior[key]['source_sha256'] and row['region_label']==prior[key]['recognition']['label'],
             'ORDER_READER_TARGET')
        for spec in corpus.specs:
            native=np.concatenate([corpus.field(task['population'],f['id'],spec.name) for f in fields]).astype('<f4',copy=False)
            signatures['ordered'][spec.name]=dict(shape=list(native.shape),sha256=sha(native.tobytes()))
            signatures['multiset'][spec.name]=dict(shape=list(native.shape),sha256=digest(sorted(map(sha,[x.tobytes() for x in native]))))
        need(signatures==row['signatures'] and row['total_windows']==sum(v['shape'][0] for v in signatures['ordered'].values()),
             'ORDER_READER_NATIVE')
        for mode in groups:groups[mode][digest(dict(operation=task['kind'],lanes=signatures[mode]))].append(key)
    need(groups==census['groups'],'ORDER_READER_GROUPS')
    index={r['task']:r for r in rows}
    for split,families in census['families'].items():
        for family,cell in families.items():
            selected={r['task'] for r in rows if r['family']==family and (split=='all' or r['split']==split)}
            need(cell['tasks']==len(selected),'ORDER_READER_DENOMINATOR')
            for mode,targets in cell['aliases'].items():
                sets=[[index[k] for k in keys if k in selected] for keys in groups[mode].values()]
                sets=[x for x in sets if x]
                for target,value in targets.items():
                    counts=[Counter(r[target] for r in group) for group in sets]
                    expected=dict(affected_tasks=sum(sum(c.values()) for c in counts if len(c)>1),
                        conflict_groups=sum(len(c)>1 for c in counts),denominator=len(selected),
                        deterministic_empirical_max_correct=sum(max(c.values()) for c in counts))
                    need(expected==value,'ORDER_READER_ALIAS_CEILING')
    return rows


def check_output(output, saved, row, condition):
    actual=dict(hidden=output['placement_hidden'].cpu(),joint=output['constellation_regions'].cpu(),
        logits={n:x.cpu() for n,x in output['regions'].items()},
        selected=int(next(iter(output['world_owner'].values()))[0]),
        slots={n:x.values.cpu() for n,x in output['predicted_constellation'].slots.items()})
    need(fingerprint(actual)==fingerprint(saved)==row['output_hashes'][condition],'ORDER_READER_OUTPUT')
    base=row['_baseline']
    if condition!='baseline':
        for category in ('hidden','joint','logits','slots'):
            pairs={category:(base[category],actual[category])} if category in ('hidden','joint') else {
                n:(x,actual[category][n]) for n,x in base[category].items()}
            for name,(x,y) in pairs.items():
                expected=dict(shape_equal=x.shape==y.shape,exact=torch.equal(x,y),
                              max_absolute=float((x-y).abs().max()) if x.shape==y.shape else None)
                reported=row['differences'][condition][category]
                need(expected==(reported if category in ('hidden','joint') else reported[name]),'ORDER_READER_DELTA')
        need(row['differences'][condition]['selected_unchanged']==(base['selected']==actual['selected']),'ORDER_READER_REGION')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'ORDER_READER_EXISTS'); ref=dict(path=str(a.root/'report.json'),pin=a.pin)
    report=pinned(ref); plan=report['plan']
    need(plan==json.loads((a.root/'plan.json').read_bytes()),'ORDER_READER_PLAN')
    for name,pin in plan['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'ORDER_READER_CODE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-order-readback');configure()
    model,corpus,parent,prior=load(plan['source']);world=model['world'];before=fingerprint(model.state_dict())
    census_rows=census_readback(corpus,plan['census'])
    raw=(a.root/'tasks.jsonl').read_bytes();need(sha(raw)==report['tasks_sha256'],'ORDER_READER_ROWS')
    rows=[json.loads(x) for x in raw.splitlines()];need([r['task'] for r in rows]==plan['selected'],'ORDER_READER_SELECTION')
    need(sha((a.root/'outputs.pt').read_bytes())==report['outputs_sha256'],'ORDER_READER_TENSORS')
    tensors=torch.load(a.root/'outputs.pt',weights_only=True);cases=[];controls=[]
    with torch.no_grad():
        for row in rows:
            key=row['task'];task=corpus.tasks[key];native=corpus.batch(key,'context',world.config,'cuda')
            need(row['native_hashes']=={n:fingerprint(s.values) for n,s in native.items()},'ORDER_READER_INPUT')
            raw,_=corpus.target(key,world.config,'cuda',fitting=False);row['_baseline']=tensors[key]['baseline']
            for condition in row['output_hashes']:
                changed=native
                if condition in ('reverse_fields','reverse_within_fields'):
                    changed={}
                    for name,slot in native.items():
                        lengths=[len(corpus.field(task['population'],f['id'],name)) for f in task['selections']['context']['fields']]
                        spans=list(torch.split(slot.values,lengths,dim=1))
                        pieces=spans[::-1] if condition=='reverse_fields' else [x.flip(1) for x in spans]
                        changed[name]=replace(slot,values=torch.cat(pieces,dim=1))
                if condition in ('attested_sequence','reversed_sequence_diagnostic'):
                    history=source_prefix(corpus,key,world.config,'cuda')
                    # Source-to-frame conservation independently checks every window.
                    for spec in world.specs:
                        need(torch.equal(torch.cat([f.slots[spec.name].values for f in history.frames],dim=1),native[spec.name].values),
                             'ORDER_READER_PREFIX_CONSERVATION')
                    if condition=='reversed_sequence_diagnostic':history=replace(history,frames=history.frames[::-1],rows=history.rows[::-1])
                    state=model['request'](world.start('capability:'+key),operation=task['kind'])
                    args=dict(allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),phase=0,
                        event_ids=[digest(dict(key=key,operation=task['kind']))],evidence_origin=['source_observation'],advance_phase=False)
                    output=world(native,(state,),history=history,**args)
                    bare=world(native,(state,),**args); expected=[]
                    for lane,spec in enumerate(world.specs):
                        current=torch.zeros_like(bare['placement_hidden'][:,lane])
                        for frame in history.frames:
                            slot=frame.slots[spec.name]; projected=F.normalize(world.core.projectors[spec.name](slot.values),dim=-1)
                            for index in range(projected.shape[1]):current=torch.where(slot.padding[:,index,None],current,
                                torch.tanh(world.query[spec.name](current+projected[:,index])))
                        expected.append(bare['placement_hidden'][:,lane]+current)
                    need(torch.equal(torch.stack(expected,dim=1),output['placement_hidden']),'ORDER_READER_RECURRENCE')
                    if condition=='attested_sequence':
                        check_output(predict_sequence(world,model['request'],corpus,key),tensors[key][condition],row,condition)
                else:output=predict(world,model['request'],changed,key=key,operation=task['kind'])
                check_output(output,tensors[key][condition],row,condition)
                measured=independent_score(model['decoder'],raw,output['predicted_constellation'].slots,
                                           skipped=False,size=row['scores'][condition]['physical_batch'])
                need(measured==row['scores'][condition]['loss'],'ORDER_READER_SURFACE')
                cases.append(dict(task=key,condition=condition,outputs_and_surface_exact=True))
        key=next(r['task'] for r in rows if r['operation']=='continue_recorded_steps');selection=corpus.tasks[key]['selections']['context']
        for mode in ('reversed','missing','future','role'):
            damaged=copy.deepcopy(selection)
            if mode=='reversed':damaged['fields'].reverse()
            elif mode=='missing':damaged['fields'].pop()
            elif mode=='future':damaged['fields']+=corpus.tasks[key]['selections']['targets']['fields']
            else:damaged['fields'][0]['role']='recorded_step'
            controls.append(refused('SEQUENCE_SELECTION',lambda:source_prefix(corpus,key,world.config,'cuda',selection=damaged)))
        other=next(r['task'] for r in rows if r['family']=='coding')
        controls.append(refused('SEQUENCE_OPERATION',lambda:source_prefix(corpus,other,world.config,'cuda')))
        damaged=copy.deepcopy(tensors[key]['baseline']);damaged['joint'][0,0]+=1
        baseline_output=predict(world,model['request'],corpus.batch(key,'context',world.config,'cuda'),key=key,operation=corpus.tasks[key]['kind'])
        row=next(r for r in rows if r['task']==key)
        controls.append(refused('ORDER_READER_OUTPUT',lambda:check_output(baseline_output,damaged,row,'baseline')))
    need(fingerprint(model.state_dict())==before==report['model_identity'],'ORDER_READER_WEIGHTS');world.check_reference()
    need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'ORDER_READER_JOURNAL')
    result=dict(schema='capability-order-readback-v1',source=ref,census_rows=len(census_rows),cases=cases,controls=controls,
        weights_and_journal_unchanged=True,gpu_before=gpu,number_status='measured',acceptance='author-checked',
        model_updates=0,scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result))),flush=True)


if __name__=='__main__':run_cli(main)

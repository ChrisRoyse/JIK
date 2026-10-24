"""Rebuild conditioning interventions and independently read every reported statistic."""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import sources, select_donors, load_arm, fingerprint
from audit.capability_kernel_learning import system
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.stream_surface import batches, position_batch


def read(path): return json.loads(Path(path).read_bytes())


def verify_statistics(corpus, rows, report, donors):
    need([(r['arm'],r['task']) for r in rows] == [(a,k) for a in ('correct','shuffled') for k in corpus.query]
         and len(rows) == report['rows'], 'CONDITION_READER_POPULATION')
    for row in rows:
        task=corpus.tasks[row['task']]; field=task['selections']['targets']['fields'][0]['id']
        raw=corpus.sources[task['population']].population.fields[field]['text'].encode()
        need(row['source_sha256'] == sha(raw) and row['source_bytes'] == len(raw)
             and row['family'] == task['family'] and row['group'] == task['split_group']
             and row['operation'] == task['kind'] and row['donor'] == donors[row['task']], 'CONDITION_READER_TARGET')
        names={'predicted','wrong_context','current','target','wrong_target','bypass'} | {'without:'+s.name for s in corpus.specs}
        need(set(row['scores']) == names, 'CONDITION_READER_ROSTER')
        need(all(v['positions'] == len(raw)+1 and math.isfinite(v['loss']) for v in row['scores'].values()),
             'CONDITION_READER_COVERAGE')
    need(not any(report[k] for k in ('scientific_admission','learned_capability','serving','generated_surface_evaluated'))
         and report['model_updates'] == 0, 'CONDITION_READER_SCOPE')
    count=0
    for family in {r['family'] for r in rows}:
        for arm in ('correct','shuffled'):
            values=[r for r in rows if r['family']==family and r['arm']==arm]
            cell=report['families'][family][arm]
            need(cell['tasks']==len(values) and cell['groups']==len({r['group'] for r in values}), 'CONDITION_READER_GROUPS')
            for condition in values[0]['scores']:
                losses=[r['scores'][condition]['loss'] for r in values]
                deltas=[r['scores'][condition]['loss']-r['scores']['predicted']['loss'] for r in values]
                expected=dict(mean_loss=math.fsum(losses)/len(losses),mean_delta_from_predicted=math.fsum(deltas)/len(deltas),
                              better=sum(v<0 for v in deltas),worse=sum(v>0 for v in deltas),ties=sum(v==0 for v in deltas))
                need(expected==cell['conditions'][condition], 'CONDITION_READER_AGGREGATE'); count+=1
    return count


def independent_score(decoder, raw, native, *, skipped, size):
    """No intervention hooks: explicitly execute or skip cross branches in the block."""
    total=0.; positions=0; c=decoder.config
    for prefix, labels, offsets, slots, items in batches(raw,native,c,batch_size=size):
        memories,_=decoder.memories(slots)
        h=decoder.tokens(prefix)+position_batch(offsets,prefix.shape[1],c.width)
        for block in decoder.blocks:
            q=block.norm(h); h=h+block.self_attention(q,q,causal=True)
            if not skipped:
                for spec in decoder.specs: h=block.branches[spec.name](h,memories[spec.name],slots[spec.name])
            h=h+block.ff(block.ff_norm(h))
        logits=decoder.output(h)
        total+=float(F.cross_entropy(logits.reshape(-1,c.vocabulary),labels.reshape(-1),ignore_index=-100,reduction='sum')/(len(raw)+1))
        positions+=sum(i['scored'][1]-i['scored'][0] for i in items)
    need(positions==len(raw)+1,'CONDITION_READER_REPLAY_COVERAGE')
    return total


def refused(code, call):
    try: call()
    except (ValueError,RuntimeError) as exc:
        need(code in str(exc),'CONDITION_READER_WRONG_REFUSAL:'+str(exc))
        return dict(expected=code,actual=str(exc))
    raise AssertionError('CONDITION_READER_ACCEPTED:'+code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); need(not a.output.exists(),'CONDITION_READER_EXISTS')
    report=pinned(dict(path=str(a.root/'report.json'),pin=a.pin)); source=report['source']
    need(source['code_sha256']==sha(Path(__file__).with_name('capability_conditioning.py').read_bytes()),'CONDITION_READER_CODE')
    pinned(source['config']); pinned(source['experiments'])
    config,corpus,experiments,reports,prior=sources(source['config']['path'],source['experiments']['path'],source['evaluation'])
    plan=read(a.root/'plan.json'); donors=select_donors(corpus)
    need(plan['donors']==donors and plan['queries']==list(corpus.query) and plan['source']==source,'CONDITION_READER_PLAN')
    raw_rows=(a.root/'queries.jsonl').read_bytes()
    need(sha(raw_rows)==report['query_records_sha256'],'CONDITION_READER_ROWS')
    rows=[json.loads(x) for x in raw_rows.splitlines()]
    aggregate_cells=verify_statistics(corpus,rows,report,donors); controls=[]
    for name,code in [('missing_query','POPULATION'),('changed_target','TARGET'),('changed_score','AGGREGATE')]:
        damaged=copy.deepcopy(rows)
        if name=='missing_query': damaged.pop()
        elif name=='changed_target': damaged[0]['source_sha256']='0'*64
        else: damaged[0]['scores']['target']['loss']+=1
        controls.append(refused('CONDITION_READER_'+code,lambda:verify_statistics(corpus,damaged,report,donors)))
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('29','capability-conditioning-readback'); configure(); model=system(config,corpus)
    selected={f:min(k for k in corpus.query if corpus.tasks[k]['family']==f) for f in ('coding','conversation','reasoning')}
    cases=[]
    with torch.no_grad():
        for arm in ('correct','shuffled'):
            before=load_arm(model,arm,experiments,reports); world=model['world']; decoder=model['decoder']
            for family,key in selected.items():
                saved=next(r for r in rows if r['arm']==arm and r['task']==key); donor=donors[key]
                native=corpus.batch(key,'context',world.config,'cuda'); other=corpus.batch(donor,'context',world.config,'cuda')
                args=dict(key=key,operation=corpus.tasks[key]['kind'])
                actual=predict(world,model['request'],native,**args); wrong=predict(world,model['request'],other,**args)
                raw,truth=corpus.target(key,world.config,'cuda',fitting=False)
                _,wrong_truth=corpus.target(donor,world.config,'cuda',fitting=True)
                predicted=actual['predicted_constellation'].slots
                conditions=dict(predicted=predicted,wrong_context=wrong['predicted_constellation'].slots,current=native,
                                target=truth,wrong_target=wrong_truth,bypass=predicted)
                for spec in world.specs:
                    masked=dict(predicted); x=predicted[spec.name]
                    masked[spec.name]=replace(x,values=torch.zeros_like(x.values),padding=torch.ones_like(x.padding),absent=torch.ones_like(x.absent))
                    conditions['without:'+spec.name]=masked
                for name,condition in conditions.items():
                    pin=fingerprint({n:dict(identity=x.identity,values=x.values,padding=x.padding,absent=x.absent) for n,x in condition.items()})
                    need(pin==saved['scores'][name]['condition_sha256'],'CONDITION_READER_INPUT')
                    value=independent_score(decoder,raw,condition,skipped=name=='bypass',size=saved['scores'][name]['physical_batch'])
                    need(value==saved['scores'][name]['loss'],'CONDITION_READER_FORWARD')
                need(saved['scores']['predicted']['loss']==prior[key]['arms'][arm]['surface_loss'],'CONDITION_READER_PRIOR')
                if not cases:
                    empty={n:replace(x,values=torch.zeros_like(x.values),padding=torch.ones_like(x.padding),absent=torch.ones_like(x.absent)) for n,x in native.items()}
                    controls.append(refused('MNLM_MODEL_NO_OBSERVATION',lambda:decoder.validate_inputs(empty)))
                    bad=dict(native); name=world.specs[0].name; bad[name]=replace(bad[name],identity='wrong-instrument')
                    controls.append(refused('MNLM_MODEL_IDENTITY',lambda:decoder.validate_inputs(bad)))
                cases.append(dict(arm=arm,family=family,task=key,conditions=len(conditions),all_losses_exact=True,all_condition_hashes_exact=True))
            need(fingerprint(model.state_dict())==before,'CONDITION_READER_WEIGHT_DRIFT')
    result=dict(schema='capability-conditioning-readback-v1',source_report=a.pin,source_rows=sha(raw_rows),
        producer_sha256=sha(Path(__file__).read_bytes()),cases=cases,controls=controls,aggregate_cells=aggregate_cells,
        all_rows=len(rows),independent_block_execution=True,weights_unchanged=True,gpu_before=gpu,
        scientific_admission=False,learned_capability=False,serving=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output,result),cases=len(cases),controls=len(controls))),flush=True)


if __name__=='__main__':run_cli(main)

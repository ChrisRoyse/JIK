"""Trace two real training conditions without fitting or changing their journal."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_grounded_rehearsal import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.constellation_learning import configure as trainable_configure, trainable
from model.constellation_world import ConstellationWorld
from model.loop_training_checkpoint import tensor_fingerprint
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from model.tool_rehearsal import RehearsalData, RehearsalToolLoop
from model.window_maxsim import maxsim
from model import process_anchor, world_checkpoint


def compare(a,b):
    x,y = a.detach().double().reshape(-1),b.detach().double().reshape(-1)
    check(x.shape == y.shape,'CONDITION_PATH_COMPARE_SHAPE')
    norm = torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y)
    return dict(equal=torch.equal(x,y),l2=float(torch.linalg.vector_norm(x-y)),
                maximum=float((x-y).abs().max()), cosine=None if float(norm) == 0 else float(x@y/norm))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    raw = (a.source/'seal.json').read_bytes()
    check(sha(raw) == a.pin,'CONDITION_PATH_SEAL')
    seal = json.loads(raw)
    for name,entry in seal['files'].items():
        raw = (a.source/name).read_bytes()
        check(sha(raw) == entry['sha256'] and len(raw) == entry['bytes'],'CONDITION_PATH_SOURCE')
    provenance = json.loads((a.source/'source.json').read_bytes())
    evidence = Path(provenance['root'])
    authenticate(evidence,provenance['seal'])
    source_summary = json.loads((evidence/'summary.json').read_bytes())
    c = source_summary['config']
    root = a.output
    root.mkdir(exist_ok=False)
    main_before = sha((a.source/'experience.sqlite3').read_bytes())
    shutil.copyfile(a.source/'experience.sqlite3',root/'experience.sqlite3')
    check(sha((root/'experience.sqlite3').read_bytes()) == main_before,'CONDITION_PATH_PREFIX')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','condition-path')
    (root/'gpu.txt').write_text(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source,Path(c['population']),Path(c['native']),
                        population_pin=c['population_pin'],native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']),inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'],'CONDITION_PATH_PARENT')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
    saved = json.loads((a.source/'snapshot.json').read_bytes())
    session = RehearsalToolLoop(root/'experience.sqlite3',source,inputs,parent_root/'initial',model_pin=parent['initial'],
        standard=parent['standard'],world=parent['world'],contexts={'train:1':c['providers'][0]['task']},
        device='cuda',condition_inputs=conditions,routing=routing,growth_root=Path(c['output'])/'growth',
        candidate_root=a.source/'candidates',expected_head=saved['head'],expected_contract=saved['contract'])
    check(session.snapshot() == saved,'CONDITION_PATH_REPLAY')
    train = [v['forecast'] for v in source_summary['measured']['scored'][:2]]
    evaluate = [source_summary['measured']['scored'][2]['forecast']]
    data = RehearsalData(session,train=train,evaluate=evaluate)
    examples = [data.examples[k] for k in train]
    target_regions = [data.binding['training_views'][k]['target_region'] for k in train]
    target_positions = {k:[data.examples[k]['units'].index(u) for u in target_regions] for k in train}
    check([data.outcomes[k] for k in train] == [1,0],'CONDITION_PATH_TRAIN_OUTCOMES')
    input_records = []
    for e in examples:
        b = e['request']['execution_conditions']['native']
        provider = conditions.for_condition(b['condition'])
        input_records.append(dict(binding=b,input=provider.conditions.read(b['condition']),
            reference=e['request']['reference'],history=e['record']['region_context']['definition']['history']))
    check(input_records[0]['input']['program_sha256'] == input_records[1]['input']['program_sha256'],
          'CONDITION_PATH_PAIRED_PROGRAM')
    phases = dict(authenticate_and_replay=time.perf_counter()-started)
    tick = time.perf_counter()
    arrays,models = {},[]
    def keep(name,value):
        arrays[name] = value.detach().cpu().numpy()
        return name
    def trace(model,key,prefix,gradients=False):
        output = data.forward(model,key)
        e = data.examples[key]
        prediction = process_anchor.predict(output,e['view'].execution_anchor_bank,e['allowed'])
        values = dict(core_hidden=torch.stack([s.hidden for s in output['states']]),
                      interpreted=output['placement_hidden'],common=output['constellation_regions'])
        result = dict(arrays={n:keep(prefix+'_'+n,v) for n,v in values.items()},lanes={},prediction=prediction)
        for i,spec in enumerate(model.specs):
            slot = e['native'][spec.name]
            ref = e['view'].references()[spec.name]
            projected = F.normalize(model.core.projectors[spec.name](slot.values),dim=-1)
            context = F.normalize(output['placement_hidden'][:,i],dim=-1)[:,None,:]
            raw_query = model.query[spec.name](context+projected)
            query = F.normalize(raw_query,dim=-1)
            keys = F.normalize(model.core.projectors[spec.name](ref.values),dim=-1)
            matches = maxsim(query,keys,ref.padding,model.config)
            reduction = matches.masked_fill(slot.padding[:,:,None],0.).sum(1)/(~slot.padding).sum(1)[:,None]
            check(torch.equal(reduction,output['regions'][spec.name]),'CONDITION_PATH_REDUCTION')
            tensors = dict(native=slot.values,padding=slot.padding,projected=projected,context=context,
                raw_query=raw_query,query=query,keys=keys,reference_padding=ref.padding,matches=matches,
                regions=output['regions'][spec.name])
            result['lanes'][spec.name] = dict(instrument=spec.identity,
                arrays={n:keep(prefix+'_l'+str(i)+'_'+n,v) for n,v in tensors.items()},
                windows=int((~slot.padding).sum()),reference_windows=(~ref.padding).sum(1).tolist())
        if gradients:
            masses = process_anchor.tensors(output,data.labels[key],e['allowed'])
            truth = masses['common'][1].new_tensor([float(i == data.outcomes[key]) for i in range(3)])
            brier = (masses['common'][1]-truth).square().sum()
            positions = target_positions[key]
            margin = output['constellation_regions'][0,positions[1]]-output['constellation_regions'][0,positions[0]]
            params = trainable(model)
            for label,loss in (('brier',brier),('margin',margin)):
                grads = torch.autograd.grad(loss,[p for _,p in params],retain_graph=label == 'brier',allow_unused=True)
                result[label+'_gradient'] = keep(prefix+'_'+label+'_gradient',torch.cat([
                    (g if g is not None else torch.zeros_like(p)).reshape(-1) for g,(_,p) in zip(grads,params)]))
            result.update(brier=float(brier.detach()),margin=float(margin.detach()))
        return result
    checkpoints = [('parent',None),('first',a.source/'candidates/rehearsal-1'),('second',a.source/'candidates/rehearsal-2')]
    for index,(name,path) in enumerate(checkpoints):
        if path is None:
            model = copy.deepcopy(session.model)
            pin = session.active_checkpoint
        else:
            pin = (path/'COMMITTED').read_text()
            raw = (path/'manifest.json').read_bytes()
            check(sha(raw) == pin,'CONDITION_PATH_CHECKPOINT')
            m = json.loads(raw)
            loaded,_,_ = world_checkpoint.load(path/'model',expected=m['model'],standard=m['binding']['standard'],
                                              world=m['binding']['world'],device='cuda')
            model = ConstellationWorld.from_world(loaded)
        trainable_configure(model)
        before = digest(tensor_fingerprint(model.state_dict()))
        actual = [trace(model,k,'m'+str(index)+'_k'+str(i),True) for i,k in enumerate(train)]
        if name == 'second':
            original = json.loads((a.source/'rehearsal-2.json').read_bytes())['after']
            check(all(v['prediction'] == original[k]['termination']['prediction'] for k,v in zip(train,actual)),
                  'CONDITION_PATH_SAVED_PREDICTION')
        counterfactuals = {}
        native0,state0 = examples[0]['native'],examples[0]['state']
        history0 = examples[0]['view'].context_history
        try:
            for mode in ('condition_only','action_only','state_only','history_only'):
                examples[0]['native'],examples[0]['state'] = native0,state0
                examples[0]['view'].context_history = history0
                if mode in ('condition_only','action_only'):
                    ref = examples[int(mode == 'action_only')]['request']['reference']
                    binding = input_records[int(mode == 'condition_only')]['binding']
                    examples[0]['native'] = conditions.merge(ref,binding,model.config,'cuda').slots
                if mode == 'state_only':
                    examples[0]['state'] = replace(state0,hidden=examples[1]['state'].hidden)
                if mode == 'history_only':
                    examples[0]['view'].context_history = examples[1]['view'].context_history
                with torch.no_grad():
                    counterfactuals[mode] = trace(model,train[0],'m'+str(index)+'_'+mode)
        finally:
            examples[0]['native'],examples[0]['state'] = native0,state0
            examples[0]['view'].context_history = history0
        native1 = examples[1]['native']
        try:
            examples[1]['native'] = conditions.merge(examples[1]['request']['reference'],input_records[0]['binding'],model.config,'cuda').slots
            with torch.no_grad():
                counterfactuals['condition_reverse'] = trace(model,train[1],'m'+str(index)+'_condition_reverse')
        finally:
            examples[1]['native'] = native1
        with torch.no_grad():
            repeated = trace(model,train[0],'m'+str(index)+'_repeat')
        check(repeated['prediction'] == actual[0]['prediction'],'CONDITION_PATH_REPEAT')
        check(before == digest(tensor_fingerprint(model.state_dict())),'CONDITION_PATH_PARAMETER_MUTATION')
        grad_compare = {kind:compare(torch.from_numpy(arrays[actual[0][kind+'_gradient']]),
                                    torch.from_numpy(arrays[actual[1][kind+'_gradient']])) for kind in ('brier','margin')}
        models.append(dict(name=name,checkpoint=pin,actual=actual,counterfactuals=counterfactuals,
            gradients=grad_compare,parameters=before,parameter_order=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in trainable(model)]))
        print(json.dumps(dict(stage=name,margins=[v['margin'] for v in actual],gradients=grad_compare)),flush=True)
    phases['actual_forward_and_substitution_trace'] = time.perf_counter()-tick
    tick = time.perf_counter()
    np.savez(root/'arrays.npz',**arrays)
    check(session.snapshot() == saved,'CONDITION_PATH_STATE_MUTATION')
    session.close()
    check(sha((a.source/'experience.sqlite3').read_bytes()) == main_before,'CONDITION_PATH_SOURCE_MUTATION')
    phases['persist_and_source_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True,acceptance='author-checked',source=dict(root=str(a.source),seal=a.pin,journal=main_before),
        head=saved['head'],keys=train,inputs=input_records,regions={k:data.examples[k]['units'] for k in train},target_positions=target_positions,
        models=models,arrays=dict(sha256=sha((root/'arrays.npz').read_bytes()),bytes=(root/'arrays.npz').stat().st_size),
        code={n:sha((Path('scripts')/n).read_bytes()) for n in ('audit/condition_path.py',)},
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        substitution_scope='diagnostic_only_no_occurrence_or_training',scientific_admission=False,serving=False)
    write(root/'report.json',report)
    print(json.dumps(dict(complete=True,seconds=wall,arrays=report['arrays'])),flush=True)


if __name__ == '__main__':
    run_cli(main)

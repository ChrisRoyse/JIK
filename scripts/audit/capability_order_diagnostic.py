"""Frozen native-order interventions and an attested reasoning-prefix comparison."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_conditioning import fingerprint, score
from audit.capability_region_diagnostic import load
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.capability_sequence import source_prefix, predict_sequence


def permute(corpus, key, native, mode):
    task = corpus.tasks[key]; fields = task['selections']['context']['fields']; result = {}
    for name, slot in native.items():
        lengths = [len(corpus.field(task['population'], f['id'], name)) for f in fields]
        end = 0; pieces = []
        for length in lengths:
            piece = slot.values[:, end:end+length]
            pieces.append(piece.flip(1) if mode == 'reverse_within_fields' else piece)
            end += length
        need(end == slot.values.shape[1] and not bool(slot.padding.any()), 'ORDER_COVERAGE')
        if mode == 'reverse_fields': pieces.reverse()
        need(mode in ('reverse_within_fields', 'reverse_fields'), 'ORDER_MODE')
        result[name] = replace(slot, values=torch.cat(pieces, dim=1))
    return result


def snapshot(output):
    return dict(hidden=output['placement_hidden'].detach().cpu(),
        logits={n:x.detach().cpu() for n,x in output['regions'].items()},
        joint=output['constellation_regions'].detach().cpu(),
        selected=int(next(iter(output['world_owner'].values()))[0]),
        slots={n:x.values.detach().cpu() for n,x in output['predicted_constellation'].slots.items()})


def difference(before, after):
    def delta(x,y):
        return dict(shape_equal=x.shape == y.shape, exact=torch.equal(x,y),
                    max_absolute=float((x-y).abs().max()) if x.shape == y.shape else None)
    return dict(hidden=delta(before['hidden'],after['hidden']), joint=delta(before['joint'],after['joint']),
        selected_unchanged=before['selected'] == after['selected'],
        logits={n:delta(x,after['logits'][n]) for n,x in before['logits'].items()},
        slots={n:delta(x,after['slots'][n]) for n,x in before['slots'].items()})


def history_intervention(model, corpus, key, history):
    world=model['world']; task=corpus.tasks[key]
    state=model['request'](world.start('capability:'+key),operation=task['kind'])
    return world(corpus.batch(key,'context',world.config,'cuda'),(state,),
        allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),phase=0,
        event_ids=[digest(dict(key=key,operation=task['kind']))],evidence_origin=['source_observation'],
        advance_phase=False,history=history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--census',type=Path,required=True); p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    need(not a.output.exists(),'ORDER_EXISTS'); a.output.mkdir(); start=tick=time.perf_counter(); phases={}
    try:
        census=pinned(dict(path=str(a.census),pin=a.pin)); diagnostic=pinned(census['source'])
        gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-order-diagnostic'); configure()
        model,corpus,parent,prior=load(diagnostic['plan']['source']); world=model['world']; before=fingerprint(model.state_dict())
        code={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in
              ('audit/capability_context_census.py','audit/capability_order_diagnostic.py','model/capability_sequence.py')}
        plan=dict(census=dict(path=str(a.census),pin=a.pin),source=diagnostic['plan']['source'],
            selected=census['selected_queries'],code=code,sequence_scope='continue_recorded_steps_only',
            other_metadata_scope='roles_clocks_and_unordered_field_structure_remain_unwired')
        write_committed(a.output/'plan.json',plan); phases['authenticate']=time.perf_counter()-tick; tick=time.perf_counter()
        rows=[]; tensors={}
        with torch.no_grad():
            for key in plan['selected']:
                task=corpus.tasks[key]; native=corpus.batch(key,'context',world.config,'cuda')
                raw,_=corpus.target(key,world.config,'cuda',fitting=False)
                outputs={'baseline':predict(world,model['request'],native,key=key,operation=task['kind'])}
                outputs['repeat']=predict(world,model['request'],native,key=key,operation=task['kind'])
                inputs={n:fingerprint(x.values) for n,x in native.items()}
                for mode in ('reverse_within_fields','reverse_fields'):
                    altered=permute(corpus,key,native,mode)
                    outputs[mode]=predict(world,model['request'],altered,key=key,operation=task['kind'])
                if task['kind']=='continue_recorded_steps':
                    outputs['attested_sequence']=predict_sequence(world,model['request'],corpus,key)
                    history=source_prefix(corpus,key,world.config,'cuda')
                    reversed_history=replace(history,frames=history.frames[::-1],rows=history.rows[::-1])
                    outputs['reversed_sequence_diagnostic']=history_intervention(model,corpus,key,reversed_history)
                states={n:snapshot(o) for n,o in outputs.items()}; tensors[key]=states
                need(fingerprint(states['baseline'])==fingerprint(states['repeat']),'ORDER_REPEAT_DRIFT')
                need(inputs=={n:fingerprint(x.values) for n,x in native.items()},'ORDER_SOURCE_DRIFT')
                row=dict(task=key,family=task['family'],operation=task['kind'],native_hashes=inputs,
                    fields=len(task['selections']['context']['fields']),windows={n:x.values.shape[1] for n,x in native.items()},
                    output_hashes={n:fingerprint(s) for n,s in states.items()},
                    differences={n:difference(states['baseline'],s) for n,s in states.items() if n!='baseline'},
                    scores={n:score(model['decoder'],raw,o['predicted_constellation'].slots) for n,o in outputs.items()})
                need(row['scores']['baseline']==prior[key]['scores']['predicted'],'ORDER_PARENT_DRIFT')
                rows.append(row)
        phases['interventions']=time.perf_counter()-tick; tick=time.perf_counter()
        torch.save(tensors,a.output/'outputs.pt')
        with (a.output/'tasks.jsonl').open('x') as out:
            for row in rows:out.write(json.dumps(row,sort_keys=True)+'\n')
        need(fingerprint(model.state_dict())==before,'ORDER_WEIGHT_DRIFT'); world.check_reference()
        config=pinned(parent['source']['config']); need(sha(Path(config['journal']['path']).read_bytes())==config['journal']['pin'],'ORDER_JOURNAL_DRIFT')
        report=dict(schema='capability-order-diagnostic-v1',plan=plan,model_identity=before,
            tasks_sha256=sha((a.output/'tasks.jsonl').read_bytes()),outputs_sha256=sha((a.output/'outputs.pt').read_bytes()),
            model_updates=0,new_parameters=0,weights_and_journal_unchanged=True,gpu_before=gpu,
            scientific_admission=False,learned_capability=False,serving=False,generated_surface_evaluated=False,
            exposure='development_exposed_group_holdout',number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick; wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'ORDER_PHASES');report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':run_cli(main)

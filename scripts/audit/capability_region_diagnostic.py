"""Frozen region/placement interventions; future target inputs are diagnostic only."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint, score
from audit.capability_kernel_learning import system, parameter_hash
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, pinned, predict
from model.capability_objective import mathematical_labels, native_similarity


CONDITIONS = ('predicted', 'selected_base', 'oracle_placed', 'oracle_base', 'target')


def load(source):
    report = pinned(source); config = pinned(report['source']['config'])
    parent = pinned(report['source']['parent'])
    for owner in (report, parent):
        for name, pin in owner['source']['code'].items():
            need(sha((Path(__file__).resolve().parents[1]/name).read_bytes()) == pin, 'REGION_SOURCE_CODE')
    corpus = KernelLearningCorpus(config); model = system(config, corpus)
    path = Path(source['path']).parent/'final.pt'
    need(sha(path.read_bytes()) == report['final_checkpoint'], 'REGION_CHECKPOINT')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(saved['source'] == report['source'] and fingerprint(saved['optimizer']) == saved['optimizer_hash'],
         'REGION_CHECKPOINT_BINDING')
    model.load_state_dict(saved['model'], strict=True); model.requires_grad_(False).eval()
    need(parameter_hash(model) == report['final_parameters'], 'REGION_PARAMETERS')
    raw = (path.parent/'queries.jsonl').read_bytes()
    need(sha(raw) == report['queries_sha256'], 'REGION_PRIOR_ROWS')
    return model, corpus, report, {r['task']:r for r in map(json.loads, raw.splitlines())}


def base_slots(world, region, template):
    result = {}
    for i, spec in enumerate(world.specs):
        book = getattr(world, f'world_values_{i}')[region:region+1]
        mask = getattr(world, f'world_padding_{i}')[region:region+1]
        width = int((~mask).sum()); book, mask = book[:, :width], mask[:, :width]
        result[spec.name] = replace(template[spec.name], values=F.normalize(book, dim=-1), padding=mask)
    return result


def interventions(model, corpus, key, output, target, label):
    world = model['world']; predicted = output['predicted_constellation'].slots
    selected = int(next(iter(output['world_owner'].values()))[0])
    # This offline experiment deliberately exercises the training-only path.
    # No optimizer exists, no weights require gradients, and no oracle state is served.
    with torch.enable_grad():
        oracle = world.supervised_placement(output, label,
            allowed=torch.ones(len(corpus.regions), dtype=torch.bool, device='cuda'),
            event_ids=[digest(dict(key=key, operation=corpus.tasks[key]['kind']))]).slots
    return dict(predicted=predicted, selected_base=base_slots(world, selected, predicted),
                oracle_placed=oracle, oracle_base=base_slots(world, int(label), oracle), target=target)


def lane_metrics(conditions, target):
    result = {}
    for name, truth in target.items():
        actual = truth.values[0, ~truth.padding[0]]; cell = dict(target_windows=len(actual), conditions={})
        for condition, native in conditions.items():
            slot = native[name]; values = slot.values[0, ~slot.padding[0]]
            cell['conditions'][condition] = dict(windows=len(values), coverage_cosine=float(native_similarity(values, actual)))
        for placed, base in (('predicted','selected_base'), ('oracle_placed','oracle_base')):
            left, right = conditions[placed][name], conditions[base][name]
            need(left.padding.shape == right.padding.shape and bool(torch.equal(left.padding,right.padding)), 'REGION_WINDOW_LOSS')
            distance = torch.linalg.vector_norm(left.values-right.values, dim=-1)[~left.padding]
            cell[placed+'_max_base_distance'] = float(distance.max())
        result[name] = cell
    return result


def recognition(output, label, assignments):
    index = int(label); regions = output['constellation_regions'][0]
    joint = torch.stack(list(assignments.values())).mean(0)
    ordered = joint.sort(descending=True).values
    result = dict(label=index, selected=int(regions.argmax()), target_ties=int((joint == joint[index]).sum()),
                  target_margin=float(ordered[0]-ordered[1]), instruments={})
    for name, values in output['regions'].items():
        logits = values[0]; p = logits.softmax(-1); logp = logits.log_softmax(-1)
        result['instruments'][name] = dict(logits=logits.cpu().tolist(), assignments=assignments[name].cpu().tolist(),
            target_probability=float(p[index]), entropy_nats=float(-(p*logp).sum()),
            target_rank=1+int((logits > logits[index]).sum()), logit_span=float(logits.max()-logits.min()))
    return result


def paired(values):
    return dict(mean=math.fsum(values)/len(values), better=sum(x < 0 for x in values),
                worse=sum(x > 0 for x in values), ties=sum(x == 0 for x in values))


def summarize(rows):
    result = {}
    for split in ('train','query'):
        result[split] = {}
        for family in ('coding','conversation','reasoning'):
            selected = [r for r in rows if r['split'] == split and r['family'] == family]
            cell = dict(tasks=len(selected), groups=len({r['group'] for r in selected}),
                region_hits=sum(r['recognition']['label'] == r['recognition']['selected'] for r in selected),
                selected_regions=dict(Counter(str(r['recognition']['selected']) for r in selected)),
                target_regions=dict(Counter(str(r['recognition']['label']) for r in selected)), instruments={})
            for name in selected[0]['recognition']['instruments']:
                cell['instruments'][name] = {metric:math.fsum(r['recognition']['instruments'][name][metric]
                    for r in selected)/len(selected) for metric in ('target_probability','entropy_nats','target_rank','logit_span')}
                if split == 'query':
                    cell['instruments'][name]['coverage_cosine'] = {condition:math.fsum(
                        r['lanes'][name]['conditions'][condition]['coverage_cosine'] for r in selected)/len(selected)
                        for condition in CONDITIONS}
            if split == 'query':
                cell['surface'] = {condition:dict(mean_loss=math.fsum(r['scores'][condition]['loss'] for r in selected)/len(selected),
                    vs_predicted=paired([r['scores'][condition]['loss']-r['scores']['predicted']['loss'] for r in selected]))
                    for condition in CONDITIONS}
            result[split][family] = cell
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--source-pin', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    need(not a.output.exists(), 'REGION_EXISTS'); a.output.mkdir()
    start = tick = time.perf_counter(); phases = {}; source = dict(path=str(a.source), pin=a.source_pin)
    try:
        gpu = subprocess.check_output(['nvidia-smi'], text=True); take_lease('29','capability-region-diagnostic'); configure()
        model, corpus, parent, prior = load(source); world = model['world']; before = fingerprint(model.state_dict())
        plan = dict(source=source, train=corpus.train, query=corpus.query, regions=corpus.regions, conditions=CONDITIONS,
                    code_sha256=sha(Path(__file__).read_bytes()), label_policy='offline_future_target_diagnostic_only')
        write_committed(a.output/'plan.json', plan)
        phases['authenticate_restore'] = time.perf_counter()-tick; tick = time.perf_counter(); rows=[]
        with torch.no_grad(), (a.output/'tasks.jsonl').open('x') as out:
            for split, keys in (('train',corpus.train), ('query',corpus.query)):
                for index, key in enumerate(keys):
                    at = time.perf_counter(); cpu = time.process_time(); task = corpus.tasks[key]
                    raw, target = corpus.target(key,world.config,'cuda',fitting=split=='train')
                    native = corpus.batch(key,'context',world.config,'cuda')
                    output = predict(world,model['request'],native,key=key,operation=task['kind'])
                    label, assignments = mathematical_labels(world,target)
                    row = dict(task=key,split=split,family=task['family'],group=task['split_group'],source_bytes=len(raw),
                        source_sha256=sha(raw),recognition=recognition(output,label,assignments))
                    if split == 'query':
                        conditions = interventions(model,corpus,key,output,target,label)
                        row['lanes'] = lane_metrics(conditions,target)
                        row['scores'] = {n:score(model['decoder'],raw,slots) for n,slots in conditions.items()}
                        for name in ('predicted','target'):
                            need(row['scores'][name] == prior[key]['scores'][name], 'REGION_PRIOR_DRIFT')
                    row.update(seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                    out.write(json.dumps(row,sort_keys=True)+'\n'); out.flush(); rows.append(row)
                    print(json.dumps(dict(split=split,index=index)),flush=True)
                phases[split] = time.perf_counter()-tick; tick = time.perf_counter()
            os.fsync(out.fileno())
        need(fingerprint(model.state_dict()) == before, 'REGION_WEIGHT_DRIFT'); world.check_reference()
        journal = parent['source']['config']; config = pinned(journal)
        need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin'], 'REGION_JOURNAL_DRIFT')
        report = dict(schema='capability-region-diagnostic-v1',plan=plan,tasks_sha256=sha((a.output/'tasks.jsonl').read_bytes()),
            families=summarize(rows),model_identity=before,model_updates=0,weights_and_journal_unchanged=True,
            gpu_before=gpu,scientific_admission=False,learned_capability=False,serving=False,generated_surface_evaluated=False,
            exposure='development_exposed_group_holdout',number_status='measured',acceptance='author-checked')
        phases['finalize'] = time.perf_counter()-tick; wall=time.perf_counter()-start
        need(sum(phases.values())/wall >= .9,'REGION_PHASE_COVERAGE'); report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc())); raise


if __name__ == '__main__': run_cli(main)

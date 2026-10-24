"""Independent arithmetic, cold update replay and query readback for experience growth."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch.nn import functional as F

from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import refused, independent_score
from audit.capability_experience_learning import load
from audit.capability_kernel_learning import parameter_hash
from audit.capability_recognition_learning import frozen
from audit.capability_structure_readback import independent_context
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_experience_learning import ARMS, PURPOSE, check_step, schedules
from model.capability_kernel_learning import pinned
from model.capability_structure_input import StructuredContexts
from model.capability_supervision import CapabilitySupervision


def geometry(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    left = np.full(len(a), -np.inf); right = np.full(len(b), -np.inf)
    for i in range(0, len(a), 256):
        for j in range(0, len(b), 256):
            scores = a[i:i+256] @ b[j:j+256].T
            left[i:i+256] = np.maximum(left[i:i+256], scores.max(axis=1))
            right[j:j+256] = np.maximum(right[j:j+256], scores.max(axis=0))
    return (math.fsum(left)/len(left) + math.fsum(right)/len(right))/2


def check_trace(row, arrays):
    a, b = arrays['correct'], arrays['shuffled']
    need(a.shape == b.shape == (row['source_bytes']+1,) and np.isfinite(a).all() and np.isfinite(b).all(),
         'EXPERIENCE_COLD_TRACE_SHAPE')
    delta = b.astype(np.float64)-a.astype(np.float64)
    expected = dict(positions=len(a), bytes=len(a)-1, mean_byte_delta=math.fsum(delta[:-1])/(len(a)-1),
        eos_delta=float(delta[-1]), mean_position_delta=math.fsum(delta)/len(delta),
        positive=int((delta > 0).sum()), negative=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    need(expected == row['delta'], 'EXPERIENCE_COLD_TRACE_DELTA')
    drift = max(abs(math.fsum(x)/len(x)-row[n]['canonical_loss']) for n,x in [('correct',a),('shuffled',b)])
    need(drift < 1e-6, 'EXPERIENCE_COLD_TRACE_NLL')
    return drift


def summary(rows):
    cells = defaultdict(list)
    for row in rows:
        for group in ('all', row['group']): cells[row['arm'], row['operation'], group].append(row)
    result = []
    for (arm, operation, group), rs in sorted(cells.items()):
        cell = dict(arm=arm, operation=operation, group=group, tasks=len(rs),
            groups=len({r['group'] for r in rs}), eligible=sum(r['eligible'] for r in rs),
            region_hits=sum(r['label'] == r['selected'] for r in rs),
            mean_nll=math.fsum(r['correct']['canonical_loss'] for r in rs)/len(rs),
            conditioning_delta=math.fsum(r['delta']['mean_byte_delta'] for r in rs)/len(rs), lanes={})
        for name in rs[0]['lanes']:
            xs = [r['lanes'][name] for r in rs]
            signs = Counter((x['target'] > x['distractor'])-(x['target'] < x['distractor']) for x in xs)
            cell['lanes'][name] = dict(mean_target=math.fsum(x['target'] for x in xs)/len(xs),
                                       wins=signs[1], ties=signs[0], losses=signs[-1])
        result.append(cell)
    return result


def replay(model, optimizer, contexts, targets, labels, plan, row):
    corpus = contexts.corpus; key, target_key = row['task'], row['target']
    check_step(corpus, plan, row['arm'], row['index'], key, target_key)
    optimizer.zero_grad(set_to_none=True); world = model['world']; kind = corpus.tasks[key]['kind']
    output, _ = independent_context(model, contexts.get(key, world.config, 'cuda'), kind, 'structured')
    label = torch.tensor([labels[target_key]['label']], device='cuda')
    chosen = world.supervised_placement(output, label,
        allowed=torch.ones(len(corpus.regions), dtype=torch.bool, device='cuda'),
        event_ids=[digest(dict(key=key, operation=kind))])
    region = []; placement = []
    for spec in world.specs:
        r = F.cross_entropy(output['regions'][spec.name], label)
        x, y = chosen.slots[spec.name], targets(target_key)[1][spec.name]
        matrix = F.normalize(x.values[0, ~x.padding[0]], dim=-1) @ F.normalize(y.values[0, ~y.padding[0]], dim=-1).T
        # Preserve the training loss's first-max subgradient at exact window ties.
        p = 1-(matrix.max(0).values.mean()+matrix.max(1).values.mean())/2
        need(row['objective']['lanes'][spec.name] == dict(region=float(r.detach()), placement=float(p.detach())),
             'EXPERIENCE_COLD_OBJECTIVE')
        region.append(r); placement.append(p)
    loss = torch.stack(region).mean()+torch.stack(placement).mean()
    need(float(loss.detach()) == row['objective']['loss'] and chosen.regions[0] == row['objective']['placement_region'],
         'EXPERIENCE_COLD_LOSS')
    loss.backward()
    need(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'EXPERIENCE_COLD_PROTECTED_GRADIENT')
    for branch, evidence in row['gradients'].items():
        def lane(n): return next((s.name for s in world.specs if '.'+s.name+'.' in n), 'shared')
        gradients = {n:p.grad for n,p in model.named_parameters() if p.grad is not None and lane(n) == branch}
        parameters = {n:p for n,p in model.named_parameters() if lane(n) == branch}
        need(fingerprint(gradients) == evidence['sha256'] and fingerprint(parameters) == evidence['parameters'],
             'EXPERIENCE_COLD_GRADIENT')
    optimizer.step()


def paired(rows):
    indexed = {(r['arm'],r['task']):r for r in rows}; result = []
    for a,b in [('full_correct','repeat_correct'), ('full_correct','full_shuffled'),
                ('repeat_correct','repeat_shuffled'), ('full_correct','initial')]:
        left = [r for r in rows if r['arm'] == a]
        groups = defaultdict(list)
        for r in left:
            groups[r['operation'],'all'].append(r); groups[r['operation'],r['group']].append(r)
        for (operation,group), rs in sorted(groups.items()):
            ds = [r['correct']['canonical_loss']-indexed[b,r['task']]['correct']['canonical_loss'] for r in rs]
            result.append(dict(left=a, right=b, operation=operation, group=group, tasks=len(rs),
                groups=len({r['group'] for r in rs}), mean_nll_delta=math.fsum(ds)/len(ds),
                lower=sum(d < 0 for d in ds), higher=sum(d > 0 for d in ds), ties=sum(d == 0 for d in ds),
                lane_mean_delta={n:math.fsum(r['lanes'][n]['target']-indexed[b,r['task']]['lanes'][n]['target']
                    for r in rs)/len(rs) for n in rs[0]['lanes']}))
    return result


def modal_baselines(corpus, plan, labels):
    result = []
    for arm in ('full_correct', 'repeat_correct'):
        for operation in plan['operation_counts']:
            training = [x['target'] for x in plan['arms'][arm] if corpus.tasks[x['task']]['kind'] == operation]
            counts = Counter(labels[k]['label'] for k in training)
            chosen = min(counts, key=lambda label: (-counts[label], label))
            queries = [k for k in corpus.query if corpus.tasks[k]['kind'] == operation]
            result.append(dict(arm=arm, operation=operation, label=chosen, train_updates=len(training),
                train_modal_count=counts[chosen], query_tasks=len(queries),
                query_region_hits=sum(labels[k]['label'] == chosen for k in queries),
                selection='TRAIN target frequencies; smallest label breaks ties; no query fitting'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pin', required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    need(not a.output.exists(), 'EXPERIENCE_COLD_EXISTS'); started = time.perf_counter()
    ref = dict(path=str(a.report), pin=a.pin); report = pinned(ref); root = a.report.parent
    for name,pin in report['source']['code'].items():
        need(sha((Path(__file__).resolve().parents[1]/name).read_bytes()) == pin, 'EXPERIENCE_COLD_CODE')
    for name,pin in report['files'].items(): need(sha((root/name).read_bytes()) == pin, 'EXPERIENCE_COLD_FILE')
    gpu = subprocess.check_output(['nvidia-smi'], text=True); take_lease('29', 'experience-learning-readback'); configure()
    model, optimizer, corpus, previous, parent, saved, identities = load(report['source'])
    supervisor = CapabilitySupervision(corpus); expected = schedules(corpus, supervisor)
    plan = json.loads((root/'plan.json').read_bytes()); labels = json.loads((root/'labels.json').read_bytes())
    need(all(plan[k] == v for k,v in expected.items()) and plan['query'] == list(corpus.query)
         and plan['region_identity'] == digest(corpus.regions) and plan['training_identity'] == corpus.training_identity,
         'EXPERIENCE_COLD_PLAN')
    need(set(labels) == set(corpus.train)|set(corpus.query), 'EXPERIENCE_COLD_LABEL_ROSTER')
    for key, label in labels.items():
        need(label['native_identity'] == identities[key], 'EXPERIENCE_COLD_LABEL_NATIVE')
        values = torch.tensor([label['assignments'][s.name] for s in corpus.specs], device='cuda')
        need(values.shape == (len(corpus.specs),len(corpus.regions)) and bool(torch.isfinite(values).all()),
             'EXPERIENCE_COLD_LABEL_SHAPE')
        joint = values.mean(0)
        need(int(joint.argmax()) == label['label'] and int((joint == joint[label['label']]).sum()) == label['ties'],
             'EXPERIENCE_COLD_LABEL')
        if key in previous:
            need(label['label'] == previous[key]['recognition']['label'] and all(label['assignments'][n] ==
                 x['assignments'] for n,x in previous[key]['recognition']['instruments'].items()), 'EXPERIENCE_COLD_OLD_LABEL')
    contexts = StructuredContexts(corpus); target_cache = {}
    def target(key):
        if key not in target_cache: target_cache[key] = corpus.target(key,model['world'].config,'cuda',fitting=key in corpus.train)
        return target_cache[key]
    # Independent bank arithmetic for one newly covered target per operation.
    label_checks = []
    for operation in plan['operation_counts']:
        key = next(k for k in corpus.train if k not in previous and corpus.tasks[k]['kind'] == operation)
        for i,spec in enumerate(corpus.specs):
            truth = F.normalize(target(key)[1][spec.name].values[0],dim=-1)
            values = getattr(model['world'],f'world_values_{i}'); masks = getattr(model['world'],f'world_padding_{i}')
            scores = []
            for vector,mask in zip(values,masks):
                matrix = truth @ F.normalize(vector[~mask],dim=-1).T
                scores.append(float((matrix.amax(0).mean()+matrix.amax(1).mean())/2))
            need(scores == labels[key]['assignments'][spec.name], 'EXPERIENCE_COLD_NEW_LABEL')
        label_checks.append(key)
    initial = torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
    need(sha((root/'initial.pt').read_bytes()) == report['initial_checkpoint'] and
         fingerprint(initial['model']) == fingerprint(model.state_dict()) and
         fingerprint(initial['optimizer']) == initial['optimizer_hash'] == fingerprint(optimizer.state_dict()) and
         initial['updates'] == saved['updates'] == report['inherited_updates'], 'EXPERIENCE_COLD_INITIAL')
    protected = frozen(model,optimizer); need(protected == report['protected_identity'], 'EXPERIENCE_COLD_PROTECTED')
    updates = []; cold_queries = []; records = []; controls = []; max_geometry = 0.; max_trace = 0.; costs = {}
    for arm in ('initial', *ARMS):
        if arm == 'initial':
            final = initial
        else:
            r = report['arms'][arm]; entries = plan['arms'][arm]; directory = root/arm
            data = (directory/'steps.jsonl').read_bytes(); need(sha(data) == r['steps_sha256'], 'EXPERIENCE_COLD_STEPS_PIN')
            steps = list(map(json.loads,data.splitlines())); need(len(steps) == len(entries) == r['updates'], 'EXPERIENCE_COLD_UPDATES')
            chain = report['initial_checkpoint']; before_hash = initial['parameter_hash']
            for index,row in enumerate(steps):
                check_step(corpus,plan,arm,index,row['task'],row['target'])
                need(row['index'] == index and row['arm'] == arm and row['previous'] == chain and row['before'] == before_hash
                     and digest({k:v for k,v in row.items() if k != 'sha256'}) == row['sha256'], 'EXPERIENCE_COLD_CHAIN')
                need(row['objective']['purpose'] == plan['decisions'][row['target']], 'EXPERIENCE_COLD_PURPOSE')
                chain, before_hash = row['sha256'], row['after']
            boundaries = sorted({0,len(entries)-1,*[next(i for i,x in enumerate(entries) if corpus.tasks[x['task']]['kind'] == k)
                                                      for k in plan['operation_counts']]})
            need(r['boundaries'] == boundaries and r['last_step'] == chain, 'EXPERIENCE_COLD_BOUNDARIES')
            for index in boundaries:
                before = initial if index == 0 else torch.load(directory/f'before-{index}.pt',map_location='cpu',weights_only=True)
                after = torch.load(directory/f'after-{index}.pt',map_location='cpu',weights_only=True)
                need(before['source'] == after['source'] == report['source'] and before['updates'] == saved['updates']+index
                     and after['updates'] == before['updates']+1 and after['previous'] == steps[index]['sha256'], 'EXPERIENCE_COLD_CHECKPOINT')
                model.load_state_dict(before['model'],strict=True); optimizer.load_state_dict(before['optimizer'])
                need(parameter_hash(model) == steps[index]['before'] == before['parameter_hash'] and
                     fingerprint(optimizer.state_dict()) == before['optimizer_hash'] and frozen(model,optimizer) == protected,
                     'EXPERIENCE_COLD_BEFORE')
                replay(model,optimizer,contexts,target,labels,plan,steps[index])
                need(fingerprint(model.state_dict()) == fingerprint(after['model']) and parameter_hash(model) == steps[index]['after']
                     and fingerprint(optimizer.state_dict()) == after['optimizer_hash'] == fingerprint(after['optimizer'])
                     and frozen(model,optimizer) == protected, 'EXPERIENCE_COLD_AFTER')
                updates.append(dict(arm=arm,index=index,loss_gradient_parameters_moments_exact=True))
            need(sha((directory/'final.pt').read_bytes()) == r['final_checkpoint'], 'EXPERIENCE_COLD_FINAL_PIN')
            final = torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
            need(final['previous'] == chain and final['parameter_hash'] == before_hash == r['final_parameters']
                 and fingerprint(final['model']) == fingerprint(model.state_dict())
                 and final['optimizer_hash'] == fingerprint(final['optimizer']) == r['optimizer'], 'EXPERIENCE_COLD_FINAL')
            costs[arm] = dict(step_wall_seconds=math.fsum(x['seconds'] for x in steps),
                             step_cpu_seconds=math.fsum(x['cpu_seconds'] for x in steps))
        model.load_state_dict(final['model'],strict=True); optimizer.load_state_dict(final['optimizer'])
        need(frozen(model,optimizer) == protected, 'EXPERIENCE_COLD_PARENT_DRIFT')
        data = (root/arm/'queries.jsonl').read_bytes()
        pin = report['files']['initial/queries.jsonl'] if arm == 'initial' else report['arms'][arm]['queries_sha256']
        need(sha(data) == pin, 'EXPERIENCE_COLD_QUERY_PIN'); rows = list(map(json.loads,data.splitlines()))
        need([r['task'] for r in rows] == list(corpus.query), 'EXPERIENCE_COLD_QUERY_ROSTER')
        selected = {min(k for k in corpus.query if corpus.tasks[k]['kind'] == kind) for kind in plan['operation_counts']}
        with torch.no_grad():
            for row in rows:
                key = row['task']; raw, truth = target(key); donor = plan['donors'][key]; task = corpus.tasks[key]
                need(row['arm'] == arm and row['donor'] == donor and row['source_sha256'] == sha(raw) and row['source_bytes'] == len(raw)
                     and row['label'] == labels[key]['label'] and row['label_ties'] == labels[key]['ties']
                     and row['group'] == task['split_group'] and row['operation'] == task['kind'] and row['family'] == task['family']
                     and row['eligible'] == plan['decisions'][key]['allowed'] and row['purpose'] == PURPOSE[task['kind']],
                     'EXPERIENCE_COLD_QUERY_BINDING')
                need(0 <= row['selected'] < len(corpus.regions) and set(row['lanes']) == {s.name for s in corpus.specs},
                     'EXPERIENCE_COLD_LANE_ROSTER')
                path = root/arm/row['file']; need(path.name == key+'.npz' and sha(path.read_bytes()) == row['file_sha256'], 'EXPERIENCE_COLD_ARRAY_PIN')
                with np.load(path,allow_pickle=False) as arrays:
                    need(set(arrays.files) == {'correct','shuffled'} | {tag+'__'+s.name for tag in ('predicted','wrong') for s in corpus.specs},
                         'EXPERIENCE_COLD_ARRAY_ROSTER')
                    for tag in ('predicted','wrong'):
                        for spec in corpus.specs:
                            v = arrays[tag+'__'+spec.name]
                            need(v.ndim == 2 and v.shape[0] > 0 and v.shape[1] == spec.dims and np.isfinite(v).all(),
                                 'EXPERIENCE_COLD_ARRAY_SHAPE')
                    max_trace = max(max_trace,check_trace(row,arrays))
                    for n in row['lanes']:
                        for metric, slots in [('target',truth),('distractor',target(donor)[1])]:
                            slot = slots[n]; actual = slot.values[0,~slot.padding[0]].cpu().numpy()
                            difference = abs(geometry(arrays['predicted__'+n],actual)-row['lanes'][n][metric])
                            need(difference < 1e-6,'EXPERIENCE_COLD_GEOMETRY'); max_geometry = max(max_geometry,difference)
                    if key in selected:
                        for tag,address,score in [('predicted',key,'correct'),('wrong',donor,'shuffled')]:
                            output,_ = independent_context(model,contexts.get(address,model['world'].config,'cuda'),task['kind'],'structured')
                            predicted = output['predicted_constellation'].slots
                            for n,slot in predicted.items():
                                need(np.array_equal(arrays[tag+'__'+n],slot.values[0,~slot.padding[0]].cpu().numpy()), 'EXPERIENCE_COLD_FORWARD')
                            need(independent_score(model['decoder'],raw,predicted,skipped=False,size=row[score]['physical_batch']) ==
                                 row[score]['canonical_loss'], 'EXPERIENCE_COLD_DECODER')
                            if tag == 'predicted': need(int(next(iter(output['world_owner'].values()))[0]) == row['selected'], 'EXPERIENCE_COLD_SELECTION')
                        cold_queries.append(dict(arm=arm,task=key,forward_arrays_decoder_exact=True))
            need(frozen(model,optimizer) == protected, 'EXPERIENCE_COLD_EVAL_DRIFT')
        records.extend(rows)
    need(summary(records) == report['summary'] and len(records) == report['rows'], 'EXPERIENCE_COLD_SUMMARY')
    controls.append(refused('EXPERIENCE_FIT_QUERY',lambda:check_step(corpus,plan,ARMS[0],0,corpus.query[0],corpus.query[0])))
    wrong = copy.deepcopy(plan); wrong['arms'][ARMS[0]][0]['target'] = corpus.query[0]
    entry = plan['arms'][ARMS[0]][0]
    controls.append(refused('EXPERIENCE_FIT_SCHEDULE',lambda:check_step(corpus,wrong,ARMS[0],0,entry['task'],entry['target'])))
    denied = copy.deepcopy(plan); denied['decisions'][entry['target']]['allowed'] = False
    controls.append(refused('EXPERIENCE_FIT_PURPOSE',lambda:check_step(corpus,denied,ARMS[0],0,entry['task'],entry['target'])))
    failed = next(k for k in corpus.train if supervisor.evidence(k)['outcome'] == 0)
    controls.append(refused('SUPERVISION_FAILED_ACTION',lambda:supervisor.training_target(failed,'successful_patch',model['world'].config,'cuda')))
    damaged = copy.deepcopy(final['optimizer']); next(iter(damaged['state'].values()))['exp_avg'].reshape(-1)[0] += 1
    controls.append(refused('EXPERIENCE_COLD_MOMENT',lambda:need(fingerprint(damaged) == final['optimizer_hash'],'EXPERIENCE_COLD_MOMENT')))
    row = records[0]
    with np.load(root/row['arm']/row['file'],allow_pickle=False) as a0: damaged = {n:a0[n].copy() for n in a0.files}
    damaged['correct'][0] += 1
    controls.append(refused('EXPERIENCE_COLD_TRACE_DELTA',lambda:check_trace(row,damaged)))
    need(sha(Path(corpus.config['journal']['path']).read_bytes()) == corpus.config['journal']['pin'], 'EXPERIENCE_COLD_JOURNAL')
    result = dict(schema='capability-experience-learning-readback-v1',source=ref,reader_sha256=sha(Path(__file__).read_bytes()),
        rows=len(records),steps=sum(r['updates'] for r in report['arms'].values()),updates=updates,cold_queries=cold_queries,
        direct_new_label_checks=label_checks,all_label_assignments_recounted=len(labels),controls=controls,
        independent_geometry_max_absolute=max_geometry,trace_nll_max_absolute=max_trace,summary=summary(records),
        comparisons=paired(records),modal_baselines=modal_baselines(corpus,plan,labels),
        cost=costs,seconds=time.perf_counter()-started,gpu_before=gpu,
        protected_state_exact=True,scientific_admission=False,learned_capability=False,serving=False,
        number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output,result),rows=len(records),steps=result['steps'])),flush=True)


if __name__ == '__main__': run_cli(main)

"""Cold reconstruction without the new update, forward or saved-tensor hooks."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import full_history_source as source
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from model.native_placement_checked import validate_optimizer
from model.native_requested_answer import update as old_update
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need


def same(a, b, code):
    need(fingerprint(a) == fingerprint(b), code)


def replay(s, model, opt, value, request, step, full, saved):
    need(fingerprint(model.state_dict()) == saved['record']['before'], 'FULL_COLD_BEFORE')
    opt.zero_grad(set_to_none=True); c = len(value['context']); a = len(value['answers'])
    noise = torch.randn(a, s['policy']['samples'], s['policy']['noise_width'],
                        generator=torch.Generator().manual_seed(s['policy']['noise_seed']+step))
    first = {}; handles = []
    def capture(name):
        def hook(module, args, output):
            if name not in first: first[name] = output
        return hook
    for n,projection in model.forecaster.predictor.projectors.items():
        handles.append(projection.register_forward_hook(capture(n)))
    stream = model.stream(request); cache = None; losses = []; count = 0
    try:
        for position, frame in enumerate((*value['context'], *value['answers'])):
            with torch.set_grad_enabled(full or position >= c-1):
                h, cache = stream.step((frame,), context='cold-complete-answer', offset=position,
                                       state=cache, detach_history=not full)
            if position < c-1: continue
            index = position-c+1; end = index == a; recorded = saved['record']['records'][count]; count += 1
            need(recorded['position'] == position and recorded['answer_index'] == (None if end else index)
                 and recorded['end'] == end, 'FULL_COLD_SCHEDULE')
            stop = model.stop_loss(h, end); same(stop, recorded['stop'], 'FULL_COLD_STOP')
            loss = stop['loss']/(a+1)
            if not end:
                cont = model.forecaster.losses(h, value['codes'][index:index+1],
                    {n:x[index:index+1] for n,x in value['counts'].items()}, (value['answers'][index],),
                    noise[index:index+1], target=s['target'])
                need(float(cont['loss'].detach()) == recorded['continuation'], 'FULL_COLD_PLACEMENT')
                loss = loss+cont['loss']/a
            else: need(recorded['continuation'] is None, 'FULL_COLD_END')
            need(float(loss.detach()) == recorded['weighted_loss'], 'FULL_COLD_WEIGHTED_LOSS')
            if full: losses.append(loss)
            else: loss.backward()
        need(count == saved['record']['decisions'] == a+1, 'FULL_COLD_COVERAGE')
        if full:
            total = torch.stack(losses).sum()
            end_grads = torch.autograd.grad(losses[-1], tuple(first.values()), retain_graph=True)
            early_grads = torch.autograd.grad(total, tuple(first.values()), retain_graph=True)
            early = {n:dict(value=cpu(t), requires_grad=t.requires_grad, end_gradient=cpu(e), gradient=cpu(g))
                     for (n,t),e,g in zip(first.items(), end_grads, early_grads)}
            total.backward()
        else:
            early = {n:dict(value=cpu(t), requires_grad=t.requires_grad, end_gradient=None, gradient=None)
                     for n,t in first.items()}
        same(early, saved['early'], 'FULL_COLD_EARLY_CREDIT')
        gradients = {n:p.grad for n,p in model.named_parameters()}
        same(gradients, saved['gradients'], 'FULL_COLD_GRADIENTS')
        need(fingerprint(gradients) == saved['record']['gradients'], 'FULL_COLD_GRADIENT_PIN')
        for prefix,pin in saved['record']['branch_gradients'].items():
            group = {n:g for n,g in gradients.items() if n.startswith(prefix)}
            need(group and any(bool(g.abs().any()) for g in group.values()) and fingerprint(group) == pin,
                 'FULL_COLD_BRANCH')
        opt.step(); validate_optimizer(model, opt, s['policy'], step+1)
        same(model.state_dict(), saved['model'], 'FULL_COLD_MODEL')
        same(opt.state_dict(), saved['optimizer'], 'FULL_COLD_OPTIMIZER')
        need(fingerprint(model.state_dict()) == saved['record']['after'] and
             fingerprint(opt.state_dict()) == saved['record']['optimizer'], 'FULL_COLD_FINAL_PIN')
    finally:
        for h in handles: h.remove()
        first.clear()
    return count


def score(left, right):
    a, b = left.double(), right.double()
    need(bool((a.norm(dim=1)>0).all()) and bool((b.norm(dim=1)>0).all()), 'FULL_COLD_ZERO_NATIVE')
    similarity = (a/a.norm(dim=1, keepdim=True))@(b/b.norm(dim=1, keepdim=True)).T
    similarity = similarity.clamp(-1, 1)
    return min(float(similarity.max(0).values.mean()), float(similarity.max(1).values.mean()))


def autonomous(s, model, saved):
    model.eval(); groups = defaultdict(list)
    expected = {(k,p,m) for k,d in s['items'].items() if d['side'] != 'train'
                for p in model.proposals for m in ('correct','reversed','train_donor','wrong_operation')}
    seen = set()
    for row in saved:
        d = s['items'][row['document']]; key = (d['document'],row['proposal'],row['mode'])
        need(key in expected and key not in seen, 'FULL_COLD_EVAL_SUPPORT'); seen.add(key)
        prefix = tuple(d['frames'][:2]); desc = legal(2)
        if row['mode'] == 'reversed': prefix = prefix[::-1]
        elif row['mode'] == 'train_donor':
            donor = sorted(k for k in s['train'] if s['items'][k]['component'] != d['component'])[0]
            prefix = tuple(s['items'][donor]['frames'][:2])
        elif row['mode'] == 'wrong_operation': desc['operation'] = 'respond_to_source_question'
        generator = model.rollout(prefix, request=NativeRequest(desc), target=s['target'], proposal=row['proposal'],
                                  context='count-answer-proof', seed=20261024+d['document'])
        events = []; refusal = None
        try:
            for _ in range(len(d['frames'])-1):
                event = next(generator); events.append(cpu(event))
                if event['kind'] == 'end': break
        except ValueError as exc: refusal = str(exc)
        finally: generator.close()
        same(events, row['events'], 'FULL_COLD_AUTONOMOUS'); need(refusal == row['refusal'], 'FULL_COLD_REFUSAL')
        m = row['metrics']; predicted = [e for e in events if e['kind'] == 'constellation']
        future = d['frames'][2:]; ended = bool(events and events[-1]['kind'] == 'end')
        need(m['expected_atoms'] == len(future) and m['generated_atoms'] == len(predicted)
             and m['ended'] == ended and m['ended_at_reference_boundary'] == (ended and len(predicted)==len(future))
             and m['missing'] == max(0,len(future)-len(predicted)) and m['extra'] == max(0,len(predicted)-len(future)),
             'FULL_COLD_SEQUENCE_COUNTS')
        for n in model.roster:
            for i,f in enumerate(future):
                x = score(predicted[i]['slots'][n], f[n]) if i < len(predicted) else -1.
                need(abs(x-m['per_slot'][n][i]) < 1e-12 and
                     abs(score(d['frames'][1][n], f[n])-m['copy_current'][n][i]) < 1e-12, 'FULL_COLD_NATIVE_SCORE')
        groups[(row['side'], row['proposal'], row['mode'])].append(m)
    need(seen == expected, 'FULL_COLD_EVAL_DENOMINATOR')
    return {'/'.join(key):dict(n=len(rows), correct_end=sum(r['ended_at_reference_boundary'] for r in rows),
        ended=sum(r['ended'] for r in rows), censored=sum(r['censored'] for r in rows),
        missing=sum(r['missing'] for r in rows), extra=sum(r['extra'] for r in rows),
        per_slot={n:sum(sum(r['per_slot'][n])/r['expected_atoms'] for r in rows)/len(rows) for n in model.roster},
        copy_current={n:sum(sum(r['copy_current'][n])/r['expected_atoms'] for r in rows)/len(rows) for n in model.roster})
        for key,rows in groups.items()}


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.pin, 'FULL_COLD_PIN')
    report = json.loads((a.input/'report.json').read_bytes()); need(report['complete'], 'FULL_COLD_COMPLETE')
    for n,h in report['files'].items(): need(file_sha(a.input/n) == h, 'FULL_COLD_BYTES')
    s = source.load(a.root); need(s['code'] == report['code'] and s['train'] == report['train'], 'FULL_COLD_SOURCE')
    phases['authentication'] = time.perf_counter()-tick; total = rollouts = updates = 0; results = {}
    load = lambda p:torch.load(p, map_location='cpu', weights_only=True)
    for architecture in source.prior.ARCHITECTURES:
        tick = time.perf_counter(); model, opt = source.prior.make(s, architecture)
        value, request = source.prior.item(s, s['train'][0])
        row, grad = old_update(model, opt, value, 5, s['policy'], request=request, target=s['target'])
        same(dict(records=row['records'], gradients=grad, model=model.state_dict(), optimizer=opt.state_dict()),
             load(a.input/(architecture+'-original.pt')), 'FULL_COLD_ORIGINAL')
        phases[architecture+'_original'] = time.perf_counter()-tick
        first_rows = None
        for full in (False, True):
            key = architecture+('/full' if full else '/detached'); model, opt = source.prior.make(s, architecture)
            need(fingerprint((model.state_dict(),opt.state_dict())) == report['arms'][key]['initial'], 'FULL_COLD_INITIAL')
            tick = time.perf_counter()
            for cursor,doc in enumerate(s['train']):
                saved = load(a.input/key/f'update-{6+cursor}.pt'); value, request = source.prior.item(s, doc)
                same(saved['binding'], dict(source=source.prior.PARENT[1], region=s['pin'], code=s['code'],
                    architecture=architecture, full=full, document=doc, step=6+cursor), 'FULL_COLD_BINDING')
                if cursor == 0:
                    if not full: first_rows = saved['record']['records']
                    else: same(saved['record']['records'], first_rows, 'FULL_COLD_INITIAL_FORWARD')
                total += replay(s, model, opt, value, request, 5+cursor, full, saved); updates += 1
                if cursor == 1: model, opt = source.prior.make(s, architecture, saved, step=7)
            phases[key+'_updates'] = time.perf_counter()-tick; tick = time.perf_counter()
            evaluation = load(a.input/key/'evaluation.pt'); results[key] = autonomous(s,model,evaluation); rollouts += len(evaluation)
            phases[key+'_autonomous'] = time.perf_counter()-tick
            need(fingerprint(model.state_dict()) == report['arms'][key]['final'] and
                 fingerprint(opt.state_dict()) == report['arms'][key]['optimizer'], 'FULL_COLD_FINAL')
    tick = time.perf_counter(); negative = {}
    sample = load(a.input/'shared/full/update-6.pt')
    for field in ('early','gradients','model','optimizer'):
        bad = deepcopy(sample[field])
        if field == 'early': next(iter(bad.values()))['end_gradient'].add_(1)
        elif field in ('gradients','model'): next(iter(bad.values())).add_(1)
        else: next(iter(bad['state'].values()))['exp_avg'].add_(1)
        negative[field] = rejected(lambda:same(bad,sample[field],'FULL_COLD_CHANGED_'+field), 'FULL_COLD_CHANGED_'+field)
    s['fresh'](); need((updates,total,rollouts) == (30,396,640), 'FULL_COLD_TOTAL')
    phases['final_source_controls'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'FULL_COLD_PHASES')
    result = dict(complete=True, source=dict(path=str(a.input),pin=a.pin), code=s['code'], updates=updates,
        decisions=total,rollouts=rollouts,results=results,controls=negative,all_updates_exact=True,
        saved_tensor_hooks_not_used=True,all_native_scores_recounted=True,all_autonomous_trajectories_exact=True,
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        capability_demonstrated=False,acceptance='author-checked',promoted=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

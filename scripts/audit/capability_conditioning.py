"""Frozen-weight decoder conditioning interventions on retained capability experience."""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from torch.nn import functional as F
from audit.capability_kernel_learning import system
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, pinned, predict
from model.loop_training_checkpoint import tensor_fingerprint
from model.stream_surface import batches


def read(path): return json.loads(Path(path).read_bytes())
def fingerprint(value): return digest(tensor_fingerprint(value))


def select_donors(corpus):
    result = {}
    for key in corpus.query:
        task = corpus.tasks[key]
        eligible = [d for d in corpus.train if corpus.tasks[d]['kind'] == task['kind']
                    and corpus.tasks[d]['split_group'] != task['split_group']]
        need(eligible, 'CONDITIONING_NO_DISJOINT_DONOR')
        result[key] = min(eligible, key=lambda d:sha((key + ':' + d).encode()))
    return result


def absent(native, name):
    need(name in native, 'CONDITIONING_ABSENT_NAME')
    result = dict(native); x = native[name]
    result[name] = replace(x, values=torch.zeros_like(x.values), padding=torch.ones_like(x.padding),
                           absent=torch.ones_like(x.absent))
    return result


@contextmanager
def bypass(decoder, enabled):
    """Diagnostic intervention at branch output; never changes parameter bytes."""
    hooks = []
    try:
        if enabled:
            for block in decoder.blocks:
                for branch in block.branches.values():
                    hooks.append(branch.register_forward_hook(lambda module, args, output:args[0]))
        yield
    finally:
        for hook in hooks: hook.remove()


def score(decoder, raw, native, *, bypass_branches=False):
    # Same arithmetic and batch of eight as training wherever it physically fits.
    # A longer real context reduces physical batch size, never its native coverage.
    elements = sum(x.values.numel() for x in native.values())
    widest = max(x.values.shape[1] for x in native.values())
    c = decoder.config
    size = min(8, c.max_native_elements//elements,
               c.max_attention_elements//(c.heads*c.max_prefix*max(c.max_prefix, widest)))
    need(size > 0, 'CONDITIONING_PHYSICAL_BATCH')
    value = 0.; ranges = []; windows = 0
    with bypass(decoder, bypass_branches):
        for prefix, labels, offsets, slots, items in batches(raw, native, c, batch_size=size):
            logits = decoder(prefix, slots, offsets=offsets)['logits']
            loss = F.cross_entropy(logits.reshape(-1, c.vocabulary), labels.reshape(-1),
                                   ignore_index=-100, reduction='sum') / (len(raw)+1)
            need(bool(torch.isfinite(loss)), 'CONDITIONING_NONFINITE')
            value += float(loss); windows += len(items)
            ranges.extend(i['scored'] for i in items)
    count = sum(b-a for a,b in ranges)
    need(count == len(raw)+1 and ranges[0][0] == 0 and ranges[-1][1] == count
         and all(a[1] == b[0] for a,b in zip(ranges,ranges[1:])), 'CONDITIONING_COVERAGE')
    return dict(loss=value, positions=count, windows=windows, physical_batch=size,
                native_windows={n:int((~x.padding).sum()) for n,x in native.items()},
                condition_sha256=fingerprint({n:dict(identity=x.identity, values=x.values,
                    padding=x.padding, absent=x.absent) for n,x in native.items()}))


def inputs(model, corpus, key, donor):
    world = model['world']; c = world.config; task = corpus.tasks[key]
    native = corpus.batch(key, 'context', c, 'cuda')
    other = corpus.batch(donor, 'context', c, 'cuda')
    actual = predict(world, model['request'], native, key=key, operation=task['kind'])
    wrong = predict(world, model['request'], other, key=key, operation=task['kind'])
    raw, truth = corpus.target(key, c, 'cuda', fitting=False)
    _, wrong_truth = corpus.target(donor, c, 'cuda', fitting=True)
    predicted = actual['predicted_constellation'].slots
    conditions = dict(predicted=predicted, wrong_context=wrong['predicted_constellation'].slots,
                      current=native, target=truth, wrong_target=wrong_truth, bypass=predicted)
    conditions.update({'without:'+s.name:absent(predicted,s.name) for s in world.specs})
    return raw, conditions, dict(actual=int(actual['constellation_regions'].argmax()),
                                wrong_context=int(wrong['constellation_regions'].argmax()))


def summarize(rows):
    result = {}
    for family in sorted({r['family'] for r in rows}):
        cell = {}; selected = [r for r in rows if r['family'] == family]
        for arm in sorted({r['arm'] for r in selected}):
            values = [r for r in selected if r['arm'] == arm]; metrics = {}
            for condition in values[0]['scores']:
                losses = [r['scores'][condition]['loss'] for r in values]
                delta = [r['scores'][condition]['loss']-r['scores']['predicted']['loss'] for r in values]
                metrics[condition] = dict(mean_loss=math.fsum(losses)/len(losses),
                    mean_delta_from_predicted=math.fsum(delta)/len(delta),
                    better=sum(x < 0 for x in delta), worse=sum(x > 0 for x in delta), ties=sum(x == 0 for x in delta))
            cell[arm] = dict(tasks=len(values), groups=len({r['group'] for r in values}), conditions=metrics)
        result[family] = cell
    return result


def sources(config_path, experiments_path, evaluation_ref):
    config = read(config_path); corpus = KernelLearningCorpus(config)
    experiments = read(experiments_path); reports = {n:pinned(r) for n,r in experiments.items()}
    need(set(reports) == {'correct','shuffled'}, 'CONDITIONING_ARMS')
    prior = pinned(evaluation_ref)
    need(prior['experiments'] == experiments and prior['config_sha256'] == sha(Path(config_path).read_bytes()),
         'CONDITIONING_PRIOR_BINDING')
    path = Path(evaluation_ref['path']).parent/'queries.jsonl'
    need(sha(path.read_bytes()) == prior['query_records_sha256'], 'CONDITIONING_PRIOR_ROWS')
    previous = {r['task']:r for r in map(json.loads,path.read_bytes().splitlines())}
    need(set(previous) == set(corpus.query), 'CONDITIONING_PRIOR_POPULATION')
    for report in reports.values():
        need(report['source']['config'] == sha(Path(config_path).read_bytes()), 'CONDITIONING_CONFIG')
        for name,pin in report['source']['code'].items():
            need(sha((Path(__file__).resolve().parents[1]/name).read_bytes()) == pin, 'CONDITIONING_CODE')
    return config, corpus, experiments, reports, previous


def load_arm(model, arm, experiments, reports):
    path = Path(experiments[arm]['path']).parent/'final.pt'
    need(sha(path.read_bytes()) == reports[arm]['final_checkpoint'], 'CONDITIONING_CHECKPOINT')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(saved['source'] == reports[arm]['source'], 'CONDITIONING_CHECKPOINT_SOURCE')
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False).eval()
    return fingerprint(model.state_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--experiments', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--evaluation-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'CONDITIONING_EXISTS'); a.output.mkdir()
    started = tick = time.perf_counter(); phases = {}
    try:
        evaluation_ref = dict(path=str(a.evaluation), pin=a.evaluation_pin)
        config,corpus,experiments,reports,prior = sources(a.config,a.experiments,evaluation_ref)
        donors = select_donors(corpus)
        source = dict(config=dict(path=str(a.config),pin=sha(a.config.read_bytes())),
            experiments=dict(path=str(a.experiments),pin=sha(a.experiments.read_bytes())), evaluation=evaluation_ref,
            code_sha256=sha(Path(__file__).read_bytes()))
        plan = dict(source=source, queries=list(corpus.query), donors=donors,
            donor_frequency=dict(Counter(donors.values())), selection='sha256_query_donor_within_operation_train_disjoint_group',
            no_bijective_permutation_claim=True, conditions=12, model_updates=0)
        write_committed(a.output/'plan.json', plan)
        phases['authenticate'] = time.perf_counter()-tick; tick=time.perf_counter()
        gpu = subprocess.check_output(['nvidia-smi'],text=True)
        take_lease('29','capability-conditioning'); configure(); model=system(config,corpus)
        phases['initialize'] = time.perf_counter()-tick; tick=time.perf_counter(); records=[]
        with torch.no_grad(), (a.output/'queries.jsonl').open('x') as out:
            for arm in ('correct','shuffled'):
                before = load_arm(model,arm,experiments,reports)
                for index,key in enumerate(corpus.query):
                    at=time.perf_counter(); cpu=time.process_time(); donor=donors[key]; task=corpus.tasks[key]
                    raw, conditions, regions = inputs(model,corpus,key,donor)
                    row = dict(task=key,arm=arm,family=task['family'],group=task['split_group'],operation=task['kind'],
                        donor=donor,source_sha256=sha(raw),source_bytes=len(raw),regions=regions,scores={})
                    for name,native in conditions.items():
                        row['scores'][name] = score(model['decoder'],raw,native,bypass_branches=name=='bypass')
                    need(row['scores']['predicted']['loss'] == prior[key]['arms'][arm]['surface_loss'],
                         'CONDITIONING_BASELINE_DRIFT')
                    row.update(seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                    out.write(json.dumps(row,sort_keys=True)+'\n'); out.flush(); records.append(row)
                    print(json.dumps(dict(arm=arm,query=index,family=task['family'])),flush=True)
                need(fingerprint(model.state_dict()) == before, 'CONDITIONING_WEIGHT_DRIFT')
        phases['interventions'] = time.perf_counter()-tick; tick=time.perf_counter()
        need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin'], 'CONDITIONING_JOURNAL')
        metrics=summarize(records); phases['finalize']=time.perf_counter()-tick
        wall=time.perf_counter()-started; need(sum(phases.values())/wall >= .9,'CONDITIONING_PHASES')
        result=dict(schema='capability-conditioning-v1',source=source,query_records_sha256=sha((a.output/'queries.jsonl').read_bytes()),
            families=metrics,rows=len(records),all_baseline_losses_exact=True,weights_and_journal_unchanged=True,
            phases=phases,seconds=wall,gpu_before=gpu,model_updates=0,scientific_admission=False,learned_capability=False,
            serving=False,generated_surface_evaluated=False,exposure='development_exposed_group_holdout',
            oracle_input_is_not_an_upper_bound=True,number_status='measured',acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',result))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()))
        raise


if __name__ == '__main__': run_cli(main)

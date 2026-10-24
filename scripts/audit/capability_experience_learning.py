"""Matched experience expansion/repetition with protected parent state and exact receipts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit.capability_conditioning import fingerprint
from audit.capability_experience_evaluation import evaluate, summarize
from audit.capability_kernel_learning import branch_evidence, parameter_hash, save_checkpoint
from audit.capability_recognition_learning import frozen
from audit.capability_structure_learning import restore
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_experience import ExperienceCorpus
from model.capability_experience_learning import ARMS, schedules, objective
from model.capability_kernel_learning import pinned
from model.capability_objective import mathematical_labels
from model.capability_structure_input import StructuredContexts
from model.capability_supervision import CapabilitySupervision


def load(source):
    coverage = pinned(source['coverage']); reader = pinned(source['coverage_readback'])
    need(reader['source'] == source['coverage'] and reader['kernel_scope_compatibility']['shared_native_exact'],
         'EXPERIENCE_LEARN_COVERAGE')
    for name, pin in coverage['source']['code'].items():
        need(sha((Path(__file__).resolve().parents[1]/name).read_bytes()) == pin, 'EXPERIENCE_LEARN_SOURCE_CODE')
    config = pinned(coverage['source']['learning'])
    model, optimizer, original, previous, parent, saved = restore(source['structure'], source['structure_readback'])
    corpus = ExperienceCorpus(config, training_scope='all_measured')
    need(original.config == corpus.config and original.regions == corpus.regions and original.query == corpus.query,
         'EXPERIENCE_LEARN_PARENT_BANK')
    raw = (Path(source['coverage']['path']).parent/'experience.jsonl').read_bytes()
    need(sha(raw) == coverage['files']['experience.jsonl'], 'EXPERIENCE_LEARN_NATIVE_IDENTITIES')
    identities = {r['task']:r['target_native'] for r in map(json.loads, raw.splitlines())}
    return model, optimizer, corpus, previous, parent, saved, identities


def materialize(model, corpus, previous, identities):
    contexts = StructuredContexts(corpus); census = contexts.census(); c = model['world'].config
    target_bytes = sum(sum(corpus.field(corpus.tasks[k]['population'], f['id'], s.name).nbytes
        for f in corpus.tasks[k]['selections']['targets']['fields'] for s in corpus.specs)
        for k in (*corpus.train, *corpus.query))
    free, total = torch.cuda.mem_get_info()
    need(census['native_bytes'] + target_bytes + c.max_native_elements*4 < free, 'EXPERIENCE_LEARN_MEMORY')
    census.update(target_native_bytes=target_bytes, free_before=free, device_total=total)
    targets = {}; labels = {}; cache = {}
    for key, row in previous.items():
        r = row['recognition']; assignments = {n:x['assignments'] for n,x in r['instruments'].items()}
        value = dict(label=r['label'], ties=r['target_ties'], assignments=assignments)
        identity = identities[key]
        need(identity not in cache or cache[identity] == value, 'EXPERIENCE_LEARN_LABEL_ALIAS')
        cache[identity] = value
    reused = len(cache)
    with torch.no_grad():
        for index, key in enumerate((*corpus.train, *corpus.query)):
            contexts.get(key, c, 'cuda')
            targets[key] = corpus.target(key, c, 'cuda', fitting=key in corpus.train)
            identity = identities[key]
            if identity not in cache:
                label, assignments = mathematical_labels(model['world'], targets[key][1])
                joint = torch.stack(list(assignments.values())).mean(0)
                cache[identity] = dict(label=int(label), ties=int((joint == joint[int(label)]).sum()),
                                       assignments={n:v.cpu().tolist() for n,v in assignments.items()})
            labels[key] = dict(native_identity=identity, **cache[identity])
            if key in previous: need(labels[key]['label'] == previous[key]['recognition']['label'], 'EXPERIENCE_LEARN_OLD_LABEL')
            if index % 100 == 0: print(json.dumps(dict(stage='materialize', index=index)), flush=True)
    census.update(unique_target_labels=len(cache), reused_labels=reused)
    return contexts, targets, labels, census


def step(model, optimizer, contexts, targets, labels, plan, arm, index, entry):
    optimizer.zero_grad(set_to_none=True); before = parameter_hash(model)
    loss, metrics = objective(model, contexts, targets, labels, plan, arm, index, entry['task'], entry['target'])
    loss.backward()
    need(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'EXPERIENCE_LEARN_PROTECTED_GRADIENT')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'EXPERIENCE_LEARN_GRADIENT')
    gradients = branch_evidence(model); optimizer.step(); after = parameter_hash(model)
    need(before != after and all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'EXPERIENCE_LEARN_UPDATE')
    return dict(**entry, arm=arm, index=index, before=before, after=after, objective=metrics, gradients=gradients)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    need(not a.output.exists(), 'EXPERIENCE_LEARN_EXISTS'); a.output.mkdir()
    start = tick = time.perf_counter(); phases = {}
    source = json.loads(a.config.read_bytes()); source['config'] = dict(path=str(a.config), pin=sha(a.config.read_bytes()))
    source['code'] = {n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in (
        'audit/capability_experience_learning.py', 'audit/capability_experience_evaluation.py',
        'model/capability_experience_learning.py', 'model/capability_experience.py', 'model/capability_supervision.py')}
    try:
        gpu = subprocess.check_output(['nvidia-smi'], text=True)
        take_lease('29', 'capability-experience-learning'); configure(); torch.cuda.reset_peak_memory_stats()
        model, optimizer, corpus, previous, parent, saved, identities = load(source)
        protected = frozen(model, optimizer); base = saved['updates']
        supervisor = CapabilitySupervision(corpus); plan = schedules(corpus, supervisor)
        plan.update(source=source, query=list(corpus.query), region_identity=digest(corpus.regions),
            training_identity=corpus.training_identity, inherited_updates=base,
            trainable=[n for n,p in model.named_parameters() if p.requires_grad],
            protected=[n for n,p in model.named_parameters() if not p.requires_grad],
            control_collisions={arm:dict(same_group=sum(corpus.tasks[x['task']]['split_group'] ==
                corpus.tasks[x['target']]['split_group'] for x in plan['arms'][arm]),
                same_native_target=sum(identities[x['task']] == identities[x['target']] for x in plan['arms'][arm]))
                for arm in ARMS if arm.endswith('shuffled')})
        need(plan['full_unique'] == 851 and plan['repeat_unique'] == 265, 'EXPERIENCE_LEARN_DECLARED_POPULATION')
        write_committed(a.output/'plan.json', plan)
        print(json.dumps(dict(stage='plan', full=plan['full_unique'], repeat=plan['repeat_unique'],
                              operations=plan['operation_counts'])), flush=True)
        phases['authenticate_restore_plan'] = time.perf_counter()-tick; tick = time.perf_counter()
        contexts, targets, labels, census = materialize(model, corpus, previous, identities)
        write_committed(a.output/'labels.json', labels); write_committed(a.output/'census.json', census)
        initial_pin = save_checkpoint(a.output/'initial.pt', model, optimizer, base, source, parent['final_checkpoint'])
        initial = torch.load(a.output/'initial.pt', map_location='cpu', weights_only=True)
        phases['materialize_labels'] = time.perf_counter()-tick; tick = time.perf_counter()
        initial_dir = a.output/'initial'; initial_dir.mkdir()
        records = evaluate(model, contexts, targets, labels, plan, 'initial', initial_dir)
        phases['initial_evaluate'] = time.perf_counter()-tick; tick = time.perf_counter(); arms = {}
        for arm in ARMS:
            model.load_state_dict(initial['model'], strict=True); optimizer.load_state_dict(initial['optimizer'])
            optimizer.zero_grad(set_to_none=True)
            need(fingerprint(model.state_dict()) == fingerprint(initial['model']) and
                 fingerprint(optimizer.state_dict()) == initial['optimizer_hash'], 'EXPERIENCE_LEARN_MATCHED_INITIAL')
            root = a.output/arm; root.mkdir(); entries = plan['arms'][arm]
            boundaries = sorted({0, len(entries)-1, *[next(i for i,x in enumerate(entries)
                if corpus.tasks[x['task']]['kind'] == kind) for kind in plan['operation_counts']]})
            chain = initial_pin
            with (root/'steps.jsonl').open('x') as stream:
                for index, entry in enumerate(entries):
                    if index in boundaries and index:
                        save_checkpoint(root/f'before-{index}.pt', model, optimizer, base+index, source, chain)
                    at = time.perf_counter(); cpu = time.process_time()
                    row = step(model, optimizer, contexts, targets, labels, plan, arm, index, entry)
                    row.update(previous=chain, seconds=time.perf_counter()-at, cpu_seconds=time.process_time()-cpu)
                    chain = digest(row); stream.write(json.dumps(dict(row, sha256=chain), sort_keys=True)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                    if index in boundaries: save_checkpoint(root/f'after-{index}.pt', model, optimizer, base+index+1, source, chain)
                    if index % 100 == 0: print(json.dumps(dict(stage='fit', arm=arm, index=index)), flush=True)
            phases[arm+'_fit'] = time.perf_counter()-tick; tick = time.perf_counter()
            need(frozen(model, optimizer) == protected, 'EXPERIENCE_LEARN_PARENT_DRIFT'); model['world'].check_reference()
            final = save_checkpoint(root/'final.pt', model, optimizer, base+len(entries), source, chain)
            records.extend(evaluate(model, contexts, targets, labels, plan, arm, root))
            need(frozen(model, optimizer) == protected, 'EXPERIENCE_LEARN_EVAL_DRIFT')
            arms[arm] = dict(final_checkpoint=final, final_parameters=parameter_hash(model), updates=len(entries),
                optimizer=fingerprint(optimizer.state_dict()), boundaries=boundaries, last_step=chain,
                steps_sha256=sha((root/'steps.jsonl').read_bytes()), queries_sha256=sha((root/'queries.jsonl').read_bytes()))
            phases[arm+'_evaluate_checkpoint'] = time.perf_counter()-tick; tick = time.perf_counter()
        need(sha(Path(corpus.config['journal']['path']).read_bytes()) == corpus.config['journal']['pin'], 'EXPERIENCE_LEARN_JOURNAL')
        files = {n:sha((a.output/n).read_bytes()) for n in ('plan.json', 'labels.json', 'census.json', 'initial/queries.jsonl')}
        report = dict(schema='capability-experience-learning-v1', source=source, arms=arms, files=files,
            initial_checkpoint=initial_pin, inherited_updates=base, protected_identity=protected, summary=summarize(records),
            rows=len(records), gpu_before=gpu, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            scientific_admission=False, learned_capability=False, serving=False, generated_surface_evaluated=False,
            exposure='development_exposed_group_holdout', number_status='measured', acceptance='author-checked')
        phases['finalize'] = time.perf_counter()-tick; wall = time.perf_counter()-start
        need(sum(phases.values())/wall >= .9, 'EXPERIENCE_LEARN_PHASES'); report.update(phases=phases, seconds=wall)
        print(json.dumps(dict(stage='complete', pin=write_committed(a.output/'report.json', report))), flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json', dict(error=str(exc), trace=traceback.format_exc())); raise


if __name__ == '__main__': run_cli(main)

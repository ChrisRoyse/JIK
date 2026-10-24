"""Physical whole-continuation/ending qualification on authenticated legal atoms."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_grouped_placement_training as training
from audit import legal_placement_runtime_readback as runtime
from claims.process_conditions import write_committed
from model.native_answer_sequence import NativeAnswerSequence, update_answer
from model.native_placement_checked import optimizer_for, validate_optimizer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
NEW = ('model/native_answer_sequence.py', 'audit/legal_answer_sequence.py', 'audit/legal_answer_sequence_readback.py')


def code(source):
    base = Path(__file__).resolve().parents[1]
    return dict(training.code(source), **{n:file_sha(base/n) for n in NEW})


def prepare(root):
    source, bundle = training.source_bundle(root)
    data = training.load_data(argparse.Namespace(root=root, scope='small'), bundle)
    need(len(data['items']) == 15 and sum(len(i['frames']) for i in data['items']) == 215, 'ANSWER_SOURCE_CENSUS')
    need(len(data['train']) == 5 and sum(len(i['codes']) for i in data['items'] if i['side'] == 'train') == 66,
         'ANSWER_TRAIN_CENSUS')
    return source, bundle, data


def make(bundle, target, architecture):
    forecaster, _ = runtime.make(bundle, target, architecture, 'cpu')
    model = NativeAnswerSequence(forecaster)
    return model, optimizer_for(model, dict(bundle['policy'], device='cpu'))


def observe(model, target, prefix, proposal, seed, decisions):
    generator = model.rollout(prefix, target=target, proposal=proposal, context='answer-proof', seed=seed)
    rows = []
    try:
        for _ in range(decisions):
            row = next(generator); rows.append(cpu(row))
            if row['kind'] == 'end': break
    finally:
        generator.close(); need(generator.gi_frame is None, 'ANSWER_GENERATOR_NOT_CLOSED')
    return rows


def native_similarity(x, y):
    a, b = x.double().numpy(), y.double().numpy()
    an, bn = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
    need(np.isfinite(a).all() and np.isfinite(b).all() and np.all(an > 0) and np.all(bn > 0), 'ANSWER_SCORE_NATIVE')
    score = np.clip((a/an)@(b/bn).T, -1, 1)
    return float(min(score.max(1).mean(), score.max(0).mean()))


def metrics(events, future, roster, copy_frame):
    generated = [v for v in events if v['kind'] == 'constellation']
    ended = events[-1]['kind'] == 'end'
    expected = len(future); covered = min(len(generated), expected)
    scores = {n:[native_similarity(generated[i]['slots'][n], f[n]) if i < covered else -1.
                 for i, f in enumerate(future)] for n in roster}
    copy = {n:[native_similarity(copy_frame[n], f[n]) for f in future] for n in roster}
    return dict(expected_atoms=expected, generated_atoms=len(generated), ended=ended,
                ended_at_reference_boundary=ended and len(generated) == expected,
                censored=not ended, matched_positions=covered, missing=expected-covered,
                extra=max(0, len(generated)-expected), per_slot=scores, copy_current=copy,
                missing_similarity=-1., semantic_validity_established=False)


def contexts(items, item):
    donors = [i for i in items if i['side'] == 'train' and i['component'] != item['component']]
    need(donors and len(item['frames']) >= 3, 'ANSWER_DONOR_CONTEXT')
    donor = min(donors, key=lambda i:i['document'])
    prefix = tuple(item['frames'][:2])
    return dict(correct=prefix, reversed=prefix[::-1], train_donor=tuple(donor['frames'][:2])), donor['document']


def controls(model, optimizer, target, bundle, train, held, step):
    policy = dict(bundle['policy'], device='cpu'); before = fingerprint((model.state_dict(), optimizer.state_dict()))
    p = model.proposals[0]; prefix = tuple(held['frames'][:2])
    def reject(fn):
        try: fn()
        except (ValueError, RuntimeError, TypeError) as e: return type(e).__name__+':'+str(e)
        raise ValueError('ANSWER_NEGATIVE_ACCEPTED')
    result = dict(nontrain=reject(lambda:update_answer(model,optimizer,held,step,policy,target=target)),
        future_argument=reject(lambda:model.rollout(prefix,target=target,proposal=p,context='control',seed=1,target_frames=held['frames'])),
        length_argument=reject(lambda:model.rollout(prefix,target=target,proposal=p,context='control',seed=1,target_length=len(held['frames']))),
        wrong_target=reject(lambda:next(model.rollout(prefix,target=deepcopy(target),proposal=p,context='control',seed=1))))
    # A separate model tests ownership/staleness without modifying qualified weights.
    probe = deepcopy(model); base = observe(probe,target,prefix,p,31,2)
    generator = probe.rollout(prefix,target=target,proposal=p,context='control',seed=31)
    row = next(generator)
    if row['kind'] == 'constellation':
        row['slots'][probe.roster[0]].fill_(999)
        actual = cpu(next(generator))
        need(fingerprint(actual) == fingerprint(base[1]), 'ANSWER_YIELDED_ALIAS')
        result['yielded_copy'] = 'exact_next_event_after_caller_mutation'
    else: result['yielded_copy'] = 'not_exercised_early_end'
    generator.close()
    probe = deepcopy(model); generator = probe.rollout(prefix,target=target,proposal=p,context='control',seed=31)
    row = next(generator)
    if row['kind'] == 'constellation':
        with torch.no_grad(): next(probe.parameters()).add_(1)
        result['weights_changed'] = reject(lambda:next(generator))
    else: result['weights_changed'] = 'not_exercised_early_end'
    generator.close()
    if model.forecaster.architecture == 'whole_proposals':
        mixed = tuple({n:(f[n] if n == p else train['frames'][i][n]) for n in model.roster} for i,f in enumerate(prefix))
        need(fingerprint(observe(model,target,prefix,p,41,3)) == fingerprint(observe(model,target,mixed,p,41,3)),
             'ANSWER_PROPOSAL_OTHER_HISTORY_LEAK')
        result['source_only'] = 'other_five_histories_changed_same_candidate_exact'
    need(fingerprint((model.state_dict(), optimizer.state_dict())) == before, 'ANSWER_CONTROL_MUTATION')
    return result


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    need(a.cpu_proof, 'ANSWER_EXPLICIT_CPU_QUALIFICATION_ONLY')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    source,bundle,data = prepare(a.root); pins = code(source); target = data['labels'].model
    frozen = fingerprint(target.state_dict()); policy = dict(bundle['policy'], device='cpu')
    items = data['items']; fitting = [i for i in items if i['side'] == 'train']; held = [i for i in items if i['side'] != 'train']
    phases = dict(authentication=time.perf_counter()-tick); results = {}
    for architecture in runtime.original.ARCHITECTURES:
        tick = time.perf_counter(); directory = a.output/architecture; directory.mkdir()
        model,optimizer = make(bundle,target,architecture); initial = runtime.persist(directory/'initial.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        negative = controls(model,optimizer,target,bundle,fitting[0],held[0],0); ledger = []
        for step,item in enumerate(fitting):
            record,gradients = update_answer(model,optimizer,item,step,policy,target=target); ledger.append(record)
            runtime.persist(directory/f'update-{step+1}.pt',dict(record=record,gradients=gradients,model=model.state_dict(),optimizer=optimizer.state_dict()))
        validate_optimizer(model,optimizer,policy,len(fitting))
        phases[architecture+'_train'] = time.perf_counter()-tick; tick = time.perf_counter()
        rows = []; output = []
        for item in held:
            prefixes,donor = contexts(items,item); future = item['frames'][2:]
            for mode,prefix in prefixes.items():
                for proposal in model.proposals:
                    seed = 20261023+item['document']
                    events = observe(model,target,prefix,proposal,seed,len(future)+1)
                    row = dict(document=item['document'],side=item['side'],component=item['component'],proposal=proposal,mode=mode,
                               prefix_atoms=2,seed=seed,observation_decisions=len(future)+1,donor_document=donor,
                               metrics=metrics(events,future,model.roster,prefix[-1]))
                    rows.append(row); output.append(dict(row=row,events=events))
        evidence = runtime.persist(directory/'evaluation.pt',output)
        need(fingerprint(target.state_dict()) == frozen,'ANSWER_FROZEN_DRIFT')
        results[architecture] = dict(initial=initial,parameters=sum(p.numel() for p in model.parameters()),
            updates=len(ledger),train_transitions=sum(r['transitions'] for r in ledger),train_ends=sum(r['end_labels'] for r in ledger),
            controls=negative,evaluation=evidence,rows=rows,final=fingerprint((model.state_dict(),optimizer.state_dict())))
        phases[architecture+'_evaluation'] = time.perf_counter()-tick
        print(json.dumps(dict(architecture=architecture,updates=len(ledger),trajectories=len(rows),
                             ends=sum(r['metrics']['ended'] for r in rows),correct_ends=sum(r['metrics']['ended_at_reference_boundary'] for r in rows))),flush=True)
    tick = time.perf_counter(); data['close'](); need(pins == code(source),'ANSWER_CODE_CHANGED')
    for name,pin in pins.items():
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(p)==pin,'ANSWER_CODE_COPY')
    result_pin = write_committed(a.output/'results.json',results)
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick; seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9,'ANSWER_PHASE_COVERAGE')
    report=dict(complete=True,code=pins,source=training.SOURCE[1],files=files,results_pin=result_pin,policy=policy,
                frozen_target=frozen,phases=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                documents=15,atoms=215,train_documents=5,exposed_evaluation_documents=10,
                target_semantics='attested_document_continuation_not_question_answer',generation_future_visible=False,
                observation_end_is_not_model_end=True,scientific_admission=False,learned_capability=False,
                gpu_qualified=False,serving=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

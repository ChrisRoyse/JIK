"""Source-bound capability schedules and real legal masked-update qualification."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_sequence as sequence
from claims.process_conditions import write_committed
from model.answer_training_schedule import AnswerSchedule
from model.native_answer_sequence import update_answer
from model.native_conditioned_answer import document_item, update
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import canonical, digest, file_sha, need

ROOT = sequence.ROOT
PLAN = dict(path=str(ROOT/'answer-native-plan-v1-20261023'),
            pin='22993ed7704f1d31ca48df404c0af14acff1d466624b412a9c55be2132cf1098')
NEW = ('model/answer_training_schedule.py', 'model/native_conditioned_answer.py',
       'audit/conditioned_answer.py', 'audit/conditioned_answer_readback.py')


def code(source):
    base = Path(__file__).resolve().parents[1]
    return dict(sequence.code(source), **{n:file_sha(base/n) for n in NEW})


def reject(expected, call):
    try: call()
    except (ValueError, RuntimeError) as exc:
        need(expected in str(exc), 'CONDITIONED_CONTROL_REASON:'+str(exc))
        return dict(code=expected, observed=str(exc))
    raise ValueError('CONDITIONED_CONTROL_ACCEPTED:'+expected)


def schedules(view):
    rows = []; controls = []
    for key, row in view.rows.items():
        request = view.request(key); private = view.supervision(key, fitting=row['split']=='train')
        need(set(request) == {'operation','fields','position_kind'}, 'CONDITIONED_REQUEST_SCHEMA')
        c, a = len(request['fields']), len(private['targets'])
        need(len(private['decisions']) == a+1 and private['decisions'][0]['position'] == c-1,
             'CONDITIONED_SCHEDULE_START')
        rows.append(dict(task=key, family=row['family'], split=row['split'], group=row['group'],
                         request=request, private=private))
    query = next(k for k,r in view.rows.items() if r['split']=='query')
    controls.append(reject('ANSWER_SCHEDULE_FIT_SPLIT', lambda:view.supervision(query, fitting=True)))
    key = next(iter(view.rows)); original = deepcopy(view.rows[key])
    view.rows[key]['private']['decisions'][0]['position'] += 1
    controls.append(reject('ANSWER_SCHEDULE_CHANGED', lambda:view.request(key)))
    view.rows[key] = original
    returned = view.request(key); returned['fields'].clear()
    need(view.request(key)['fields'], 'CONDITIONED_REQUEST_ALIAS')
    controls.append(dict(code='OWNED_REQUEST', observed='caller mutation leaves stored source request unchanged'))
    return rows, controls


def compatibility(bundle, target, architecture, item):
    left, lo = sequence.make(bundle, target, architecture)
    right, ro = sequence.make(bundle, target, architecture)
    need(fingerprint(left.state_dict()) == fingerprint(right.state_dict()), 'CONDITIONED_COMPAT_INITIAL')
    policy = dict(bundle['policy'], device='cpu')
    _, lg = update_answer(left, lo, item, 0, policy, target=target)
    _, rg = update(right, ro, document_item(item, 1), 0, policy, target=target)
    a = fingerprint((left.state_dict(), lo.state_dict(), lg))
    b = fingerprint((right.state_dict(), ro.state_dict(), rg))
    need(a == b, 'CONDITIONED_SINGLE_CONTEXT_EQUIVALENCE')
    return dict(exact=True, fingerprint=a)


def controls(model, optimizer, train, held, policy, target):
    initial = fingerprint((model.state_dict(), optimizer.state_dict())); output = []
    variants = [('CONDITIONED_NONTRAIN', document_item(held, 2))]
    bad = deepcopy(train); bad['codes'] = bad['codes'][:-1]
    variants.append(('CONDITIONED_LABEL_COUNT', bad))
    bad = deepcopy(train); first = next(iter(bad['context'][0].values()))
    first[0,0] = torch.nextafter(first[0,0], torch.tensor(float('inf')))
    variants.append(('CONDITIONED_NATIVE_PIN', bad))
    bad = deepcopy(train); bad['codes'][0,0] += 1
    variants.append(('CONDITIONED_LABEL_PIN', bad))
    bad = deepcopy(train); bad['provenance']['observed'] += 1
    variants.append(('CONDITIONED_SOURCE_BINDING', bad))
    for expected, bad in variants:
        output.append(reject(expected, lambda:update(model, optimizer, bad, 0, policy, target=target)))
    need(fingerprint((model.state_dict(), optimizer.state_dict())) == initial, 'CONDITIONED_CONTROL_MUTATION')
    return output


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter()
    need(a.cpu_proof, 'CONDITIONED_CPU_QUALIFICATION_ONLY')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    view = AnswerSchedule(PLAN['path'], PLAN['pin']); rows, negative = schedules(view)
    schedule_pin = write_committed(a.output/'schedules.json', rows)
    write_committed(a.output/'schedule-controls.json', negative)
    counts = {side:dict(tasks=sum(r['split']==side for r in rows),
        contexts=sum(len(r['request']['fields']) for r in rows if r['split']==side),
        targets=sum(len(r['private']['targets']) for r in rows if r['split']==side),
        decisions=sum(len(r['private']['decisions']) for r in rows if r['split']==side)) for side in ('train','query')}
    phases = dict(capability_schedules=time.perf_counter()-tick); tick = time.perf_counter()
    source,bundle,data = sequence.prepare(a.root); target = data['labels'].model
    pins = code(source); frozen = fingerprint(target.state_dict()); policy = dict(bundle['policy'], device='cpu')
    fitting = [i for i in data['items'] if i['side']=='train']; held = [i for i in data['items'] if i['side']!='train']
    phases['native_authentication'] = time.perf_counter()-tick; results = {}
    for architecture in sequence.runtime.original.ARCHITECTURES:
        tick = time.perf_counter(); dest = a.output/architecture; dest.mkdir()
        equivalent = compatibility(bundle, target, architecture, fitting[0])
        model, optimizer = sequence.make(bundle, target, architecture)
        sequence.runtime.persist(dest/'initial.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
        negative = controls(model, optimizer, document_item(fitting[0],2), held[0], policy, target)
        ledger = []
        for step, original in enumerate(fitting):
            record, gradients = update(model, optimizer, document_item(original,2), step, policy, target=target)
            sequence.runtime.persist(dest/f'update-{step+1}.pt',
                dict(record=record, gradients=gradients, model=model.state_dict(), optimizer=optimizer.state_dict()))
            ledger.append(record)
        phases[architecture+'_train_and_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
        evaluated = []
        for item in held:
            contexts, donor = sequence.contexts(data['items'], item)
            for mode, prefix in contexts.items():
                for proposal in model.proposals:
                    seed = 20261023+item['document']; future = item['frames'][2:]
                    events = sequence.observe(model, target, prefix, proposal, seed, len(future)+1)
                    row = dict(document=item['document'], side=item['side'], component=item['component'],
                        mode=mode, proposal=proposal, seed=seed, observation_decisions=len(future)+1,
                        donor_document=donor, metrics=sequence.metrics(events,future,model.roster,prefix[-1]))
                    evaluated.append(dict(row=row, events=events))
        sequence.runtime.persist(dest/'evaluation.pt', evaluated)
        results[architecture] = dict(compatibility=equivalent, controls=negative,
            parameters=sum(p.numel() for p in model.parameters()), updates=len(ledger),
            answer_targets=sum(r['answer_targets'] for r in ledger), ends=sum(r['end_labels'] for r in ledger),
            prompt_transition_labels=sum(r['prompt_transition_labels'] for r in ledger),
            trajectories=len(evaluated), ended=sum(v['row']['metrics']['ended'] for v in evaluated),
            correct_ends=sum(v['row']['metrics']['ended_at_reference_boundary'] for v in evaluated),
            final=fingerprint((model.state_dict(),optimizer.state_dict())))
        phases[architecture+'_rollout'] = time.perf_counter()-tick
        print(json.dumps(dict(architecture=architecture, **results[architecture])), flush=True)
    tick=time.perf_counter();data['close']();need(frozen==fingerprint(target.state_dict()),'CONDITIONED_TARGET_CHANGED')
    need(pins==code(source),'CONDITIONED_CODE_CHANGED')
    for n,h in pins.items():
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes());need(file_sha(dest)==h,'CONDITIONED_SOURCE_COPY')
    write_committed(a.output/'results.json',results)
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'CONDITIONED_PHASE_COVERAGE')
    report=dict(complete=True,plan=PLAN,source=sequence.training.SOURCE[1],code=pins,files=files,
        schedule_pin=schedule_pin,counts=counts,policy=policy,frozen_target=frozen,phases=phases,seconds=seconds,
        cpu_qualification_only=True,capability_native_fit=False,learned_capability=False,scientific_admission=False,
        target_semantics='attested_legal_continuations; capability_schedules_only',
        no_future_in_inference=True,no_prompt_transition_labels=True,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=seconds,counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise

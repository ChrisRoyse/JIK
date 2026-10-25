"""Independent byte replay with the producer's exact declared reduction order.

Flat math.fsum remains a separately reported diagnostic. It is not used as a
bit-identical comparison to a differently grouped floating-point reduction.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch.nn import functional as F

from audit import legal_surface_inversion as producer
from audit import surface_device_authority as authority
from audit.capability_surface_fsv import check_schedule
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import mixed_surface_batches as mixed
from model import surface_inversion_training as training
from model.native_population_training import cpu, fingerprint
from model.stream_surface import schedule
from senses.contract import digest
from text.constellation_corpus import file_sha, need


def refused(fn):
    try: fn()
    except (ValueError, RuntimeError) as e: return str(e)
    raise ValueError('INVERSION_READBACK_NEGATIVE_ACCEPTED')


def score_totals(values):
    grouped = {n:sum(float(x[n]['losses'].double().sum()) for x in values) for n in ('correct','shuffled')}
    flat = {n:math.fsum(float(v) for x in values for v in x[n]['losses']) for n in grouped}
    need(all(math.isfinite(v) for v in (*grouped.values(), *flat.values())), 'INVERSION_READBACK_NONFINITE_SUM')
    return grouped, flat


def denominator(ref, positions, sums, matches, exact):
    need(positions == ref['positions'] and sums['correct'] == ref['correct_sum']
         and sums['shuffled'] == ref['shuffled_sum'] and matches == ref['byte_matches']
         and exact == ref['exact_atoms'], 'INVERSION_READBACK_DENOMINATOR')


def score(model, examples, backward=False):
    # Independent packer and explicit loss calculation, same declared batch geometry.
    result = [dict(losses=torch.full((len(e.raw)+1,), float('nan')),
                   argmax=torch.full((len(e.raw)+1,), -1, dtype=torch.int64)) for e in examples]
    device = next(model.parameters()).device; objective = 0.
    for b in mixed.batches(examples, model.config, model.specs, rows=128):
        v = b.to(device); logits = model(v.prefix, v.native, offsets=v.offsets)['logits']
        losses = F.cross_entropy(logits.reshape(-1,258), v.labels.reshape(-1), ignore_index=-100,
                                reduction='none').reshape(v.labels.shape)
        loss = (losses.sum(1)/v.denominators).sum()/len(examples)
        if backward: loss.backward()
        objective += float(loss.detach()); values = losses.detach().cpu(); predicted = logits.detach().argmax(-1).cpu()
        for j,r in enumerate(b.records):
            a,z = r['scored']; left = a-r['offset']; right = z-r['offset']; item = result[r['atom']]
            need(bool(item['losses'][a:z].isnan().all()), 'INVERSION_READBACK_DUPLICATE_BYTE')
            item['losses'][a:z] = values[j,left:right]; item['argmax'][a:z] = predicted[j,left:right]
    need(all(bool(x['losses'].isfinite().all()) and bool((x['argmax']>=0).all()) for x in result),
         'INVERSION_READBACK_MISSING_BYTE')
    return result, objective


def run(a):
    began = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False); leased = False; s = None
    readback_code = file_sha(__file__)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        r = authority.pinned(a.input, a.input_pin); p = authority.pinned(a.plan, a.plan_pin)
        s = producer.sources(a.root); need(r['plan'] == a.plan_pin and r['code'] == s['code'] == p['code'], 'INVERSION_READBACK_CODE')
        train = np.load(a.plan/'train.npy', mmap_mode='r', allow_pickle=False)
        # Recount document ranges, independently of the plan's atom-side selection.
        cursor = size = byte_count = prefix_rows = 0; offsets = s['surfaces'].offsets
        for d,(lo,hi) in enumerate(s['data'].documents):
            if s['data'].doc_sides[d] != 0: continue
            lo,hi = int(lo),int(hi); expected = np.arange(lo,hi,dtype=np.int64)
            need(np.array_equal(train[cursor:cursor+hi-lo], expected), 'INVERSION_READBACK_TRAIN_COVERAGE')
            u = s['data'].mapping[lo:hi]; lengths = offsets[u+1]-offsets[u]
            cursor += hi-lo; size += hi-lo; byte_count += int(lengths.sum()); prefix_rows += int(((lengths+16)//16).sum())
        need(cursor == len(train) and dict(atoms=size, bytes=byte_count, prefix_rows=prefix_rows) == p['train'],
             'INVERSION_READBACK_CENSUS')
        need(producer.select(s['data']) == {k:v['documents'] for k,v in p['evaluation'].items()}, 'INVERSION_READBACK_COHORT')
        device = torch.device('cpu' if a.cpu_proof else 'cuda:0')
        need(r['binding']['device'] == str(device), 'INVERSION_READBACK_DEVICE')
        if not a.cpu_proof:
            import subprocess
            (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
            need(torch.cuda.is_available(), 'INVERSION_READBACK_CUDA_REQUIRED')
            take_lease('29', 'surface-inversion-readback'); leased = True
        model, optimizer = training.create(s['config'], s['specs'], s['initial'], device)
        rows = [json.loads(x) for x in (a.input/'steps.jsonl').read_text().splitlines()]
        selected = train if r['full_training'] else np.array([e.occurrence for e in s['examples'] if e.side=='train'])
        state_size = len(selected); cursor = 0; ledger = digest(r['binding']); previous_state = fingerprint(s['initial'])
        first_step = 0
        if r['resume']:
            cursor,first_step,ledger = training.restore(r['resume'],model,optimizer,r['binding'],size=state_size)
            previous_state = fingerprint(cpu(model.state_dict()))
        controls = {}; replayed = 0
        controls['held_fit'] = refused(lambda:s['adapter'].fetch([next(e.occurrence for e in s['examples'] if e.side=='query')], side='query', fit=True))
        example=next(e for e in s['examples'] if e.side=='train')
        frame={n:x.values[0] for n,x in example.native.items()}
        controls['changed_surface']=refused(lambda:s['adapter'].wrap(example.occurrence,example.uid,b'x'+example.raw,frame,side='train'))
        controls['missing_instrument']=refused(lambda:s['adapter'].wrap(example.occurrence,example.uid,example.raw,dict(list(frame.items())[1:]),side='train'))
        controls['missing_byte_tail']=refused(lambda:check_schedule(example.raw,list(schedule(example.raw))[:-1]))
        for index,record in enumerate(rows):
            row = record['row']; ids = selected[cursor:cursor+r['binding']['tile']].tolist()
            need(row['step']==first_step+index+1 and row['occurrences']==ids and row['cursor']==cursor+len(ids)
                 and row['previous']==ledger and digest(row)==record['pin'] and row['before']==previous_state,
                 'INVERSION_READBACK_LEDGER')
            cursor += len(ids); ledger=record['pin']; previous_state=row['after']
            if not r['full_training']:
                examples=s['adapter'].fetch(ids, side='train', fit=True)
                # Compare to separately persisted native/source observations too.
                reference={e.occurrence:e for e in s['examples']}
                for e in examples:
                    old=reference[e.occurrence]
                    need(e.raw==old.raw and all(torch.equal(e.native[n].values,old.native[n].values) for n in e.native),
                         'INVERSION_READBACK_SOURCE')
                model.train(); optimizer.zero_grad(set_to_none=True); scores,objective=score(model,examples,True)
                gradients=cpu({n:v.grad for n,v in model.named_parameters()})
                need(fingerprint(gradients)==row['gradient'] and fingerprint(scores)==row['scores'] and objective==row['objective'],
                     'INVERSION_READBACK_GRADIENT')
                optimizer.step()
                need(fingerprint(cpu(model.state_dict()))==row['after'] and fingerprint(cpu(optimizer.state_dict()))==row['optimizer'],
                     'INVERSION_READBACK_UPDATE')
                replayed+=1
                if index==0:
                    ref=r['checkpoints'][0]; other, opt=training.create(s['config'],s['specs'],s['initial'],device)
                    controls['wrong_binding']=refused(lambda:training.restore(ref,other,opt,{},size=state_size))
                    bad=dict(ref,cursor=ref['cursor']+1)
                    controls['wrong_cursor']=refused(lambda:training.restore(bad,other,opt,r['binding'],size=state_size))
                    controls['wrong_pin']=refused(lambda:training.restore(dict(ref,pin='0'*64),other,opt,r['binding'],size=state_size))
                    bad_state=torch.load(ref['path'],map_location='cpu',weights_only=True)
                    bad_state['optimizer']['state']={}
                    bad_path=a.output/'negative-weights-only.pt';torch.save(bad_state,bad_path)
                    bad_ref=dict(ref,path=str(bad_path),pin=file_sha(bad_path),state=fingerprint(bad_state))
                    controls['weights_only_not_continuation']=refused(lambda:training.restore(bad_ref,other,opt,r['binding'],size=state_size))
                    training.restore(ref,other,opt,r['binding'],size=state_size)
                    need(fingerprint(cpu(other.state_dict()))==row['after'] and fingerprint(cpu(opt.state_dict()))==row['optimizer'],
                         'INVERSION_READBACK_RESUME_STATE')
                    # All later independently calculated updates now use restored weights AND moments.
                    model,optimizer=other,opt
        need(cursor==r['cursor']==len(selected) and len(rows)+first_step==r['steps'] and ledger==r['ledger'], 'INVERSION_READBACK_COMPLETE')
        last=r['checkpoints'][-1]; training.restore(last,model,optimizer,r['binding'],size=state_size)
        need(fingerprint(cpu(model.state_dict()))==previous_state, 'INVERSION_READBACK_FINAL')
        evaluations={}; atom_exact=0; reductions=[]
        for phase in ['initial','final']:
            if phase=='initial' and r['resume']: training.restore(r['resume'],model,optimizer,r['binding'],size=state_size)
            elif phase=='initial': model.load_state_dict(s['initial'],strict=True)
            else: training.restore(last,model,optimizer,r['binding'],size=state_size)
            totals={}
            for ref in r[phase]:
                need(file_sha(ref['path'])==ref['pin'], 'INVERSION_READBACK_EVAL_PIN')
                saved=torch.load(ref['path'],map_location='cpu',weights_only=True); meta=saved['metadata']; actual=[]
                byte_matches=exact_atoms=0
                for start in range(0,len(meta),128):
                    m=meta[start:start+128]; ids=[v['occurrence'] for v in m]; donors=[v['donor'] for v in m]
                    need(donors==producer.donor_ids(s,ids,train),'INVERSION_READBACK_DONORS')
                    examples=s['adapter'].fetch(ids,side=ref['side']); other=s['adapter'].fetch(donors,side='train')
                    model.eval()
                    with torch.no_grad():
                        correct,_=score(model,examples); shuffled,_=score(model,[replace(e,native=d.native) for e,d in zip(examples,other)])
                    actual.extend(dict(correct=x,shuffled=y) for x,y in zip(correct,shuffled))
                    for e,x in zip(examples,correct):
                        labels=torch.tensor([*e.raw,257]); exact=int(torch.equal(labels,x['argmax']))
                        atom_exact+=exact;exact_atoms+=exact;byte_matches+=int((labels==x['argmax']).sum())
                need(fingerprint(actual)==fingerprint(saved['scores']),'INVERSION_READBACK_EVERY_BYTE')
                positions=sum(len(x['correct']['losses']) for x in actual)
                sums, flat = score_totals(actual); correct, shuffled = sums['correct'], sums['shuffled']
                denominator(ref, positions, sums, byte_matches, exact_atoms)
                reductions.append(dict(phase=phase,side=ref['side'],document=ref['document'],
                                       grouped=sums,flat_fsum=flat,delta={n:flat[n]-sums[n] for n in sums}))
                if not controls.get('changed_denominator'):
                    controls['changed_denominator'] = refused(lambda:denominator(ref,positions+1,sums,byte_matches,exact_atoms))
                    bad = dict(sums,correct=sums['correct']+1.)
                    controls['changed_sum'] = refused(lambda:denominator(ref,positions,bad,byte_matches,exact_atoms))
                    changed = deepcopy(actual); changed[0]['correct']['losses'][0] += 1.
                    altered, _ = score_totals(changed)
                    controls['changed_byte_score'] = refused(lambda:denominator(ref,positions,altered,byte_matches,exact_atoms))
                t=totals.setdefault(ref['side'],dict(documents=0,atoms=0,positions=0,correct_sum=0.,shuffled_sum=0.))
                t['documents']+=1;t['atoms']+=len(actual);t['positions']+=positions;t['correct_sum']+=correct;t['shuffled_sum']+=shuffled
            evaluations[phase]=totals
        producer.close(s);s=None
        need(file_sha(__file__)==readback_code,'INVERSION_CANONICAL_READER_CHANGED')
        out=dict(complete=True,input_pin=a.input_pin,plan=a.plan_pin,code=r['code'],train_census=p['train'],
            readback_code=readback_code,reduction='per_atom_float64_torch_sum_then_ordered_python_sum',
            flat_sum_diagnostics=reductions,comparison_tolerance=0.,
            full_ledger_coverage=True, device=str(device), cold_optimizer_updates=replayed, all_update_replay_exact=not r['full_training'],
            restored_midpoint_continuation_exact=not r['full_training'], every_evaluation_byte_exact=True,
            controls=controls,evaluation=evaluations,teacher_forced_exact_atoms_across_both_stages=atom_exact,
            scientific_admission=False,capability_demonstrated=False,seconds=time.perf_counter()-began,
            acceptance='author-checked',number_status='measured')
        pin=write_committed(a.output/'report.json',out);print(json.dumps(dict(pin=pin,replayed=replayed,evaluation=evaluations)),flush=True)
    finally:
        if s is not None:producer.close(s)
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=authority.ROOT);p.add_argument('--output',type=Path,required=True)
    for name in ['input','plan']:p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

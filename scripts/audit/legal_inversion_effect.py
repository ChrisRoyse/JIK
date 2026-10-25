"""Read complete-source inversion effects from sealed paired byte predictions.

This is CPU analysis of persisted observations, not inference or a sense assay.
The comparison remains teacher-forced and uses true native readings.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_inversion as producer
from audit.legal_surface_inversion_canonical_readback import denominator, score_totals, refused
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha, need

STRATA = ('all', 'seen_in_train', 'unseen_in_train')


def read(path, pin):
    need(file_sha(path) == pin, 'INVERSION_EFFECT_PIN'); return json.loads(Path(path).read_bytes())


def blank():
    return dict(atoms=0,positions=0,correct_sum=0.,shuffled_sum=0.,correct_atom_mean_sum=0.,
                shuffled_atom_mean_sum=0.,byte_matches=0,exact_atoms=0,condition_wins=0,condition_ties=0,condition_losses=0)


def combine(rows):
    # Python's sum uses a different float reduction from repeated +=.
    # Match the producer's per-atom sum, including its order.
    return {name:sum(row[name] for row in rows) for name in blank()}


def rates(total):
    n, p = total['atoms'], total['positions']; need(n > 0 and p > 0, 'INVERSION_EFFECT_EMPTY')
    return dict(total,correct_nats_per_byte=total['correct_sum']/p,shuffled_nats_per_byte=total['shuffled_sum']/p,
        conditioning_gain_nats_per_byte=(total['shuffled_sum']-total['correct_sum'])/p,
        correct_mean_atom_loss=total['correct_atom_mean_sum']/n,
        shuffled_mean_atom_loss=total['shuffled_atom_mean_sum']/n,
        byte_accuracy=total['byte_matches']/p,teacher_forced_exact_atom_rate=total['exact_atoms']/n,
        paired_atom_preference=total['condition_wins']/n)


def atom(metadata, scores, raw, uid, presence):
    need(metadata['uid'] == uid and metadata['bytes'] == len(raw)
         and metadata['train_present'] == bool(presence & 1), 'INVERSION_EFFECT_SOURCE_METADATA')
    labels = torch.tensor([*raw,257], dtype=torch.int64); size = len(labels)
    need(set(scores) == {'correct','shuffled'}, 'INVERSION_EFFECT_ARMS')
    for values in scores.values():
        need(set(values) == {'losses','argmax'} and values['losses'].dtype == torch.float32
             and values['argmax'].dtype == torch.int64 and values['losses'].shape == values['argmax'].shape == labels.shape
             and bool(values['losses'].isfinite().all()) and bool((values['losses'] >= 0).all())
             and bool(((values['argmax'] >= 0) & (values['argmax'] < 258)).all()), 'INVERSION_EFFECT_BYTE_LAYOUT')
    c, s = (float(scores[n]['losses'].double().sum()) for n in ('correct','shuffled'))
    matched = scores['correct']['argmax'] == labels
    return dict(atoms=1,positions=size,correct_sum=c,shuffled_sum=s,
        correct_atom_mean_sum=c/size,shuffled_atom_mean_sum=s/size,byte_matches=int(matched.sum()),
        exact_atoms=int(bool(matched.all())),condition_wins=int(c < s),condition_ties=int(c == s),condition_losses=int(c > s))


def paired_components(rows, side, stratum):
    paired = {}; initial = {}; final = {}
    for row in rows:
        if row['side'] != side or not row['strata'][stratum]['positions']: continue
        value = rates(row['strata'][stratum]); key = row['component']
        target = initial if row['phase'] == 'initial' else final
        need(key not in target, 'INVERSION_EFFECT_DUPLICATE_COMPONENT'); target[key] = value
    need(initial.keys() == final.keys() and initial, 'INVERSION_EFFECT_COMPONENT_PAIR')
    keys = sorted(initial); rng = np.random.default_rng(20261024)
    indices = rng.integers(0,len(keys),size=(1000,len(keys)))
    metrics = dict(correct_loss_improvement=[initial[k]['correct_nats_per_byte']-final[k]['correct_nats_per_byte'] for k in keys],
        final_conditioning_gain=[final[k]['conditioning_gain_nats_per_byte'] for k in keys],
        conditioning_gain_change=[final[k]['conditioning_gain_nats_per_byte']-initial[k]['conditioning_gain_nats_per_byte'] for k in keys],
        byte_accuracy_improvement=[final[k]['byte_accuracy']-initial[k]['byte_accuracy'] for k in keys])
    for name, values in metrics.items():
        x = np.asarray(values,dtype=np.float64); means = x[indices].mean(1)
        paired[name] = dict(component_mean=float(x.mean()),components=len(keys),up=int((x > 0).sum()),
            down=int((x < 0).sum()),tie=int((x == 0).sum()),bootstrap_95=np.quantile(means,[.025,.975]).tolist())
    return dict(metrics=paired,components=keys,resamples=1000,seed=20261024,scope='exposed_development_diagnostic')


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); source = None; phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        r = read(a.input/'report.json',a.input_pin); p = read(a.plan/'report.json',a.plan_pin)
        cold = read(a.readback/'report.json',a.readback_pin)
        need(r['complete'] and r['full_training'] and r['cursor'] == p['train']['atoms'] == 6547176
             and r['steps'] == p['updates'] == 51150 and r['binding']['device'] == 'cuda:0'
             and r['resume'] is None and r['plan'] == a.plan_pin, 'INVERSION_EFFECT_TRAIN_SCOPE')
        need(cold['complete'] and cold['input_pin'] == a.input_pin and cold['plan'] == a.plan_pin
             and cold['every_evaluation_byte_exact'] and cold['full_ledger_coverage'] and cold['device'] == 'cuda:0'
             and cold['comparison_tolerance'] == 0. and cold['cold_optimizer_updates'] == 0
             and not cold['all_update_replay_exact'], 'INVERSION_EFFECT_READBACK_SCOPE')
        base = Path(__file__).resolve().parents[1]
        code = {n:file_sha(base/n) for n in ('audit/legal_inversion_effect.py',
                                            'audit/legal_surface_inversion_canonical_readback.py')}
        for n,h in code.items():
            path = a.output/'source'/n; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((base/n).read_bytes()); need(file_sha(path)==h,'INVERSION_EFFECT_CODE_COPY')
        need(file_sha(base/'audit/legal_surface_inversion_canonical_readback.py') == cold['readback_code'],
             'INVERSION_EFFECT_READER_CODE')
        source = producer.sources(a.root)
        need(source['code'] == r['code'] == p['code'] == cold['code'], 'INVERSION_EFFECT_SOURCE_CODE')
        for n,h in r['files'].items(): need(file_sha(a.input/n) == h, 'INVERSION_EFFECT_TRAIN_BYTES')
        need(file_sha(a.plan/'train.npy') == p['files']['train.npy'], 'INVERSION_EFFECT_PLAN_BYTES')
        train = np.load(a.plan/'train.npy',mmap_mode='r',allow_pickle=False)
        phases['authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
        ledger = digest(r['binding']); cursor = step = positions = 0; model = r['binding']['initial']
        states = {v['step']:v for v in r['checkpoints']}; checked = []
        update_profile = dict(calls=0,positions=0,seconds=0.,phases={})
        with (a.input/'steps.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line); row = record['row']; ids = train[cursor:cursor+128].tolist()
                profile = record['profile']
                for n in ('calls','positions','seconds'): update_profile[n] += profile[n]
                for n,v in profile['phases'].items():
                    update_profile['phases'][n] = update_profile['phases'].get(n,0.)+v
                need(row['occurrences'] == ids and row['step'] == step+1 and row['cursor'] == cursor+len(ids)
                     and row['previous'] == ledger and row['before'] == model and digest(row) == record['pin'],
                     'INVERSION_EFFECT_LEDGER')
                uids = source['data'].mapping[ids]; lengths = source['surfaces'].offsets[uids+1]-source['surfaces'].offsets[uids]
                need(row['positions'] == int(lengths.sum())+len(ids), 'INVERSION_EFFECT_TRAIN_DENOMINATOR')
                cursor += len(ids); step += 1; ledger = record['pin']; model = row['after']; positions += row['positions']
                if step in states:
                    ref = states[step]; value = torch.load(ref['path'],map_location='cpu',weights_only=True)
                    need(file_sha(ref['path']) == ref['pin'] and fingerprint(value) == ref['state']
                         and value['binding'] == r['binding'] and value['cursor'] == cursor and value['step'] == step
                         and value['ledger'] == ledger and fingerprint(value['model']) == model
                         and fingerprint(value['optimizer']) == row['optimizer'], 'INVERSION_EFFECT_CHECKPOINT')
                    need(value['optimizer']['state'] and all(float(v['step']) == step for v in value['optimizer']['state'].values()),
                         'INVERSION_EFFECT_ADAM_COUNTER'); checked.append(step)
        need(cursor == r['cursor'] and step == r['steps'] and ledger == r['ledger'] and set(checked) == set(states)
             and positions == p['train']['bytes']+p['train']['atoms'] == update_profile['positions'],
             'INVERSION_EFFECT_TRAIN_COVERAGE')
        need(.9 <= sum(update_profile['phases'].values())/update_profile['seconds'] <= 1.000001,
             'INVERSION_EFFECT_PROFILE_COVERAGE')
        train._mmap.close(); phases['ledger_and_checkpoints'] = time.perf_counter()-tick; tick = time.perf_counter()
        indices = {phase:{(v['side'],v['document']):v for v in r[phase]} for phase in ('initial','final')}
        need(indices['initial'].keys() == indices['final'].keys(), 'INVERSION_EFFECT_EVALUATION_PAIR')
        rows = []; controls = {}; total_atoms = total_positions = 0
        with (a.output/'atoms.jsonl').open('x') as atoms:
            for key in sorted(indices['initial']):
                side, doc = key; first,last = map(int,source['data'].documents[doc])
                uids = list(map(int,source['data'].mapping[first:last])); surfaces = []
                for start in range(0,len(uids),128): surfaces.extend(source['surfaces'].read(uids[start:start+128]))
                for phase in ('initial','final'):
                    ref = indices[phase][key]; value = torch.load(ref['path'],map_location='cpu',weights_only=True)
                    need(file_sha(ref['path']) == ref['pin'] and len(value['metadata']) == len(value['scores']) == len(uids)
                         and ref['component'] == int(source['data'].components[doc]), 'INVERSION_EFFECT_DOCUMENT')
                    records = {s:[] for s in STRATA}
                    for ordinal,(meta,scores,raw,uid) in enumerate(zip(value['metadata'],value['scores'],surfaces,uids,strict=True)):
                        need(meta['occurrence'] == first+ordinal, 'INVERSION_EFFECT_ORDER')
                        counts = atom(meta,scores,raw,uid,int(source['continuation'].presence[uid]))
                        stratum = 'seen_in_train' if meta['train_present'] else 'unseen_in_train'
                        records['all'].append(counts); records[stratum].append(counts)
                        atoms.write(json.dumps(dict(phase=phase,side=side,document=doc,component=ref['component'],
                            occurrence=meta['occurrence'],uid=uid,stratum=stratum,**counts),sort_keys=True)+'\n')
                    group = {s:combine(v) for s,v in records.items()}
                    sums, _ = score_totals(value['scores']); g = group['all']
                    denominator(ref,g['positions'],sums,g['byte_matches'],g['exact_atoms'])
                    need(g['correct_sum'] == sums['correct'] and g['shuffled_sum'] == sums['shuffled'], 'INVERSION_EFFECT_REDUCTION')
                    if not controls:
                        controls['denominator'] = refused(lambda:denominator(ref,g['positions']+1,sums,g['byte_matches'],g['exact_atoms']))
                        bad = dict(value['metadata'][0],train_present=not value['metadata'][0]['train_present'])
                        controls['identity_stratum'] = refused(lambda:atom(bad,value['scores'][0],surfaces[0],uids[0],int(source['continuation'].presence[uids[0]])))
                        changed = deepcopy(value['scores']); changed[0]['correct']['losses'][0] += 1.
                        altered, _ = score_totals(changed)
                        controls['byte_score'] = refused(lambda:denominator(ref,g['positions'],altered,g['byte_matches'],g['exact_atoms']))
                    total_atoms += g['atoms']; total_positions += 2*g['positions']
                    rows.append(dict(phase=phase,side=side,document=doc,component=ref['component'],strata=group))
            atoms.flush(); os.fsync(atoms.fileno())
        phases['source_and_scores'] = time.perf_counter()-tick; tick = time.perf_counter(); summary = {}; paired = {}
        for side in ('development','query'):
            paired[side] = {s:paired_components(rows,side,s) for s in STRATA}; summary[side] = {}
            for phase in ('initial','final'):
                selected = [v for v in rows if v['side']==side and v['phase']==phase]
                need(len(selected) == p['evaluation'][side]['components'] == 100, 'INVERSION_EFFECT_COMPONENT_CENSUS')
                groups = {s:combine([row['strata'][s] for row in selected]) for s in STRATA}
                expected = p['evaluation'][side]
                need(groups['all']['atoms'] == expected['atoms'] and groups['all']['positions'] == expected['bytes']+expected['atoms']
                     and groups['seen_in_train']['atoms'] == expected['train_present'], 'INVERSION_EFFECT_STRATUM_CENSUS')
                c = cold['evaluation'][phase][side]
                need(groups['all']['positions'] == c['positions'] and groups['all']['atoms'] == c['atoms'], 'INVERSION_EFFECT_COLD_DENOMINATOR')
                summary[side][phase] = {s:rates(v) for s,v in groups.items()}
        need(total_atoms == 27700 and total_positions == 6151148, 'INVERSION_EFFECT_COMPLETE_EVALUATION')
        write_committed(a.output/'components.json',rows); write_committed(a.output/'summary.json',dict(summary=summary,paired=paired))
        producer.close(source); source = None
        for n,h in code.items():
            need(file_sha(base/n)==h,'INVERSION_EFFECT_CODE_CHANGED')
            need(file_sha(a.output/'source'/n)==h,'INVERSION_EFFECT_CODE_COPY')
        phases['summary_and_seal'] = time.perf_counter()-tick; tick = time.perf_counter()
        files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(sum(phases.values())/seconds >= .9 and not torch.cuda.is_initialized(), 'INVERSION_EFFECT_RUNTIME')
        report = dict(complete=True,input_pin=a.input_pin,readback_pin=a.readback_pin,plan=a.plan_pin,code=code,files=files,
            train=p['train'],steps=step,checkpoints=len(checked),ledger=ledger,controls=controls,summary=summary,paired=paired,
            training_update_profile=update_profile,
            evaluation_atom_records=total_atoms,paired_byte_positions=total_positions,phases=phases,seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            full_optimizer_updates_replayed=False,model_inference_calls=0,encoder_calls=0,gpu_used=False,
            free_generation_tested=False,predicted_conditioning_tested=False,scientific_admission=False,
            capability_demonstrated=False,acceptance='author-checked',exposure='previously_exposed_group_holdout')
        pin = write_committed(a.output/'report.json',report)
        print(json.dumps(dict(pin=pin,seconds=seconds,summary=summary,paired=paired)),flush=True)
    finally:
        if source is not None: producer.close(source)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=producer.authority.ROOT)
    p.add_argument('--output',type=Path,required=True)
    for name in ('input','plan','readback'): p.add_argument('--'+name,type=Path,required=True); p.add_argument('--'+name+'-pin',required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

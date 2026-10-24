"""Real, typed, TRAIN-only successor retrieval on complete predictor probe cases."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_training_statistics as training, legal_region_predictor_readback as q
from audit.legal_predictor_training import distance
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane
from model.legal_breadth_schedule import Metadata
from model.legal_native_successor import NativeSuccessor, scope
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need

PROOF = ('legal-region-predictor-qualification-v2-20261023',
         'd5d4477fb98face26b4556d7101351a5ea4cdfd5271af912e35a0df776a54570')
PROFILE = ('legal-specter-candidate-profiles-v2-20261023',
           '0dac2f88d78f75be7aeb5e3cece395290af45b420e9202d8cff39803a9f478d4')
PROFILE_COLD = ('legal-specter-candidate-profiles-cold-v2-20261023',
                'fab2a76207402a6dd716a31dd82ebe7e7d2c06040568ce2c07a088ebe4a68354')


def code():
    root = Path(__file__).resolve().parents[1]
    names = ['audit/legal_native_successor_readback.py', 'model/legal_native_successor.py',
             'kernel/certified_index_graph.py', 'kernel/native_window_index.py', 'audit/native_index_reference.py',
             'audit/legal_predictor_training.py', 'model/legal_breadth_schedule.py', 'model/constellation.py']
    return dict(q.code(), **{n:file_sha(root/n) for n in names})


def summarize(rows):
    out = {}
    for side in ['train','development','query']:
        for novel in [False, True]:
            selected = [r for r in rows if r['side'] == side and (not novel or r['target_absent_from_train'])]
            key = side+(':target_absent_from_train' if novel else ':all')
            if not selected:
                out[key] = dict(rows=0, status='unmeasured_empty_support'); continue
            available = [r for r in selected if r['retrieved_distance'] is not None]
            exact = [r for r in selected if r['exact_input_distance'] is not None]
            out[key] = dict(rows=len(selected), documents=len({r['document'] for r in selected}),
                retrieval_coverage=len(available)/len(selected), exact_input_coverage=len(exact)/len(selected),
                retrieved_distance_on_available=sum(r['retrieved_distance'] for r in available)/len(available) if available else None,
                exact_input_distance_on_available=sum(r['exact_input_distance'] for r in exact)/len(exact) if exact else None,
                **{k:sum(r[k] for r in selected)/len(selected) for k in
                   ['copy_distance','train_mean_distance','retrieval_or_copy_distance','exact_input_or_copy_distance']})
    return out


def run(a):
    began = tick = time.perf_counter(); cpu_start = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); need(not torch.cuda.is_initialized(), 'NATIVE_SUCCESSOR_GPU_CONTEXT')
    for name,pin in code().items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'NATIVE_SUCCESSOR_SOURCE_COPY')
    _, _, data, continuation, _, _, _, _ = training.authenticate(a.root)
    report = training.pinned(a.root/PROOF[0]/'report.json', PROOF[1])
    plan = training.pinned(a.root/PROOF[0]/'selection.json', report['binding']['selection'])
    need(plan['source'] == data.binding, 'NATIVE_SUCCESSOR_SOURCE_BINDING')
    lane = CertifiedIndexLane(a.certificate, a.certificate_pin)
    need(lane.source['pin'] == data.source.pin and lane.name in data.roster, 'NATIVE_SUCCESSOR_INDEX_SOURCE')
    meta = Metadata(data); profile = (512,1024) if lane.name == 'specter2-base' else (128,256)
    profile_proof = None
    if lane.name == 'specter2-base':
        pr = training.pinned(a.root/PROFILE[0]/'report.json', PROFILE[1])
        pc = training.pinned(a.root/PROFILE_COLD[0]/'report.json', PROFILE_COLD[1])
        cell = 'candidates512-expansion1024'
        need(pr['complete'] and pc['complete'] and pc['previous_pin'] == PROFILE[1]
             and pr['plan']['certificate'] == a.certificate_pin
             and pr['profiles'][cell]['groups'] == pc['results'][cell]['groups']
             and all(v['recall_gate_pass'] for v in pr['profiles'][cell]['groups'].values()),
             'NATIVE_SUCCESSOR_PROFILE_AUTHORITY')
        for name,pin in pr['files'].items(): need(file_sha(a.root/PROFILE[0]/name) == pin, 'NATIVE_SUCCESSOR_PROFILE_BYTES')
        profile_proof = dict(publication=PROFILE[1], cold=PROFILE_COLD[1])
    reader = NativeSuccessor(lane, continuation, candidates=profile[0], expansion=profile[1])
    stats_path = a.root/q.initialization.STATS[0]
    stats_report = training.pinned(stats_path/'report.json', q.initialization.STATS[1])
    need(file_sha(stats_path/'statistics.pt') == stats_report['files']['statistics.pt'], 'NATIVE_SUCCESSOR_STATISTICS_PIN')
    stats = torch.load(stats_path/'statistics.pt', map_location='cpu', weights_only=True)
    uids = sorted({int(u) for row in plan['documents'] for u in data.mapping[row['start']:row['stop']-1]})
    binding = dict(code=code(), source=data.binding, selection=report['binding']['selection'], index=lane.binding,
                   candidates=profile[0], expansion=profile[1], train_fit=continuation.pin, metadata=meta.binding,
                   current_inputs=len(uids), source_index_transductive=True, target_donors_train_only=True,
                   candidate_profile_proof=profile_proof)
    previous = training.pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding'] == binding, 'NATIVE_SUCCESSOR_COLD_BINDING')
        for name,pin in previous['files'].items(): need(file_sha(a.previous/name) == pin, 'NATIVE_SUCCESSOR_COLD_PHYSICAL_PIN')
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    rows = [reader.query(uid) for uid in uids]; scope(rows,uids)
    reversed_rows = [reader.query(uid) for uid in reversed(uids)]
    need(rows == list(reversed(reversed_rows)), 'NATIVE_SUCCESSOR_REVERSE_REPLAY')
    by_uid = {r['uid']:r for r in rows}; queries_pin = write_committed(a.output/'queries.json', rows)
    if previous:
        need(queries_pin == previous['files']['queries.json'], 'NATIVE_SUCCESSOR_COLD_QUERIES')
    phases['query_reverse_and_independent_scores'] = time.perf_counter()-tick; tick = time.perf_counter()
    native = {}
    def value(uid):
        if uid not in native: native[uid] = torch.from_numpy(np.array(lane([uid])[0], copy=True))
        return native[uid]
    evaluated = []
    for row in plan['documents']:
        start, stop = row['start'], row['stop']; ids = data.mapping[start:stop].tolist()
        data.indices(list(range(start,stop)), row['side'], fit=False)
        data.validate_next(np.arange(start,stop-1), np.arange(start+1,stop))
        for pos, (current, target) in enumerate(zip(ids[:-1],ids[1:])):
            predicted = by_uid[current]['successor_uid']; exact = int(continuation.modal[current])
            need(exact < 0 or bool(int(continuation.presence[exact]) & 1), 'NATIVE_SUCCESSOR_EXACT_INPUT_TARGET_TRAIN')
            truth = value(target); copy_distance = distance(truth, value(current))
            retrieval_distance = distance(truth, value(predicted)) if predicted is not None else None
            exact_distance = distance(truth,value(exact)) if exact >= 0 else None
            evaluated.append(dict(document=row['document'], side=row['side'], position=pos, current_uid=current, target_uid=target,
                source_collection=meta.rows[row['document']]['source_path'], donor_uid=by_uid[current]['donor_uid'],
                predicted_uid=predicted, exact_input_uid=exact if exact >= 0 else None,
                target_absent_from_train=not bool(int(continuation.presence[target]) & 1),
                copy_distance=copy_distance, train_mean_distance=distance(truth,stats['centers'][lane.name].float()[None]),
                retrieved_distance=retrieval_distance, exact_input_distance=exact_distance,
                retrieval_or_copy_distance=retrieval_distance if retrieval_distance is not None else copy_distance,
                exact_input_or_copy_distance=exact_distance if exact_distance is not None else copy_distance))
    evaluation_pin = write_committed(a.output/'evaluation.json', evaluated); summary = summarize(evaluated)
    torch.save(native, a.output/'native-proof.pt')
    need(fingerprint(torch.load(a.output/'native-proof.pt',map_location='cpu',weights_only=True)) == fingerprint(native),
         'NATIVE_SUCCESSOR_NATIVE_READBACK')
    if previous:
        need(evaluation_pin == previous['files']['evaluation.json'] and summary == previous['summary'], 'NATIVE_SUCCESSOR_COLD_SCORES')
    phases['source_successor_scoring_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    uid = uids[0]; accepted = next(r for r in rows if r['donor_uid'] is not None)
    held = int(np.flatnonzero((continuation.presence & 1) == 0)[0])
    negative = dict(held_donor=rejected(lambda:reader.donor(uid,held)), self_donor=rejected(lambda:reader.donor(uid,uid)),
        unknown_donor=rejected(lambda:reader.donor(uid,continuation.u)),
        changed_successor=rejected(lambda:reader.donor(accepted['uid'],accepted['donor_uid'],-1)),
        missing_query=rejected(lambda:scope(rows[:-1],uids)), duplicate_query=rejected(lambda:scope(rows+rows[:1],uids)),
        changed_authority=rejected(lambda:CertifiedIndexLane(a.certificate,'0'*64)))
    reader.close(); meta.fresh(); need(not torch.cuda.is_initialized(),'NATIVE_SUCCESSOR_GPU_CONTEXT')
    if previous: need(negative == previous['controls'], 'NATIVE_SUCCESSOR_COLD_CONTROLS')
    phases['controls_and_final_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'NATIVE_SUCCESSOR_PHASE_ATTRIBUTION')
    result = dict(complete=True, mode='cold' if previous else 'qualify', previous_pin=a.previous_pin, binding=binding,
                  summary=summary, controls=negative, native_proof_state=fingerprint(native), files=files,
                  queries=len(rows), successors=len(evaluated), phases_seconds=phases, seconds=seconds,
                  cpu_seconds=time.process_time()-cpu_start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  native_encoder_calls=0, target_model_calls=0, consensus_required=False, scientific_admission=False,
                  acceptance='author-checked', number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),instrument=lane.name,summary=summary,seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--certificate',type=Path,required=True);p.add_argument('--certificate-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise

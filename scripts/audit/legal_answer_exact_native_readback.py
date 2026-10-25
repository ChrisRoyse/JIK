"""Independent raw-shard/NumPy recount of whole-answer native retrieval scores."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_answer_exact_native import selection,ROOT,BASE,PROOF
from audit.legal_answer_public_pack import authenticate
from audit.complete_answer_comparison import direct
from claims.process_conditions import write_committed
from model import legal_answer_exact_scores as scoring
from model.native_population_training import fingerprint
from text.constellation_corpus import digest,file_sha,need
from text.legal_native_store import SHARD_INPUTS


def checked_cases(path,pin):
    need(file_sha(path)==pin,'EXACT_NATIVE_RECOUNT_CASES_PIN')
    with path.open('rb') as f:
        for line in f:yield json.loads(line)


def native(data,ids):
    values=[{} for _ in ids];groups={}
    for i,u in enumerate(ids):groups.setdefault(u//SHARD_INPUTS*SHARD_INPUTS,[]).append((i,u))
    for name in data.roster:
        for start,entries in groups.items():
            d=data.native.descriptor(name,start);data.native.descriptor_fresh(d)
            array=np.load(d['root']/'vectors.npy',mmap_mode='r',allow_pickle=False)
            try:
                for i,u in entries:
                    lo,hi=map(int,d['offsets'][u-start:u-start+2])
                    values[i][name]=torch.from_numpy(np.array(array[lo:hi],copy=True))
            finally:array._mmap.close()
            data.native.descriptor_fresh(d)
    return tuple(values)


def recompute(donor,ref,expected,roster,*,refusal=None):
    slots={n:[-1.]*expected for n in roster};generated=len(donor)
    for j,right in enumerate(ref):
        for n in roster:
            a=donor[j][n].numpy().astype(np.float64);b=right[n].numpy().astype(np.float64)
            matrix=np.clip((a/np.linalg.norm(a,axis=1,keepdims=True))@
                (b/np.linalg.norm(b,axis=1,keepdims=True)).T,-1,1)
            slots[n][j]=float(min(matrix.max(0).mean(),matrix.max(1).mean()))
    extra=max(0,generated-expected);ended=bool(generated)
    return dict(expected_atoms=expected,generated_atoms=generated,ended=ended,
        ended_at_reference_boundary=ended and generated==expected,censored=not ended,
        matched_positions=min(expected,generated),missing=max(0,expected-generated),extra=extra,
        per_slot=slots,full_output_mean={n:(sum(v)-extra)/(expected+extra) for n,v in slots.items()},
        missing_similarity=-1.,extra_similarity=-1.,refusal=refusal,
        reference_scored_after_generation=True,semantic_validity_established=False)


def reject(fn):
    try:fn()
    except ValueError as exc:return str(exc)
    raise ValueError('EXACT_NATIVE_RECOUNT_NEGATIVE_ACCEPTED')


def main(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir()
    need(file_sha(a.input/'report.json')==a.pin,'EXACT_NATIVE_RECOUNT_REPORT_PIN')
    r=json.loads((a.input/'report.json').read_bytes());need(r['complete'],'EXACT_NATIVE_RECOUNT_COMPLETE')
    need(file_sha(a.input/'binding.json')==r['binding'],'EXACT_NATIVE_RECOUNT_BINDING_PIN')
    binding=json.loads((a.input/'binding.json').read_bytes())
    source,public_rows,data,continuation,reader=authenticate(a.root);rows,code=selection(a.root,source)
    need(r['code']==code and r['source']==source.binding==binding['source']
         and binding['mode']==r['mode'] and binding['policy']==scoring.POLICY,'EXACT_NATIVE_RECOUNT_AUTHORITY')
    chosen=rows if r['mode']=='full' else rows[np.isin(rows['document'],PROOF)]
    need(r['documents']==binding['documents']==chosen['document'].tolist(),'EXACT_NATIVE_RECOUNT_COVERAGE')
    roster=source.roster;groups={};total_bytes=compared=0;first_case=first_native=None
    phases=dict(authentication=time.perf_counter()-tick,native=0.,scores=0.,proof_trace=0.)
    records=checked_cases(a.input/'cases.jsonl',r['cases_pin']);seen=[]
    for row in chosen:
        tick=time.perf_counter();case=next(records);d=int(row['document']);seen.append(d)
        lo,hi=map(int,source.documents[d]);at=lo+int(row['observed']);expected=hi-at;status=int(row['status'])
        donor_id=int(row['donor']);ids=[] if donor_id<0 else source.mapping[int(row['donor_answer_start']):int(row['donor_stop'])].tolist()
        ref_ids=source.mapping[at:at+min(expected,len(ids))].tolist()
        donor=native(data,ids);ref=native(data,ref_ids);payload=int(source.counts[ids+ref_ids].sum())*768*4
        total_bytes+=payload;compared+=len(ref)
        need(case['native']==dict(donor=fingerprint((ids,donor)),reference=fingerprint((ref_ids,ref)),
            donor_atoms=len(ids),compared_atoms=len(ref),bytes=payload),'EXACT_NATIVE_RECOUNT_RAW_WINDOWS')
        phases['native']+=time.perf_counter()-tick;tick=time.perf_counter()
        identity=dict(document=d,component=int(source.components[d]),side=('train','development','query')[int(source.sides[d])],
            collection=source.metadata[d]['source_path'])
        need(case['identity']==identity and case['selection_status']==status and case['donor']==donor_id
             and case['selection_row']==digest(row.tobytes()),'EXACT_NATIVE_RECOUNT_CASE_BINDING')
        modal=recompute(donor,ref,expected,roster,refusal='NO_EXACT_TRAIN_PREFIX' if status==0 else None)
        strict=recompute((),(),expected,roster,refusal='CONFLICTING_TRAIN_ANSWERS') if status==2 else modal
        need(case['arms']==dict(modal_control=modal,clarify_conflicts=strict),'EXACT_NATIVE_RECOUNT_METRICS')
        for arm,m in case['arms'].items():
            for collection in ('ALL',identity['collection']):
                groups.setdefault((collection,identity['side'],arm),[]).append(dict(status=status,
                    counts={k:int(m[k]) for k in ('expected_atoms','generated_atoms','matched_positions','missing','extra',
                        'ended','ended_at_reference_boundary','censored')},means=m['full_output_mean'],
                    sums={n:sum(m['per_slot'][n])-m['extra'] for n in roster}))
        if donor and first_native is None:first_native=donor;first_case=case
        phases['scores']+=time.perf_counter()-tick;tick=time.perf_counter()
        if r['mode']=='proof':
            p=a.input/('document-'+str(d));saved=r['proofs'][str(d)]
            need(file_sha(p/'native.pt')==saved['native'] and file_sha(p/'retrieved/events.bin')==saved['trace'],
                 'EXACT_NATIVE_RECOUNT_PROOF_PINS')
            full=native(data,source.mapping[at:hi].tolist());v=torch.load(p/'native.pt',map_location='cpu',weights_only=True)
            need(fingerprint(v)==fingerprint(dict(donor_ids=ids,donor=donor,reference_ids=source.mapping[at:hi].tolist(),reference=full)),
                 'EXACT_NATIVE_RECOUNT_PROOF_NATIVE')
            actual=direct(p/'retrieved',full,roster)
            need(actual==dict(per_slot=modal['per_slot'],generated=len(ids),ended=bool(ids),extra=modal['extra'],
                missing=modal['missing'],means=modal['full_output_mean']),'EXACT_NATIVE_RECOUNT_EXISTING_READER')
        phases['proof_trace']+=time.perf_counter()-tick
    sentinel=object();need(next(records,sentinel) is sentinel,'EXACT_NATIVE_RECOUNT_EXTRA_CASE')
    tick=time.perf_counter();summary=[]
    for (collection,side,arm),items in sorted(groups.items()):
        counts={n:sum(i['counts'][n] for i in items) for n in items[0]['counts']};den=counts['expected_atoms']+counts['extra']
        summary.append(dict(collection=collection,side=side,arm=arm,cases=len(items),counts=counts,
            selection_status={str(i):sum(x['status']==i for x in items) for i in (0,1,2)},
            per_slot={n:dict(case_mean=math.fsum(i['means'][n] for i in items)/len(items),
                atom_mean=math.fsum(i['sums'][n] for i in items)/den) for n in roster}))
    need(r['summary']==dict(cases=len(seen),groups=summary,semantic_validity_established=False,
        architecture_winner=False,uncertainty_estimated=False),'EXACT_NATIVE_RECOUNT_SUMMARY')
    need(total_bytes==r['native_bytes'],'EXACT_NATIVE_RECOUNT_BYTES')
    controls={}
    if r['mode']=='proof':
        bad=deepcopy(first_native);bad[0].pop(roster[0]);controls['missing_slot']=reject(lambda:scoring.native(bad,roster))
        bad=deepcopy(first_native);bad[0][roster[0]]=bad[0][roster[0]][:0]
        controls['lost_window']=reject(lambda:scoring.native(bad,roster))
        controls['reference_overrun']=reject(lambda:scoring.metrics(first_native,first_native,expected=1,roster=roster,status=1))
        bad=deepcopy(first_case);bad['arms'].pop('clarify_conflicts')
        controls['missing_arm']=reject(lambda:scoring.validate_case(bad,roster))
        s=scoring.Summary(roster);s.add(first_case);controls['duplicate_case']=reject(lambda:s.add(first_case))
        wrong=a.output/'corrupt-cases.jsonl';raw=(a.input/'cases.jsonl').read_bytes();wrong.write_bytes(bytes([raw[0]^1])+raw[1:])
        controls['same_size_corrupt_scores']=reject(lambda:next(checked_cases(wrong,r['cases_pin'])))
    phases['summary_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    reader.close();source.close();public_rows._mmap.close();continuation.fresh()
    need(all(file_sha(BASE/n)==p for n,p in code.items()) and not torch.cuda.is_initialized(),'EXACT_NATIVE_RECOUNT_FINAL')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'EXACT_NATIVE_RECOUNT_TIMING')
    result=dict(complete=True,acceptance='author-checked',mode=r['mode'],producer=a.pin,code=code,source=source.binding,
        documents=seen,all_native_exact=True,all_scores_exact=True,summary_exact=True,existing_reader_exact=r['mode']=='proof',
        controls=controls,native_bytes=total_bytes,compared_positions=compared,summary=r['summary'],
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,encoder_calls=0,gpu_used=False,admitted_senses=0,semantic_validity_established=False)
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds,cases=len(seen))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:main(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

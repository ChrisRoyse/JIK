"""Independent tuple-based readback of full legal continuation prefix controls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.process_conditions import write_committed
from model.legal_answer_evaluation_plan import authenticate
from model.legal_answer_exact_control import DTYPE,POLICY
from text.constellation_corpus import file_sha,need


def signature(answer):
    return hashlib.sha256(b'legal-answer-uid-sequence-v1\0'+struct.pack('<Q',len(answer))+
        struct.pack('<'+'q'*len(answer),*answer)).hexdigest()


def read(root,pin):
    need(file_sha(root/'report.json')==pin,'EXACT_READBACK_REPORT_PIN')
    r=json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['policy']==POLICY,'EXACT_READBACK_REPORT')
    for name,ref in r['files'].items():
        need(Path(name).name==name and (root/name).stat().st_size==ref['bytes']
             and file_sha(root/name)==ref['sha256'],'EXACT_READBACK_FILE_PIN')
    return r


def main(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir()
    r=read(a.producer,a.pin);cold=read(a.cold,a.cold_pin)
    for key in ('source','code','policy','files','selection_seal','train','census','controls','ambiguity','order'):
        need(r[key]==cold[key],'EXACT_READBACK_COLD_'+key)
    base=Path(__file__).resolve().parents[1]
    need(all(file_sha(base/n)==p for n,p in r['code'].items()),'EXACT_READBACK_CODE')
    source=authenticate(a.root);need(source.binding==r['source'],'EXACT_READBACK_SOURCE')
    rows=np.load(a.producer/'selection.npy',allow_pickle=False)
    train_file=json.loads((a.producer/'train-index.json').read_bytes())
    seal=json.loads((a.producer/'selection.json').read_bytes())
    need(file_sha(a.producer/'selection.json')==r['selection_seal']
         and seal['private_references_read'] is False and seal['source']==source.binding
         and seal['train_index']==file_sha(a.producer/'train-index.json')
         and seal['selection']==file_sha(a.producer/'selection.npy'),'EXACT_READBACK_SEAL')
    private=[json.loads(s) for s in (a.producer/'private-cases.jsonl').read_bytes().splitlines()]
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    # No producer indexing, prefix, hashing, selection or census functions are used.
    groups=defaultdict(lambda:defaultdict(list)); observed={}; answers={}; boundaries={}
    for d in range(len(source.documents)):
        lo,hi=map(int,source.documents[d]); n=1 if hi-lo==2 else 2
        observed[d]=tuple(int(v) for v in source.mapping[lo:lo+n])
        boundaries[d]=(lo+n,hi)
        if int(source.sides[d])==0:
            answer=tuple(int(v) for v in source.mapping[lo+n:hi]);answers[d]=answer
            groups[observed[d]][answer].append(d)
    expected_groups=[dict(prefix=list(k),answers={signature(answer):dict(donor=min(ds),count=len(ds))
        for answer,ds in g.items()}) for k,g in sorted(groups.items())]
    need(train_file['groups']==expected_groups and train_file['source']==source.binding
         and train_file['policy']==POLICY,'EXACT_READBACK_TRAIN_INDEX')
    train_summary=dict(train_documents=len(answers),train_prefix_groups=len(groups),
        conflicting_train_prefix_groups=sum(len(g)>1 for g in groups.values()),
        train_documents_in_conflicting_groups=sum(sum(map(len,g.values())) for g in groups.values() if len(g)>1),
        unique_prefix_input_deterministic_exact_ceiling_numerator=sum(max(map(len,g.values())) for g in groups.values()),
        ceiling_denominator=len(answers),train_answer_occurrences=sum(map(len,answers.values())),
        answer_bytes_hashed=8*sum(map(len,answers.values())))
    need(train_summary==r['train']==train_file['summary'],'EXACT_READBACK_TRAIN_COUNTS')
    phases['independent_train_tuples']=time.perf_counter()-tick;tick=time.perf_counter()
    ids=[i for i in range(len(source.documents)) if source.sides[i]!=0]
    need(rows.dtype==DTYPE and rows['document'].tolist()==ids and len(private)==len(ids)
         and seal['cases']==len(ids),'EXACT_READBACK_ALL_CASES')
    counts=Counter();collections={};held=defaultdict(Counter);donors=Counter()
    for j,d in enumerate(ids):
        group=groups.get(observed[d],{}); selection=np.zeros((),dtype=DTYPE)
        selection['uids']=-1;selection['uids'][:len(observed[d])]=observed[d]
        donor=-1;pin='00'*32;at=hi=-1;count=0
        if group:
            answer=min(group,key=lambda v:(-len(group[v]),signature(v),min(group[v])))
            donor=min(group[answer]);pin=signature(answer);count=len(group[answer]);at,hi=boundaries[donor]
            need(source.sides[donor]==0 and source.components[donor]!=source.components[d],
                 'EXACT_READBACK_DONOR_SPLIT')
            donors[donor]+=1
        status=0 if not group else 1 if len(group)==1 else 2
        fields=dict(ordinal=j,document=d,side=int(source.sides[d]),component=int(source.components[d]),
            observed=len(observed[d]),matches=sum(map(len,group.values())),distinct_answers=len(group),
            modal_count=count,donor=donor,donor_answer_start=at,donor_stop=hi,status=status)
        for k,v in fields.items():selection[k]=v
        selection['answer']=bytes.fromhex(pin)
        need(selection.tobytes()==rows[j].tobytes(),'EXACT_READBACK_SELECTION')
        a0,b0=boundaries[d];reference=tuple(int(v) for v in source.mapping[a0:b0]);held[observed[d]][reference]+=1
        exact=donor>=0 and answers[donor]==reference
        expected_case=dict(ordinal=j,document=d,reference_answer=signature(reference),reference_atoms=len(reference),
            donor=donor,status=('unavailable','candidate','clarification_required')[status],modal_exact=exact)
        need(private[j]==expected_case,'EXACT_READBACK_PRIVATE_CASE')
        delta=dict(cases=1,prefix_seen_in_train=int(bool(group)),unambiguous_train_prefix=int(status==1),
            ambiguous_train_prefix=int(status==2),unavailable=int(status==0),modal_exact_recorded_answer=int(exact),
            unambiguous_exact_recorded_answer=int(exact and status==1),reference_answer_occurrences=len(reference))
        counts.update(delta);collections.setdefault(source.metadata[d]['source_path'],Counter()).update(delta)
    actual=dict(counts=dict(counts),by_collection={k:dict(v) for k,v in sorted(collections.items())},
        heldout_prefix_groups=len(held),conflicting_heldout_prefix_groups=sum(len(g)>1 for g in held.values()),
        heldout_deterministic_exact_ceiling_numerator=sum(max(g.values()) for g in held.values()),
        ceiling_denominator=len(ids),reference_scored_after_selection=True,
        ceiling_scope='observed_exact_uid_prefix_to_recorded_uid_sequence_on_this_finite_population',
        distinct_valid_answers_not_errors=True,semantic_validity_established=False)
    need(actual==r['census'],'EXACT_READBACK_PRIVATE_COUNTS')
    phases['selection_and_private_recount']=time.perf_counter()-tick;tick=time.perf_counter()
    source.close();need(all(file_sha(base/n)==p for n,p in r['code'].items()),'EXACT_READBACK_FINAL_CODE')
    phases['final_source_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'EXACT_READBACK_TIMING')
    pin=write_committed(a.output/'report.json',dict(complete=True,acceptance='author-checked',
        producer=a.pin,cold=a.cold_pin,source=r['source'],code=r['code'],selection=r['selection_seal'],
        train_index_exact=True,all_selections_exact=True,all_private_cases_exact=True,
        train=train_summary,census=actual,distinct_selected_donors=len(donors),
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,native_encoder_calls=0,gpu_used=False,admitted_senses=0,
        capability_demonstrated=False))
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('producer','cold','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--cold-pin',required=True);a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

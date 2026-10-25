"""Publish full legal TRAIN prefix lookup and only then score private references."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.process_conditions import write_committed
from model import legal_answer_exact_control as control
from model.legal_answer_evaluation_plan import authenticate
from text.constellation_corpus import canonical,file_sha,need

BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
CODE=('model/legal_answer_exact_control.py','audit/legal_answer_exact_control.py',
      'audit/legal_answer_exact_readback.py','model/legal_complete_answer_plan.py',
      'model/legal_answer_evaluation_plan.py','text/constellation_corpus.py',
      'text/constellation_input.py','claims/process_conditions.py')


def negative(fn, label):
    try:fn()
    except (ValueError,AssertionError) as exc:return dict(control=label,refused=str(exc))
    raise ValueError('MNLM_EXACT_CONTROL_NEGATIVE_ACCEPTED_'+label)


def main(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir()
    source=authenticate(a.root); code={n:file_sha(BASE/n) for n in CODE}
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    index=control.ExactControl(source)
    train_pin=write_committed(a.output/'train-index.json',dict(policy=control.POLICY,
        source=source.binding,summary=index.summary,groups=[dict(prefix=list(k),answers=g)
            for k,g in sorted(index.groups.items())]))
    phases['train_index']=time.perf_counter()-tick;tick=time.perf_counter()
    ids=np.flatnonzero(source.sides!=0)
    rows=np.zeros(len(ids),dtype=control.DTYPE)
    for j,d in enumerate(ids):rows[j]=index.row(j,int(d))
    np.save(a.output/'selection.npy',rows,allow_pickle=False)
    with (a.output/'selection.npy').open('rb') as f:os.fsync(f.fileno())
    selection_pin=file_sha(a.output/'selection.npy')
    seal=write_committed(a.output/'selection.json',dict(policy=control.POLICY,source=source.binding,
        code=code,train_index=train_pin,selection=selection_pin,cases=len(rows),
        private_references_read=False,model_loads=0,encoder_calls=0))
    phases['public_selection_and_seal']=time.perf_counter()-tick;tick=time.perf_counter()
    # Only now inspect the evaluation continuation. The fixed selector never sees it.
    need(file_sha(a.output/'selection.json')==seal,'EXACT_CONTROL_SEAL')
    census,cases=control.private_census(source,rows)
    with (a.output/'private-cases.jsonl').open('xb') as f:
        for row in cases:f.write(canonical(row)+b'\n')
        f.flush();os.fsync(f.fileno())
    phases['private_census']=time.perf_counter()-tick;tick=time.perf_counter()
    train=np.flatnonzero(source.sides==0).tolist()
    controls=[negative(lambda:control.ExactControl(source,training=train+[int(ids[0])]),'heldout_fit'),
        negative(lambda:control.ExactControl(source,training=train[:-1]),'incomplete_train'),
        negative(lambda:index.choose((True,)),'boolean_address'),
        negative(lambda:index.choose((len(source.counts),)),'unknown_address'),
        negative(lambda:control.validate_selection(source,index,rows[:-1]),'missing_case')]
    bad=rows.copy();bad[0]['donor']=int(ids[0])
    controls.append(negative(lambda:control.validate_selection(source,index,bad),'heldout_donor'))
    bad=rows.copy();bad[0]['uids'][0]=len(source.counts)
    controls.append(negative(lambda:control.validate_selection(source,index,bad),'changed_public'))
    # Real repeated-prefix conflict, if available, is the positive ambiguity boundary.
    conflicts=[k for k,g in index.groups.items() if len(g)>1]
    ambiguity=None
    if conflicts:
        key=min(conflicts);choice=index.choose(key)
        need(choice['status']==2 and choice['distinct_answers']>1,'EXACT_CONTROL_AMBIGUITY')
        ambiguity=dict(prefix=list(key),choice=choice)
    pair=next(k for k in index.groups if len(k)==2 and k[0]!=k[1])
    # Public order is significant even when both directions happen to be unavailable.
    forward=index.choose(pair);reverse=index.choose(pair[::-1])
    need(pair!=pair[::-1] and (forward['donor']!=reverse['donor']), 'EXACT_CONTROL_ORDER')
    control.validate_selection(source,index,rows)
    phases['controls_and_selection_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    source.close();need(all(file_sha(BASE/n)==p for n,p in code.items()),'EXACT_CONTROL_CODE_CHANGED')
    need(file_sha(a.output/'selection.npy')==selection_pin,'EXACT_CONTROL_SELECTION_CHANGED')
    phases['final_source_readback']=time.perf_counter()-tick
    seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'EXACT_CONTROL_TIMING')
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    report=dict(schema=control.POLICY['schema'],complete=True,acceptance='author-checked',
        source=source.binding,code=code,policy=control.POLICY,files=files,selection_seal=seal,
        train=index.summary,census=census,controls=controls,ambiguity=ambiguity,
        order=dict(prefix=list(pair),forward=forward,reversed=reverse),
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,native_encoder_calls=0,gpu_used=False,admitted_senses=0,
        architecture_winner=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=pin,train=index.summary,census=census,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

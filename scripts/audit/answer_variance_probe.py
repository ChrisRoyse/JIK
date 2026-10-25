"""Matched complete-answer updates with original or scoped variance reads."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from model import frozen_variance_scope as candidate
from text.constellation_corpus import file_sha,need

NEW=('model/frozen_variance_scope.py','audit/answer_variance_probe.py')


def controls(source):
    result={}
    def case(label,code,action):
        model=deepcopy(source);heads=tuple(model.forecaster.placement.values())
        result[label]=proof.reject(lambda:action(model,heads),code)
        need(all('variances' not in vars(h) for h in heads),'VARIANCE_PROBE_NOT_RESTORED:'+label)
    def mutate(model,heads,kind):
        with candidate.frozen_variances(model):
            head=heads[0];value=head._buffers['variance_0']
            if kind=='inplace':value.add_(1)
            elif kind=='replacement':head._buffers['variance_0']=value.clone()
            elif kind=='roster':head.roster=head.roster[::-1]
            elif kind=='raw_data':
                version=value._version;value.data.add_(1)
                need(value._version==version,'VARIANCE_PROBE_RAW_VERSION')
            head.variances()
    for label,code in (('inplace','BUFFER_CHANGED'),('replacement','BUFFER_CHANGED'),
                       ('roster','ROSTER_CHANGED'),('raw_data','VALUES_CHANGED')):
        case(label,'VARIANCE_SCOPE_'+code,lambda m,h,k=label:mutate(m,h,k))
    def invalid(model,heads,kind):
        head=heads[0]
        if kind=='missing':del head._buffers['variance_0']
        elif kind=='dtype':head._buffers['variance_0']=torch.tensor(1,dtype=torch.int64)
        else:head._buffers['variance_0'].fill_(float('nan') if kind=='nonfinite' else 0.)
        with candidate.frozen_variances(model):pass
    for label,code in (('missing','MISSING_BUFFER'),('dtype','LAYOUT'),
                       ('nonfinite','INVALID_VARIANCE'),('zero','INVALID_VARIANCE')):
        case(label,'VARIANCE_SCOPE_'+code,lambda m,h,k=label:invalid(m,h,k))
    def nested(model,heads):
        with candidate.frozen_variances(model):
            with candidate.frozen_variances(model):pass
    case('nested','VARIANCE_SCOPE_METHOD',nested)
    def partial(model,heads):
        heads[-1].variances=lambda:None
        try:
            try:
                with candidate.frozen_variances(model):pass
            finally:need(all('variances' not in vars(h) for h in heads[:-1]),'VARIANCE_PROBE_PARTIAL_INSTALL')
        finally:del heads[-1].variances
    case('partial_install','VARIANCE_SCOPE_METHOD',partial)
    def interrupted(model,heads):
        with candidate.frozen_variances(model):raise ValueError('VARIANCE_PROBE_INTERRUPTED')
    case('exception','VARIANCE_PROBE_INTERRUPTED',interrupted)
    def changed(model,heads):
        with candidate.frozen_variances(model):heads[0].variances=lambda:None
    case('ownership','VARIANCE_SCOPE_OWNERSHIP_CHANGED',changed)
    model=deepcopy(source);heads=tuple(model.forecaster.placement.values())
    expected=[{n:float(v) for n,v in h.variances().items()} for h in heads]
    with candidate.frozen_variances(model) as caches:
        for head,values in zip(heads,expected):
            observed=head.variances()
            need(tuple(observed)==tuple(values) and all(struct.pack('!d',observed[n])==struct.pack('!d',v)
                 for n,v in values.items()),'VARIANCE_PROBE_SCALAR_BYTES')
            observed[next(iter(observed))]=-1.
            need(head.variances()==values,'VARIANCE_PROBE_RETURNED_DICT_MUTATION')
    result['closed']=proof.reject(caches[0].read,'VARIANCE_SCOPE_CLOSED')
    need(all(c.report()['batch_reads']==2 and c.report()['scalar_reads']==2*len(c.roster) for c in caches),
         'VARIANCE_PROBE_BOUNDARY_READS')
    need(all('variances' not in vars(h) for h in heads),'VARIANCE_PROBE_NOT_RESTORED')
    result.update(exact_scalar_bytes=True,returned_dictionary_isolated=True,scope_restored=True)
    return result


def run(a):
    a.output.mkdir(exist_ok=False);original=proof.learner.update;negative={};observations=[]
    code={name:file_sha(proof.source.BASE/name) for name in NEW};began=time.perf_counter()
    for name in code:
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((proof.source.BASE/name).read_bytes())
    def observe(*args,**kwargs):
        model,_,item,step,_=args;arch=model.forecaster.architecture
        if arch not in negative:negative[arch]=controls(model)
        cached=proof.candidate.ACTIVE;tick=time.perf_counter();caches=()
        with candidate.frozen_variances(model) if cached else proof.candidate.captured_backward() as context:
            if cached:caches=context
            value=original(*args,**kwargs)
        seconds=time.perf_counter()-tick
        observations.append(dict(architecture=arch,document=item['document'],step=step,
            variant='scoped_variances' if cached else 'original_variances',seconds=seconds,
            decisions=value[0]['decisions'],cache=[c.report() for c in caches]))
        return value
    proof.learner.update=observe
    try:
        previous=a.previous/'numerical' if a.previous else None
        prior=proof.source.read(a.previous/'report.json',a.previous_pin) if a.previous else None
        if prior:need(prior['complete'] and prior['code']==code,'VARIANCE_PROBE_COLD_BINDING')
        proof.run(argparse.Namespace(root=a.root,output=a.output/'numerical',cpu_proof=True,
            previous=previous,previous_pin=prior['numerical_report'] if prior else None))
    finally:
        intact=proof.learner.update is observe;proof.learner.update=original
        need(intact,'VARIANCE_PROBE_UPDATE_OWNERSHIP')
    report=json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and len(observations)==30
         and sum(x['decisions'] for x in observations)==396,'VARIANCE_PROBE_COVERAGE')
    groups={}
    for arch in proof.source.ARCHITECTURES:
        groups[arch]={v:sum(x['seconds'] for x in observations if x['architecture']==arch and x['variant']==v)
                      for v in ('original_variances','scoped_variances')}
    if prior:need(negative==prior['controls'],'VARIANCE_PROBE_COLD_CONTROLS')
    need(all(file_sha(proof.source.BASE/n)==h for n,h in code.items()),'VARIANCE_PROBE_CODE_CHANGED')
    result=dict(complete=True,code=code,controls=negative,observations=observations,groups=groups,
        numerical_report=file_sha(a.output/'numerical/report.json'),seconds=time.perf_counter()-began,
        captured_backward_in_both_arms=True,cold_reproduced=prior is not None,previous=a.previous_pin,
        gpu_used=False,promoted=False,acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result),groups=groups)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=proof.source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

"""Matched whole-answer updates with direct or state_dict-based version checks."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from model import direct_state_versions as candidate
from text.constellation_corpus import file_sha,need

NEW=('model/direct_state_versions.py','audit/answer_version_probe.py')


def controls(model):
    model=deepcopy(model);reference=candidate.ORIGINALS[0];result={}
    def equal(label):
        need(candidate.versions(model)==reference(model),'VERSION_PROBE_TUPLE:'+label)
        result[label]=True
    equal('initial');before=reference(model)
    value=next(model.parameters())
    with torch.no_grad():value.add_(1)
    need(reference(model)!=before,'VERSION_PROBE_INPLACE_UNOBSERVED');equal('inplace')
    child=next(m for m in model.modules() if m._parameters)
    name=next(iter(child._parameters));before=reference(model)
    child._parameters[name]=torch.nn.Parameter(child._parameters[name].detach().clone())
    need(reference(model)!=before,'VERSION_PROBE_REPLACEMENT_UNOBSERVED');equal('replacement')
    before=reference(model);model.register_buffer('_probe_persistent',torch.ones(2))
    need(reference(model)!=before,'VERSION_PROBE_BUFFER_UNOBSERVED');equal('persistent_buffer')
    before=reference(model);model.register_buffer('_probe_ephemeral',torch.ones(2),persistent=False)
    need(reference(model)==before,'VERSION_PROBE_EPHEMERAL_REFERENCE');equal('nonpersistent_buffer')
    before=reference(model);model.add_module('_probe_alias',child)
    need(reference(model)!=before,'VERSION_PROBE_ALIAS_UNOBSERVED');equal('shared_alias')
    model.register_parameter('_probe_none',None);model.add_module('_probe_none_child',None)
    equal('none_entries')
    handle=model.register_state_dict_pre_hook(lambda *args:None)
    try:result['serialization_hook']=proof.reject(lambda:candidate.versions(model),'DIRECT_VERSIONS_CUSTOM_SERIALIZATION')
    finally:handle.remove()
    model.state_dict=lambda **kwargs:None
    result['serialization_override']=proof.reject(lambda:candidate.versions(model),'DIRECT_VERSIONS_CUSTOM_SERIALIZATION')
    del model.state_dict
    def nested():
        with candidate.direct_versions():
            with candidate.direct_versions():pass
    result['nested']=proof.reject(nested,'DIRECT_VERSIONS_ALREADY_INSTALLED')
    def interrupted():
        with candidate.direct_versions():raise ValueError('VERSION_PROBE_INTERRUPTED')
    result['exception']=proof.reject(interrupted,'VERSION_PROBE_INTERRUPTED')
    def changed():
        with candidate.direct_versions():candidate.BINDINGS[0][0].versions=reference
    result['ownership']=proof.reject(changed,'DIRECT_VERSIONS_OWNERSHIP_CHANGED')
    need(not candidate.ACTIVE and all(getattr(m,n) is old for (m,n),old in zip(candidate.BINDINGS,candidate.ORIGINALS)),
         'VERSION_PROBE_BINDINGS_NOT_RESTORED')
    result['scope_restored']=True
    return result


def run(a):
    a.output.mkdir(exist_ok=False);original=proof.learner.update;negative={};observations=[]
    code={name:file_sha(proof.source.BASE/name) for name in NEW};began=time.perf_counter()
    def observe(*args,**kwargs):
        model,_,item,step,_=args;arch=model.forecaster.architecture
        if arch not in negative:negative[arch]=controls(model)
        direct=proof.candidate.ACTIVE;tick=time.perf_counter()
        with candidate.direct_versions() if direct else proof.candidate.captured_backward():
            value=original(*args,**kwargs)
        observations.append(dict(architecture=arch,document=item['document'],step=step,
            variant='direct_versions' if direct else 'state_dict_versions',
            seconds=time.perf_counter()-tick,decisions=value[0]['decisions']))
        return value
    proof.learner.update=observe
    try:
        previous=a.previous/'numerical' if a.previous else None
        prior=proof.source.read(a.previous/'report.json',a.previous_pin) if a.previous else None
        if prior:need(prior['complete'] and prior['code']==code,'VERSION_PROBE_COLD_BINDING')
        proof.run(argparse.Namespace(root=a.root,output=a.output/'numerical',cpu_proof=True,
            previous=previous,previous_pin=prior['numerical_report'] if prior else None))
    finally:
        intact=proof.learner.update is observe;proof.learner.update=original
        need(intact,'VERSION_PROBE_UPDATE_OWNERSHIP')
    report=json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and len(observations)==30
         and sum(x['decisions'] for x in observations)==396,'VERSION_PROBE_COVERAGE')
    groups={}
    for arch in proof.source.ARCHITECTURES:
        groups[arch]={v:sum(x['seconds'] for x in observations if x['architecture']==arch and x['variant']==v)
                      for v in ('state_dict_versions','direct_versions')}
    if prior:need(negative==prior['controls'],'VERSION_PROBE_COLD_CONTROLS')
    need(all(file_sha(proof.source.BASE/n)==h for n,h in code.items()),'VERSION_PROBE_CODE_CHANGED')
    result=dict(complete=True,code=code,controls=negative,observations=observations,groups=groups,
        numerical_report=file_sha(a.output/'numerical/report.json'),seconds=time.perf_counter()-began,
        captured_backward_in_both_arms=True,cold_reproduced=prior is not None,previous=a.previous_pin,
        gpu_used=False,promoted=False,acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result),groups=groups)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=proof.source.ROOT)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path)
    parser.add_argument('--previous-pin');args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists():write_committed(args.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

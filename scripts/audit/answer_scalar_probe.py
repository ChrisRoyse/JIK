"""Compare scalar checkpoint updates; retain the first numerical refusal."""
import argparse
from contextlib import ExitStack
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
from model import scalar_answer_checkpoint as candidate
from text.constellation_corpus import file_sha,need

NEW=('model/scalar_answer_checkpoint.py','audit/answer_scalar_probe.py')


def controls(source):
    model=deepcopy(source);result={};original=model.forecaster.losses
    def nested():
        with candidate.scalar_heads(model):
            with candidate.scalar_heads(model):pass
    result['nested']=proof.reject(nested,'SCALAR_HEAD_ALREADY_INSTALLED')
    def interrupted():
        with candidate.scalar_heads(model):raise ValueError('SCALAR_PROBE_INTERRUPTED')
    result['exception']=proof.reject(interrupted,'SCALAR_PROBE_INTERRUPTED')
    def changed():
        with candidate.scalar_heads(model):model.forecaster.losses=original
    result['ownership']=proof.reject(changed,'SCALAR_HEAD_OWNERSHIP_CHANGED')
    pin=candidate.versions(model);value=next(model.parameters())
    with torch.no_grad():value.add_(1)
    result['weights']=proof.reject(lambda:candidate.fresh(model,pin),'SCALAR_HEAD_WEIGHTS_CHANGED')
    owner=candidate.STORAGE(1<<20,'cpu');replay=candidate.Replay(owner)
    tensor=value.detach().flatten()[:4];replay.record(tensor)
    result['metadata']=proof.reject(lambda:replay.retain(tensor.double()),'SCALAR_HEAD_METADATA_CHANGED')
    result['forward_graph']=proof.reject(lambda:replay.impossible(None),'SCALAR_HEAD_FORWARD_GRAPH_REUSED')
    replay.primal=tensor[0].clone()
    result['count']=proof.reject(lambda:replay.finish(replay.primal),'SCALAR_HEAD_METADATA_COUNT')
    saved=replay.retain(tensor);need(owner.current>0,'SCALAR_PROBE_STORAGE_NOT_TRACKED')
    result['overflow']=proof.reject(lambda:replay.retain(tensor),'SCALAR_HEAD_METADATA_CHANGED')
    result['value']=proof.reject(lambda:replay.finish(replay.primal+1),'SCALAR_HEAD_VALUE_CHANGED')
    replay.finish(replay.primal);need(torch.equal(owner.unpack(saved),tensor),'SCALAR_PROBE_STORAGE_VALUES')
    del saved
    need(owner.current==0 and model.forecaster.losses is original and proof.learner.SavedStorage is candidate.STORAGE,
         'SCALAR_PROBE_NOT_RESTORED')
    result.update(scalar_exact=True,storage_released=True,scope_restored=True)
    return result


def run(a):
    a.output.mkdir(exist_ok=False);original=proof.learner.update;negative={};observations=[]
    code={name:file_sha(proof.source.BASE/name) for name in NEW};began=time.perf_counter()
    for name in code:
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((proof.source.BASE/name).read_bytes())
    prior=proof.source.read(a.previous/'report.json',a.previous_pin) if a.previous else None
    if prior:need(prior['code']==code,'SCALAR_PROBE_COLD_CODE')
    reverse=bool(prior and prior['refusal'])
    def observe(*args,**kwargs):
        model,_,item,step,_=args;arch=model.forecaster.architecture;active=proof.candidate.ACTIVE
        scalar=active!=reverse
        with ExitStack() as stack:
            if not active:stack.enter_context(proof.candidate.captured_backward())
            if arch not in negative:
                negative[arch]=controls(model);write_committed(a.output/f'controls-{arch}.json',negative[arch])
            tick=time.perf_counter();replays=stack.enter_context(candidate.scalar_heads(model)) if scalar else ()
            value=original(*args,**kwargs)
        row=dict(architecture=arch,document=item['document'],step=step,inner='captured' if active else 'original',
            variant='scalar_checkpoint' if scalar else 'original_checkpoint',seconds=time.perf_counter()-tick,
            decisions=value[0]['decisions'],replays=[r.report() for r in replays],storage=value[0]['storage'])
        observations.append(row);write_committed(a.output/f'observation-{len(observations):02}.json',row)
        return value
    proof.learner.update=observe;refusal=None
    try:
        previous=a.previous/'numerical' if prior and not prior['refusal'] else None
        proof.run(argparse.Namespace(root=a.root,output=a.output/'numerical',cpu_proof=True,
            previous=previous,previous_pin=prior['numerical_report'] if previous else None))
    except ValueError as exc:
        if str(exc)!='MNLM_TEXT_CORPUS_CAPTURE_PROOF_NUMERICAL_UPDATE':raise
        refusal=str(exc);write_committed(a.output/'numerical/refusal.json',dict(error=refusal))
    finally:
        intact=proof.learner.update is observe;proof.learner.update=original
        need(intact and proof.learner.SavedStorage is candidate.STORAGE,'SCALAR_PROBE_RESTORE')
    numerical=a.output/'numerical/report.json'
    binding=json.loads((a.output/'numerical/binding.json').read_bytes())
    for name,pin in binding['code'].items():
        need(file_sha(proof.source.BASE/name)==pin,'SCALAR_PROBE_INPUT_CODE')
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((proof.source.BASE/name).read_bytes())
    groups={}
    for arch in negative:
        groups[arch]={v:sum(x['seconds'] for x in observations if x['architecture']==arch and x['variant']==v)
                      for v in ('original_checkpoint','scalar_checkpoint')}
    if prior:
        need(prior['refusal']==refusal and prior['controls']==negative,'SCALAR_PROBE_COLD_REFUSAL')
        old={(r['architecture'],r['document'],r['variant']):r for r in prior['observations']}
        need(len(old)==len(observations),'SCALAR_PROBE_COLD_COVERAGE')
        for row in observations:
            peer=old[row['architecture'],row['document'],row['variant']]
            need(row['replays']==peer['replays'] and row['decisions']==peer['decisions'],'SCALAR_PROBE_COLD_REPLAYS')
    need(all(file_sha(proof.source.BASE/n)==h for n,h in code.items()),'SCALAR_PROBE_CODE_CHANGED')
    result=dict(diagnostic_complete=True,all_updates_exact=refusal is None,refusal=refusal,code=code,controls=negative,
        observations=observations,groups=groups,numerical_report=file_sha(numerical) if numerical.exists() else None,
        binding=file_sha(a.output/'numerical/binding.json'),seconds=time.perf_counter()-began,
        captured_backward_in_both_arms=True,cold_reproduced=prior is not None,previous=a.previous_pin,
        reversed_assignment=reverse,gpu_used=False,promoted=False,acceptance='author-checked')
    result['files']={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    print(json.dumps(dict(diagnostic_complete=True,pin=write_committed(a.output/'report.json',result),
                         refusal=refusal,groups=groups)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=proof.source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

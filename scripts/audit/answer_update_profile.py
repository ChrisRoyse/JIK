"""Observe qualified whole-answer updates without replacing their arithmetic."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need


def rows(stats):
    return [dict(file=key[0],line=key[1],function=key[2],primitive_calls=value[0],
                 calls=value[1],self_seconds=value[2],cumulative_seconds=value[3])
            for key,value in sorted(stats.stats.items(),key=lambda kv:(-kv[1][3],kv[0]))]


def run(a):
    a.output.mkdir(exist_ok=False);original=proof.learner.update;observations=[]
    code=file_sha(__file__);began=time.perf_counter()
    def observe(*args,**kwargs):
        model,_,item,step,_=args
        profiler=cProfile.Profile();tick=time.perf_counter()
        try:
            profiler.enable()
            value=original(*args,**kwargs)
        finally:
            profiler.disable()
            path=a.output/f'update-{len(observations):02d}.pstats'
            profiler.dump_stats(path)
        stats=pstats.Stats(str(path))
        observations.append(dict(architecture=model.forecaster.architecture,document=item['document'],
            step=step,variant='captured' if proof.candidate.ACTIVE else 'original',
            decisions=value[0]['decisions'],seconds=time.perf_counter()-tick,
            total_calls=stats.total_calls,primitive_calls=stats.prim_calls,
            profiler_seconds=stats.total_tt,profile=path.name,pin=file_sha(path)))
        return value
    proof.learner.update=observe
    try:
        proof.run(argparse.Namespace(root=a.root,output=a.output/'numerical',cpu_proof=True,
                                     previous=None,previous_pin=None))
    finally:
        intact=proof.learner.update is observe;proof.learner.update=original
        need(intact,'UPDATE_PROFILE_OWNERSHIP_CHANGED')
    report=json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and len(observations)==30
         and sum(x['decisions'] for x in observations)==396,'UPDATE_PROFILE_COVERAGE')
    groups={}
    for arch in proof.source.ARCHITECTURES:
        for variant in ('original','captured'):
            selected=[x for x in observations if x['architecture']==arch and x['variant']==variant]
            need(len(selected)==5,'UPDATE_PROFILE_GROUP')
            stats=pstats.Stats(*(str(a.output/x['profile']) for x in selected))
            groups[arch+':'+variant]=dict(total_calls=stats.total_calls,primitive_calls=stats.prim_calls,
                profiler_seconds=stats.total_tt,functions=rows(stats))
    need(file_sha(__file__)==code and proof.learner.update is original,'UPDATE_PROFILE_SOURCE_CHANGED')
    result=dict(complete=True,code=code,numerical_report=file_sha(a.output/'numerical/report.json'),
        observations=observations,groups=groups,seconds=time.perf_counter()-began,
        profiler_overhead_included=True,speedup_claim=False,gpu_used=False,promoted=False,
        acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,report=pin,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=proof.source.ROOT)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists():write_committed(args.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

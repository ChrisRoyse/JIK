"""Execute a frozen depth increment using each task's unchanged original oracle."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_depth import DepthCohort,inputs
from oracle.bound_rows import BoundRows
from oracle.repository_cohort_actions import ATTEMPT,META,action,digest,open_witness
from oracle.replay_swe import verify_prepared
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE='MNLM_REPOSITORY_DEPTH_ACTION'


def source_memory(bound):return {k:bound[k] for k in ('head','journal','snapshot')}


def run(proof_path,proof_pin,output):
    start=time.perf_counter();proof=read_pinned(proof_path,proof_pin)
    require(proof['schema']=='repository-depth-selection-readback-v1' and
            proof['code']==digest(Path(__file__).parents[1]/'audit/repository_depth_selection.py') and
            proof['new_outcomes_read'] is False,CODE,'selection readback required')
    cohort=DepthCohort(**proof['selection']);bound=inputs(cohort.config);root=bound['root']
    require(proof['source_memory']==source_memory(bound) and
            output.resolve().is_relative_to(root) and not output.exists(),CODE,'fresh output and fixed memory required')
    requests=[]
    for case in cohort.cases:
        requests.append((case['benchmark'],META))
        requests.extend((a['binding'],ATTEMPT) for a in case['attempts'][case['prior_attempts']:])
    rows=BoundRows(root,requests)
    output.mkdir(parents=True);publish(output/'bound-rows.json',rows.receipts)
    oracle=Path(__file__).parents[1]/'oracle';code={p.name:digest(p) for p in oracle.glob('*.py')}
    results=[];binding_seconds=time.perf_counter()-start
    for case in cohort.cases:
        source=cohort.sources[case['task']];first=source['item']
        witness=read_pinned(Path(first['path']),first['sha256'])
        destination=output/case['task'];destination.mkdir()
        plan,prepared,prep_pin,spec,before,modes,files,shared=open_witness(root,first,witness,code)
        actions=[]
        for ordinal in range(case['prior_attempts'],len(case['attempts'])):
            attempt=case['attempts'][ordinal];path=destination/str(ordinal)
            try:
                result=action(root,cohort,case,attempt,rows,plan,prepared,spec,files,modes,shared,path)
            except Exception as exc:
                error=exc.envelope() if isinstance(exc,SenseError) else dict(code=CODE+'_EXECUTION',
                    message='No outcome; inspect retained evidence.',exception=type(exc).__name__)
                path.mkdir(exist_ok=True)
                pin=publish(path/'gap.json',dict(task=case['task'],attempt=attempt,error=error))
                result=dict(status='gap',path=str(path/'gap.json'),sha256=pin,error=error)
            actions.append(dict(ordinal=ordinal,**result))
            print(json.dumps(dict(task=case['task'],ordinal=ordinal,**result)),flush=True)
        require(verify_prepared(prepared,prep_pin)[1]==before,CODE,'original witness changed')
        result=dict(task=case['task'],partition=case['partition'],witness=shared,actions=actions)
        publish(destination/'result.json',result);results.append(result)
    require(source_memory(inputs(cohort.config))==source_memory(bound),CODE,'source memory changed')
    report=dict(schema='repository-depth-actions-v1',selection=proof['selection'],
        selection_readback=dict(path=str(proof_path),sha256=proof_pin),source_memory=source_memory(bound),
        source_memory_unchanged=True,results=results,code=digest(Path(__file__)),oracle_code=code,
        action_counts=dict(Counter(a['status'] for r in results for a in r['actions'])),
        wall_seconds=time.perf_counter()-start,binding_seconds=binding_seconds,
        reserved_evaluation_used=False,learned_skill=False,admitted=False,acceptance='author-checked')
    pin=publish(output/'report.json',report)
    print(json.dumps(dict(path=str(output/'report.json'),pin=pin,counts=report['action_counts'],
                         wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-readback',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.selection_readback,a.pin,a.output)

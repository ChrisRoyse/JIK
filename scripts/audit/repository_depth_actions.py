"""Read back every depth action, its isolated process and unchanged original oracle."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_depth import DepthCohort,inputs
from claims.repository_depth_actions import source_memory
from claims.repository_cohort_population import CohortPopulation
from oracle.repository_cohort_actions import digest,open_witness
from oracle.repository_cohort_readback import controls,read_action
from oracle.replay_swe import verify_prepared,recount
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE='MNLM_REPOSITORY_DEPTH_READBACK'


def run(path,pin,output):
    start=time.perf_counter();report=read_pinned(path,pin)
    scripts=Path(__file__).resolve().parents[1]
    require(report['schema']=='repository-depth-actions-v1' and
            report['code']==digest(scripts/'claims/repository_depth_actions.py'),CODE,'writer pin')
    code={p.name:digest(p) for p in (scripts/'oracle').glob('*.py')}
    require(report['oracle_code']==code,CODE,'original oracle code changed')
    cohort=DepthCohort(**report['selection']);bound=inputs(cohort.config)
    proof=read_pinned(Path(report['selection_readback']['path']),report['selection_readback']['sha256'])
    require(proof['selection']==report['selection'] and
            proof['code']==digest(scripts/'audit/repository_depth_selection.py') and
            proof['actions']==cohort.report['selected_actions'],CODE,'selection proof differs')
    require(source_memory(bound)==report['source_memory']==proof['source_memory'],CODE,'memory source changed')
    parent=read_pinned(Path(cohort.config['source_config']),cohort.config['source_config_pin'])
    old_ids=set();by_task=defaultdict(lambda:dict(prior=Counter(),added=Counter()))
    for item in parent['populations']:
        spec=item['native'];pop=CohortPopulation(spec['population'],spec['population_pin'])
        for entry in pop.observations.entries.values():
            require(entry['container'] not in old_ids,CODE,'duplicated prior process')
            old_ids.add(entry['container']);by_task[entry['task']]['prior'][entry['outcome']]+=1
    output.mkdir(parents=True,exist_ok=False);results=[];new_ids=set();control_target=None
    for case,item in zip(cohort.cases,report['results'],strict=True):
        require(case['task']==item['task'] and case['partition']==item['partition'] and
                [a['ordinal'] for a in item['actions']]==list(range(case['prior_attempts'],len(case['attempts']))),
                CODE,'task/action denominator differs')
        first=cohort.sources[case['task']]['item'];witness=read_pinned(Path(first['path']),first['sha256'])
        plan,prepared,prep_pin,spec,before,modes,files,shared=open_witness(bound['root'],first,witness,code)
        require(item['witness']==shared,CODE,'original witness differs')
        actions=[]
        for action in item['actions']:
            if action['status']=='gap':
                gap=read_pinned(Path(action['path']),action['sha256'])
                require(gap['task']==case['task'] and gap['attempt']==case['attempts'][action['ordinal']],
                        CODE,'gap binding')
                checked=dict(ordinal=action['ordinal'],status='gap',error=gap['error'])
                directory=Path(action['path']).parent
                execution=directory/'execution/execution.json'
                if execution.exists():
                    raw=json.loads(execution.read_bytes())
                    try:recount(raw,spec['required_tests'],allow_uncollectable=True)
                    except SenseError as exc:
                        require(exc.code==gap['error']['code'],CODE,'gap cause differs')
                    else:raise SenseError(CODE,'evaluable execution was presented as gap')
                    checked['retained_execution']=dict(sha256=digest(execution),
                        stdout=digest(directory/'execution/stdout.bin'),stderr=digest(directory/'execution/stderr.bin'))
                    require(raw['container_id'] not in old_ids|new_ids,CODE,'gap reuses process')
                    new_ids.add(raw['container_id'])
            else:
                checked=read_action(action,case,cohort,spec,prepared,shared)
                raw=json.loads((Path(action['path']).parent/'execution/execution.json').read_bytes())
                identifier=raw['container_id']
                require(identifier not in old_ids|new_ids,CODE,'reused process presented as new occurrence')
                new_ids.add(identifier)
            by_task[case['task']]['added'][checked.get('outcome','gap')]+=1;actions.append(checked)
        require(verify_prepared(prepared,prep_pin)[1]==before,CODE,'original witness changed during readback')
        if control_target is None:control_target=(prepared,spec,Path(witness['execution']))
        results.append(dict(task=case['task'],partition=case['partition'],actions=actions))
    checks=controls(cohort,*control_target,output,path)
    require(source_memory(inputs(cohort.config))==source_memory(bound),CODE,'memory changed during readback')
    require(dict(Counter(a['status'] for r in results for a in r['actions']))==report['action_counts'],
            CODE,'action status census differs')
    census={t:{s:{str(k):n for k,n in v.items()} for s,v in sides.items()} for t,sides in by_task.items()}
    value=dict(schema='repository-depth-actions-readback-v1',source=dict(path=str(path),sha256=pin),
        selection=report['selection'],results=results,checks=checks,census=census,
        processes=dict(prior=len(old_ids),new=len(new_ids),overlap=0),
        outcomes=dict(Counter(str(a.get('outcome','gap')) for r in results for a in r['actions'])),
        source_memory=source_memory(bound),source_memory_unchanged=True,original_oracles_unchanged=True,
        code=digest(Path(__file__)),wall_seconds=time.perf_counter()-start,
        learned_skill=False,admitted=False,acceptance='author-checked')
    result=publish(output/'report.json',value)
    print(json.dumps(dict(path=str(output/'report.json'),pin=result,processes=value['processes'],
        outcomes=value['outcomes'],controls=len(checks),census=census,wall_seconds=value['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actions',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.actions,a.pin,a.output)

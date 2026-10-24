"""Reconstruct the frozen depth roster from retained source rows, without labels."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_depth import DepthCohort,POLICY,inputs
from claims.process_conditions import write_committed
from senses.contract import SenseError,require
from senses.trajectory_population import open_bound

CODE='MNLM_REPOSITORY_DEPTH_SELECTION'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def reconstruct(bound):
    eligible={task:{} for task in bound['cases']};counts=Counter();rows=Counter()
    old={t:{a['sha256'] for a in c['attempts']} for t,c in bound['cases'].items()}
    for shard in bound['scope']['shards']:
        with open_bound(bound['root']/'raw/swe-agent-trajectories',shard) as stream:
            table=pq.read_table(stream,columns=['instance_id','generated_patch'])
        require(table.num_rows==shard['rows'],CODE,'source row count')
        for index,row in enumerate(table.to_pylist()):
            counts['rows']+=1;task=row['instance_id']
            if task not in eligible:continue
            counts['connected_task_rows']+=1;text=row['generated_patch']
            if not isinstance(text,str) or not text.strip():counts['missing_patch']+=1;continue
            raw=text.encode('utf-8');counts['connected_patch_bytes']+=len(raw)
            if len(raw)>POLICY['patch_limit']:counts['oversize_patch']+=1;continue
            pin=digest(raw)
            if pin in old[task]:counts['already_retained']+=1;continue
            rows[task]+=1
            binding=dict(shard,path='raw/swe-agent-trajectories/'+shard['path'],row=index)
            item=dict(sha256=pin,bytes=len(raw),raw=raw,binding=binding,
                      rank=digest((POLICY['seed']+'\0'+task+'\0'+pin).encode('utf-8')))
            previous=eligible[task].get(pin)
            if previous is None or (binding['path'],index)<(previous['binding']['path'],previous['binding']['row']):
                eligible[task][pin]=item
    cases,patches,coverage=[],{},{}
    for task,old_case in sorted(bound['cases'].items()):
        case=deepcopy(old_case);case['prior_attempts']=len(old_case['attempts'])
        ordered=sorted(eligible[task].values(),key=lambda v:(v['rank'],v['sha256']))
        selected=ordered[:POLICY['additional_per_task']]
        coverage[task]=dict(rows=rows[task],distinct=len(ordered),selected=len(selected))
        for item in selected:
            item=dict(item);raw=item.pop('raw');name='patches/'+item['sha256']+'.diff'
            patches[name]=raw;case['attempts'].append(dict(item,path=name))
        cases.append(case)
    return cases,patches,dict(counts),coverage


def match(actual,expected):
    require(actual==expected,CODE,'roster differs from independent full-sort reconstruction')


def run(root,pin,output):
    tick=time.perf_counter();cohort=DepthCohort(root,pin);bound=inputs(cohort.config)
    cases,patches,counts,coverage=reconstruct(bound);match(cohort.cases,cases)
    require(cohort.sources==bound['bindings'] and cohort.report['counts']==counts and
            cohort.report['coverage']==coverage and cohort.report['selected_actions']==len(
                [a for c in cases for a in c['attempts'][c['prior_attempts']:]]),CODE,'source/count binding')
    require(set(cohort.report['files'])==set(patches)|{'config.json','cases.json','sources.json'},
            CODE,'file coverage')
    for name,raw in patches.items():require((root/name).read_bytes()==raw,CODE,'source patch bytes')
    before=dict(head=bound['head'],journal=bound['journal'],snapshot=bound['snapshot'])
    require(cohort.report['source_memory']==before,CODE,'memory lineage differs')
    output.mkdir(parents=True,exist_ok=False);checks=[]

    def refuses(name,edit):
        changed=deepcopy(cases);edit(changed)
        write_committed(output/(name+'.json'),changed)
        try:match(changed,cases)
        except SenseError as exc:
            require(exc.code==CODE,CODE,'unexpected negative-control failure')
            checks.append(dict(name=name,code=exc.code))
        else:raise SenseError(CODE,'accepted invalid '+name)

    refuses('omitted-project',lambda c:c.pop())
    refuses('old-action-as-new',lambda c:c[0]['attempts'].__setitem__(-1,c[0]['attempts'][0]))
    refuses('reserved-project',lambda c:c[0].update(partition='reserved_eval'))
    refuses('changed-source-row',lambda c:c[0]['attempts'][-1]['binding'].update(row=0))
    refuses('changed-selection-rank',lambda c:c[0]['attempts'][-1].update(rank='0'*64))
    again=inputs(cohort.config)
    require(before==dict(head=again['head'],journal=again['journal'],snapshot=again['snapshot']),
            CODE,'source memory changed')
    report=dict(schema='repository-depth-selection-readback-v1',selection=dict(root=str(root),pin=pin),
        cases=len(cases),actions=cohort.report['selected_actions'],coverage=coverage,counts=counts,
        checks=checks,source_memory=before,source_memory_unchanged=True,new_outcomes_read=False,
        code=digest(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-tick,
        scientific_admission=False,serving=False,acceptance='author-checked')
    result=write_committed(output/'report.json',report)
    print(json.dumps(dict(path=str(output/'report.json'),pin=result,cases=len(cases),
                         actions=report['actions'],controls=len(checks),wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.selection,a.pin,a.output)

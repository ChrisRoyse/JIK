"""Outcome-blind, order-neutral additional actions on already connected tasks."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain
from claims.codata_source import check,sha
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from senses.trajectory_population import open_bound,read_pinned
from senses.contract import require

POLICY=dict(seed='repository-depth-v1:20261022',additional_per_task=4,patch_limit=1<<20,
            selection='lowest_seed_task_patch_hash',new_outcomes_read=False,
            scope='exposed_project_depth_not_independent_project_growth')


def read(path):return json.loads(path.read_bytes())


def inputs(config):
    parent=read_pinned(Path(config['source_config']),config['source_config_pin'])
    memory=Path(parent['source']);proof=read_pinned(Path(parent['proof']),parent['proof_pin'])
    check(sha((memory/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((memory/'snapshot-final.json').read_bytes())==proof['snapshot'],'DEPTH_MEMORY_PIN')
    _,events=chain(memory/'experience.sqlite3');snapshot=read(memory/'snapshot-final.json')
    check(len(events)==snapshot['events'] and events[-1][2]==snapshot['head'],'DEPTH_MEMORY_HEAD')
    saved={k:v for intake in snapshot['repository_cohort_intakes'].values()
           for k,v in intake['bindings'].items()}
    cases,bindings,connected={}, {},set()
    scope=None;root=None
    for p in parent['populations']:
        spec=p['native'];native=read_pinned(Path(p['proof']),p['proof_pin'])
        check(native['native']==spec,'DEPTH_NATIVE_PROOF')
        pop=CohortPopulation(spec['population'],spec['population_pin']);ob=pop.observations
        current_scope=ob.cohort.report['scope']
        check(scope is None or current_scope==scope,'DEPTH_SOURCE_SCOPE');scope=current_scope
        actions=read_pinned(ob.path,ob.pin)
        witness=read_pinned(Path(actions['witness']['path']),actions['witness']['sha256'])
        check(root is None or root==Path(witness['root']),'DEPTH_SOURCE_ROOT');root=Path(witness['root'])
        for key,e in ob.entries.items():
            check(key not in connected and key in saved and saved[key]['episode']==ob.episode(key)['id'] and
                  {s:saved[key][s] for s in ('input','echo')}==native['receipts'][key],
                  'DEPTH_CONNECTED_SOURCE')
            connected.add(key);task=e['task']
            if task in cases:continue
            case=next(c for c in ob.cohort.cases if c['task']==task)
            check(case['partition'] in ('learn_1','learn_2'),'DEPTH_RESERVED_TASK')
            item=next(v for v in witness['results'] if v['task']==task)
            check(read_pinned(Path(item['path']),item['sha256'])['status']=='observed','DEPTH_WITNESS')
            cases[task]=copy.deepcopy(case)
            bindings[task]=dict(cohort=dict(root=str(ob.cohort.root),pin=ob.cohort.pin),
                witness=actions['witness'],item=item,source_population=spec,
                source_observations=ob.binding)
    check(connected==set(saved) and cases,'DEPTH_COMPLETE_ROSTER')
    return dict(root=root,scope=scope,cases=cases,bindings=bindings,head=snapshot['head'],
                journal=proof['journal'],snapshot=proof['snapshot'],connected=sorted(connected))


def rank(task,pin):return sha((POLICY['seed']+'\0'+task+'\0'+pin).encode())


def offer(selected,item):
    pin=item['sha256']
    if pin in selected:
        a,b=selected[pin]['binding'],item['binding']
        if (b['path'],b['row'])<(a['path'],a['row']):selected[pin]=item
    else:selected[pin]=item
    if len(selected)>POLICY['additional_per_task']:
        del selected[max(selected,key=lambda p:(selected[p]['rank'],p))]


def collect(bound):
    chosen={t:{} for t in bound['cases']};seen={t:set() for t in chosen}
    old={t:{a['sha256'] for a in c['attempts']} for t,c in bound['cases'].items()}
    counts=Counter();new_rows=Counter()
    for shard in bound['scope']['shards']:
        with open_bound(bound['root']/'raw/swe-agent-trajectories',shard) as stream:
            p=pq.ParquetFile(stream);check(p.metadata.num_rows==shard['rows'],'DEPTH_SHARD_ROWS')
            offset=0
            for batch in p.iter_batches(batch_size=512,columns=['instance_id','generated_patch']):
                for local,row in enumerate(batch.to_pylist()):
                    counts['rows']+=1;task=row['instance_id']
                    if task not in chosen:continue
                    counts['connected_task_rows']+=1;text=row['generated_patch']
                    if not isinstance(text,str) or not text.strip():
                        counts['missing_patch']+=1;continue
                    raw=text.encode();counts['connected_patch_bytes']+=len(raw)
                    if len(raw)>POLICY['patch_limit']:
                        counts['oversize_patch']+=1;continue
                    pin=sha(raw)
                    if pin in old[task]:counts['already_retained']+=1;continue
                    new_rows[task]+=1;seen[task].add(pin)
                    item=dict(sha256=pin,bytes=len(raw),rank=rank(task,pin),raw=raw,
                        binding=dict(shard,path='raw/swe-agent-trajectories/'+shard['path'],row=offset+local))
                    offer(chosen[task],item)
                offset+=batch.num_rows
            check(offset==shard['rows'],'DEPTH_SCAN_COMPLETE')
    return chosen,dict(counts),{t:dict(rows=new_rows[t],distinct=len(seen[t]),selected=len(chosen[t]))
                               for t in chosen}


def build(config,output):
    start=time.perf_counter();check(not output.exists(),'DEPTH_OUTPUT_EXISTS')
    bound=inputs(config);bound_at=time.perf_counter()
    chosen,counts,coverage=collect(bound);selected_at=time.perf_counter()
    output.mkdir(parents=True);(output/'patches').mkdir()
    cases,files=[],{}
    for task in sorted(bound['cases']):
        case=copy.deepcopy(bound['cases'][task]);case['prior_attempts']=len(case['attempts'])
        for item in sorted(chosen[task].values(),key=lambda v:(v['rank'],v['sha256'])):
            item=copy.deepcopy(item);raw=item.pop('raw');name='patches/'+item['sha256']+'.diff'
            if name not in files:
                (output/name).write_bytes(raw);files[name]=dict(bytes=len(raw),sha256=sha(raw))
            case['attempts'].append(dict(item,path=name))
        cases.append(case)
    for name,value in (('config.json',config),('cases.json',cases),('sources.json',bound['bindings'])):
        pin=write_committed(output/name,value);files[name]=dict(bytes=(output/name).stat().st_size,sha256=pin)
    end=time.perf_counter()
    report=dict(schema='repository-depth-selection-v1',policy=POLICY,files=files,counts=counts,
        coverage=coverage,scope=bound['scope'],source_root=str(bound['root']),
        source_memory=dict(head=bound['head'],journal=bound['journal'],snapshot=bound['snapshot']),
        prior_actions=len(bound['connected']),projects=len(cases),selected_actions=sum(v['selected'] for v in coverage.values()),
        code=sha(Path(__file__).read_bytes()),new_outcomes_read=False,serving=False,scientific_admission=False,
        wall_seconds=end-start,phases=dict(binding=bound_at-start,selection=selected_at-bound_at,
        publication=end-selected_at),acceptance='author-checked')
    pin=write_committed(output/'report.json',report)
    write_committed(output/'COMMITTED.json',dict(sha256=pin))
    print(json.dumps(dict(pin=pin,counts=counts,coverage=coverage,selected=report['selected_actions'],
                         wall_seconds=report['wall_seconds'])),flush=True)


class DepthCohort:
    def __init__(self,root,pin):
        self.root=Path(root);self.pin=pin;self.report=read_pinned(self.root/'report.json',pin)
        r=self.report
        check(r['schema']=='repository-depth-selection-v1' and r['policy']==POLICY and
              r['code']==sha(Path(__file__).read_bytes()) and
              read(self.root/'COMMITTED.json')==dict(sha256=pin),'DEPTH_SELECTION_PIN')
        for n,v in r['files'].items():
            check(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n and
                  not (self.root/n).is_symlink() and (self.root/n).resolve().is_relative_to(self.root.resolve()),
                  'DEPTH_PATH')
            raw=(self.root/n).read_bytes();check(dict(bytes=len(raw),sha256=sha(raw))==v,'DEPTH_FILE_PIN')
        self.cases=read(self.root/'cases.json');self.sources=read(self.root/'sources.json')
        self.config=read(self.root/'config.json')
        check(len({c['task'] for c in self.cases})==len(self.cases)==r['projects'],'DEPTH_CASE_ROSTER')
        check(all(c['partition'] in ('learn_1','learn_2') for c in self.cases),'DEPTH_RESERVED_TASK')

    def training_cases(self,partition):
        require(partition in ('learn_1','learn_2'),'MNLM_CODATA_COHORT_TRAINING_PARTITION',
                'Only previously exposed training projects are in this depth increment.')
        return [c for c in self.cases if c['partition']==partition]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(read(a.config),a.output)

"""Source metadata and exact native-input aliases on the retained capability cohort."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.capability_kernel_learning import KernelLearningCorpus, pinned


def native_rows(corpus,key):
    task=corpus.tasks[key]
    return {s.name:np.concatenate([corpus.field(task['population'],f['id'],s.name)
        for f in task['selections']['context']['fields']]).astype('<f4',copy=False) for s in corpus.specs}


def signature(values,kind):
    lanes={}
    for name,a in values.items():
        lanes[name]=dict(shape=list(a.shape),sha256=sha(a.tobytes()) if kind=='ordered' else
                        digest(sorted(sha(row.tobytes()) for row in a)))
    return lanes


def same_input(corpus,left,right,kind):
    a,b=native_rows(corpus,left),native_rows(corpus,right)
    need(corpus.tasks[left]['kind']==corpus.tasks[right]['kind'],'CONTEXT_ALIAS_OPERATION')
    for name in a:
        need(a[name].shape==b[name].shape,'CONTEXT_ALIAS_SHAPE')
        if kind=='ordered': need(a[name].tobytes()==b[name].tobytes(),'CONTEXT_ALIAS_HASH_COLLISION')
        else: need(Counter(row.tobytes() for row in a[name])==Counter(row.tobytes() for row in b[name]),'CONTEXT_ALIAS_HASH_COLLISION')


def summarize(rows,groups):
    result={}
    for split in ('train','query','all'):
        result[split]={}
        for family in ('coding','conversation','reasoning'):
            selected=[r for r in rows if r['family']==family and (split=='all' or r['split']==split)]
            keys={r['task'] for r in selected}; cells={}
            for kind,partitions in groups.items():
                sets=[[r for r in selected if r['task'] in members] for members in partitions.values() if keys.intersection(members)]
                cells[kind]={target:dict(conflict_groups=sum(len({r[target] for r in group})>1 for group in sets),
                    affected_tasks=sum(len(group) for group in sets if len({r[target] for r in group})>1),
                    deterministic_empirical_max_correct=sum(max(Counter(r[target] for r in group).values()) for group in sets),
                    denominator=len(selected)) for target in ('target_sha256','region_label')}
            result[split][family]=dict(tasks=len(selected),split_groups=len({r['group'] for r in selected}),aliases=cells,
                role_counts=dict(Counter(f['role'] for r in selected for f in r['fields'])),
                unknown_clock_fields=sum(f['clock'] is None for r in selected for f in r['fields']),
                max_fields=max(len(r['fields']) for r in selected),max_native_windows=max(r['total_windows'] for r in selected))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,required=True); p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args(); need(not a.output.exists(),'CONTEXT_CENSUS_EXISTS'); a.output.mkdir()
    started=tick=time.perf_counter(); phases={}; diagnostic=pinned(dict(path=str(a.diagnostic),pin=a.pin))
    source=pinned(diagnostic['plan']['source']); config=pinned(source['source']['config']); corpus=KernelLearningCorpus(config)
    raw=(a.diagnostic.parent/'tasks.jsonl').read_bytes(); need(sha(raw)==diagnostic['tasks_sha256'],'CONTEXT_CENSUS_PARENT_ROWS')
    prior={r['task']:r for r in map(json.loads,raw.splitlines())}
    phases['authenticate']=time.perf_counter()-tick; tick=time.perf_counter()
    rows=[]; groups={k:defaultdict(list) for k in ('ordered','multiset')}
    for split,keys in (('train',corpus.train),('query',corpus.query)):
        for key in keys:
            task=corpus.tasks[key]; population=corpus.sources[task['population']].population
            values=native_rows(corpus,key); fields=[]
            for item in task['selections']['context']['fields']:
                f=population.fields[item['id']]
                fields.append({**{k:f[k] for k in ('id','role','clock','source')},
                    'windows':{s.name:len(corpus.field(task['population'],f['id'],s.name)) for s in corpus.specs}})
            signatures={kind:signature(values,kind) for kind in groups}
            row=dict(task=key,split=split,family=task['family'],group=task['split_group'],operation=task['kind'],fields=fields,
                source_provenance=population.tasks[key]['provenance'],target_sha256=prior[key]['source_sha256'],
                region_label=prior[key]['recognition']['label'],signatures=signatures,total_windows=sum(len(x) for x in values.values()))
            for kind in groups:
                identity=digest(dict(operation=task['kind'],lanes=signatures[kind]))
                if groups[kind][identity]: same_input(corpus,key,groups[kind][identity][0],kind)
                groups[kind][identity].append(key)
            rows.append(row)
    phases['census']=time.perf_counter()-tick; tick=time.perf_counter()
    selected=[]
    for family in ('coding','conversation','reasoning'):
        choices=[r for r in rows if r['split']=='query' and r['family']==family]
        for criterion in ('windows','fields'):
            key=min(choices,key=lambda r:(-(r['total_windows'] if criterion=='windows' else len(r['fields'])),r['task']))['task']
            if key not in selected:selected.append(key)
    with (a.output/'tasks.jsonl').open('x') as out:
        for row in rows:out.write(json.dumps(row,sort_keys=True)+'\n')
    result=dict(schema='capability-context-census-v1',source=dict(path=str(a.diagnostic),pin=a.pin),
        code_sha256=sha(Path(__file__).read_bytes()),tasks_sha256=sha((a.output/'tasks.jsonl').read_bytes()),
        families=summarize(rows,groups),groups=groups,selected_queries=selected,selection='max_native_windows_and_max_fields_then_lowest_task_hash',
        rows=len(rows),number_status='measured',acceptance='author-checked',scientific_admission=False,serving=False,
        alias_ceiling_scope='finite_empirical_deterministic_mapping_only_not_information_or_generalization',
        source_ids_are_not_model_features=True)
    phases['finalize']=time.perf_counter()-tick; wall=time.perf_counter()-started
    need(sum(phases.values())/wall>=.9,'CONTEXT_CENSUS_PHASES'); result.update(phases=phases,seconds=wall)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),selected=selected,families=result['families'])),flush=True)


if __name__=='__main__':main()

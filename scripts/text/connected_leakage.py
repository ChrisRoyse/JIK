"""Bounded reuse of the exact overlap engine, then distinct evaluation policies."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import numpy as np
from text.evaluation_overlap import Features,prefix_join,components,CONTRACT,canonical,file_sha
from text.connected_observations import identity
from text.connected_training_census import check,norm,sha
from senses.trajectory_population import publish

PAIR_VISIT_BOUND=2000000
SEED='connected-grounding-v1|25'


def overlap(catalog,output):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();features=Features();source=output/'source';source.mkdir();rows=[]
    for r in catalog:
        text=r['text'];row=dict(id=r['id'],domain='catalog',group_keys=[],fields=[dict(name='grouping-only',text=text,sha256=sha(text),characters=len(text),utf8_bytes=len(text.encode()))])
        features.add(row);rows.append(row)
    with (source/'catalog.jsonl').open('xb') as f:
        for r in rows:f.write(canonical(r)+b'\n')
    publish(source/'report.json',dict(schema='evaluation-source-capture-v1',complete=True,errors=[],sealed_evaluation=False,
        sources={'catalog':dict(file='catalog.jsonl',bytes=(source/'catalog.jsonl').stat().st_size,sha256=file_sha(source/'catalog.jsonl'),counts=dict(selected_documents=len(rows)))}))
    phases=dict(source_and_features=time.perf_counter()-start);tick=time.perf_counter()
    frequency=Counter(v for s in features.sets for v in s);buckets=Counter()
    for s in features.sets:
        prefix=sorted(s,key=lambda v:(frequency[v],v))[:len(s)-(3*len(s)+4)//5+1] if s else []
        buckets.update(prefix)
    visits=sum(n*(n-1)//2 for n in buckets.values())
    publish(output/'cost-precount.json',dict(unique_texts=len(rows),prefix_pair_visits=visits,bound=PAIR_VISIT_BOUND,within_bound=visits<=PAIR_VISIT_BOUND))
    check(visits<=PAIR_VISIT_BOUND,'PREFIX_PAIR_VISIT_BOUND')
    phases['prefix_precount']=time.perf_counter()-tick;tick=time.perf_counter()
    pairs,near,stats=prefix_join(features.sets);links,groups=components(features.documents,near)
    phases['existing_join_and_components']=time.perf_counter()-tick;tick=time.perf_counter();result=output/'overlap';result.mkdir();features.save(result)
    np.save(result/'candidates.npy',np.array(pairs,dtype=np.uint32).reshape(-1,2))
    for name,values in (('links.jsonl',links),('groups.jsonl',groups)):
        with (result/name).open('xb') as f:
            for v in values:f.write(canonical(v)+b'\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in result.iterdir() if p.is_file()}
    counts=dict(features.counts,unique_words=len(features.words),unique_shingles=len(features.shingles),**stats,links=len(links),components=len(groups),
                grouped_documents=sum(len(g['members']) for g in groups if len(g['members'])>1),largest_component=max(map(lambda g:len(g['members']),groups)))
    phases['persist']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema=CONTRACT,complete=True,errors=[],source_report_sha256=file_sha(source/'report.json'),files=files,counts=counts,
        threshold=dict(numerator=3,denominator=5),shingle_words=5,sealed_evaluation=False,scientific_admission=False,served=False,observations_removed=0,
        cost_precount=visits,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(result/'report.json',report)
    return groups,dict(**counts,prefix_pair_visits=visits,wall_seconds=wall)


def partition_groups(nodes,edges,policy):
    parent={n:n for n in nodes}
    def root(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    for a,b,kind in edges:
        check(a in parent and b in parent,'SPLIT_NODE');x,y=sorted((root(a),root(b)));parent[y]=x
    groups=defaultdict(list)
    for n in nodes:groups[root(n)].append(n)
    result=[]
    for members in groups.values():
        members=sorted(members);gid=identity(['connected-component-v1',policy,members]);bucket=int(sha(SEED+'|'+policy+'|'+gid)[:8],16)%100
        result.append(dict(id=gid,members=members,partition='train' if bucket<70 else 'calibration' if bucket<85 else 'evaluation'))
    largest=min(result,key=lambda r:(-len(r['members']),r['id']));largest['partition']='train'
    return sorted(result,key=lambda r:r['id']),largest['id']


def splits(data,catalogs,overlaps):
    edges={k:[] for k in ('multirc','qasc-fact-disjoint','qasc-composition')}
    def link_members(members,policies,kind):
        ms=sorted(set(members))
        for key in policies:
            for other in ms[1:]:edges[key].append([ms[0],other,kind])
    for cohort,groups in overlaps.items():
        policies=['multirc'] if cohort=='multirc' else ['qasc-fact-disjoint','qasc-composition'] if cohort=='qasc-query-target' else ['qasc-fact-disjoint']
        for group in groups:
            owners=[owner for i in group['members'] for owner in catalogs[cohort][i]['owners']]
            link_members(owners,policies,cohort+'-exact-or-near')
    aliases=defaultdict(list)
    for d in data['documents']:aliases[d['source_alias']['key']].append(d['id'])
    for owners in aliases.values():link_members(owners,['multirc'],'source-identifier-alias')
    ref={r['id']:r for r in data['references']};pairs=defaultdict(list);qnodes=[]
    for c in data['cases']:
        if c['family']!='qasc':continue
        qnodes.append(c['id']);key=tuple(sorted(norm(data['units'][ref[r]['unit']]['text']) for r in c['facts']));pairs[key].append(c['id'])
    for owners in pairs.values():link_members(owners,['qasc-fact-disjoint','qasc-composition'],'unordered-fact-pair')
    policies={};rows=[]
    for policy in edges:
        nodes=[d['id'] for d in data['documents']] if policy=='multirc' else qnodes
        links=sorted({tuple(v) for v in edges[policy]});groups,forced=partition_groups(nodes,links,policy)
        membership={m:g for g in groups for m in g['members']};counts=Counter();eligible=Counter();targets={t['case']:t for t in data['targets']}
        for c in data['cases']:
            if (c['family']=='multirc')!=(policy=='multirc'):continue
            member=c['document'] if policy=='multirc' else c['id'];g=membership[member];quarantine=bool(targets[c['id']]['quarantine_reasons'])
            rows.append(dict(policy=policy,case=c['id'],group=g['id'],partition=g['partition'],eligible=not quarantine))
            counts[g['partition']]+=1;eligible[g['partition']]+=not quarantine
        policies[policy]=dict(groups=groups,links=[list(v) for v in links],largest_forced_train=forced,cases=dict(counts),eligible=dict(eligible),
            groups_by_partition=dict(Counter(g['partition'] for g in groups)),largest_component=max(len(g['members']) for g in groups),
            split_support_present=all(eligible[s]>0 for s in ('train','calibration','evaluation')))
    return rows,policies

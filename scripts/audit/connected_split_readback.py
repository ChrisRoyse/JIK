"""Rebuild owner closure by graph traversal and independently audit split exposure."""
from collections import Counter,defaultdict
import json
import unicodedata
import numpy as np
from audit.connected_population_readback import check,sha,identity,rows


def normalize(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def read_splits(root,data):
    tables={k:{} for k in ('multirc','qasc-query-target','qasc-fact')};ref={r['id']:r for r in data['references']};units=data['units']
    def text(rid):return units[ref[rid]['unit']]['text']
    def add(cohort,value,owner,role):
        s=normalize(value);key=sha(s);entry=tables[cohort].setdefault(key,dict(text=s,owners=set(),roles=set()));check(entry['text']==s,'HASH_COLLISION')
        entry['owners'].add(owner);entry['roles'].add(role)
    for r in data['references']:
        if r['family']=='multirc' and r['role']!='option':add('multirc',text(r['id']),r['owner'],r['role'])
    for c,t in zip(data['cases'],data['targets']):
        if c['family']=='qasc':
            add('qasc-query-target',text(c['question']),c['id'],'question');add('qasc-query-target',t['combined_statement'],c['id'],'private-target')
            for rid in c['facts']:add('qasc-fact',text(rid),c['id'],'fact')
    edges={k:set() for k in ('multirc','qasc-fact-disjoint','qasc-composition')};precounts={};catalog_counts={}
    def join(owners,policies,reason):
        members=sorted(set(owners))
        for policy in policies:
            for n in members[1:]:edges[policy].add((members[0],n,reason))
    for cohort,table in tables.items():
        base=root/'cohorts'/cohort;stored=json.loads((base/'owners.json').read_bytes())
        expected=[dict(id=k,text=v['text'],owners=sorted(v['owners']),roles=sorted(v['roles'])) for k,v in table.items()]
        check(stored==expected,'CATALOG_OWNERS');source=rows(base/'source/catalog.jsonl');check(len(source)==len(stored),'CATALOG_SOURCE_COUNT')
        for r,v in zip(source,stored):
            s=v['text'];check(r==dict(id=v['id'],domain='catalog',group_keys=[],fields=[dict(name='grouping-only',text=s,sha256=sha(s),characters=len(s),utf8_bytes=len(s.encode()))]),'CATALOG_SOURCE')
        ptr=np.load(base/'overlap/indptr.npy',allow_pickle=False);flat=np.load(base/'overlap/members.npy',allow_pickle=False)
        frequency=Counter(map(int,flat));bucket=Counter()
        for i in range(len(ptr)-1):
            values=list(map(int,flat[ptr[i]:ptr[i+1]]));keep=len(values)-(3*len(values)+4)//5+1 if values else 0
            bucket.update(sorted(values,key=lambda v:(frequency[v],v))[:keep])
        visits=sum(v*(v-1)//2 for v in bucket.values());precounts[cohort]=visits
        check(json.loads((base/'cost-precount.json').read_bytes())==dict(unique_texts=len(stored),prefix_pair_visits=visits,bound=2000000,within_bound=True) and visits<=2000000,'PREFIX_BOUND')
        over=json.loads((base/'overlap/report.json').read_bytes());catalog_counts[cohort]={**over['counts'],'prefix_pair_visits':visits,'wall_seconds':over['wall_seconds']}
        check(over['cost_precount']==visits and .9<=over['phase_coverage']<=1 and abs(sum(over['phases_seconds'].values())/over['wall_seconds']-over['phase_coverage'])<1e-12,'COHORT_COST')
        policies=['multirc'] if cohort=='multirc' else ['qasc-fact-disjoint','qasc-composition'] if cohort=='qasc-query-target' else ['qasc-fact-disjoint']
        for g in rows(base/'overlap/groups.jsonl'):join([o for i in g['members'] for o in stored[i]['owners']],policies,cohort+'-exact-or-near')
    source_groups=defaultdict(list)
    for d in data['documents']:source_groups[d['source_alias']['key']].append(d['id'])
    for owners in source_groups.values():join(owners,['multirc'],'source-identifier-alias')
    pair_groups=defaultdict(list)
    for c in data['cases']:
        if c['family']=='qasc':
            pair=tuple(sorted(normalize(text(r)) for r in c['facts']))
            pair_groups[pair].append(c['id'])
    for owners in pair_groups.values():join(owners,['qasc-fact-disjoint','qasc-composition'],'unordered-fact-pair')
    expected_policies={};expected_rows=[];exposures={};targets={t['case']:t for t in data['targets']}
    for policy,links in edges.items():
        nodes=[d['id'] for d in data['documents']] if policy=='multirc' else [c['id'] for c in data['cases'] if c['family']=='qasc']
        adjacency={n:set() for n in nodes}
        for a,b,_ in links:adjacency[a].add(b);adjacency[b].add(a)
        remaining=set(nodes);groups=[]
        while remaining:
            seed=min(remaining);remaining.remove(seed);stack=[seed];members=[]
            while stack:
                n=stack.pop();members.append(n)
                for other in adjacency[n]:
                    if other in remaining:remaining.remove(other);stack.append(other)
            members.sort();gid=identity(['connected-component-v1',policy,members]);v=int(sha('connected-grounding-v1|25|'+policy+'|'+gid)[:8],16)%100
            groups.append(dict(id=gid,members=members,partition='train' if v<70 else 'calibration' if v<85 else 'evaluation'))
        forced=min(groups,key=lambda g:(-len(g['members']),g['id']));forced['partition']='train';groups.sort(key=lambda g:g['id'])
        lookup={n:g for g in groups for n in g['members']};counts=Counter();eligible=Counter();selection=[]
        for c in data['cases']:
            if (c['family']=='multirc')!=(policy=='multirc'):continue
            member=c['document'] if policy=='multirc' else c['id'];g=lookup[member];ok=not targets[c['id']]['quarantine_reasons']
            expected_rows.append(dict(policy=policy,case=c['id'],group=g['id'],partition=g['partition'],eligible=bool(ok)));counts[g['partition']]+=1;eligible[g['partition']]+=bool(ok)
            selection.append((c,g['partition']))
        expected_policies[policy]=dict(groups=groups,links=[list(v) for v in sorted(links)],largest_forced_train=forced['id'],cases=dict(counts),eligible=dict(eligible),
             groups_by_partition=dict(Counter(g['partition'] for g in groups)),largest_component=max(len(g['members']) for g in groups),split_support_present=all(eligible[s]>0 for s in ('train','calibration','evaluation')))
        if policy.startswith('qasc'):
            trainfacts={normalize(text(r)) for c,s in selection if s=='train' for r in c['facts']}
            trainpairs={tuple(sorted(normalize(text(r)) for r in c['facts'])) for c,s in selection if s=='train'}
            ex={}
            for split in ('calibration','evaluation'):
                cs=[c for c,s in selection if s==split];hist=Counter(sum(normalize(text(r)) in trainfacts for r in c['facts']) for c in cs)
                repeated=sum(tuple(sorted(normalize(text(r)) for r in c['facts'])) in trainpairs for c in cs);check(repeated==0,'PAIR_SPLIT_LEAK')
                if policy=='qasc-fact-disjoint':check(hist[1]==hist[2]==0,'FACT_SPLIT_LEAK')
                ex[split]=dict(cases=len(cs),train_facts_seen_histogram={str(k):v for k,v in sorted(hist.items())},pairs_seen_in_train=repeated)
            exposures[policy]=ex
    check(json.loads((root/'partitions.json').read_bytes())==expected_policies,'PARTITIONS');check(rows(root/'assignments.jsonl')==expected_rows,'ASSIGNMENTS')
    counts=dict(public_unique_texts=len(units),public_unique_characters=sum(len(u['text']) for u in units),references=len(data['references']),missing_surfaces=sum(r['unit'] is None for r in data['references']),
        roles=dict(Counter(r['family']+':'+r['role'] for r in data['references'])),cases=dict(Counter(c['family'] for c in data['cases'])),
        structural_edges=sum(len(d['edges']) for d in data['documents']),source_alias_methods=dict(Counter(d['source_alias']['method'] for d in data['documents'])),
        private_targets=len(data['targets']),quarantined_cases=sum(bool(t['quarantine_reasons']) for t in data['targets']),overlap=catalog_counts,
        partitions={k:{n:v for n,v in p.items() if n not in ('links','groups')} for k,p in expected_policies.items()})
    report=json.loads((root/'report.json').read_bytes());check(report['counts']==counts==json.loads((root/'counts.json').read_bytes()),'COUNTS')
    check(report['split_seed']=='connected-grounding-v1|25' and report['prefix_pair_visit_bound']==2000000,'SPLIT_CONTRACT')
    return dict(policies=counts['partitions'],exposure=exposures,prefix_pair_visits=precounts,declared_lexical_rule_sealed=True,world_entity_source_time_sealed=False)

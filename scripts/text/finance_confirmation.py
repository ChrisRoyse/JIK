"""One fixed new-entity confirmation population with exact lexical separation (#12)."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canon.common import iter_canon_docs
from text.chunk_spans import chunk_spans, CONTRACT
from text.evaluation_overlap import Features, components, NUM, DEN
from audit.fixed_anchor_census import sha, check, write


def overlap(sets, fresh):
    """Complete prefix join, omitting irrelevant prior-prior comparisons."""
    frequency=Counter(x for s in sets for x in s)
    prefixes=[sorted(s,key=lambda x:(frequency[x],x))[:len(s)-(NUM*len(s)+DEN-1)//DEN+1] if s else [] for s in sets]
    indices=[defaultdict(deque),defaultdict(deque)]; links=[]; counts=Counter()
    for i in sorted(range(len(sets)),key=lambda i:(len(sets[i]),i)):
        if not sets[i]:continue
        minimum=(NUM*len(sets[i])+DEN-1)//DEN; found=set()
        for token in prefixes[i]:
            for kind in ((0,1) if fresh[i] else (1,)):
                posting=indices[kind][token]
                while posting and len(sets[posting[0]])<minimum:posting.popleft()
                counts['posting_visits']+=len(posting);found.update(posting)
            indices[int(fresh[i])][token].append(i);counts['prefix_entries']+=1
        for j in sorted(found):
            counts['candidate_pairs']+=1
            intersection=len(sets[i]&sets[j]);union=len(sets[i])+len(sets[j])-intersection
            if DEN*intersection>=NUM*union:
                a,b=sorted((i,j));links.append(dict(left=a,right=b,kind='near_exact_jaccard',intersection=intersection,union=union))
    return sorted(links,key=lambda r:(r['left'],r['right'])),dict(counts)


def run(a):
    start=time.perf_counter();check(not a.output.exists(),'CONFIRM_EXISTS')
    census=json.loads((a.census/'report.json').read_bytes());manifest=json.loads((a.census/'manifest.json').read_bytes())
    check(sha(a.census/'report.json')==manifest['report_sha256'] and census['complete'],'CONFIRM_CENSUS')
    config=json.loads((a.census/'contract.json').read_bytes());check(sha(a.census/'contract.json')==census['config_sha256'],'CONFIRM_CONTRACT')
    check(sha(a.calibration/'report.json')=='9c5034106722fbc73dc9bc38add30010b0af536fba80109f0ddf909d774429ae','CONFIRM_CALIBRATION')
    summary=json.loads((a.calibration/'finance/summary.json').read_bytes())
    sources={str(a.calibration/'finance/rows.jsonl'):summary['rows_sha256']}
    sources.update({str(a.records/'records'/f/'rows.jsonl'):v['rows_sha256'] for f,v in config['families'].items()})
    prior={};used_text=set()
    for filename,pin in sources.items():
        path=Path(filename);check(sha(path)==pin,'CONFIRM_PRIOR_PIN')
        for line in path.read_bytes().splitlines():
            row=json.loads(line);entity=row['entity'];text=row['text']
            check(entity not in prior or prior[entity]['text']==text,'CONFIRM_PRIOR_ENTITY')
            prior[entity]=dict(entity=entity,text=text,kind='prior')
            used_text.add(hashlib.sha256(text.encode()).hexdigest())
    check(sha(a.census/'private/eligible.jsonl')==census['eligible_sha256'],'CONFIRM_ORACLE')
    classes=config['families']['finance']['questions']['P159'];candidates={}
    for line in (a.census/'private/eligible.jsonl').read_bytes().splitlines():
        row=json.loads(line);values=row['values'].get('P159',[])
        if row['family']=='finance' and row['entity'] not in prior and row['text_source']['sha256'] not in used_text and len(values)==1 and values[0] in classes:
            candidates[row['entity']]=row
    check(len(prior)==42043 and len(candidates)==2312,'CONFIRM_PREFLIGHT_N')
    a.output.mkdir(parents=True)
    write(a.output/'intent.json',dict(prior_sources=sources,census_report_sha256=manifest['report_sha256'],
        classes=classes,pre_overlap_counts=[sum(r['values']['P159']==[c] for r in candidates.values()) for c in classes],
        selection='text-hash-first fresh component representative, then hash-ranked balanced max 600; minimum 384',
        lexical_overlap='exact lowercase word-5-gram Jaccard >=3/5; components touching prior data excluded',
        scientific_admission=False,served=False))
    check(sha(a.canon/'manifest.jsonl')==census['manifest_sha256'],'CONFIRM_CANON')
    declaration=census['source_manifest']['wikidata5m_text'];texts={};scanned=0
    for record in iter_canon_docs(a.canon,'wikidata5m_text',declaration['out_blake3']):
        scanned+=1;entity=record['f'].get('entity_id')
        if entity not in candidates:continue
        source=candidates[entity]['text_source'];body=record['text'].encode()
        check(entity not in texts and record['row']==source['row'] and hashlib.sha256(body).hexdigest()==source['sha256'] and len(body)==source['utf8_bytes'],'CONFIRM_TEXT')
        texts[entity]=record['text']
    check(scanned==declaration['docs'] and set(texts)==set(candidates),'CONFIRM_COMPLETE_TEXT')
    records=list(prior.values())+[dict(entity=e,text=texts[e],kind='fresh') for e in sorted(candidates)]
    features=Features();fresh=[]
    for row in records:
        text=row['text'];pin=hashlib.sha256(text.encode()).hexdigest()
        features.add(dict(id=row['entity'],domain=row['kind'],group_keys=[row['entity']],fields=[dict(
            name='prose',text=text,sha256=pin,characters=len(text),utf8_bytes=len(text.encode()))]))
        fresh.append(row['kind']=='fresh')
    tick=time.perf_counter();links,stats=overlap(features.sets,fresh)
    complete_links,groups=components(features.documents,links)
    eligible=[];excluded=0
    for group in groups:
        members=group['members'];new=[i for i in members if fresh[i]]
        if not new:continue
        if len(new)!=len(members):excluded+=len(new);continue
        chosen=min(new,key=lambda i:(hashlib.sha256(records[i]['text'].encode()).hexdigest(),records[i]['entity']))
        if features.sets[chosen]:eligible.append(records[chosen]['entity'])
    overlap_seconds=time.perf_counter()-tick
    pools={c:sorted((e for e in eligible if candidates[e]['values']['P159']==[c]),
                   key=lambda e:hashlib.sha256(('finance-confirm-v1|'+candidates[e]['text_source']['sha256']).encode()).hexdigest()) for c in classes}
    quota=min(100,min(map(len,pools.values())));check(quota*6>=384,'CONFIRM_SUPPORT')
    selected=[(e,label) for label,c in enumerate(classes) for e in pools[c][:quota]]
    dest=a.output/'finance';(dest/'chunks').mkdir(parents=True)
    documents=[];chunks=[]
    for i,(entity,label) in enumerate(selected):
        source=candidates[entity]['text_source'];text=texts[entity]
        documents.append(dict(row=i,entity=entity,text=text,text_source=source,partition='confirmation'))
        for j,span in enumerate(chunk_spans(text)):
            chunks.append(dict(**span,row=i,entity=entity,chunk=j,source_sha256=source['sha256']))
    with (dest/'rows.jsonl').open('x') as f:
        for row in documents:f.write(json.dumps(row,sort_keys=True)+'\n')
    write(dest/'anchors.json',dict(P159=dict(classes=classes,rows=list(range(len(selected))),labels=[v for e,v in selected],partitions=['confirmation']*len(selected))))
    pq.write_table(pa.Table.from_pylist(chunks),dest/'chunks/atoms.parquet',compression='zstd')
    proof=a.output/'overlap';proof.mkdir();features.save(proof)
    with (proof/'links.jsonl').open('x') as f:
        for row in complete_links:f.write(json.dumps(row,sort_keys=True)+'\n')
    write(proof/'components.json',groups)
    # A real prior document duplicated as a candidate must be excluded by the join.
    witness=next(s for s in features.sets if s)
    control,_=overlap([witness,witness],[False,True]);check(len(control)==1,'CONFIRM_DUPLICATE_CONTROL')
    summary=dict(schema='fixed-finance-confirmation-v1',family='finance',entities=len(documents),chunks=len(chunks),
        rows_sha256=sha(dest/'rows.jsonl'),anchors_sha256=sha(dest/'anchors.json'),chunks_sha256=sha(dest/'chunks/atoms.parquet'),
        chunker=CONTRACT,calibration_only=True,purpose='fixed_calibration_replication',scientific_admission=False,served=False)
    write(dest/'summary.json',summary)
    # Readback independent of construction: complete source tiling, labels, and no component overlap.
    restored=pq.read_table(dest/'chunks/atoms.parquet').to_pylist();assembled=defaultdict(list)
    for row in restored:assembled[row['entity']].append(row['text'])
    check(set(assembled)=={e for e,v in selected} and all(''.join(v)==texts[e] for e,v in assembled.items()),'CONFIRM_TILING')
    report=dict(complete=True,errors=[],calibration_only=True,purpose='fixed_calibration_replication',scientific_admission=False,served=False,intent_sha256=sha(a.output/'intent.json'),
        families=dict(finance=dict(entities=len(documents),chunks=len(chunks),ready=['P159'],unavailable=[],summary_sha256=sha(dest/'summary.json'))),
        census_report_sha256=manifest['report_sha256'],source_rows=scanned,prior_entities=len(prior),candidate_entities=len(candidates),
        source_new=True,lexical_component_separation=True,pretraining_independence=False,
        retained_class_support=[len(pools[c]) for c in classes],excluded_prior_component_candidates=excluded,
        overlap=dict(stats,seconds=overlap_seconds,features=dict(features.counts),links=len(complete_links),components=len(groups)),
        files={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
        wall_seconds=time.perf_counter()-start,code_sha256=sha(Path(__file__)))
    write(a.output/'report.json',report);write(a.output/'manifest.json',dict(report_sha256=sha(a.output/'report.json')))
    print(json.dumps({k:report[k] for k in ('families','retained_class_support','excluded_prior_component_candidates','overlap','wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('census','calibration','records','canon','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

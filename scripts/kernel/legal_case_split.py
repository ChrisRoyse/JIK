"""Freeze case/near-duplicate connected components before legal predictor fitting."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical,file_sha,need
from text.constellation_input import stamp
from kernel.legal_exact_near import load_rank,pair_map
from text.legal_shingles_input import Shingles

SIDES=('train','development','query')


def pinned(root,pin):
    need(file_sha(root/'report.json')==pin,'LEGAL_SPLIT_SOURCE_PIN')
    r=json.loads((root/'report.json').read_bytes());need(r['complete'],'LEGAL_SPLIT_SOURCE_INCOMPLETE')
    for name,ref in r.get('files',{}).items():
        expected=ref if isinstance(ref,str) else ref['sha256']
        need(file_sha(root/name)==expected,'LEGAL_SPLIT_FILE_PIN')
    return r


def strict_dictionary(values):
    """Independent strict lexicographic order check, tiled; implies no duplicate IDs."""
    for start in range(1,len(values),1<<19):
        end=min(len(values),start+(1<<19))
        left=values[start-1:end-1].view('>u4').reshape(-1,5)
        right=values[start:end].view('>u4').reshape(-1,5)
        equal=np.ones(len(left),dtype=bool);less=np.zeros(len(left),dtype=bool)
        for k in range(5):
            less|=equal&(left[:,k]<right[:,k]);equal&=left[:,k]==right[:,k]
        need(bool(less.all()),'LEGAL_SPLIT_DICTIONARY_NOT_UNIQUE')


def check_sides(sides,n,links):
    need(sides.shape==(n,) and sides.dtype==np.dtype('u1') and bool((sides<3).all()),'LEGAL_SPLIT_MEMBERSHIP')
    need(all(sides[a]==sides[b] for a,b in links),'LEGAL_SPLIT_RELATED_CASE_LEAK')


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time()
    a.output.mkdir(parents=True,exist_ok=False)
    near=pinned(a.near,a.near_pin);rankroot=Path(near['source']['path'])
    rank,arrays,stamps=load_rank(rankroot,near['source']['pin'])
    bank=Shingles(Path(rank['source']['path']),rank['source']['pin'])
    ancestry=pinned(a.ancestry,a.ancestry_pin)
    need(ancestry['source']['pin']==bank.report['source']['pin'],'LEGAL_SPLIT_ANCESTRY_SOURCE')
    corpus=pinned(a.corpus,a.corpus_pin)
    n=len(bank.documents);lookup={d['record']:i for i,d in enumerate(bank.documents)}
    need(len(lookup)==n,'LEGAL_SPLIT_DUPLICATE_RECORD')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    strict_dictionary(arrays['dictionary.npy'])
    phases['strict_exact_dictionary']=time.perf_counter()-tick;tick=time.perf_counter()
    original=json.loads((a.ancestry/'case-components.json').read_bytes())['components']
    need(len(original)==ancestry['components'] and sorted(k for group in original for k in group)==sorted(lookup),
         'LEGAL_SPLIT_ANCESTRY_COVERAGE')
    links=[]
    for group in original:
        ids=[lookup[k] for k in group];links.extend((ids[0],i) for i in ids[1:])
    alias_links=len(links)
    pairs=pair_map(a.near/'pairs.u32',n)
    need(len(pairs)==near['counts']['edges'],'LEGAL_SPLIT_NEAR_COUNT')
    links.extend(pairs)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for u,v in links:
        u,v=find(u),find(v)
        if u!=v:parent[max(u,v)]=min(u,v)
    grouped=defaultdict(list)
    for i in range(n):grouped[find(i)].append(i)
    components=[]
    for ids in grouped.values():
        records=sorted(bank.documents[i]['record'] for i in ids)
        key=hashlib.sha256(canonical(records)).hexdigest()
        draw=int(hashlib.sha256(('legal-source-growth-20261022/split/'+key).encode()).hexdigest(),16)%10
        side=0 if draw<8 else (1 if draw==8 else 2)
        components.append(dict(id=key,indices=ids,side=SIDES[side]))
    components.sort(key=lambda c:c['id'])
    sides=np.empty(n,dtype='u1');labels=np.empty(n,dtype='<u4')
    for label,c in enumerate(components):
        sides[c['indices']]=SIDES.index(c['side']);labels[c['indices']]=label
    check_sides(sides,n,links)
    phases['case_and_near_closure']=time.perf_counter()-tick;tick=time.perf_counter()
    atoms=int(corpus['counts']['atoms']);atom_sides=np.empty(atoms,dtype='u1')
    last=0;support={side:dict(documents=0,atoms=0,successors=0,components=0) for side in SIDES}
    for c in components:support[c['side']]['components']+=1
    with (a.corpus/'documents.jsonl').open('rb') as stream:
        for i,line in enumerate(stream):
            d=json.loads(line)
            need(i<n and d['source_record']==bank.documents[i]['record']
                 and d['atom_start']==last and d['atom_end']>last,'LEGAL_SPLIT_ATOM_CONTAINMENT')
            lo,hi=d['atom_start'],d['atom_end'];side=SIDES[int(sides[i])]
            atom_sides[lo:hi]=sides[i];last=hi
            support[side]['documents']+=1;support[side]['atoms']+=hi-lo;support[side]['successors']+=hi-lo-1
    need(i+1==n and last==atoms and all(support[s]['components'] for s in SIDES),'LEGAL_SPLIT_SUPPORT')
    phases['all_atom_containment']=time.perf_counter()-tick;tick=time.perf_counter()
    controls={}
    def rejected(name,fn):
        try:fn()
        except (ValueError,RuntimeError) as e:controls[name]=str(e);return
        raise AssertionError('LEGAL_SPLIT_NEGATIVE_PASSED')
    wrong=sides.copy();u,v=links[0];wrong[v]=(int(wrong[u])+1)%3
    rejected('related_case_crosses_split',lambda:check_sides(wrong,n,links))
    rejected('missing_document',lambda:check_sides(sides[:-1],n,links))
    repeated=np.concatenate([arrays['dictionary.npy'][:2],arrays['dictionary.npy'][1:2]])
    rejected('duplicate_dictionary_identity',lambda:strict_dictionary(repeated))
    bank.fresh()
    need(all(stamp(p)==value for p,value in stamps.items()),'LEGAL_SPLIT_RANK_DRIFT')
    for name,value in [('document-side.npy',sides),('component.npy',labels),('atom-side.npy',atom_sides)]:
        with (a.output/name).open('xb') as stream:np.save(stream,value,allow_pickle=False)
    write_committed(a.output/'components.json',components)
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(a.output.iterdir()) if p.is_file()}
    phases['controls_and_persistence']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'LEGAL_SPLIT_TIMING')
    result=dict(schema='legal-case-near-split-v1',complete=True,
                inputs={name:dict(path=str(getattr(a,name)),pin=getattr(a,name+'_pin')) for name in ('near','ancestry','corpus')},
                counts=dict(documents=n,atoms=atoms,original_components=len(original),components=len(components),
                            alias_links=alias_links,near_edges=len(pairs),maximum_component=max(map(len,grouped.values()))),
                support=support,files=files,controls=controls,all_document_and_atom_addresses_retained=True,
                dictionary_strictly_unique=True,declared_related_components_disjoint=True,
                split_policy='sha256(component_records) then salted hash mod10: train0..7 development8 query9',
                exposure='development_exposed_current_fit_case_disjoint',untouched_test=False,
                prior_mnlm_calyx_exposure='retained; not reset',encoder_pretraining_exposure='unknown',
                shorter_than_20_words='retained under exact/alias grouping; excluded from declared shingle relation',
                semantic_closure_complete=False,lineage_complete_beyond_retained_metadata=False,
                training_ready=False,model_updates=0,scientific_admission=False,
                acceptance='author-checked',number_status='measured',phases_seconds=phases,seconds=seconds,
                cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                code_sha256=file_sha(Path(__file__)))
    pin=write_committed(a.output/'report.json',result);write_committed(a.output/'COMMITTED',dict(report_sha256=pin))
    print(json.dumps(dict(pin=pin,counts=result['counts'],support=support,controls=controls,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('near','ancestry','corpus'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

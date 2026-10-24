"""Independent sparse-component and every-atom readback of the legal split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need


def enc(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()


def checked(root,pin):
    need(file_sha(root/'report.json')==pin,'SPLIT_COLD_PIN')
    r=json.loads((root/'report.json').read_bytes());need(r['complete'],'SPLIT_COLD_INCOMPLETE')
    for name,value in r.get('files',{}).items():
        expected=value if isinstance(value,str) else value['sha256']
        need(file_sha(root/name)==expected,'SPLIT_COLD_FILE')
    return r


def same_memberships(actual,expected):
    need(actual.dtype==expected.dtype and actual.shape==expected.shape and np.array_equal(actual,expected),
         'SPLIT_COLD_MEMBERSHIP')


def run(root,pin,output):
    start=tick=time.perf_counter();cpu=time.process_time();output.mkdir(parents=True,exist_ok=False)
    report=checked(root,pin)
    need(report['schema']=='legal-case-near-split-v1' and json.loads((root/'COMMITTED').read_bytes())==
         dict(report_sha256=pin),'SPLIT_COLD_COMMIT')
    inputs=report['inputs'];nearroot=Path(inputs['near']['path'])
    near=checked(nearroot,inputs['near']['pin'])
    ranks=checked(Path(near['source']['path']),near['source']['pin'])
    shingleroot=Path(ranks['source']['path']);shingles=checked(shingleroot,ranks['source']['pin'])
    ancestryroot=Path(inputs['ancestry']['path']);ancestry=checked(ancestryroot,inputs['ancestry']['pin'])
    corpusroot=Path(inputs['corpus']['path']);corpus=checked(corpusroot,inputs['corpus']['pin'])
    records=[json.loads(line)['record'] for line in (shingleroot/'documents.jsonl').open('rb')]
    n=len(records);lookup={key:i for i,key in enumerate(records)}
    original=json.loads((ancestryroot/'case-components.json').read_bytes())['components']
    need(len(lookup)==n and sorted(k for group in original for k in group)==sorted(records),'SPLIT_COLD_COVERAGE')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    pairs=np.fromfile(nearroot/'pairs.u32',dtype='<u4').reshape(-1,3)
    need(len(pairs)==near['counts']['edges'],'SPLIT_COLD_PAIR_COUNT')
    origins=list(map(int,pairs[:,0]));targets=list(map(int,pairs[:,1]));aliases=0
    for group in original:
        ids=[lookup[k] for k in group]
        origins.extend([ids[0]]*(len(ids)-1));targets.extend(ids[1:]);aliases+=len(ids)-1
    graph=coo_matrix((np.ones(len(origins),dtype='u1'),(origins,targets)),shape=(n,n)).tocsr()
    count,labels=connected_components(graph,directed=False)
    groups=defaultdict(list)
    for i,label in enumerate(labels):groups[int(label)].append(i)
    expected=[];names=('train','development','query')
    for indices in groups.values():
        key=hashlib.sha256(enc(sorted(records[i] for i in indices))).hexdigest()
        draw=int(hashlib.sha256(('legal-source-growth-20261022/split/'+key).encode()).hexdigest(),16)%10
        side='train' if draw<8 else ('development' if draw==8 else 'query')
        expected.append(dict(id=key,indices=indices,side=side))
    expected.sort(key=lambda c:c['id'])
    need(json.loads((root/'components.json').read_bytes())==expected,'SPLIT_COLD_COMPONENTS')
    sides=np.empty(n,dtype='u1');component=np.empty(n,dtype='<u4')
    support={name:dict(documents=0,atoms=0,successors=0,components=0) for name in names}
    for i,c in enumerate(expected):
        sides[c['indices']]=names.index(c['side']);component[c['indices']]=i
        support[c['side']]['components']+=1
    saved=np.load(root/'document-side.npy',allow_pickle=False)
    same_memberships(saved,sides);same_memberships(np.load(root/'component.npy',allow_pickle=False),component)
    need(all(sides[a]==sides[b] for a,b in zip(origins,targets)),'SPLIT_COLD_EDGE_CROSSES')
    phases['independent_sparse_components']=time.perf_counter()-tick;tick=time.perf_counter()
    lengths=[];last=0
    with (corpusroot/'documents.jsonl').open('rb') as stream:
        for i,line in enumerate(stream):
            d=json.loads(line)
            need(i<n and d['source_record']==records[i] and d['atom_start']==last,'SPLIT_COLD_ATOM_ADDRESS')
            size=d['atom_end']-d['atom_start'];need(size>0,'SPLIT_COLD_EMPTY_DOCUMENT')
            lengths.append(size);last=d['atom_end'];side=names[int(sides[i])]
            support[side]['documents']+=1;support[side]['atoms']+=size;support[side]['successors']+=size-1
    need(i+1==n and last==corpus['counts']['atoms'],'SPLIT_COLD_ATOM_CENSUS')
    all_atoms=np.repeat(sides,np.asarray(lengths,dtype='<i8'))
    same_memberships(np.load(root/'atom-side.npy',allow_pickle=False),all_atoms)
    need(support==report['support'] and count==report['counts']['components'],'SPLIT_COLD_SUPPORT')
    phases['all_atom_side_reconstruction']=time.perf_counter()-tick;tick=time.perf_counter()
    controls={}
    def reject(name,actual,expected):
        try:same_memberships(actual,expected)
        except ValueError as e:controls[name]=str(e);return
        raise AssertionError('SPLIT_COLD_NEGATIVE_PASSED')
    wrong=saved.copy();wrong[targets[0]]=(int(wrong[origins[0]])+1)%3
    reject('related_case_moved',wrong,sides);reject('document_removed',saved[:-1],sides)
    wrong_atoms=all_atoms.copy();wrong_atoms[0]=(int(wrong_atoms[0])+1)%3
    reject('atom_reassigned',wrong_atoms,all_atoms)
    selected=sorted(range(n),key=lambda i:records[i])[:33]
    consumers=[dict(index=i,record=records[i],side=names[int(saved[i])],component=expected[int(component[i])]['id']) for i in selected]
    write_committed(output/'consumer.json',consumers)
    phases['consumer_controls_and_persist']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'SPLIT_COLD_TIMING')
    result=dict(schema='legal-case-near-split-readback-v1',complete=True,source=dict(path=str(root),pin=pin),
                counts=report['counts'],support=support,all_components_independently_rederived=True,
                all_atom_sides_reconstructed=True,related_pair_crossings=0,consumer_cases=len(consumers),controls=controls,
                files={'consumer.json':file_sha(output/'consumer.json')},
                exposure='development_exposed_current_fit_case_disjoint',untouched_test=False,
                semantic_closure_complete=False,lineage_complete_beyond_retained_metadata=False,
                model_updates=0,scientific_admission=False,acceptance='author-checked',number_status='measured',
                phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,code_sha256=file_sha(Path(__file__)))
    receipt=write_committed(output/'report.json',result)
    print(json.dumps(dict(pin=receipt,counts=result['counts'],support=support,controls=controls,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args();run(a.source,a.pin,a.output)

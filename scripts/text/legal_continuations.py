"""Every legal continuation, cross-split text exposure, and TRAIN-only baselines."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_reuse_input import ReuseInputs

SCHEMA='legal-source-continuation-inventory-v1'
SIDES=('train','development','query')
PAIR_DTYPE=np.dtype([('key','<u8'),('count','<u4')])


def split_input(root,pin,cold,cold_pin,source):
    need(file_sha(root/'report.json')==pin and json.loads((root/'COMMITTED').read_bytes())==
         dict(report_sha256=pin),'CONTINUATION_SPLIT_COMMIT')
    r=json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['schema']=='legal-case-near-split-v1'
         and r['inputs']['corpus']['pin']==source.report['corpus']['pin'],'CONTINUATION_SPLIT_SCOPE')
    need(file_sha(cold/'report.json')==cold_pin,'CONTINUATION_COLD_PIN')
    proof=json.loads((cold/'report.json').read_bytes())
    need(proof['complete'] and proof['source']['pin']==pin and proof['all_atom_sides_reconstructed'],
         'CONTINUATION_COLD_SCOPE')
    seen={root/n:stamp(root/n) for n in ('report.json','COMMITTED')}
    for name,ref in r['files'].items():
        path=root/name;need(path.stat().st_size==ref['bytes'] and file_sha(path)==ref['sha256'],'CONTINUATION_SPLIT_FILE')
        seen[path]=stamp(path)
    sides=np.load(root/'atom-side.npy',mmap_mode='r',allow_pickle=False)
    docs=np.load(root/'document-side.npy',mmap_mode='r',allow_pickle=False)
    bounds=source.arrays['documents'];n=source.report['counts']['occurrences']
    need(sides.shape==(n,) and sides.dtype==np.dtype('u1') and np.all(sides<3),'CONTINUATION_SIDE_SHAPE')
    need(np.array_equal(sides,np.repeat(docs,bounds[:,1]-bounds[:,0])),'CONTINUATION_DOCUMENT_BINDING')
    return r,sides,seen


def bind_edges(documents,successors,sides):
    valid=np.ones(len(sides),dtype=bool);valid[documents[:,1]-1]=False
    need(np.array_equal(successors,np.flatnonzero(valid)),'CONTINUATION_DOCUMENT_EDGE')
    need(np.array_equal(sides[successors],sides[successors+1]),'CONTINUATION_SIDE_EDGE')


def pack(current,target,u):
    need(u<2**32 and np.all((current>=0)&(current<u)) and np.all((target>=0)&(target<u)),
         'CONTINUATION_INPUT_WIDTH')
    return (current.astype('<u8')<<np.uint64(32))|target.astype('<u8')


def grouped(keys):
    key,count=np.unique(keys,return_counts=True)
    need(int(count.max(initial=0))<2**32,'CONTINUATION_COUNT_WIDTH')
    result=np.empty(len(key),dtype=PAIR_DTYPE);result['key']=key;result['count']=count
    return result


def fit_mode(pairs,u):
    current=(pairs['key']>>np.uint64(32)).astype('<i8')
    boundaries=np.r_[0,np.flatnonzero(current[1:]!=current[:-1])+1,len(current)]
    maxima=np.maximum.reduceat(pairs['count'],boundaries[:-1])
    winners=np.flatnonzero(pairs['count']==np.repeat(maxima,np.diff(boundaries)))
    winners=winners[np.r_[True,current[winners][1:]!=current[winners][:-1]]]
    modal=np.full(u,-1,dtype='<i8');modal[current[winners]]=(pairs['key'][winners]&np.uint64(0xffffffff)).astype('<i8')
    stats=dict(current_inputs=len(boundaries)-1,multiple_successors=int((np.diff(boundaries)>1).sum()),
               maximum_successors=int(np.diff(boundaries).max()),
               empirical_current_only_oracle_hits=int(maxima.sum(dtype='<u8')),
               oracle_scope='best deterministic exact next-text lookup given only current-text identity; uses this side targets')
    return modal,stats


def membership(keys,gallery):
    indices=np.searchsorted(gallery,keys);valid=indices<len(gallery);result=np.zeros(len(keys),dtype=bool)
    result[valid]=gallery[indices[valid]]==keys[valid];return result


def score(pairs,train,presence,modal,global_mode):
    current=(pairs['key']>>np.uint64(32)).astype('<i8');target=(pairs['key']&np.uint64(0xffffffff)).astype('<i8')
    weights=pairs['count'];prediction=modal[current]
    current_seen=prediction>=0;target_seen=(presence[target]&1)!=0;pair_seen=membership(pairs['key'],train['key'])
    cells={}
    for c in (0,1):
        for t in (0,1):
            for p in (0,1):
                mask=(current_seen==c)&(target_seen==t)&(pair_seen==p)
                cells[f'current{c}_target{t}_pair{p}']=dict(
                    occurrences=int(weights[mask].sum(dtype='<u8')),distinct_pairs=int(mask.sum()),
                    train_lookup_hits=int(weights[mask&(prediction==target)].sum(dtype='<u8')),
                    copy_current_hits=int(weights[mask&(current==target)].sum(dtype='<u8')),
                    global_target_hits=int(weights[mask&(target==global_mode)].sum(dtype='<u8')))
    return dict(occurrences=int(weights.sum(dtype='<u8')),distinct_pairs=len(pairs),
                train_lookup_hits=int(weights[prediction==target].sum(dtype='<u8')),
                train_lookup_misses=int(weights[prediction<0].sum(dtype='<u8')),
                copy_current_hits=int(weights[current==target].sum(dtype='<u8')),
                global_target_hits=int(weights[target==global_mode].sum(dtype='<u8')),
                current_atom_seen=int(weights[(presence[current]&1)!=0].sum(dtype='<u8')),
                current_has_train_successor=int(weights[current_seen].sum(dtype='<u8')),
                target_atom_seen=int(weights[target_seen].sum(dtype='<u8')),
                pair_seen=int(weights[pair_seen].sum(dtype='<u8')),cells=cells)


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time();source=ReuseInputs(a.source,a.pin)
    split,sides,stamps=split_input(a.split,a.split_pin,a.cold,a.cold_pin,source)
    mapping=source.arrays['mapping'];successors=source.arrays['successors'];u=source.report['counts']['unique_inputs']
    bind_edges(source.arrays['documents'],successors,sides)
    a.output.mkdir(parents=True,exist_ok=False)
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    presence=np.zeros(u,dtype='u1');np.bitwise_or.at(presence,mapping,np.left_shift(np.uint8(1),sides))
    need(np.all(presence>0),'CONTINUATION_UNUSED_INPUT')
    phases['exact_text_exposure']=time.perf_counter()-tick;tick=time.perf_counter()
    keys=pack(mapping[successors],mapping[successors+1],u);edge_sides=sides[successors]
    pairs={name:grouped(keys[edge_sides==i]) for i,name in enumerate(SIDES)}
    for name in SIDES:need(int(pairs[name]['count'].sum(dtype='<u8'))==split['support'][name]['successors'],
                          'CONTINUATION_COMPLETE_DENOMINATOR')
    phases['all_pair_grouping']=time.perf_counter()-tick;tick=time.perf_counter()
    modal,train_stats=fit_mode(pairs['train'],u)
    global_counts=np.bincount(mapping[successors[edge_sides==0]+1],minlength=u)
    global_mode=int(np.argmax(global_counts))
    phases['train_only_baselines']=time.perf_counter()-tick;tick=time.perf_counter()
    metrics={name:score(values,pairs['train'],presence,modal,global_mode) for name,values in pairs.items()}
    ambiguity={'train':train_stats}
    for name in SIDES[1:]:
        _,ambiguity[name]=fit_mode(pairs[name],u)
    exposure={str(mask):int((presence==mask).sum()) for mask in range(1,8)}
    phases['metrics_and_input_ambiguity']=time.perf_counter()-tick;tick=time.perf_counter()
    arrays={'presence.npy':presence,'train-modal.npy':modal,**{name+'-pairs.npy':v for name,v in pairs.items()}}
    for name,value in arrays.items():
        with (a.output/name).open('xb') as stream:np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    source.fresh();need(all(stamp(p)==s for p,s in stamps.items()),'CONTINUATION_SPLIT_DRIFT')
    files={name:dict(bytes=(a.output/name).stat().st_size,sha256=file_sha(a.output/name)) for name in arrays}
    phases['publication_and_pins']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'CONTINUATION_TIMING')
    result=dict(schema=SCHEMA,complete=True,source=dict(path=str(a.source),pin=a.pin),
                split=dict(path=str(a.split),pin=a.split_pin),cold=dict(path=str(a.cold),pin=a.cold_pin),
                counts=source.report['counts'],files=files,exposure=exposure,metrics=metrics,ambiguity=ambiguity,
                global_train_target=global_mode,global_train_target_count=int(global_counts[global_mode]),
                policy=dict(input='current atom exact text only',target='attested next atom exact text',
                            ties='lowest UID; lexicographic text SHA256',denominator='all attested successors; lookup misses counted',
                            presence_bits={'train':1,'development':2,'query':4},oracle_is_model_ceiling=False),
                baseline_fit_side='train_only',model_updates=0,scientific_admission=False,training_ready=False,
                exposure_status='development_exposed_current_fit_case_disjoint',untouched_test=False,
                phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                acceptance='author-checked',number_status='measured',code_sha256=file_sha(Path(__file__)))
    pin=write_committed(a.output/'report.json',result);write_committed(a.output/'COMMITTED',dict(report_sha256=pin))
    print(json.dumps(dict(pin=pin,exposure=exposure,metrics={k:{a:b for a,b in v.items() if a!='cells'} for k,v in metrics.items()},seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','split','cold','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pin','split-pin','cold-pin'):p.add_argument('--'+name,required=True)
    run(p.parse_args())

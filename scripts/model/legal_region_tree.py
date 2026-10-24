"""Fit a TRAIN-only complete-window prototype hierarchy on a declared legal prefix."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from claims.process_conditions import write_committed
from model.native_region_tree import POLICY,NativeRegionTree,distances,fit,publish
from text.constellation_corpus import file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs


class NativeRows:
    def __init__(self,source,bank,pin):
        self.bank=NativeBatchSession(bank,pin,source);self.roster=tuple(self.bank.config['roster'])
        self.calls=0;self.inputs=0

    def __call__(self,uids):
        result={}
        for name in self.roster:
            batch=self.bank.read_many(name,uids)
            result[name]=[batch.values[batch.offsets[i]:batch.offsets[i+1]] for i in range(len(uids))]
        self.calls+=1;self.inputs+=len(uids);return result


def fit_members(presence,prefix):
    need(type(prefix) is int and 0<prefix<=len(presence),'LEGAL_TREE_PREFIX')
    uids=np.flatnonzero(presence[:prefix]&1).astype('<i8')
    need(len(uids)>0,'LEGAL_TREE_NO_TRAIN');return uids


def check_training(uids,presence):
    need(uids.dtype==np.int64 and uids.ndim==1 and len(uids)>0 and np.all(uids>=0)
         and np.all(uids<len(presence)) and np.all(presence[uids]&1),'LEGAL_TREE_NON_TRAIN_MEMBER')


def selected_distances(read,uids,paths,tree):
    leaves=np.max(np.where(paths>=0,np.arange(paths.shape[1])[None],-1),axis=1)
    selected=paths[np.arange(len(paths)),leaves];result=np.empty((len(uids),len(tree.roster)),dtype='<f8')
    for node in np.unique(selected):
        indices=np.flatnonzero(selected==node);prototype=tree.prototypes[tree.nodes[int(node)]['prototype']]
        for lo in range(0,len(indices),POLICY['tile']):
            part=indices[lo:lo+POLICY['tile']];values=distances(read(uids[part].tolist()),[prototype],tree.roster)
            for lane,name in enumerate(tree.roster):result[part,lane]=values[name][:,0]
    return result,selected


def support(tree,paths,presence):
    groups={}
    for name,mask in [('train',(presence&1)>0),('development',(presence&2)>0),('query',(presence&4)>0),
                      ('development_without_train',((presence&2)>0)&((presence&1)==0)),
                      ('query_without_train',((presence&4)>0)&((presence&1)==0))]:
        levels=[]
        for depth in range(1,paths.shape[1]):
            current=paths[mask,depth];ids,counts=np.unique(current[current>=0],return_counts=True)
            levels.append(dict(depth=depth,total=int(mask.sum()),active=int((current>=0).sum()),
                               terminal_before_level=int((current<0).sum()),nodes=ids.tolist(),counts=counts.tolist()))
        groups[name]=levels
    leaves=[n['support'] for n in tree.nodes if not n['children']]
    return dict(groups=groups,leaf_train_support=dict(minimum=min(leaves),median=float(np.median(leaves)),
                maximum=max(leaves),sum=sum(leaves),count=len(leaves)),nodes=len(tree.nodes))


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    source=ReuseInputs(a.source,a.source_pin);continuations=Continuations(a.continuations,a.continuations_pin)
    need(continuations.u==source.report['counts']['unique_inputs'],'LEGAL_TREE_SOURCE_COUNT')
    # Both separately authenticated manifests must name this exact measurement population.
    need(continuations.report['source']['pin']==source.pin,'LEGAL_TREE_CONTINUATION_SOURCE')
    read=NativeRows(source,a.bank,a.bank_pin);train=fit_members(continuations.presence,a.prefix_inputs)
    check_training(train,continuations.presence);uids=np.arange(a.prefix_inputs,dtype='<i8')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    nodes,prototypes,fit_counts=fit(read,train,read.roster)
    phases['fit_native_hierarchy']=time.perf_counter()-tick;tick=time.perf_counter()
    np.save(a.output/'fit-uids.npy',train,allow_pickle=False);np.save(a.output/'query-uids.npy',uids,allow_pickle=False)
    binding=dict(source_pin=source.pin,continuation_pin=a.continuations_pin,bank_pin=a.bank_pin,
                 prefix_inputs=a.prefix_inputs,fit_members_sha256=file_sha(a.output/'fit-uids.npy'),
                 instruments={n:read.bank.config['instruments'][n]['lens_id'] for n in read.roster},
                 policy=POLICY,scope='complete_declared_prefix_engineering_proof',fitting='TRAIN_unique_inputs_only')
    tree_pin=publish(a.output/'tree',nodes,prototypes,read.roster,binding)
    tree=NativeRegionTree(a.output/'tree',tree_pin,binding)
    phases['publish_and_cold_load']=time.perf_counter()-tick;tick=time.perf_counter()
    codes,paths=tree.encode(read,uids);native_distance,selected=selected_distances(read,uids,paths,tree)
    phases['all_prefix_assignment_and_native_distances']=time.perf_counter()-tick;tick=time.perf_counter()
    np.save(a.output/'codes.npy',codes,allow_pickle=False);np.save(a.output/'paths.npy',paths,allow_pickle=False)
    np.save(a.output/'selected-distance.npy',native_distance,allow_pickle=False)
    census=support(tree,paths,continuations.presence[:len(uids)])
    source.fresh();continuations.fresh();tree.fresh()
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(a.output.iterdir()) if p.is_file()}
    phases['support_and_publication']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'LEGAL_TREE_PHASE_ATTRIBUTION')
    result=dict(schema='legal-native-region-tree-proof-v1',complete=True,tree=dict(path=str(a.output/'tree'),pin=tree_pin),
        source=dict(path=str(a.source),pin=source.pin),continuations=dict(path=str(a.continuations),pin=a.continuations_pin),
        bank=dict(path=str(a.bank),pin=a.bank_pin),binding=binding,files=files,fit_counts=fit_counts,
        query_unique_inputs=len(uids),train_unique_inputs=len(train),support=census,
        mean_selected_distance={name:float(native_distance[:,i].mean()) for i,name in enumerate(read.roster)},
        reader_stats=read.bank.stats,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code_sha256=file_sha(Path(__file__)),tree_code_sha256=file_sha(Path(__file__).with_name('native_region_tree.py')),
        device='cpu_index_construction',new_encoder_calls=0,predictor_updates=0,
        scientific_admission=False,training_ready=False,full_legal_corpus_tree=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=pin,tree_pin=tree_pin,support=census,seconds=seconds,cpu_seconds=result['cpu_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','continuations','bank'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-pin',required=True)
    parser.add_argument('--prefix-inputs',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())

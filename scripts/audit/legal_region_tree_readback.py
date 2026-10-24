"""Cold native hierarchy audit: flat prototype comparisons validate every persisted route."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.constellation import ordered_distance
from model.native_region_tree import NativeRegionTree,POLICY,distances
from model.legal_region_tree import NativeRows,check_training,fit_members
from text.constellation_corpus import digest,file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs


def variable_windows(bank,output):
    root=Path(bank.config['runtime_proof']['path']);pin=bank.config['runtime_proof']['pin']
    need(file_sha(root/'report.json')==pin,'TREE_AUDIT_NATIVE_PROOF')
    report=json.loads((root/'report.json').read_bytes());roster=tuple(bank.config['roster']);slots={};prototypes=[{}, {}, {}]
    lengths={}
    for name in roster:
        need(file_sha(root/name/'report.json')==report['instruments'][name]['pin'],'TREE_AUDIT_LANE_PROOF')
        lane=json.loads((root/name/'report.json').read_bytes())
        for filename,ref in lane['files'].items():need(file_sha(root/name/filename)==ref['sha256'],'TREE_AUDIT_LANE_FILE')
        with np.load(root/name/'reference.npz',allow_pickle=False) as saved:
            values,bounds=saved['vectors'],saved['bounds'];long=int(np.diff(bounds).argmax())
            chosen=[0,long,len(bounds)-2];rows=[values[bounds[j]:bounds[j+1]].copy() for j in chosen]
        slots[name]=rows+[rows[1].copy()]
        for j in range(3):prototypes[j][name]=rows[j]
        lengths[name]=[len(r) for r in rows]
    measured=distances(slots,prototypes,roster);errors={};order_change={};missing_change={}
    for name in roster:
        expected=np.empty((4,3),dtype=np.float64)
        for i,raw in enumerate(slots[name]):
            x=raw.astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
            for j,prototype in enumerate(prototypes):
                y=prototype[name].astype(np.float64);y/=np.linalg.norm(y,axis=1,keepdims=True)
                expected[i,j]=ordered_distance(np.clip(x@y.T,-1,1))
        error=float(np.max(np.abs(expected-measured[name])));need(error<=5e-13,'TREE_AUDIT_DISTANCE_ARITHMETIC')
        errors[name]=error
    reversed_rows={n:list(reversed(rows)) for n,rows in slots.items()}
    reverse=distances(reversed_rows,prototypes,roster)
    one=distances({n:[rows[1]] for n,rows in slots.items()},prototypes,roster)
    for name in roster:
        need(np.array_equal(reverse[name],measured[name][::-1]) and np.array_equal(one[name][0],measured[name][1]),
             'TREE_AUDIT_BATCH_CONTEXT')
    # Real long readings change when native windows are removed or reversed.
    long={n:[slots[n][1]] for n in roster};base=distances(long,[prototypes[1]],roster)
    backwards=distances({n:[long[n][0][::-1].copy()] for n in roster},[prototypes[1]],roster)
    missing=distances({n:[long[n][0][:-1].copy()] for n in roster},[prototypes[1]],roster)
    for name in roster:
        order_change[name]=float(backwards[name][0,0]-base[name][0,0])
        missing_change[name]=float(missing[name][0,0]-base[name][0,0])
        need(lengths[name][1]>1 and order_change[name]>0 and missing_change[name]>0,'TREE_AUDIT_NATIVE_WINDOW_SENSITIVITY')
    result=dict(source_pin=pin,lengths=lengths,maximum_error=errors,exact_batch_context=True,
                native_window_reversal_change=order_change,removed_window_change=missing_change)
    write_committed(output/'variable-window-distance.json',result);return result


def all_paths(tree,read,uids,codes,paths,selected_distance,presence):
    prototype_uids=sorted(tree.prototypes);columns={uid:i for i,uid in enumerate(prototype_uids)}
    prototypes=[tree.prototypes[i] for i in prototype_uids];route_counts=Counter();max_distance_error=0.
    for lo in range(0,len(uids),POLICY['tile']):
        hi=min(lo+POLICY['tile'],len(uids));values=distances(read(uids[lo:hi].tolist()),prototypes,tree.roster)
        combined=np.stack(list(values.values()),axis=0).mean(0)
        for row in range(hi-lo):
            index=lo+row;node=0;expected_path=[0];expected_code=[]
            while tree.nodes[node]['children']:
                children=tree.nodes[node]['children']
                score=[combined[row,columns[tree.nodes[c]['prototype']]] for c in children]
                chosen=min(range(len(children)),key=lambda j:(score[j],j));node=children[chosen]
                expected_code.append(chosen);expected_path.append(node)
            expected_path+=[-1]*(POLICY['depth']+1-len(expected_path));expected_code+=[-1]*(POLICY['depth']-len(expected_code))
            need(paths[index].tolist()==expected_path and codes[index].tolist()==expected_code,'TREE_AUDIT_COLD_ROUTE')
            if presence[index]&1:
                for ancestor in expected_path:
                    if ancestor>=0:route_counts[ancestor]+=1
            column=columns[tree.nodes[node]['prototype']]
            delta=max(abs(values[name][row,column]-selected_distance[index,lane]) for lane,name in enumerate(tree.roster))
            max_distance_error=max(max_distance_error,float(delta))
    need(max_distance_error<=5e-13,'TREE_AUDIT_SELECTED_DISTANCE')
    for node_id,node in enumerate(tree.nodes):
        members=uids[((presence&1)>0)&(paths[:,node['depth']]==node_id)]
        need(route_counts[node_id]==node['support']==len(members) and digest(members.tobytes())==node['members_sha256'],
             'TREE_AUDIT_TRAIN_SUPPORT')
    return dict(inputs=len(uids),all_codes_and_paths_exact=True,all_train_support_and_member_hashes_exact=True,
                maximum_selected_distance_error=max_distance_error)


def controls(tree,continuations,source,output):
    wrong=deepcopy(tree.report['binding']);wrong['source_pin']='0'*64
    nontrain=int(np.flatnonzero((continuations.presence[:tree.report['binding']['prefix_inputs']]&1)==0)[0])
    result=dict(wrong_tree_pin=rejected(lambda:NativeRegionTree(tree.root,'0'*64,tree.report['binding'])),
        wrong_source_binding=rejected(lambda:NativeRegionTree(tree.root,tree.pin,wrong)),
        non_train_fit=rejected(lambda:check_training(np.array([nontrain],dtype='<i8'),continuations.presence)))
    copy=output/'corrupted-real-tree';copy.mkdir()
    for filename in ['report.json','COMMITTED',*tree.report['files']]:shutil.copy2(tree.root/filename,copy/filename)
    prior=NativeRegionTree(copy,tree.pin,tree.report['binding']);prior.fresh();path=copy/'instrument-0.npz'
    before=file_sha(path)
    with path.open('r+b') as handle:
        handle.seek(-1,2);old=handle.read(1);handle.seek(-1,2);handle.write(bytes([old[0]^1]));handle.flush();os.fsync(handle.fileno())
    result['warm_prototype_change']=rejected(prior.fresh)
    result['cold_prototype_corruption']=rejected(lambda:NativeRegionTree(copy,tree.pin,tree.report['binding']))
    result['mutation']=dict(path=str(path),before=before,after=file_sha(path))
    bad=deepcopy(tree.nodes);bad[0]['support']+=1;original=tree.nodes;tree.nodes=bad
    try:result['incorrect_support']=rejected(tree.validate)
    finally:tree.nodes=original
    bad=deepcopy(tree.nodes);bad[0]['children']=[0];tree.nodes=bad
    try:result['cycle']=rejected(tree.validate)
    finally:tree.nodes=original
    bad=deepcopy(tree.nodes);bad[0]['members_sha256']='0'*64;tree.nodes=bad
    try:result['changed_memory_nodes']=rejected(tree.fresh)
    finally:tree.nodes=original
    first=tree.prototypes[min(tree.prototypes)][tree.roster[0]];first.flags.writeable=True
    try:result['writable_memory_prototype']=rejected(tree.fresh)
    finally:first.flags.writeable=False
    return result


def run(a):
    started=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    need(file_sha(a.publication/'report.json')==a.pin,'TREE_AUDIT_PUBLICATION_PIN')
    report=json.loads((a.publication/'report.json').read_bytes());need(report['complete'],'TREE_AUDIT_INCOMPLETE')
    for filename,ref in report['files'].items():need(file_sha(a.publication/filename)==ref['sha256'],'TREE_AUDIT_PUBLICATION_FILE')
    source=ReuseInputs(Path(report['source']['path']),report['source']['pin'])
    continuations=Continuations(Path(report['continuations']['path']),report['continuations']['pin'])
    read=NativeRows(source,Path(report['bank']['path']),report['bank']['pin'])
    tree=NativeRegionTree(Path(report['tree']['path']),report['tree']['pin'],report['binding'])
    uids=np.load(a.publication/'query-uids.npy',allow_pickle=False);train=np.load(a.publication/'fit-uids.npy',allow_pickle=False)
    need(np.array_equal(train,fit_members(continuations.presence,len(uids)))
         and np.array_equal(uids,np.arange(report['query_unique_inputs'],dtype='<i8')),'TREE_AUDIT_SCOPE')
    original=read(sorted(tree.prototypes))
    for j,uid in enumerate(sorted(tree.prototypes)):
        for name in tree.roster:need(np.array_equal(tree.prototypes[uid][name],original[name][j]),'TREE_AUDIT_REAL_PROTOTYPE')
    phases=dict(authenticate_and_prototype_replay=time.perf_counter()-tick);tick=time.perf_counter()
    variable=variable_windows(read.bank,a.output)
    phases['independent_variable_window_arithmetic']=time.perf_counter()-tick;tick=time.perf_counter()
    codes=np.load(a.publication/'codes.npy',allow_pickle=False);paths=np.load(a.publication/'paths.npy',allow_pickle=False)
    measured=np.load(a.publication/'selected-distance.npy',allow_pickle=False)
    route=all_paths(tree,read,uids,codes,paths,measured,continuations.presence[:len(uids)])
    phases['all_prefix_independent_routes']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(tree,continuations,source,a.output);source.fresh();continuations.fresh();tree.fresh()
    phases['controls']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'TREE_AUDIT_PHASE_ATTRIBUTION')
    result=dict(schema='legal-native-region-tree-cold-readback-v1',complete=True,
        source=dict(path=str(a.publication),pin=a.pin),tree_pin=tree.pin,prototypes_replayed=len(tree.prototypes),
        variable_windows=variable,route=route,controls=negative,phases_seconds=phases,seconds=seconds,
        code_sha256=file_sha(Path(__file__)),tree_code_sha256=file_sha(Path(__file__).parents[1]/'model/native_region_tree.py'),
        new_encoder_calls=0,predictor_updates=0,
        scientific_admission=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=pin,route=route,controls=negative,seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication',type=Path,required=True);parser.add_argument('--pin',required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

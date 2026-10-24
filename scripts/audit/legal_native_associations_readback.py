"""Cold-read every indexed native vector and independently reconstruct exact query rankings."""
import argparse
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
from kernel.legal_native_associations import BankLane,ProofLane
from kernel.native_window_index import NativeWindowIndex,POLICY,window_keys
from text.constellation_corpus import file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs


def vectors(index,read):
    total=0
    for lo in range(0,len(index.uids),512):
        uids=index.uids[lo:lo+512].tolist();rows=read(uids)
        need(np.array_equal([len(x) for x in rows],index.lengths[lo:lo+len(rows)]),'ASSOCIATION_AUDIT_WINDOW_LOSS')
        # Independent reversible addressing; no use of the producer's key helper.
        keys=np.array([uid*4294967296+j for uid,row in zip(uids,rows) for j in range(len(row))],dtype=np.uint64)
        saved=index.index.get(keys,dtype='f32');expected=np.concatenate(rows)
        need(len(saved)==len(expected) and all(x is not None and np.array_equal(x,y) for x,y in zip(saved,expected)),
             'ASSOCIATION_AUDIT_NATIVE_VECTOR')
        total+=len(keys)
    need(total==len(index.index)==index.report['windows'],'ASSOCIATION_AUDIT_WINDOW_COUNT')
    return dict(inputs=len(index.uids),windows=total,all_keys_and_native_vectors_exact=True)


def independent_scores(queries,rows):
    q=[x.astype(np.float64) for x in queries];r=[x.astype(np.float64) for x in rows]
    q=[x/np.linalg.norm(x,axis=1,keepdims=True) for x in q];r=[x/np.linalg.norm(x,axis=1,keepdims=True) for x in r]
    out=np.empty((len(q),len(r)));oneq=[i for i,x in enumerate(q) if len(x)==1];oner=[i for i,x in enumerate(r) if len(x)==1]
    if oneq and oner:out[np.ix_(oneq,oner)]=np.clip(np.stack([q[i][0] for i in oneq])@np.stack([r[i][0] for i in oner]).T,-1,1)
    for i,a in enumerate(q):
        for j,b in enumerate(r):
            if len(a)==len(b)==1:continue
            dots=np.clip(a@b.T,-1,1)
            out[i,j]=min(sum(max(row) for row in dots)/len(a),sum(max(col) for col in dots.T)/len(b))
    return out


def rankings(index,read,assessment):
    rows=assessment['rows'];uids=[r['uid'] for r in rows];queries=read(uids);best=[[] for _ in uids]
    for lo in range(0,len(index.uids),512):
        ids=index.uids[lo:lo+512].tolist();scores=independent_scores(queries,read(ids))
        for j,uid in enumerate(uids):
            entries=best[j]+[(other,float(s)) for other,s in zip(ids,scores[j]) if other!=uid]
            best[j]=sorted(entries,key=lambda x:(-x[1],x[0]))[:POLICY['top_k']]
    maximum=0.;recovered=0
    for row,expected in zip(rows,best):
        need([u for u,_ in row['exact_hits']]==[u for u,_ in expected],'ASSOCIATION_AUDIT_EXACT_RANK')
        maximum=max(maximum,max(abs(a[1]-b[1]) for a,b in zip(row['exact_hits'],expected)))
        found=len({u for u,_ in expected}&{u for u,_ in row['hits']})
        need(found==row['recovered'] and row['recall']==found/POLICY['top_k'],'ASSOCIATION_AUDIT_RECALL_ROW')
        recovered+=found
    recall=recovered/(len(rows)*POLICY['top_k'])
    need(maximum<=5e-13 and recall==assessment['recall'] and assessment['qualified']==(recall>=.95),
         'ASSOCIATION_AUDIT_RECALL')
    # Reversed query order and a repeat must cold-reproduce candidate membership and exact reranking.
    original={r['uid']:r for r in rows}
    for uid in list(reversed(uids))+uids[:1]:
        fresh=index.query(uid,read([uid])[0],read,lens_id=index.report['binding']['lens_id'])
        need(fresh['hits']==original[uid]['hits'] and fresh['candidate_uids']==original[uid]['candidate_uids'],
             'ASSOCIATION_AUDIT_QUERY_REPLAY')
    return dict(queries=len(rows),recall=recall,qualified=recall>=.95,maximum_scalar_error=maximum,
                all_exact_rankings_reproduced=True,cold_query_order_and_repeat_exact=True)


def union_check(union,lanes):
    expected={}
    for name,rows in lanes.items():
        for row in rows:
            for rank,(uid,score) in enumerate(row['hits'],1):expected[(row['uid'],uid,name)]=(rank,score)
    actual={}
    for edge in union['edges']:
        for name,value in edge['readings'].items():
            key=(edge['origin'],edge['target'],name);need(key not in actual,'ASSOCIATION_AUDIT_DUPLICATE_EDGE')
            actual[key]=(value['rank'],value['score'])
    need(actual==expected and union['consensus_required'] is False
         and union['absence_means']=='not_materialized_not_no_association','ASSOCIATION_AUDIT_LOST_VOTE')
    return dict(edges=len(union['edges']),instrument_votes=len(actual),
                single_instrument_edges=sum(len(x['readings'])==1 for x in union['edges']),every_vote_exact=True)


def controls(index,read,proof_index,proof_read,union,lanes,output):
    uid=int(index.uids[0]);native=read([uid])[0];binding=index.report['binding'];wrong={**binding,'lens_id':'0'*64}
    result=dict(wrong_pin=rejected(lambda:NativeWindowIndex(index.root,'0'*64,binding)),
                wrong_binding=rejected(lambda:NativeWindowIndex(index.root,index.pin,wrong)),
                wrong_lens=rejected(lambda:index.query(uid,native,read,lens_id='0'*64)),
                wrong_dimensions=rejected(lambda:index.query(uid,native[:,:1],read,lens_id=binding['lens_id'])),
                bad_window_address=rejected(lambda:window_keys([2**32],[native])))
    long=int(proof_index.uids[int(np.argmax(proof_index.lengths))]);need(len(proof_read([long])[0])>1,'ASSOCIATION_AUDIT_LONG')
    result['changed_query_identity']=rejected(lambda:index.query(int(index.uids[1]),native,read,lens_id=binding['lens_id']))
    result['dropped_query_window']=rejected(lambda:proof_index.query(long,proof_read([long])[0][:-1],proof_read,
                                                                    lens_id=binding['lens_id']))
    def dropped(ids):return [x[:-1] if uid==long else x for uid,x in zip(ids,proof_read(ids))]
    result['lost_real_window']=rejected(lambda:vectors(proof_index,dropped))
    lost=deepcopy(union);single=next(i for i,e in enumerate(lost['edges']) if len(e['readings'])==1)
    lost['edges'].pop(single);result['lost_single_instrument_edge']=rejected(lambda:union_check(lost,lanes))
    copy=output/'corrupted-real-index';copy.mkdir()
    for filename in ['report.json','COMMITTED',*index.report['files']]:shutil.copy2(index.root/filename,copy/filename)
    copied=NativeWindowIndex(copy,index.pin,binding);path=copy/'windows.usearch';before=file_sha(path)
    with path.open('r+b') as stream:
        stream.seek(-1,2);old=stream.read(1);stream.seek(-1,2);stream.write(bytes([old[0]^1]));stream.flush();os.fsync(stream.fileno())
    result['warm_file_change']=rejected(copied.fresh)
    result['cold_file_corruption']=rejected(lambda:NativeWindowIndex(copy,index.pin,binding))
    result['mutation']=dict(path=str(path),before=before,after=file_sha(path));return result


def run(a):
    started=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    need(file_sha(a.publication/'report.json')==a.pin,'ASSOCIATION_AUDIT_PUBLICATION')
    report=json.loads((a.publication/'report.json').read_bytes());need(report['complete'],'ASSOCIATION_AUDIT_COMPLETE')
    source=ReuseInputs(Path(report['source']['path']),report['source']['pin'])
    continuation=Continuations(Path(report['continuations']['path']),report['continuations']['pin'])
    bank=NativeBatchSession(Path(report['bank']['path']),report['bank']['pin'],source)
    query=np.flatnonzero((continuation.presence[:report['prefix_inputs']]&1)==0)[:32].tolist()
    need(query==report['query_uids'],'ASSOCIATION_AUDIT_QUERY_SELECTION')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter();lanes={};results={};first=None
    for name in report['roster']:
        read=BankLane(bank,name);proof=ProofLane(Path(report['proof']['path']),report['proof']['pin'],source,name,
                                               bank.config['instruments'][name]['lens_id'])
        record={};loaded={}
        for kind,reader in [('prefix',read),('multiwindow',proof)]:
            ref=report['lanes'][name][kind];root=Path(ref['path']);r=json.loads((root/'report.json').read_bytes())
            index=NativeWindowIndex(root,ref['pin'],r['binding']);loaded[kind]=index
            expected=np.arange(report['prefix_inputs'],dtype='<i8') if kind=='prefix' else proof.uids
            need(np.array_equal(index.uids,expected),'ASSOCIATION_AUDIT_GALLERY')
            path=root.parent/(kind+'-assessment.json');need(file_sha(path)==ref['assessment_pin'],'ASSOCIATION_AUDIT_ASSESSMENT_PIN')
            assessment=json.loads(path.read_bytes())
            need([r['uid'] for r in assessment['rows']]==(query if kind=='prefix' else proof.uids.tolist()),
                 'ASSOCIATION_AUDIT_QUERY_SCOPE')
            record[kind]=dict(vectors=vectors(index,reader),rankings=rankings(index,reader,assessment))
            if kind=='prefix':lanes[name]=assessment['rows']
        if first is None:first=(loaded['prefix'],read,loaded['multiwindow'],proof)
        results[name]=record;print(json.dumps(dict(instrument=name,**record)),flush=True)
    phases['all_native_vectors_and_independent_rankings']=time.perf_counter()-tick;tick=time.perf_counter()
    path=a.publication/'multiplex.json';need(file_sha(path)==report['multiplex_pin'],'ASSOCIATION_AUDIT_MULTIPLEX_PIN')
    union=json.loads(path.read_bytes());union_result=union_check(union,lanes)
    negative=controls(*first,union,lanes,a.output);bank.fresh();continuation.fresh()
    phases['union_and_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'ASSOCIATION_AUDIT_PHASE_ATTRIBUTION')
    result=dict(schema='legal-native-associations-readback-v1',complete=True,source=dict(path=str(a.publication),pin=a.pin),
        lanes=results,multiplex=union_result,controls=negative,phases_seconds=phases,seconds=seconds,
        code_sha256=file_sha(Path(__file__)),index_code_sha256=file_sha(Path(__file__).parents[1]/'kernel/native_window_index.py'),
        scientific_admission=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,seconds=seconds,multiplex=union_result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

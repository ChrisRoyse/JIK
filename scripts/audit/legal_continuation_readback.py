"""Independent SQL recount from original document occurrences and baseline replay."""
import argparse
import json
from pathlib import Path
import resource
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need
from text.legal_reuse_input import ReuseInputs
from text.legal_continuation_input import Continuations
from text.legal_continuations import split_input,bind_edges,SIDES


def same_pair(key,count,value):
    need(int(value['key'])==key and int(value['count'])==count,'CONTINUATION_SQL_PAIR_MISMATCH')


def same_modal(actual,expected):
    need(np.array_equal(actual,expected),'CONTINUATION_MODAL_MISMATCH')


def run(root,pin,output):
    start=tick=time.perf_counter();cpu=time.process_time();output.mkdir(parents=True,exist_ok=False)
    bank=Continuations(root,pin);r=bank.report
    source=ReuseInputs(Path(r['source']['path']),r['source']['pin'])
    split,sides,stamps=split_input(Path(r['split']['path']),r['split']['pin'],Path(r['cold']['path']),r['cold']['pin'],source)
    mapping=source.arrays['mapping'];documents=source.arrays['documents'];u=bank.u
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    presence=np.zeros(u,dtype='u1')
    for side in range(3):presence[np.unique(mapping[sides==side])]|=np.uint8(1<<side)
    need(np.array_equal(presence,bank.presence),'CONTINUATION_PRESENCE_RECOUNT')
    db=sqlite3.connect(':memory:');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE edges(side INTEGER,x INTEGER,y INTEGER)')
    inserted=0
    for lo,hi in documents:
        lo,hi=int(lo),int(hi);side=int(sides[lo])
        db.executemany('INSERT INTO edges VALUES(?,?,?)',((side,int(x),int(y)) for x,y in zip(mapping[lo:hi-1],mapping[lo+1:hi])))
        inserted+=hi-lo-1
    need(inserted==source.report['counts']['successors'],'CONTINUATION_SQL_EDGE_COUNT');db.commit()
    phases['independent_presence_and_sql_occurrences']=time.perf_counter()-tick;tick=time.perf_counter()
    modal=np.full(u,-1,dtype='<i8');maximum=np.zeros(u,dtype='<u4');global_counts=np.zeros(u,dtype='<i8');train_keys=set()
    positions=[0,0,0];amb=[dict(current_inputs=0,multiple_successors=0,maximum_successors=0,
                              empirical_current_only_oracle_hits=0) for _ in range(3)]
    last=None;alternatives=best=0
    def finish():
        if last is not None:
            stats=amb[last[0]];stats['current_inputs']+=1;stats['multiple_successors']+=alternatives>1
            stats['maximum_successors']=max(stats['maximum_successors'],alternatives)
            stats['empirical_current_only_oracle_hits']+=best
    query='SELECT side,x,y,COUNT(*) FROM edges GROUP BY side,x,y ORDER BY side,x,y'
    plan=list(db.execute('EXPLAIN QUERY PLAN '+query))
    for side,current,target,count in db.execute(query):
        key=current*4294967296+target;values=bank.pairs[SIDES[side]]
        need(positions[side]<len(values),'CONTINUATION_SQL_EXTRA_PAIR')
        same_pair(key,count,values[positions[side]]);positions[side]+=1
        if last!=(side,current):finish();last=(side,current);alternatives=best=0
        alternatives+=1;best=max(best,count)
        if side==0:
            train_keys.add(key);global_counts[target]+=count
            if count>maximum[current]:maximum[current]=count;modal[current]=target
    finish();db.close()
    need(positions==[len(bank.pairs[s]) for s in SIDES],'CONTINUATION_SQL_MISSING_PAIR')
    same_modal(bank.modal,modal);global_mode=int(np.argmax(global_counts))
    need(global_mode==r['global_train_target'] and int(global_counts[global_mode])==r['global_train_target_count'],
         'CONTINUATION_GLOBAL_MODE')
    for side,name in enumerate(SIDES):
        need(all(r['ambiguity'][name][k]==v for k,v in amb[side].items()),'CONTINUATION_AMBIGUITY_RECOUNT')
    phases['sql_group_recount_and_train_modes']=time.perf_counter()-tick;tick=time.perf_counter()
    successors=source.arrays['successors'];metrics={}
    for side,name in enumerate(SIDES):
        indices=successors[sides[successors]==side];current=mapping[indices];target=mapping[indices+1]
        pred=modal[current];seen=pred>=0;target_seen=(presence[target]&1)!=0
        pair_seen=np.fromiter((int(x)*4294967296+int(y) in train_keys for x,y in zip(current,target)),dtype=bool,count=len(current))
        cells={}
        # Pair counts in a cell come from SQL-recounted unique rows; hits/denominators use raw occurrences.
        unique=bank.pairs[name];left=(unique['key']>>np.uint64(32)).astype('<i8');right=(unique['key']&np.uint64(0xffffffff)).astype('<i8')
        unique_flags=((modal[left]>=0).astype('u1')*4+(((presence[right]&1)!=0).astype('u1')*2)
                      +np.fromiter((int(k) in train_keys for k in unique['key']),dtype='u1',count=len(unique)))
        for c in (0,1):
            for t in (0,1):
                for p in (0,1):
                    mask=(seen==c)&(target_seen==t)&(pair_seen==p)
                    cells[f'current{c}_target{t}_pair{p}']=dict(occurrences=int(mask.sum()),
                        distinct_pairs=int((unique_flags==c*4+t*2+p).sum()),train_lookup_hits=int((mask&(pred==target)).sum()),
                        copy_current_hits=int((mask&(current==target)).sum()),global_target_hits=int((mask&(target==global_mode)).sum()))
        metrics[name]=dict(occurrences=len(indices),distinct_pairs=len(unique),train_lookup_hits=int((pred==target).sum()),
            train_lookup_misses=int((pred<0).sum()),copy_current_hits=int((current==target).sum()),
            global_target_hits=int((target==global_mode).sum()),current_atom_seen=int(((presence[current]&1)!=0).sum()),
            current_has_train_successor=int(seen.sum()),target_atom_seen=int(target_seen.sum()),pair_seen=int(pair_seen.sum()),cells=cells)
    need(metrics==r['metrics'],'CONTINUATION_RAW_METRIC_RECOUNT')
    need({str(mask):int((presence==mask).sum()) for mask in range(1,8)}==r['exposure'],'CONTINUATION_EXPOSURE_RECOUNT')
    phases['all_occurrence_metric_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    controls={}
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError) as exc:controls[name]=str(exc);return
        raise AssertionError('CONTINUATION_NEGATIVE_PASSED')
    example=bank.pairs['train'][0];key=int(example['key']);count=int(example['count'])
    reject('changed_pair',lambda:same_pair(key+1,count,example))
    reject('changed_count',lambda:same_pair(key,count+1,example))
    changed=np.array(modal,copy=True);uid=int(np.flatnonzero(modal>=0)[0]);changed[uid]=(int(modal[uid])+1)%u
    reject('changed_lookup_target',lambda:same_modal(changed,modal))
    bad_edge=np.array(successors,copy=True);bad_edge[0]=int(documents[0,1]-1)
    reject('cross_document_edge',lambda:bind_edges(documents,bad_edge,sides))
    wrong=np.array(sides,copy=True);wrong[int(successors[0])+1]=(int(wrong[int(successors[0])])+1)%3
    reject('cross_split_edge',lambda:bind_edges(documents,successors,wrong))
    reject('implicit_training',bank.training_authority)
    selected=sorted(set([0,u-1,uid]+[int(i) for i in np.linspace(0,u-1,31)]));consumers=[]
    for i in selected:
        expected=None if modal[i]<0 else int(modal[i]);need(bank.lookup(i)==expected,'CONTINUATION_CONSUMER')
        rows=bank.successors(i,'train');need(all(int(k)>>32==i for k in rows['key']),'CONTINUATION_CONSUMER_PAIRS')
        consumers.append(dict(uid=i,target=expected,train_successors=len(rows)))
    bank.fresh();source.fresh();write_committed(output/'consumer.json',consumers)
    phases['consumer_and_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'CONTINUATION_READBACK_TIMING')
    result=dict(schema='legal-source-continuation-readback-v1',complete=True,source=dict(path=str(root),pin=pin),
                sql_occurrences=inserted,sql_distinct_pairs=dict(zip(SIDES,positions)),sql=query,sql_plan=plan,
                all_presence_flags_recounted=True,all_pair_counts_recounted=True,train_modes_recounted=True,
                all_occurrence_metrics_replayed=True,controls=controls,consumer_cases=len(consumers),
                metrics=metrics,exposure=r['exposure'],ambiguity=r['ambiguity'],files={'consumer.json':file_sha(output/'consumer.json')},
                model_updates=0,scientific_admission=False,acceptance='author-checked',number_status='measured',
                phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,code_sha256=file_sha(Path(__file__)))
    receipt=write_committed(output/'report.json',result)
    print(json.dumps(dict(pin=receipt,counts=result['sql_distinct_pairs'],controls=controls,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args();run(a.source,a.pin,a.output)

"""Real-native bounded proof of complete graph timing and singleton query layout."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel import native_graph_accounted as current
from kernel import native_graph_batch as legacy
from kernel.certified_index_graph import CertifiedIndexLane,pinned
from kernel.native_window_index import NativeWindowIndex
from kernel.native_window_bounded import publish
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp

QUAL=('legal-region-predictor-qualification-v2-20261023','d5d4477fb98face26b4556d7101351a5ea4cdfd5271af912e35a0df776a54570')
COLD=('legal-region-predictor-cold-v2-20261023','a9aab63aa684030d7445446df93e814fd803615346442453b1c2045ea0d511d2')
BANK=Path('/zfs/hot/mnlm/readings/legal-source-v1-20261022/configuration.json')


def code():
    root=Path(__file__).resolve().parents[1]
    return {n:file_sha(root/n) for n in ['audit/legal_graph_accounting_readback.py','kernel/native_graph_accounted.py',
        'kernel/native_graph_batch.py','kernel/native_search_layout.py','kernel/native_window_index.py','kernel/native_window_bounded.py',
        'kernel/certified_index_graph.py','text/constellation_corpus.py','text/constellation_input.py',
        'model/native_population_training.py','model/loop_training_checkpoint.py','claims/process_conditions.py']}


def source(root):
    q=pinned(root/QUAL[0]/'report.json',QUAL[1]);c=pinned(root/COLD[0]/'report.json',COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin']==QUAL[1] and q['binding']==c['binding']
         and q['results_pin']==c['results_pin'],'GRAPH_CLOCK_SOURCE_PROOF')
    selection=pinned(root/QUAL[0]/'selection.json',q['binding']['selection']);authority=selection['source']
    raw=root/'legal-input-reuse-v1-20261022';report=pinned(raw/'report.json',authority['source'])
    for name in ['mapping.npy','documents.npy']:
        ref=report['files'][name];need(file_sha(raw/name)==(ref['sha256'] if isinstance(ref,dict) else ref),'GRAPH_CLOCK_SOURCE_ARRAY')
    mapping=np.load(raw/'mapping.npy',mmap_mode='r');documents=np.load(raw/'documents.npy',mmap_mode='r')
    bank=pinned(BANK,authority['bank']);native={};origins=[]
    for row in selection['documents']:
        d=row['document'];need(tuple(map(int,documents[d]))==(row['start'],row['stop']),'GRAPH_CLOCK_SOURCE_DOCUMENT')
        name=f'document-{d}/outputs.pt';f=root/QUAL[0]/name
        need(file_sha(f)==q['files'][name],'GRAPH_CLOCK_NATIVE_BYTES')
        saved=torch.load(f,map_location='cpu',weights_only=True);frames=saved['frames']
        need(fingerprint(frames)==saved['native_pin'] and len(frames)==row['stop']-row['start'],'GRAPH_CLOCK_NATIVE_PIN')
        for position,(uid,frame) in enumerate(zip(mapping[row['start']:row['stop']],frames)):
            uid=int(uid);need(set(frame)==set(bank['roster']),'GRAPH_CLOCK_NATIVE_ROSTER')
            if uid in native:need(all(torch.equal(native[uid][n],v) for n,v in frame.items()),'GRAPH_CLOCK_ALIAS_CHANGED')
            else:native[uid]=frame
            origins.append(dict(document=d,position=position,uid=uid))
    mapping._mmap.close();documents._mmap.close()
    need(len(origins)==215 and len(native)==208,'GRAPH_CLOCK_SOURCE_SCOPE')
    return native,bank,dict(proof=QUAL[1],cold=COLD[1],selection=q['binding']['selection'],authority=authority,origins=origins)


class BoundedNativeLane:
    """Use the real certified-index vector read method without claiming a full certificate."""
    __call__=CertifiedIndexLane.__call__

    def __init__(self,index,native_path):
        self.index=index;self.stamps={native_path:stamp(native_path)};self.delay=0.;self.observed_delays=[]
        self.stats=dict(calls=0,inputs=0,windows=0,read_seconds=0.,maximum_payload_bytes=0)

    def fresh(self):
        self.index.fresh()
        need(all(stamp(p)==s for p,s in self.stamps.items()),'GRAPH_CLOCK_NATIVE_CHANGED')
        if self.delay:
            delay=self.delay;self.delay=0.;tick=time.perf_counter();time.sleep(delay)
            self.observed_delays.append(time.perf_counter()-tick)


def same(a,b):
    need(a['rows']==b['rows'],'GRAPH_CLOCK_CHANGED_NEIGHBOURS')
    need(a['work']==b['work'],'GRAPH_CLOCK_CHANGED_SEARCH_WORK')


def recount(result,read):
    largest=0.
    for row in result['rows']:
        a=read([row['uid']])[0].astype(np.float64);a/=np.sqrt(np.einsum('ij,ij->i',a,a))[:,None];scores=[]
        for uid,b in zip(row['candidate_uids'],read(row['candidate_uids'])):
            b=b.astype(np.float64);b/=np.sqrt(np.einsum('ij,ij->i',b,b))[:,None]
            dot=np.clip(np.einsum('id,jd->ij',a,b,optimize=False),-1,1)
            scores.append((uid,float(min(dot.max(0).mean(),dot.max(1).mean()))))
        expected=sorted(scores,key=lambda x:(-x[1],x[0]))[:10]
        need([x[0] for x in expected]==[x[0] for x in row['hits']],'GRAPH_CLOCK_NUMPY_RANK')
        largest=max(largest,max(abs(x[1]-y[1]) for x,y in zip(expected,row['hits'])))
    need(largest<=1e-12,'GRAPH_CLOCK_NUMPY_SCORE')
    return largest


def timing_controls(timing):
    low=deepcopy(timing);low['phases_seconds']={n:v*.1 for n,v in low['phases_seconds'].items()}
    gap=deepcopy(timing);gap['intervals'][1]['start']+=1
    missing=deepcopy(timing);missing['intervals'].pop(0)
    nonfinite=deepcopy(timing);nonfinite['intervals'][0]['stop']=float('nan')
    negative=deepcopy(timing);negative['intervals'][0]['stop']=-1
    return dict(underaccounted=rejected(lambda:current.validate_timing(low)),gap=rejected(lambda:current.validate_timing(gap)),
        missing_boundary=rejected(lambda:current.validate_timing(missing)),nonfinite=rejected(lambda:current.validate_timing(nonfinite)),
        negative=rejected(lambda:current.validate_timing(negative)))


def run(a):
    started=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pins=code()
    for name,pin in pins.items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(dest)==pin,'GRAPH_CLOCK_CODE_COPY')
    native,bank,origin=source(a.root);uids=np.array(sorted(native),dtype='<i8')
    native_path=a.output/'native.pt';torch.save(native,native_path)
    need(fingerprint(torch.load(native_path,map_location='cpu',weights_only=True))==fingerprint(native),'GRAPH_CLOCK_NATIVE_READBACK')
    binding=dict(code=pins,source=origin,native=fingerprint(native),inputs=len(uids),roster=bank['roster'],
                 scope='bounded_208_native_input_component_proof',threads=1,memory_bytes=16<<20,delay_seconds=.1)
    previous=pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'GRAPH_CLOCK_COLD_BINDING')
        for n,pin in previous['files'].items():need(file_sha(a.previous/n)==pin,'GRAPH_CLOCK_COLD_BYTES')
        indexes=pinned(a.previous/'indexes.json',previous['indexes_pin'])
    else:indexes={}
    phases=dict(authentication=time.perf_counter()-tick);results={};runs={}
    for name in bank['roster']:
        tick=time.perf_counter();out=a.output/name;out.mkdir()
        rows={u:native[u][name].numpy() for u in native}
        for row in rows.values():row.flags.writeable=False
        read=lambda ids:[rows[u] for u in ids]
        ib=dict(instrument=name,lens_id=bank['instruments'][name]['lens_id'],source=origin,native=binding['native'],scope=binding['scope'])
        if not previous:
            index_root=out/'index';pin=publish(index_root,uids,read,ib,memory_bytes=16<<20)
            indexes[name]=dict(path=str(index_root),pin=pin)
        ref=indexes[name];index=NativeWindowIndex(Path(ref['path']),ref['pin'],ib);lane=BoundedNativeLane(index,native_path)
        loaded=lane(uids.tolist());need(all(np.array_equal(x,rows[int(u)]) for x,u in zip(loaded,uids)),'GRAPH_CLOCK_INDEX_NATIVE_EXACT')
        numeric=[];records=[];largest=0.;ordinary_seconds=0.;outer_seconds=0.
        for direction,ids in [('forward',uids.tolist()),('reverse',list(reversed(uids.tolist())))]:
            for lo in range(0,len(ids),64):
                batch=ids[lo:lo+64];old=legacy.query_batch(lane,batch,threads=1)
                outer=time.perf_counter();new=current.query_batch(lane,batch,threads=1,trace=True);elapsed=time.perf_counter()-outer
                same(old,new);current.validate_timing(new['timing']);largest=max(largest,recount(new,read))
                ordinary_seconds+=new['seconds'];outer_seconds+=elapsed
                records.append(dict(direction=direction,first=lo,old=old,new=new,outer_seconds=elapsed))
                numeric.append(dict(direction=direction,first=lo,rows=new['rows'],work=new['work']))
        # Singleton and native multi-window queries each retain all actual native windows.
        singles=[int(u) for u in uids if len(rows[int(u)])==1];multiples=[int(u) for u in uids if len(rows[int(u)])>1]
        need(singles and multiples,'GRAPH_CLOCK_WINDOW_COVERAGE')
        singleton=[]
        for uid in [singles[0],multiples[0]]:
            new=current.query_batch(lane,[uid],threads=1,trace=True)
            old=index.query(uid,rows[uid],read,lens_id=ib['lens_id'])
            need(new['rows'][0]=={k:v for k,v in old.items() if k!='ann_distance_computations'},'GRAPH_CLOCK_SINGLETON_EXACT')
            largest=max(largest,recount(new,read));singleton.append(dict(uid=uid,new=new,reference=old))
            numeric.append(dict(singleton=uid,rows=new['rows']))
        pair=uids[:2].tolist();baseline=legacy.query_batch(lane,pair,threads=1)
        lane.delay=.1;old_fault=rejected(lambda:legacy.query_batch(lane,pair,threads=1))
        need('GRAPH_BATCH_PHASE_ATTRIBUTION' in old_fault,'GRAPH_CLOCK_OLD_FAILURE_REASON')
        lane.delay=.1;delayed=current.query_batch(lane,pair,threads=1,trace=True);same(baseline,delayed)
        need(delayed['phases_seconds']['initial_freshness_and_validation']>=lane.observed_delays[-1],
             'GRAPH_CLOCK_DELAY_UNACCOUNTED')
        no_trace=current.query_batch(lane,pair,threads=1);same(baseline,no_trace)
        need('intervals' not in no_trace['timing'],'GRAPH_CLOCK_DEFAULT_TRACE_STORAGE')
        negatives=timing_controls(delayed['timing'])
        negatives.update(legacy_delayed_initial=old_fault,threads=rejected(lambda:current.query_batch(lane,pair,threads=0)),
            duplicate=rejected(lambda:current.query_batch(lane,[pair[0]]*2,threads=1)),
            oversized=rejected(lambda:current.query_batch(lane,uids[:65].tolist(),threads=1)),
            legacy_singleton=rejected(lambda:legacy.query_batch(lane,[singles[0]],threads=1)))
        need('GRAPH_BATCH_SEARCH_LAYOUT' in negatives['legacy_singleton'],'GRAPH_CLOCK_LEGACY_SINGLETON_REASON')
        measurements=dict(records=records,singletons=singleton,baseline=baseline,delayed=delayed,
            actual_delays=lane.observed_delays,default_timing=no_trace['timing'],ordinary_seconds=ordinary_seconds,
            outer_seconds=outer_seconds,ordinary_coverage=ordinary_seconds/outer_seconds)
        need(ordinary_seconds/outer_seconds>=.9,'GRAPH_CLOCK_OUTER_ACCOUNTING')
        windows=int(index.lengths.sum());memory=index.report['index_memory_bytes'];lane=None;index=None;gc.collect()
        # Alter only an owned negative-control copy. The immutable original index remains intact.
        corrupt=out/'changed-index';shutil.copytree(Path(ref['path']),corrupt)
        altered=NativeWindowIndex(corrupt,ref['pin'],ib);guarded=BoundedNativeLane(altered,native_path)
        with (corrupt/'windows.usearch').open('ab') as f:f.write(b'negative-control')
        negatives['changed_index']=rejected(lambda:current.query_batch(guarded,pair,threads=1))
        guarded=None;altered=None;gc.collect()
        write_committed(out/'measurements.json',measurements)
        numerical_pin=write_committed(out/'numeric.json',numeric)
        results[name]=dict(inputs=len(uids),native_windows=windows,index_memory_bytes=memory,queries=2*len(uids)+2,
            numerical_pin=numerical_pin,numpy_score_error=largest,controls=negatives,all_indexed_windows_exact=True,
            ranks_scores_candidates_work_exact=True,singleton_and_multiwindow_exact=True)
        runs[name]=dict(ordinary_seconds=ordinary_seconds,outer_seconds=outer_seconds,delay_seconds=measurements['actual_delays'])
        phases[name]=time.perf_counter()-tick;print(json.dumps(dict(instrument=name,windows=windows,seconds=phases[name])),flush=True)
    tick=time.perf_counter();need(not torch.cuda.is_initialized(),'GRAPH_CLOCK_UNEXPECTED_CUDA')
    indexes_pin=write_committed(a.output/'indexes.json',indexes);result_pin=write_committed(a.output/'results.json',results)
    if previous:need(previous['results_pin']==result_pin and previous['indexes_pin']==indexes_pin,'GRAPH_CLOCK_COLD_EXACT')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['readback']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'GRAPH_CLOCK_PHASE_ACCOUNTING')
    report=dict(complete=True,binding=binding,files=files,indexes_pin=indexes_pin,results_pin=result_pin,previous_pin=a.previous_pin,
        phase_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        measurements=runs,model_updates=0,new_encoder_calls=0,full_gallery_qualified=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    args=parser.parse_args();need(bool(args.previous)==bool(args.previous_pin),'GRAPH_CLOCK_PREVIOUS_ARGUMENTS');run(args)

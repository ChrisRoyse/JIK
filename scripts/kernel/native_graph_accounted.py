"""Native graph batches with adjacent measured phase boundaries and singleton layout."""
import math
import time

import numpy as np

from kernel.native_search_layout import batch_layout
from kernel.native_window_index import POLICY,score_table,unit
from text.constellation_corpus import need


def validate_timing(r):
    need(isinstance(r,dict) and {'seconds','phases_seconds','phase_boundaries','clock'}<=set(r),
         'GRAPH_TIMING_FIELDS')
    seconds=r['seconds'];phases=r['phases_seconds'];counts=r['phase_boundaries']
    number=lambda x:type(x) in (int,float) and math.isfinite(x) and x>=0
    need(number(seconds) and seconds>0 and r['clock']=='perf_counter'
         and isinstance(phases,dict) and isinstance(counts,dict) and phases and set(phases)==set(counts)
         and all(isinstance(k,str) and k and number(v) for k,v in phases.items())
         and all(type(v) is int and v>0 for v in counts.values()),'GRAPH_TIMING_VALUES')
    # Counts bound accumulated floating-point addition error; they do not relax coverage.
    tolerance=8*math.ulp(max(1.,seconds))*sum(counts.values())
    total=math.fsum(phases.values())
    need(total/seconds>=.9,'GRAPH_BATCH_PHASE_ATTRIBUTION')
    need(abs(total-seconds)<=tolerance,'GRAPH_TIMING_PHASE_SUM')
    if 'intervals' in r:
        intervals=r['intervals'];need(isinstance(intervals,list) and intervals,'GRAPH_TIMING_TRACE')
        previous=0.;observed={};seen={}
        for row in intervals:
            need(isinstance(row,dict) and set(row)=={'phase','start','stop'},'GRAPH_TIMING_INTERVAL_FIELDS')
            name,start,stop=row['phase'],row['start'],row['stop']
            need(name in phases and number(start) and number(stop) and start==previous and stop>=start,
                 'GRAPH_TIMING_INTERVAL_ORDER')
            observed[name]=observed.get(name,0.)+stop-start;seen[name]=seen.get(name,0)+1;previous=stop
        need(previous==seconds and seen==counts and all(abs(observed[k]-phases[k])<=tolerance for k in phases),
             'GRAPH_TIMING_TRACE_SUM')


class PhaseClock:
    def __init__(self,trace):
        self.started=time.perf_counter();self.last=0.;self.phases={};self.counts={}
        self.intervals=[] if trace else None

    def mark(self,name):
        stop=time.perf_counter()-self.started
        need(math.isfinite(stop) and stop>=self.last,'GRAPH_TIMING_CLOCK')
        self.phases[name]=self.phases.get(name,0.)+stop-self.last
        self.counts[name]=self.counts.get(name,0)+1
        if self.intervals is not None:self.intervals.append(dict(phase=name,start=self.last,stop=stop))
        self.last=stop

    def finish(self):
        self.mark('timing_bookkeeping')
        r=dict(seconds=self.last,phases_seconds=self.phases,phase_boundaries=self.counts,clock='perf_counter')
        if self.intervals is not None:r['intervals']=self.intervals
        validate_timing(r)
        return r


def query_batch(lane,uids,*,threads,trace=False):
    clock=PhaseClock(trace);cpu=time.process_time();lane.fresh()
    need(type(threads) is int and threads in (1,4,16,32),'GRAPH_BATCH_THREADS')
    need(isinstance(uids,list) and 0<len(uids)<=64
         and all(type(u) is int for u in uids) and len(set(uids))==len(uids),'GRAPH_BATCH_SCOPE')
    clock.mark('initial_freshness_and_validation')
    native=lane(uids)
    for row in native:unit(row)
    lengths=np.asarray([len(x) for x in native],dtype='<i8');bounds=np.r_[0,np.cumsum(lengths)]
    values=np.concatenate(native);clock.mark('source')
    count=min(POLICY['window_candidates'],len(lane.index.index))
    need(values.nbytes+len(values)*count*16<=64<<20,'GRAPH_BATCH_MEMORY')
    matches=lane.index.index.search(values,count=count,threads=threads);clock.mark('search')
    matches=batch_layout(matches,len(values),count)
    work=dict(native_windows=len(values),queries=len(uids),computed_distances=int(matches.computed_distances),
              visited_members=int(matches.visited_members),scope='whole_native_window_batch',per_query=None)
    need(work['computed_distances']>0,'GRAPH_BATCH_DISTANCE_WORK')
    owners=(matches.keys>>np.uint64(32)).astype('<i8');positions=np.searchsorted(lane.index.uids,owners)
    need(np.all(positions<len(lane.index.uids)) and np.array_equal(lane.index.uids[positions],owners),'GRAPH_BATCH_UNKNOWN_OWNER')
    need(np.all((matches.keys&np.uint64(2**32-1))<lane.index.lengths[positions]),'GRAPH_BATCH_UNKNOWN_WINDOW')
    candidates=[]
    for i,uid in enumerate(uids):
        ids=np.unique(owners[bounds[i]:bounds[i+1]]);ids=ids[ids!=uid].tolist()
        need(len(ids)>=POLICY['top_k'],'GRAPH_BATCH_SCARCITY');candidates.append(ids)
    clock.mark('candidate_owners');rows=[]
    for uid,query,ids in zip(uids,native,candidates):
        scored=[]
        for lo in range(0,len(ids),512):
            part=ids[lo:lo+512];clock.mark('candidate_bookkeeping')
            target=lane(part);clock.mark('candidate_vectors')
            scored.extend(zip(part,score_table([query],target)[0].tolist()));clock.mark('rerank')
        ranked=sorted(scored,key=lambda pair:(-pair[1],pair[0]))[:POLICY['top_k']]
        rows.append(dict(uid=uid,hits=[[u,s] for u,s in ranked],candidate_uids=ids,
                         exhaustive=False,relation='same_instrument_native_window_neighbour'))
        clock.mark('rerank')
    clock.mark('candidate_bookkeeping');lane.fresh();clock.mark('final_freshness')
    timing=clock.finish()
    return dict(rows=rows,work=work,phases_seconds=timing['phases_seconds'],seconds=timing['seconds'],
                cpu_seconds=time.process_time()-cpu,timing=timing)

"""Separate scalar-probability and case-table recount of the breadth comparison."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
import numpy as np
import torch
from audit import legal_breadth_comparison_source as source
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need


def scalar(row,subset):
    indices=np.flatnonzero(row['masks'][subset].numpy()).tolist();out={};truth=row['target'].numpy()
    for level in range(3):
        accuracy={};prefix={}
        for name,tensor in row['predictions'].items():
            if name=='root_transition' and level>0:continue
            values=tensor.numpy()
            accuracy[name]=sum(int(values[i,level])==int(truth[i,level]) for i in indices)
            prefix[name]=sum(all(int(values[i,j])==int(truth[i,j]) for j in range(level+1)) for i in indices)
        out[f'accuracy/{level}']=(len(indices),accuracy);out[f'prefix_accuracy/{level}']=(len(indices),prefix)
        probs=row['probabilities'][level].numpy();other=row['permuted_probabilities'][level].numpy()
        ce=[];bs=[];gap=[]
        for i in indices:
            y=int(truth[i,level]);loss=-math.log(float(probs[i,y]));ce.append(loss)
            delta=probs[i].copy();delta[y]-=1;bs.append(math.fsum(float(x*x) for x in delta))
            if row['masks']['permuted_target_distinct'][i]:
                q=int(row['permutation'][i]);gap.append(-math.log(float(other[i,int(truth[q,level])]))-loss)
        out[f'conditional_ce/{level}']=(len(indices),dict(model=math.fsum(ce)))
        out[f'conditional_brier/{level}']=(len(indices),dict(model=math.fsum(bs)))
        out[f'target_permutation_gap/{level}']=(len(gap),dict(model=math.fsum(gap)))
    for name,entry in row['native'].items():
        values=entry['ordered_distances'].numpy()
        out['ordered_distance/'+name]=(len(indices),{
            key:math.fsum(float(values[i,j]) for i in indices) for j,key in enumerate(('model','copy_current','train_mean'))})
        out['count_accuracy/'+name]=(len(indices),dict(model=sum(
            int(entry['predicted_count'][i])==int(entry['target_count'][i]) for i in indices)))
    return out


def near(a,b):
    need(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b)<=5e-10,'BREADTH_COLD_NUMBER')


def percentile(values,q):
    ordered=sorted(values);position=(len(ordered)-1)*q;left=int(position);weight=position-left
    return ordered[left]*(1-weight)+ordered[min(left+1,len(ordered)-1)]*weight


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    report=json.loads(source.raw(a.input/'report.json',a.pin))
    need(report['complete'] and report['code']==source.code(),'BREADTH_COLD_CODE')
    for n,h in report['files'].items():need(file_sha(a.input/n)==h,'BREADTH_COLD_FILE')
    s=source.Source(a.root);records=[json.loads(line) for line in (a.input/'documents.jsonl').read_text().splitlines()]
    indexed={(r['document'],r['subset'],r['metric']):r for r in records}
    need(len(indexed)==len(records)==15552,'BREADTH_COLD_RECORDS')
    phases['authority']=time.perf_counter()-tick;tick=time.perf_counter();checked=set()
    for selected in s.selection:
        left,right=s.pair(selected)
        for subset in source.SUBSETS:
            by_arm={arm:scalar(r,subset) for arm,r in [('breadth',left),('repeated',right)]}
            for metric in by_arm['breadth']:
                key=(left['document'],subset,metric);r=indexed[key];checked.add(key)
                need(r['case']==selected['case_component'] and r['source']==left['source_collection']
                     and r['side']==selected['side'],'BREADTH_COLD_BINDING')
                for arm,values in by_arm.items():
                    n,sums=values[metric];need(n==r['n'] and sums.keys()==r['sums'][arm].keys(),'BREADTH_COLD_SUPPORT')
                    for name,value in sums.items():near(value,r['sums'][arm][name])
    need(checked==set(indexed),'BREADTH_COLD_COVERAGE')
    phases['scalar_all_document_scores']=time.perf_counter()-tick;tick=time.perf_counter()
    groups=defaultdict(list)
    for r in records:
        for collection in (r['source'],'ALL_COLLECTIONS'):
            groups[(r['side'],collection,r['subset'],r['metric'])].append(r)
    cells=json.loads((a.input/'cells.json').read_bytes());seen=set();draw_cache={}
    for cell in cells:
        key=tuple(cell[k] for k in ('side','source','subset','metric'));need(key not in seen,'BREADTH_COLD_DUPLICATE_CELL');seen.add(key)
        all_rows=groups[key];rows=sorted((r for r in all_rows if r['n']),key=lambda r:r['document']);n=sum(r['n'] for r in rows)
        need(n==cell['rows'] and len(rows)==cell['documents'] and len(all_rows)-len(rows)==cell['excluded_empty_documents'],
             'BREADTH_COLD_CELL_SUPPORT')
        if not n:need(cell['status']=='unmeasured_empty_support','BREADTH_COLD_EMPTY');continue
        higher=key[-1].startswith(('accuracy/','prefix_accuracy/','count_accuracy/','target_permutation_gap/'))
        need(cell['higher_is_better']==higher,'BREADTH_COLD_DIRECTION');sign=1 if higher else -1
        for arm in ('breadth','repeated'):
            for name,value in cell['means'][arm].items():near(math.fsum(r['sums'][arm][name] for r in rows)/n,value)
            for name,counts in cell['baseline_comparison'][arm].items():
                diffs=[sign*(r['sums'][arm]['model']-r['sums'][arm][name]) for r in rows]
                need(counts['documents_better']==sum(x>0 for x in diffs) and counts['documents_worse']==sum(x<0 for x in diffs)
                     and counts['documents_tied']==sum(x==0 for x in diffs),'BREADTH_COLD_BASELINE_COUNTS')
                near(math.fsum(diffs)/n,counts['micro_gain'])
        deltas=[sign*(r['sums']['breadth']['model']-r['sums']['repeated']['model'])/r['n'] for r in rows]
        near(math.fsum(deltas)/len(rows),cell['breadth_gain_document_mean'])
        near(math.fsum(sign*(r['sums']['breadth']['model']-r['sums']['repeated']['model']) for r in rows)/n,cell['breadth_gain_micro'])
        need(cell['breadth_documents_better']==sum(x>0 for x in deltas)
             and cell['breadth_documents_worse']==sum(x<0 for x in deltas)
             and cell['breadth_documents_tied']==sum(x==0 for x in deltas),'BREADTH_COLD_PAIRED_COUNTS')
        name=cell['draws']
        if name not in draw_cache:
            import hashlib
            ids=[r['case'] for r in rows];digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            need(digest==name,'BREADTH_COLD_DRAW_ID')
            draws=np.load(a.input/'draws'/(name+'.npy'),allow_pickle=False)
            expected=np.random.default_rng(int(name[:16],16)).integers(len(rows),size=(2000,len(rows)),dtype=np.int64)
            need(np.array_equal(draws,expected),'BREADTH_COLD_DRAWS');draw_cache[name]=draws
        draws=draw_cache[name];samples=[math.fsum(deltas[int(i)] for i in draw)/len(draw) for draw in draws]
        if len(rows)>1:
            for value,q in zip(cell['breadth_document_interval'],(.025,.975)):near(value,percentile(samples,q))
        else:need(cell['breadth_document_interval'] is None,'BREADTH_COLD_SINGLE_CASE')
    need(seen==set(groups) and len(cells)==648,'BREADTH_COLD_CELL_CENSUS')
    phases['all_cells_and_intervals']=time.perf_counter()-tick;tick=time.perf_counter()
    sources=json.loads((a.input/'sources.json').read_bytes());need(sources['files']==s.files,'BREADTH_COLD_SOURCE_MANIFEST')
    s.fresh();need(source.code()==report['code'] and not torch.cuda.is_initialized(),'BREADTH_COLD_STATE')
    phases['source_fresh']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'BREADTH_COLD_PHASES')
    value=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),code=source.code(),documents=192,records=len(records),
        cells=len(cells),all_scalar_scores_and_intervals_recounted=True,seconds=seconds,phases=phases,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked',
        scientific_admission=False,model_promotion=False,native_forward_replayed_here=False)
    pin=write_committed(a.output/'report.json',value);print(json.dumps(dict(pin=pin,seconds=seconds,cells=len(cells))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

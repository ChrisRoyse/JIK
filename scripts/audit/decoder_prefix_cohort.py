"""Hash-selected legal decoder intervention cohort, before any model forward."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_surface_inversion as original
from audit import trained_answer_render as trained
from model.native_population_training import cpu,fingerprint
from model.trained_surface_decoder import TrainedSurfaceDecoder
from senses.contract import digest
from text.constellation_corpus import file_sha,need

SEED='decoder-prefix-v1-20261025'
SELECT=(('train',0,128,8),('development',1,16,4),('query',2,16,4))


def code():
    root=Path(__file__).resolve().parents[1]; result={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).resolve().is_relative_to(root):
            p=Path(file).resolve();result[str(p.relative_to(root))]=file_sha(p)
    return result


def selection(s):
    data=s['data'];old=original.select(data)
    excluded={int(data.components[d]) for docs in old.values() for d in docs}
    excluded.update(int(data.components[m['document']]) for m in s['metadata'])
    result=[];seen=set(excluded)
    for side,bit,count,per in SELECT:
        candidates=[]
        for d in np.flatnonzero(data.doc_sides==bit).tolist():
            a,z=map(int,data.documents[d]);component=int(data.components[d])
            if z-a>=per and component not in seen:
                candidates.append((digest(dict(seed=SEED,source=data.source.pin,component=component,document=d)),d))
        picked=0
        for _,d in sorted(candidates):
            component=int(data.components[d])
            if component in seen:continue
            a,z=map(int,data.documents[d]);seen.add(component);picked+=1
            atoms=sorted(sorted(range(a,z),key=lambda i:digest(dict(seed=SEED,document=d,occurrence=i)))[:per])
            result.extend(dict(side=side,document=d,component=component,occurrence=i) for i in atoms)
            if picked==count:break
        need(picked==count,'PREFIX_COHORT_COMPONENTS')
    return result,sorted(excluded)


def save(path,value):
    with path.open('xb') as f:torch.save(cpu(value),f);f.flush();os.fsync(f.fileno())
    actual=torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(actual)==fingerprint(value),'PREFIX_COHORT_SAVED_STATE')
    return dict(sha256=file_sha(path),state=fingerprint(actual))


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    s=original.sources(a.root);selected,excluded=selection(s)
    phases=dict(authentication_selection=time.perf_counter()-tick);tick=time.perf_counter()
    rows=[]
    try:
        for side,_,_,_ in SELECT:
            chosen=[r for r in selected if r['side']==side]
            for start in range(0,len(chosen),128):
                chunk=chosen[start:start+128]
                examples=s['adapter'].fetch([r['occurrence'] for r in chunk],side=side,fit=side=='train')
                for r,e in zip(chunk,examples):
                    rows.append(dict(r,uid=e.uid,raw=torch.tensor(list(e.raw),dtype=torch.uint8),
                        row=next(iter(e.native.values())).rows[0],
                        native={n:x.values[0].clone() for n,x in e.native.items()},
                        train_present=bool(s['continuation'].presence[e.uid]&1)))
        binding=s['adapter'].binding;initial_config=asdict(s['config']);specs=[asdict(x) for x in s['specs']]
    finally:original.close(s)
    phases['source_materialization']=time.perf_counter()-tick;tick=time.perf_counter()
    args=[]
    for key in ('full','plan','cold','final'):args.extend((a.root/trained.REFS[key][0],trained.REFS[key][1]))
    decoder=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True,retain_continuation=True)
    need(vars(decoder.model.config)==initial_config and [vars(s) for s in decoder.model.specs]==specs,'PREFIX_COHORT_MODEL')
    initial,step=decoder.export_continuation();decoder.fresh();decoder.close()
    need(step==51150,'PREFIX_COHORT_FULL_STEP')
    files={'observations.pt':save(a.output/'observations.pt',rows),'initial.pt':save(a.output/'initial.pt',initial)}
    census={}
    for side,_,count,per in SELECT:
        r=[x for x in rows if x['side']==side]
        census[side]=dict(atoms=len(r),unique_inputs=len({x['uid'] for x in r}),documents=len({x['document'] for x in r}),
            components=len({x['component'] for x in r}),bytes=sum(len(x['raw']) for x in r),
            positions=sum(len(x['raw'])+1 for x in r),prefix_rows=sum((len(x['raw'])+16)//16 for x in r),
            windows={n:sum(len(x['native'][n]) for x in r) for n in r[0]['native']},
            train_present=sum(x['train_present'] for x in r),reference_fits_96=sum(len(x['raw'])+1<=96 for x in r))
        need(len(r)==count*per and census[side]['components']==count,'PREFIX_COHORT_CENSUS')
    training=[r for r in rows if r['side']=='train'];groups={}
    for index,r in enumerate(training):groups.setdefault(tuple((n,tuple(v.shape)) for n,v in sorted(r['native'].items())),[]).append(index)
    donors=[]
    for r in rows:
        key=tuple((n,tuple(v.shape)) for n,v in sorted(r['native'].items()))
        eligible=[i for i in groups.get(key,[]) if training[i]['component']!=r['component'] and training[i]['uid']!=r['uid']]
        # Address matching is a control, not an input feature or outcome selection.
        donors.append(min(eligible,key=lambda i:digest(dict(seed=SEED,target=r['occurrence'],donor=training[i]['occurrence']))) if eligible else None)
    phases['initial_and_census']=time.perf_counter()-tick
    result=dict(complete=True,seed=SEED,selection=selected,excluded_components=excluded,binding=binding,
        code=code(),config=initial_config,instruments=specs,initial_step=step,trained_sources=trained.REFS,
        files=files,census=census,donor_train_indices=donors,missing_shape_matched_donors=sum(d is None for d in donors),
        no_forward=True,untouched_test=False,phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked only')
    with (a.output/'report.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(report=file_sha(a.output/'report.json'),census=census,missing_donors=result['missing_shape_matched_donors'],wall=result['wall'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():
            with (a.output/'failure.json').open('x') as f:json.dump(dict(error=type(e).__name__,message=str(e)),f)
        raise

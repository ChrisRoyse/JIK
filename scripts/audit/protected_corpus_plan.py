"""Reuse the full TRAIN index; materialize its first three real batches for qualification."""
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
from audit.decoder_prefix_cohort import code,save
from claims.process_conditions import write_committed
from model.contracts import ModelConfig,parameter_count
from model.protected_corpus_training import POLICY
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha,need

CAPACITY=('decoder-capacity-census-v1-20261025','c6ac3fc367e689aa550e54e6b760116a5e3e0b93b488778398b7c02ae45011f1')


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    name,pin=trained.REFS['plan'];path=a.root/name/'report.json'
    need(file_sha(path)==pin,'PROTECTED_PLAN_ORIGINAL_PIN');old=json.loads(path.read_bytes())
    train_path=a.root/name/'train.npy';need(file_sha(train_path)==old['files']['train.npy'],'PROTECTED_PLAN_TRAIN_PIN')
    ids=np.load(train_path,mmap_mode='r',allow_pickle=False)
    need(old['complete'] and old['train']==dict(atoms=6547176,bytes=669898601,prefix_rows=45357686)
         and len(ids)==old['train']['atoms'] and np.all(ids[1:]>ids[:-1]),'PROTECTED_PLAN_POPULATION')
    cp=a.root/CAPACITY[0]/'report.json';need(file_sha(cp)==CAPACITY[1],'PROTECTED_PLAN_CAPACITY_PIN')
    capacity=json.loads(cp.read_bytes());config=ModelConfig(**capacity['correction_config'])
    s=original.sources(a.root);rows=[]
    need(s['adapter'].binding==old['binding'] and all(file_sha(Path(__file__).resolve().parents[1]/n)==h for n,h in old['code'].items()),
         'PROTECTED_PLAN_ORIGINAL_BINDING')
    need(bool((s['data'].sides[ids]==0).all()),'PROTECTED_PLAN_ALL_TRAIN')
    selected=[dict(side='train',occurrence=int(i)) for i in ids[:384]]
    selected += [dict(side=r['side'],occurrence=r['occurrence']) for r in capacity['selected'] if r['side']!='train']
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    try:
        for side in ('train','development','query'):
            group=[r for r in selected if r['side']==side]
            for offset in range(0,len(group),128):
                chunk=group[offset:offset+128]
                examples=s['adapter'].fetch([r['occurrence'] for r in chunk],side=side,fit=side=='train')
                for r,e in zip(chunk,examples):
                    document=int(np.searchsorted(s['data'].documents[:,1],e.occurrence,side='right'))
                    rows.append(dict(r,document=document,component=int(s['data'].components[document]),uid=e.uid,
                        raw=torch.tensor(list(e.raw),dtype=torch.uint8),row=next(iter(e.native.values())).rows[0],
                        native={n:x.values[0].clone() for n,x in e.native.items()},train_present=bool(s['continuation'].presence[e.uid]&1)))
        need(parameter_count(config,s['specs'])==78325042,'PROTECTED_PLAN_PARAMETERS')
        binding=s['adapter'].binding;specs=[asdict(x) for x in s['specs']]
    finally:original.close(s)
    phases['real_source_materialization']=time.perf_counter()-tick;tick=time.perf_counter()
    args=[]
    for key in ('full','plan','cold','final'):args.extend((a.root/trained.REFS[key][0],trained.REFS[key][1]))
    parent=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True)
    parent_receipt=parent.receipt;parent.fresh();parent.close()
    files={'observations.pt':save(a.output/'observations.pt',rows)}
    census={}
    for side in ('train','development','query'):
        values=[r for r in rows if r['side']==side]
        census[side]=dict(atoms=len(values),bytes=sum(len(r['raw']) for r in values),
            positions=sum(len(r['raw'])+1 for r in values),prefix_rows=sum((len(r['raw'])+16)//16 for r in values),
            documents=len({r['document'] for r in values}),components=len({r['component'] for r in values}),
            windows={n:sum(len(r['native'][n]) for r in values) for n in values[0]['native']})
    phases['persistence_census']=time.perf_counter()-tick
    r=dict(complete=True,schema='protected-corpus-plan-v1',policy=POLICY,config=asdict(config),instruments=specs,
           parameters=78325042,original_plan=dict(path=str(a.root/name),pin=pin),
           train_index=dict(path=str(train_path),pin=old['files']['train.npy']),train=old['train'],updates=old['updates'],
           evaluation=old['evaluation'],binding=binding,parent=parent_receipt,capacity=CAPACITY[1],census=census,
           files=files,code=code(),no_forward=True,untouched_test=False,phases=phases,
           wall=time.perf_counter()-began,cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
           acceptance='author-checked',full_training=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',r);print(json.dumps(dict(pin=pin,census=census,wall=r['wall'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

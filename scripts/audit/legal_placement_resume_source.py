"""Reconstruct the live checkpoint's document/window cursor from sealed native metadata."""
import argparse
from copy import deepcopy
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
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as worker
from audit import legal_efficient_predictor_training as existing
from audit.legal_grouped_io_authority import authenticate
from claims.process_conditions import write_committed
from model.legal_placement_resume import Census, recount
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


INSPECTION=('live-placement-checkpoint-v2-20261024','7d06667582f44fd8454998b3d7b648eb291a144ad3a303fb2a1b36ffe231c2a5')


def run(a):
    began=tick=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    inspection_root=a.root/INSPECTION[0]; inspection=worker.pinned(inspection_root,INSPECTION[1])
    worker.physical(inspection_root,inspection)
    saved=torch.load(inspection_root/'checkpoint.pt',map_location='cpu',weights_only=True)
    job=Path(inspection['job']); initial_path=job/'shared/checkpoint-0.pt'
    need(file_sha(initial_path)==inspection['initial_checkpoint_pin'],'PLACEMENT_RESUME_FULL_INITIAL_PIN')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    for name,pin in saved['binding']['code'].items():
        need(file_sha(Path(__file__).resolve().parents[1]/name)==pin
             and file_sha(job/'source'/name)==pin,'PLACEMENT_RESUME_FULL_SOURCE')
    source,bundle=worker.source_bundle(a.root)
    need(worker.code(source)==saved['binding']['code'],'PLACEMENT_RESUME_FULL_EXECUTION')
    args=argparse.Namespace(root=a.root,scope='full',arm='breadth')
    for name,(path,pin) in worker.REFS.items():setattr(args,name,a.root/path);setattr(args,name+'_pin',pin)
    data,continuation,reader,labels,cache,meta=existing.breadth.prepare(args)
    # This path counts immutable metadata only. It never consumes native vectors.
    reader.close();schedule,receipt=existing.load_schedule(args,data,cache,meta)
    plan=existing.selection(args,data,meta,schedule);io,io_fresh=authenticate(a.root)
    binding=dict(source=data.binding,cache=cache.binding,metadata=meta.binding,
                 schedule=worker.REFS['schedule'][1],plan=plan,io=io)
    need(binding==saved['binding']['source'],'PLACEMENT_RESUME_FULL_DATA_BINDING')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    documents=plan['train_documents'];n=len(documents);roster=tuple(data.roster)
    targets=np.empty(n,dtype=np.int64);windows=np.empty((n,len(roster)),dtype=np.int64)
    for cursor,number in enumerate(documents):
        start,stop=map(int,data.documents[number]);uids=data.mapping[start:stop]
        counts=np.asarray(cache.maps[1][uids])
        need(int(data.doc_sides[number])==0 and counts.dtype==np.uint8
             and counts.shape==(stop-start,len(roster)) and bool((counts>0).all()),'PLACEMENT_RESUME_FULL_COUNTS')
        targets[cursor]=stop-start-1;windows[cursor]=counts.sum(axis=0,dtype=np.int64)
    census=Census(documents,targets,windows,roster);total=census.counts(n)
    need(n==receipt['summary']['documents'] and total['rows']==receipt['summary']['targets']
         and total['windows']==receipt['summary']['native_windows'],'PLACEMENT_RESUME_FULL_TOTAL')
    ref=worker.measure.persist(a.output/'census.pt',census.value())
    loaded=Census.load(torch.load(a.output/'census.pt',map_location='cpu',weights_only=True))
    need(loaded.pin==census.pin,'PLACEMENT_RESUME_FULL_CENSUS_READBACK')
    phases['full_metadata_census']=time.perf_counter()-tick;tick=time.perf_counter()
    prefix=inspection_root/'steps-prefix.jsonl'
    imported=recount(prefix,saved,initial,loaded)
    # Recount the prefix in reverse source order, lane by lane, independently
    # of the producer's per-document cumsums. No frozen model is invoked.
    reverse={name:0 for name in roster};rows=0
    for number in reversed(documents[:saved['step']]):
        start,stop=map(int,data.documents[number]);uids=data.mapping[start:stop];rows+=stop-start-1
        counts=cache.maps[1][uids]
        for index,name in enumerate(roster):reverse[name]+=int(np.add.reduce(counts[:,index],dtype=np.uint64))
    need(rows==imported['rows'] and reverse==imported['windows'],'PLACEMENT_RESUME_FULL_REVERSE_CENSUS')
    for key in ('step','rows','chain','prefix_sha256','prefix_bytes','model','optimizer','next_document'):
        need(imported[key]==inspection['census'][key],'PLACEMENT_RESUME_FULL_PRIOR_CENSUS')
    phases['prefix_and_independent_reverse']=time.perf_counter()-tick;tick=time.perf_counter()
    target=deepcopy(labels.model).eval().requires_grad_(False)
    model,optimizer=worker.runtime.make(bundle,target,'shared','cpu')
    reference=dict(pin=inspection['checkpoint_pin'],state=fingerprint(saved))
    worker.restore(inspection_root/'checkpoint.pt',reference,model,optimizer,saved['binding'],saved['step'],
                   saved['rows'],saved['chain'],saved['binding']['policy'])
    comparison=rd.exact(dict(model=dict(model.state_dict()),optimizer=optimizer.state_dict()),
                        dict(model=saved['model'],optimizer=saved['optimizer']))
    broken=deepcopy(saved);broken['rows']+=1
    try:recount(prefix,broken,initial,loaded)
    except ValueError as error:
        need('CHECKPOINT_CURSOR' in str(error),'PLACEMENT_RESUME_FULL_NEGATIVE_CODE');negative=str(error)
    else:raise ValueError('PLACEMENT_RESUME_FULL_NEGATIVE_ACCEPTED')
    cache.close();continuation.fresh();meta.fresh();labels.fresh();io_fresh();data.fresh()
    need(not torch.cuda.is_initialized(),'PLACEMENT_RESUME_FULL_GPU')
    phases['restore_controls_and_freshness']=time.perf_counter()-tick;tick=time.perf_counter()
    code={name:file_sha(Path(__file__).resolve().parents[1]/name) for name in
          ('model/legal_placement_resume.py','audit/legal_placement_resume_source.py')}
    for name,pin in code.items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(dest)==pin,'PLACEMENT_RESUME_FULL_CODE_COPY')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['publication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'PLACEMENT_RESUME_FULL_PHASES')
    report=dict(complete=True,inspection=INSPECTION[1],binding=binding,census=census.pin,census_file=ref,
        total=total,imported=imported,comparison=comparison,negative=negative,files=files,code=code,
        seconds=seconds,phases=phases,cpu_seconds=time.process_time()-cpu_start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_encoder_calls=0,optimizer_updates=0,gpu_used=False,live_processes_changed=False,
        cuda_continuation_qualified=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),total=total,imported=imported)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

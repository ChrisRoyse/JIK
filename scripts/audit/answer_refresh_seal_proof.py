"""Exercise the live sealer's branch algorithm on the original real small experiment."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_refresh_seal as producer
from audit import answer_refresh_seal_readback as reader
from audit import legal_grouped_placement_training as original
from audit import captured_answer_readback as rd
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model.legal_placement_resume import Census
from text.constellation_corpus import file_sha,need

REFERENCE=('placement-driver-original-v1-20261024','af334059da30b3a6cf1961b97c4c3b838fb00aeb8d2762a80f68c54a6202ce40')


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    job=a.root/REFERENCE[0];r=original.pinned(job,REFERENCE[1]);original.physical(job,r)
    _,bundle=original.source_bundle(a.root);data=original.load_data(argparse.Namespace(root=a.root,scope='small'),bundle)
    census=producer.driver.load_census(argparse.Namespace(root=a.root,scope='small'),data)
    totals=dict(bytes=0,tensors=0);results={};pins={};cuts={}
    for n,step in (('shared',5),('own_slot',2)):
        b,p,cut=producer.branch(job,a.output,n,step,r['binding'],census,data['items'],data['labels'].roster)
        results[n]=reader.inspect(a.output,job,n,b,census,cut);pins.update(p);cuts[n]=cut
        for rel in p:
            source=Path(rel);dest=a.output/source.relative_to(job)
            if source.suffix=='.pt':
                v=rd.exact(torch.load(source,map_location='cpu',weights_only=True),torch.load(dest,map_location='cpu',weights_only=True))
                for k in totals:totals[k]+=v[k]
        controls={}
        def reject(label,branch,expected):
            try:reader.inspect(a.output,job,n,branch,census,cut)
            except ValueError as e:need(expected in str(e),'ANSWER_REFRESH_SEAL_PROOF_REFUSAL');controls[label]=str(e)
            else:raise ValueError('ANSWER_REFRESH_SEAL_PROOF_ACCEPTED:'+label)
        bad=deepcopy(b);bad['tail']['end']['step']+=1;reject('changed_tail_cursor',bad,'BACK_END')
        bad=deepcopy(b);bad['counts']['prefix_sha256']='0'*64;reject('changed_prefix',bad,'BACK_SOURCE_PARTS')
        if n=='shared':
            bad=deepcopy(b);bad['completed']['files'][next(iter(bad['completed']['files']))]='0'*64
            reject('changed_completed_file',bad,'BACK_COMPLETED_BYTES')
        results[n]['controls']=controls
    need(results['shared']['checkpoint']==results['shared']['observed_step']==5 and results['shared']['completed']
         and results['own_slot']['checkpoint']==2 and results['own_slot']['observed_step']==5
         and not (a.output/'whole_proposals').exists(),'ANSWER_REFRESH_SEAL_PROOF_SCOPE')
    need(all(file_sha(p)==h for p,h in pins.items()),'ANSWER_REFRESH_SEAL_PROOF_INPUT_CHANGED')
    data['close']();need(not torch.cuda.is_initialized(),'ANSWER_REFRESH_SEAL_PROOF_GPU')
    v=dict(complete=True,roles=['completed','partial','fresh'],reference=REFERENCE,comparison=totals,results=results,
        source=file_sha(producer.__file__),reader=file_sha(reader.__file__),proof_source=file_sha(__file__),
        source_pins=pins,cuts=cuts,gpu_used=False,predictor_forwards=0,signals_sent=0,
        seconds=time.perf_counter()-began,files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()},
        acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',v),seconds=v['seconds'],comparison=totals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.contract.ROOT)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

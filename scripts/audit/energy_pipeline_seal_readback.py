"""Independent journal, checkpoint and original-byte readback of a three-owner seal."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import energy_pipeline_contract as contract
from audit.answer_refresh_seal_readback import inspect
from audit.legal_placement_successor_seal_readback import compare,digest,owners
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    m,census=publication.boundary(a.input,a.pin,execution=publication.code())
    ref=m['owner_authority'];cfg,_,_,fresh=contract.authenticate(Path(ref['path']),ref['pin'])
    need(m['owners']==list(cfg['owners'].values()) and len(m['owners'])==3 and m['retirement_ready']
         and m['source_job']==str(contract.SOURCE),'ENERGY_PIPELINE_BACK_SCOPE')
    before=owners(cfg,True);physical=0
    for path,h in m['source_pins'].items():
        need(file_sha(path)==h,'ENERGY_PIPELINE_BACK_ORIGINAL');physical+=Path(path).stat().st_size
    rows={n:inspect(a.input,contract.SOURCE,n,b,census,m['journal_cuts'][n]) for n,b in m['branches'].items() if not b.get('fresh')}
    q=m['quiescence'];need(file_sha(Path(q['path'])/'contract.json')==q['pin']
        and not (Path(q['path'])/'decision.json').exists(),'ENERGY_PIPELINE_BACK_QUIESCENCE')
    after=owners(cfg,True);fresh();need(not torch.cuda.is_initialized(),'ENERGY_PIPELINE_BACK_GPU')
    r=dict(complete=True,boundary=a.pin,authority=ref['pin'],source=file_sha(__file__),results=rows,
        all_checkpoint_bytes_exact=True,committed_prefix_exact=True,retained_tail_bytes_exact=True,
        completed_branch_bytes_exact=True,owner_closure_complete=True,retirement_ready=True,quiescence=q,
        before=before,after=after,original_bytes=physical,gpu_used=False,optimizer_updates=0,
        seconds=time.perf_counter()-began,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),seconds=r['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise

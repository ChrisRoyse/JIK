"""Exercise real write and fsync failures after a restored placement update."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import batched_continuation_proof as proof
from audit import legal_grouped_placement_training as original
from claims.process_conditions import write_committed
from model.batched_placement_continuation import BatchedPlacementContinuation
from model.legal_placement_resume import Census,recount
from model.native_population_training import fingerprint
from model.native_rq_packed import PackedNativeRQ
from model import placement_resume_tail as tail
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    previous=original.pinned(a.input,a.pin);original.physical(a.input,previous)
    cold=original.pinned(a.cold,a.cold_pin)
    need(previous['complete'] and cold['complete'] and cold['cold_readback_exact']
         and cold['previous_pin']==a.pin and cold['code']==previous['code'] and not cold['gpu_used'],
         'COMMIT_FAILURE_QUALIFICATION')
    base=Path(__file__).resolve().parents[1];code=dict(previous['code'])
    code['audit/batched_commit_failure_proof.py']=file_sha(__file__)
    need(all(file_sha(base/n)==h for n,h in code.items()),'COMMIT_FAILURE_SOURCE')
    _,bundle=original.source_bundle(a.root)
    target=PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],
                          'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True);target.eval().requires_grad_(False)
    # Reuse the already authenticated real census, never fabricate a training row.
    manifest,census=proof.publication.boundary(a.root/proof.BOUNDARY[0],proof.BOUNDARY[1])
    items={i['document']:i for i in bundle['items'] if i['side']=='train'};results={}
    for name in original.runtime.original.ARCHITECTURES:
        src=a.input/name/'checkpoint-3.pt';ref=previous['results'][name]['checkpoints']['3']
        need(file_sha(src)==ref['pin'],'COMMIT_FAILURE_CHECKPOINT')
        saved=torch.load(src,map_location='cpu',weights_only=True)
        initial=torch.load(a.root/proof.BOUNDARY[0]/name/'checkpoint-0.pt',map_location='cpu',weights_only=True)
        branch=saved['binding'];results[name]={}
        for mode in ('write','fsync'):
            out=a.output/name/mode;out.mkdir(parents=True);prefix=out/'steps.jsonl'
            counts=recount(a.input/name/'steps.jsonl',saved,initial,census,copy_to=prefix,allow_suffix=True)
            before=file_sha(prefix);full=a.root/proof.REFERENCE[0]/name/'steps.jsonl'
            witness=tail.seal(full,out,counts,census,branch,full.stat().st_size)
            engine=BatchedPlacementContinuation(bundle,target,branch,census)
            engine.restore(src,ref,prefix,initial,cursor=3,chain=saved['chain'])
            expected=(out/'retained-tail.jsonl').read_bytes().splitlines(keepends=True)[0]
            read_fd=None;captured=None
            if mode=='write':stream=open('/dev/full','wb',buffering=0);expected_errno=errno.ENOSPC
            else:
                read_fd,write_fd=os.pipe();stream=os.fdopen(write_fd,'wb',buffering=0);expected_errno=errno.EINVAL
                need(len(expected)<=fcntl.fcntl(write_fd,fcntl.F_GETPIPE_SZ),'COMMIT_FAILURE_PIPE_CAPACITY')
            try:
                with stream,tail.Replay(out/'retained-tail.jsonl',witness,3) as replay:
                    try:engine.commit(items[census.documents[3]],stream,replay)
                    except OSError as exc:
                        need(exc.errno==expected_errno,'COMMIT_FAILURE_WRONG_ERRNO');error=dict(type=type(exc).__name__,errno=exc.errno)
                    else:raise ValueError('COMMIT_FAILURE_ACCEPTED')
                    if read_fd is not None:captured=os.read(read_fd,len(expected))
            finally:
                if read_fd is not None:os.close(read_fd)
            need(file_sha(prefix)==before and file_sha(src)==ref['pin'] and engine.poisoned,'COMMIT_FAILURE_MUTATION')
            poison=proof.reject(engine.snapshot,'RESUME_POISONED')
            need(engine.step==4,'COMMIT_FAILURE_DID_NOT_EXECUTE_REAL_UPDATE')
            if mode=='fsync':
                need(captured==expected,'COMMIT_FAILURE_TRANSIENT_BYTES');(out/'transient-not-committed.bin').write_bytes(captured)
            results[name][mode]=dict(error=error,poison=poison,restored_step=3,attempted_step=4,
                committed_step=3,prefix=before,source_checkpoint=ref,transient_bytes=len(captured or b''),
                prefix_unchanged=True,checkpoint_unchanged=True)
    need(not torch.cuda.is_initialized() and fingerprint(target.state_dict())==bundle['target_fingerprint'],
         'COMMIT_FAILURE_FROZEN_RUNTIME')
    need(all(file_sha(base/n)==h for n,h in code.items()),'COMMIT_FAILURE_SOURCE_CHANGED')
    report=dict(complete=True,kind='batched-commit-physical-io-failure-proof-v1',input=a.pin,cold=a.cold_pin,
        code=code,results=results,real_update_attempts=6,committed_new_updates=0,gpu_used=False,
        process_signals_sent=0,production_promoted=False,seconds=time.perf_counter()-began,
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()},
        acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=report['seconds'],
                          attempts=6,committed=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','input','cold','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--cold-pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
        raise

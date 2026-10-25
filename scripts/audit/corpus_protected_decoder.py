"""Author-check the corpus decoder loader against persisted, original arithmetic."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import protected_corpus_program as original
from claims.process_conditions import write_committed as write
from model import protected_corpus_training as training
from model.contracts import ModelConfig
from model.corpus_protected_decoder import CorpusProtectedDecoder,scope
from model.native_population_training import fingerprint
from model.surface_device_runtime import batches
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha,need


def read(path,pin):
    need(file_sha(path)==pin,'CORPUS_LOADER_AUDIT_PIN')
    return json.loads(path.read_bytes())


def load(a):
    p=read(a.plan/'report.json',a.plan_pin);r=read(a.run/'report.json',a.run_pin)
    c=read(a.readback/'report.json',a.readback_pin)
    need(not r['full_training'] and r['cpu_proof'],'CORPUS_LOADER_AUDIT_CPU_SCOPE')
    args=[]
    for key,path in original.PARENT_PATHS.items():args.extend((a.root/path,p['parent'][key]))
    parent=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True)
    values={}
    for path,ref in ((a.plan/'observations.pt',p['files']['observations.pt']),
                     (a.run/'final-evaluation.pt',r['files']['final-evaluation.pt'])):
        need(file_sha(path)==ref['sha256'],'CORPUS_LOADER_AUDIT_SOURCE_FILE')
        v=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(v)==ref['state'],'CORPUS_LOADER_AUDIT_SOURCE_STATE');values[path.name]=v
    return p,r,c,parent,values


def negatives(a,p,r,c,parent,decoder):
    results={}
    def reject(name,fn,code):results[name]=original.rejected(fn,code)
    def check(pp,rr,cc,full=False):
        return scope(parent.receipt,pp,rr,cc,plan_pin=a.plan_pin,run_pin=a.run_pin,full=full,device='cpu')
    cases=[('plan','plan','CORPUS_DECODER_READBACK_BINDING'),
           ('cursor','cursor','CORPUS_DECODER_INCOMPLETE_TRAINING'),
           ('runtime','runtime','CORPUS_DECODER_RUNTIME'),
           ('parent','parent','CORPUS_DECODER_PARENT')]
    for name,key,code in cases:
        bad=deepcopy(r)
        if key=='cursor':bad[key]-=1
        elif key=='parent':bad[key]['model']='0'*64
        elif key=='runtime':bad[key]['threads']+=1
        else:bad[key]='0'*64
        reject(name,lambda:check(p,bad,c),code)
    bad=deepcopy(c);bad['evaluation']['independently_replayed']=False
    reject('unread_evaluation',lambda:check(p,r,bad),'CORPUS_DECODER_READBACK')
    reject('proof_claimed_as_full',lambda:check(p,r,c,True),'CORPUS_DECODER_TRAINING_SCOPE')
    bad=deepcopy(r);bad['checkpoints'][-1]['state']='0'*64
    reject('wrong_final_state',lambda:check(p,bad,c),'CORPUS_DECODER_FINAL_REFERENCE')
    # Only new evidence is corrupted; the source checkpoint remains immutable.
    root=a.output/'missing-state';root.mkdir()
    raw=(Path(r['checkpoints'][-1]['root'])/'manifest.json').read_bytes()
    with (root/'manifest.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    ref=r['checkpoints'][-1]
    from model.protected_surface_state import read as state_read
    binding=training.progress(r['binding'],ref['cursor'],ref['step'],ref['ledger'],ref['size'])
    reject('missing_physical_shard',lambda:state_read(root,ref['pin'],binding=binding,parent=parent.receipt['model']),
           'PROTECTED_STATE_SHARD_PIN')
    # A no-op in-place write still invalidates the version proof.
    with torch.no_grad():next(decoder.model.correction.parameters()).add_(0)
    reject('changed_weight_version',decoder.fresh,'CORPUS_DECODER_WEIGHTS_CHANGED')
    need(fingerprint(decoder.model.correction.state_dict())==ref['after'],'CORPUS_LOADER_AUDIT_CONTROL_CHANGED_VALUE')
    decoder.close();reject('closed',decoder.fresh,'CORPUS_DECODER_CLOSED')
    return results


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(exist_ok=False)
    need(a.cpu_proof,'CORPUS_LOADER_AUDIT_EXPLICIT_CPU_PROOF')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p,r,c,parent,source=load(a);decoder=None;opt=None
    base=Path(__file__).resolve().parents[1]
    code=dict(r['implementation'])
    code.update({n:file_sha(base/n) for n in ('model/corpus_protected_decoder.py','audit/corpus_protected_decoder.py')})
    need(all(file_sha(base/n)==h for n,h in code.items()),'CORPUS_LOADER_AUDIT_IMPLEMENTATION')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    parent_before=fingerprint(parent.model.state_dict());rng=torch.get_rng_state().clone()
    try:
        previous=None
        if a.mode=='produce':
            decoder=CorpusProtectedDecoder(parent,plan=a.plan,plan_pin=a.plan_pin,run=a.run,run_pin=a.run_pin,
                readback=a.readback,readback_pin=a.readback_pin,full=False)
            need(torch.equal(rng,torch.get_rng_state()),'CORPUS_LOADER_AUDIT_RNG')
            model=decoder.model;receipt=decoder.receipt;consumed=decoder.consumed
        else:
            need(a.input and a.pin,'CORPUS_LOADER_AUDIT_INPUT')
            previous=read(a.input/'report.json',a.pin)
            need(previous['complete'] and previous['mode']=='produce' and previous['code']==code
                 and previous['plan']==a.plan_pin and previous['run']==a.run_pin and previous['readback']==a.readback_pin,
                 'CORPUS_LOADER_AUDIT_READBACK_BINDING')
            model,opt=training.create(parent.model,ModelConfig(**p['config']))
            ref=r['checkpoints'][-1]
            need(training.restore(ref,model,opt,r['binding'],size=384)==(384,3,r['ledger']),
                 'CORPUS_LOADER_AUDIT_ORIGINAL_RESTORE')
            need(fingerprint(model.correction.state_dict())==ref['after'] and fingerprint(opt.state_dict())==ref['optimizer'],
                 'CORPUS_LOADER_AUDIT_ORIGINAL_STATE')
            del opt;opt=None;gc.collect();model.eval().requires_grad_(False)
            receipt=previous['decoder'];consumed=previous['consumed']
            for path,pin in consumed.items():need(file_sha(Path(path))==pin,'CORPUS_LOADER_AUDIT_CONSUMED')
        phases['restore']=time.perf_counter()-tick;tick=time.perf_counter()
        selected=original.examples({'rows':source['observations.pt']},model,'development')
        selected+=original.examples({'rows':source['observations.pt']},model,'query')
        need(len(selected)==4 and all(e.side!='train' for e in selected),'CORPUS_LOADER_AUDIT_HELD_SCOPE')
        expected=source['final-evaluation.pt']['teacher'];files=[];positions=0;logit_elements=0
        covered=[[] for _ in selected]
        with torch.no_grad():
            for ordinal,b in enumerate(batches(selected,model.config,model.specs,'gathered',128)):
                logits=model(b.prefix,b.native,offsets=b.offsets)['logits'].detach()
                v=dict(records=b.records,prefix=b.prefix,labels=b.labels,offsets=b.offsets,logits=logits)
                if a.mode=='produce':
                    name=f'batch-{ordinal:04d}.pt';ref=original.save(a.output/name,v);files.append(dict(name=name,**ref))
                else:
                    ref=previous['files'][ordinal];path=a.input/ref['name']
                    need(file_sha(path)==ref['sha256'],'CORPUS_LOADER_AUDIT_BATCH_PIN')
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    need(fingerprint(saved)==ref['state']==fingerprint(v),'CORPUS_LOADER_AUDIT_EVERY_LOGIT')
                    files.append(ref)
                token=F.cross_entropy(logits.reshape(-1,258),b.labels.reshape(-1),ignore_index=-100,reduction='none').reshape(b.labels.shape)
                predicted=logits.argmax(-1)
                for row,item in enumerate(b.records):
                    i=item['atom'];lo,hi=item['scored'];left=lo-item['offset']
                    need(torch.equal(token[row,left:left+hi-lo],expected[i]['losses'][lo:hi])
                         and torch.equal(predicted[row,left:left+hi-lo],expected[i]['argmax'][lo:hi]),
                         'CORPUS_LOADER_AUDIT_ORIGINAL_SCORE')
                    covered[i].extend(range(lo,hi));positions+=hi-lo
                logit_elements+=logits.numel()
        need(all(v==list(range(len(e.raw)+1)) for v,e in zip(covered,selected,strict=True)),
             'CORPUS_LOADER_AUDIT_BYTE_EOS_COVERAGE')
        if previous:need(len(files)==len(previous['files']),'CORPUS_LOADER_AUDIT_BATCH_COUNT')
        phases['held_out_forward_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        if decoder:
            decoder.fresh();controls=negatives(a,p,r,c,parent,decoder)
        else:controls={}
        parent.fresh();need(fingerprint(parent.model.state_dict())==parent_before,'CORPUS_LOADER_AUDIT_PARENT_CHANGED')
        need(all(file_sha(base/n)==h for n,h in code.items()),'CORPUS_LOADER_AUDIT_CODE_CHANGED')
        need(not torch.cuda.is_initialized(),'CORPUS_LOADER_AUDIT_CPU_ONLY')
        phases['controls_and_freshness']=time.perf_counter()-tick
        result=dict(complete=True,mode=a.mode,plan=a.plan_pin,run=a.run_pin,readback=a.readback_pin,input_pin=a.pin,
            code=code,decoder=receipt,consumed=consumed,files=files,controls=controls,
            atoms=len(selected),positions=positions,logit_elements=logit_elements,all_original_scores_exact=True,
            all_saved_logits_exact=a.mode=='read',all_byte_eos_positions=True,parent_unchanged=True,
            optimizer_restored_in_reader=a.mode=='read',free_generation_tested=False,gpu_used=False,
            full_corpus_checkpoint_tested=False,capability_demonstrated=False,promoted=False,
            phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,number_status='measured',acceptance='author-checked')
        need(.9<=sum(phases.values())/result['wall']<=1.001,'CORPUS_LOADER_AUDIT_PHASES')
        pin=write(a.output/'report.json',result)
        print(json.dumps(dict(pin=pin,mode=a.mode,positions=positions,wall=result['wall'])),flush=True)
    finally:
        if decoder:decoder.close()
        parent.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['produce','read'],required=True)
    for n in ('root','plan','run','readback','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('plan','run','readback'):p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--cpu-proof',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

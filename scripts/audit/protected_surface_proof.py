"""Full-size correction engineering proof; no capability or GPU promotion claim."""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
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
import torch
from audit import surface_condition_inputs as inputs
from audit.decoder_prefix_cohort import code,save
from audit.decoder_prefix_training import rejection
from audit.trained_answer_render import observe
from claims.process_conditions import write_committed
from model import protected_surface as protected
from model import protected_surface_state as state
from model import protected_surface_stream as stream
from model.answer_surface_stream import SurfaceStepper,render_atom
from model.contracts import ModelConfig,parameter_count
from model.legal_surface_inputs import SurfaceExample
from model.native_population_training import cpu,fingerprint
from model.surface_device_runtime import batches,evaluate
from text.constellation_corpus import file_sha,need

CENSUS=('decoder-capacity-census-v1-20261025','c6ac3fc367e689aa550e54e6b760116a5e3e0b93b488778398b7c02ae45011f1')


def load(root):
    path=root/CENSUS[0]/'report.json';need(file_sha(path)==CENSUS[1],'PROTECTED_PROOF_CENSUS_PIN')
    census=json.loads(path.read_bytes());s=inputs.load(root,continuation=True)
    need(census['source_input']==s['input_pin'] and census['parent']==s['decoder'].receipt,'PROTECTED_PROOF_SOURCE')
    rows=[];examples=[]
    for r in census['selected']:
        row=s['rows'][r['index']]
        need(all(row[k]==r[k] for k in ('side','document','occurrence','uid')) and len(row['raw'])==r['bytes']
             and {n:len(v) for n,v in row['native'].items()}==r['windows'],'PROTECTED_PROOF_SELECTION')
        rows.append(row);examples.append(SurfaceExample(row['occurrence'],row['uid'],row['side'],bytes(row['raw'].tolist()),
                                                        inputs.measured_native(row,s['decoder'].model)))
    parent,step=s['decoder'].export_continuation();need(step==51150,'PROTECTED_PROOF_PARENT_STEP')
    config=ModelConfig(**census['correction_config']);count=parameter_count(config,s['decoder'].model.specs)
    need(count==census['correction_parameters']==78_325_042,'PROTECTED_PROOF_CAPACITY')
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    need(count*64<available//2 and shutil.disk_usage(root).free>count*32,'PROTECTED_PROOF_PHYSICAL_MEMORY')
    return dict(source=s,census=census,rows=rows,examples=examples,train=[e for e in examples if e.side=='train'],
                evaluation=[e for e in examples if e.side!='train'],parent=parent,config=config,available=available)


def create(s):
    rng=torch.random.get_rng_state().clone();cuda_seed=torch.cuda.manual_seed_all;calls=[]
    def forbidden(*args,**kwargs):
        calls.append(1);raise ValueError('PROTECTED_PROOF_CUDA_RNG_TOUCHED')
    torch.cuda.manual_seed_all=forbidden
    try:model=protected.ProtectedSurfaceDecoder(s['source']['decoder'].model,s['config'],seed=171)
    finally:torch.cuda.manual_seed_all=cuda_seed
    need(not calls and torch.equal(rng,torch.random.get_rng_state()),'PROTECTED_PROOF_GLOBAL_RNG')
    return model,protected.optimizer_for(model)


def zero(s,model):
    model.eval();records=[]
    with torch.no_grad():
        for b in batches(s['examples'],model.config,model.specs,'gathered',8):
            parent=model.parent(b.prefix,b.native,offsets=b.offsets)['logits']
            combined=model(b.prefix,b.native,offsets=b.offsets)['logits']
            need(fingerprint(parent)==fingerprint(combined),'PROTECTED_PROOF_INITIAL_LOGITS')
            records.append(dict(records=b.records,shape=list(combined.shape),pin=fingerprint(combined)))
    free=[]
    for e in s['examples']:
        a=observe(render_atom(model.parent,e.native,capture_logits=True),32)
        b=observe(stream.render_atom(model,e.native,capture_logits=True),32)
        need(fingerprint(a)==fingerprint(b),'PROTECTED_PROOF_INITIAL_FREE');free.append(b)
    return dict(logits=records,free=free,initial_correction=fingerprint(model.correction.state_dict()))


def score(model,s,*,direct=False):
    model.eval()
    with torch.no_grad():teacher,profile=evaluate(model,s['evaluation'],'gathered',8)
    free=[];cache=[]
    for e in s['evaluation']:
        if direct:generator=render_atom(model,e.native,cached=False,capture_logits=True)
        else:generator=stream.render_atom(model,e.native,capture_logits=True)
        free.append(observe(generator,32))
        if not direct:
            x=stream.ProtectedSurfaceStepper(model,e.native)
            for _ in range(3):x.logits([256],0)
            need(x.parent.projection_calls==x.correction.projection_calls==1,'PROTECTED_PROOF_CACHE')
            cache.append(dict(parent=x.parent.projection_calls,correction=x.correction.projection_calls,calls=3));x.close()
    return dict(teacher=teacher,free=free),profile,cache


def controls(s,model,opt,step,ref,binding,output):
    model.eval();e=s['train'][0];result={}
    result['nontrain']=rejection(lambda:protected.update(model,opt,s['evaluation'][:1],step),'PROTECTED_SURFACE_NONTRAIN')
    result['incompatible_cache']=rejection(lambda:SurfaceStepper(model,e.native),'PROTECTED_SURFACE_USE_COMPOSITE_STEPPER_OR_DIRECT')
    result['missing_slot']=rejection(lambda:model.validate_inputs({}),'MNLM_MODEL_ROSTER')
    before=fingerprint(model.parent.state_dict());p=next(model.parent.parameters());old=p.detach().clone()
    with torch.no_grad():p.add_(1)
    result['mutated_parent']=rejection(model.fresh,'PROTECTED_SURFACE_PARENT_CHANGED')
    # This disposable model cannot recover merely by copying values back: version changed.
    with torch.no_grad():p.copy_(old)
    result['restored_but_stale_parent']=rejection(model.fresh,'PROTECTED_SURFACE_PARENT_CHANGED')
    need(fingerprint(model.parent.state_dict())==before,'PROTECTED_PROOF_CONTROL_PARENT_BYTES')
    root=Path(ref['root']);bad=output/'bad-state';bad.mkdir();manifest=json.loads((root/'manifest.json').read_bytes())
    shutil.copyfile(root/'manifest.json',bad/'manifest.json')
    first=manifest['shards'][0];path=bad/first['name'];shutil.copyfile(root/first['name'],path)
    with path.open('r+b') as f:f.seek(path.stat().st_size-1);v=f.read(1);f.seek(-1,1);f.write(bytes([v[0]^1]));f.flush();os.fsync(f.fileno())
    result['corrupt_shard']=rejection(lambda:state.read(bad,ref['pin'],binding=binding,parent=model.parent_pin),'PROTECTED_STATE_SHARD_PIN')
    result['wrong_parent_binding']=rejection(lambda:state.read(root,ref['pin'],binding=binding,parent='0'*64),'PROTECTED_STATE_BINDING')
    result['wrong_manifest_pin']=rejection(lambda:state.read(root,'0'*64,binding=binding,parent=model.parent_pin),'PROTECTED_STATE_MANIFEST_PIN')
    return result


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    s=load(a.root);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    files={'inputs.pt':save(a.output/'inputs.pt',dict(rows=s['rows'],parent=s['parent'],config=asdict(s['config'])))}
    model,opt=create(s);parameter_count_actual=sum(p.numel() for p in model.correction.parameters())
    print(json.dumps(dict(stage='allocated',parameters=parameter_count_actual)),flush=True)
    initial=zero(s,model);files['zero.pt']=save(a.output/'zero.pt',initial)
    phases['allocation_zero_preservation']=time.perf_counter()-tick;tick=time.perf_counter()
    print(json.dumps(dict(stage='zero_preservation',seconds=phases['allocation_zero_preservation'])),flush=True)
    binding=dict(census=CENSUS[1],parent=s['source']['decoder'].receipt,config=asdict(s['config']),seed=171,source=fingerprint(s['rows']))
    updates=[];checkpoints=[]
    for step in range(3):
        row,profile,scores=protected.update(model,opt,s['train'],step)
        files[f'scores-{step+1}.pt']=save(a.output/f'scores-{step+1}.pt',scores);updates.append(dict(row=row,profile=profile))
        print(json.dumps(dict(step=step+1,objective=row['objective'],positions=row['positions'])),flush=True)
        if step in (1,2):
            ref=state.save(a.output/f'checkpoint-{step+1}',model,opt,step=step+1,binding=binding);checkpoints.append(ref)
            if step==1:
                del model,opt;gc.collect();model,opt=create(s);need(state.restore(ref,model,opt,binding=binding)==2,'PROTECTED_PROOF_RESUME')
    phases['training_checkpoint_resume']=time.perf_counter()-tick;tick=time.perf_counter()
    result,profile,cache=score(model,s);files['evaluation.pt']=save(a.output/'evaluation.pt',result)
    final=fingerprint(dict(correction=model.correction.state_dict(),optimizer=opt.state_dict()))
    need(final==checkpoints[-1]['state'],'PROTECTED_PROOF_FINAL_STATE')
    phases['unassessed_rendering']=time.perf_counter()-tick;tick=time.perf_counter()
    # Optimizer controls on the actual intact post-training model.
    negatives={};wrong=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
    negatives['wrong_optimizer_membership']=rejection(lambda:protected.check(model,wrong,3),'PROTECTED_SURFACE_OPTIMIZER_MEMBERSHIP');del wrong
    entry=next(iter(opt.state.values()));old=entry['step'].clone();entry['step'].add_(1)
    negatives['wrong_moment_step']=rejection(lambda:protected.check(model,opt,3),'PROTECTED_SURFACE_MOMENTS');entry['step'].copy_(old)
    negatives.update(controls(s,model,opt,3,checkpoints[-1],binding,a.output))
    s['source']['fresh']();need(fingerprint(s['parent']['optimizer'])==s['source']['decoder'].receipt['optimizer'],'PROTECTED_PROOF_PARENT_MOMENTS')
    s['source']['decoder'].close();need(not torch.cuda.is_initialized(),'PROTECTED_PROOF_CPU_ONLY')
    phases['negative_controls_and_freshness']=time.perf_counter()-tick
    report=dict(complete=True,binding=binding,files=files,code=code(),config=asdict(s['config']),parameters=parameter_count_actual,
        initial=initial['initial_correction'],updates=updates,checkpoints=checkpoints,final=final,controls=negatives,
        cache=cache,evaluation_profile=profile,source_selection=s['census']['selected'],phases=phases,
        wall=time.perf_counter()-began,cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        available_before=s['available'],full_size_engineering=True,initialization_rng_preserved=True,cuda_qualified=False,promoted=False,
        semantic_validity=False,capability_gain_demonstrated=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,parameters=parameter_count_actual,wall=report['wall'])),flush=True)


def cold(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'PROTECTED_COLD_REPORT');r=json.loads((a.input/'report.json').read_bytes())
    base=Path(__file__).resolve().parents[1];need(all(file_sha(base/n)==p for n,p in r['code'].items()),'PROTECTED_COLD_CODE')
    def saved(name):
        path=a.input/name;ref=r['files'][name];need(file_sha(path)==ref['sha256'],'PROTECTED_COLD_FILE')
        value=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(value)==ref['state'],'PROTECTED_COLD_STATE');return value
    s=load(a.root);source=saved('inputs.pt')
    need(fingerprint(source)==fingerprint(dict(rows=s['rows'],parent=s['parent'],config=asdict(s['config']))),'PROTECTED_COLD_SOURCE')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    model,opt=create(s);need(fingerprint(zero(s,model))==fingerprint(saved('zero.pt')),'PROTECTED_COLD_ZERO')
    phases['zero_preservation']=time.perf_counter()-tick;tick=time.perf_counter()
    for step in range(3):
        row,_,scores=protected.update(model,opt,s['train'],step)
        need(row==r['updates'][step]['row'] and fingerprint(scores)==fingerprint(saved(f'scores-{step+1}.pt')),'PROTECTED_COLD_UPDATE')
        if step in (1,2):
            ref=r['checkpoints'][0 if step==1 else 1]
            actual=fingerprint(dict(correction=model.correction.state_dict(),optimizer=opt.state_dict()))
            payload,m=state.read(ref['root'],ref['pin'],binding=r['binding'],parent=model.parent_pin)
            need(actual==fingerprint(payload)==m['state']==ref['state'],'PROTECTED_COLD_CHECKPOINT');del payload
            if step==1:
                del model,opt;gc.collect();model,opt=create(s);state.restore(ref,model,opt,binding=r['binding'])
    phases['training_cold_restart']=time.perf_counter()-tick;tick=time.perf_counter()
    evaluated,_,_=score(model,s,direct=True)
    need(fingerprint(evaluated)==fingerprint(saved('evaluation.pt')),'PROTECTED_COLD_DIRECT_RENDER')
    need(fingerprint(dict(correction=model.correction.state_dict(),optimizer=opt.state_dict()))==r['final'],'PROTECTED_COLD_FINAL')
    s['source']['fresh']();s['source']['decoder'].close();need(not torch.cuda.is_initialized(),'PROTECTED_COLD_CPU_ONLY')
    phases['direct_render_and_freshness']=time.perf_counter()-tick
    result=dict(complete=True,input_pin=a.pin,source_bytes_exact=True,all_zero_logits_and_free_exact=True,
        all_updates_exact=True,all_final_moments_exact=True,midpoint_restart_exact=True,all_direct_free_logits_exact=True,
        updates=3,parameters=r['parameters'],phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,code=code(),acceptance='author-checked',cuda_qualified=False,promoted=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,wall=result['wall'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','cold'],required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--pin');a=p.parse_args()
    try:(run if a.mode=='run' else cold)(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

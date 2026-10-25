"""Full-size corpus-training path with GPU qualification and exact bounded cold replay."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_inversion as original
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model import protected_corpus_training as training
from model import protected_corpus_lineage as lineage
from model import protected_surface_state as state
from model.contracts import ModelConfig
from model.legal_surface_inputs import SurfaceExample
from model.native import NativeInput
from model.native_population_training import cpu,fingerprint
from model.protected_surface_stream import render_atom
from model.surface_device_runtime import batches,evaluate
from model.trained_surface_decoder import TrainedSurfaceDecoder
from senses.contract import digest
from text.constellation_corpus import file_sha,need


# Data references and small helpers deliberately avoid CPU-only audit entrypoints:
# importing those entrypoints changes CUDA_VISIBLE_DEVICES for the whole process.
PARENT_PATHS=dict(full='phase115-complete-answer-recovery-v1-20261024/inversion-full',
                  plan='legal-inversion-plan-v3-20261024',
                  cold='inversion-canonical-repair-v2-20261024/readback',
                  final='legal-inversion-final-readback-v1-20261024')


def save(path,value):
    with path.open('xb') as f:torch.save(cpu(value),f);f.flush();os.fsync(f.fileno())
    actual=torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(actual)==fingerprint(value),'PROTECTED_PROGRAM_SAVED_STATE')
    return dict(sha256=file_sha(path),state=fingerprint(actual))


def rejected(call,code):
    try:call()
    except (ValueError,RuntimeError,TypeError) as exc:
        need(code in str(exc),'PROTECTED_PROGRAM_WRONG_REFUSAL:'+str(exc));return str(exc)
    raise ValueError('PROTECTED_PROGRAM_CONTROL_ACCEPTED:'+code)


def measured_native(row,model):
    need(set(row['native'])=={s.name for s in model.specs},'CONDITION_ROSTER')
    result={s.name:NativeInput(s.identity,(row['row'],),row['native'][s.name].clone()[None],
        torch.zeros((1,len(row['native'][s.name])),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        for s in model.specs}
    model.validate_inputs(result);return result


def observe(generator,limit):
    events=[];decisions=0;terminal=False
    try:
        while decisions<limit or (events and events[-1].get('utf8_prefix_valid') is False):
            try:item=next(generator)
            except StopIteration:break
            events.append(item);decisions+=int('token' in item)
            if item['kind'] in ('answer_end','refuse','atom_end'):terminal=True;break
    finally:generator.close()
    return dict(events=events,decisions=decisions,censored=not terminal)


def load(a,device):
    need(file_sha(a.plan/'report.json')==a.plan_pin,'PROTECTED_PROGRAM_PLAN')
    p=json.loads((a.plan/'report.json').read_bytes());base=Path(__file__).resolve().parents[1]
    need(p['complete'] and p['schema']=='protected-corpus-plan-v1' and p['policy']==training.POLICY,'PROTECTED_PROGRAM_POLICY')
    implementation=dict(p['code']);implementation['audit/protected_corpus_program.py']=file_sha(__file__)
    implementation['model/protected_corpus_lineage.py']=file_sha(base/'model/protected_corpus_lineage.py')
    need(all(file_sha(base/n)==h for n,h in implementation.items()),'PROTECTED_PROGRAM_CODE')
    path=Path(p['train_index']['path']);need(file_sha(path)==p['train_index']['pin'],'PROTECTED_PROGRAM_INDEX')
    ids=np.load(path,mmap_mode='r',allow_pickle=False);need(len(ids)==p['train']['atoms'],'PROTECTED_PROGRAM_POPULATION')
    path=a.plan/'observations.pt';ref=p['files']['observations.pt'];need(file_sha(path)==ref['sha256'],'PROTECTED_PROGRAM_COHORT_PIN')
    rows=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(rows)==ref['state'],'PROTECTED_PROGRAM_COHORT_STATE')
    need([r['occurrence'] for r in rows if r['side']=='train']==ids[:384].tolist(),'PROTECTED_PROGRAM_REAL_PREFIX')
    args=[]
    for key,path in PARENT_PATHS.items():args.extend((a.root/path,p['parent'][key]))
    parent=TrainedSurfaceDecoder(*args,device=str(device),cpu_diagnostic=device.type=='cpu')
    need(all(parent.receipt[k]==p['parent'][k] for k in ('full','plan','cold','final','model','checkpoint','optimizer')),
         'PROTECTED_PROGRAM_PARENT')
    config=ModelConfig(**p['config']);count=p['parameters']
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    need(count==78325042 and count*64<available//2 and shutil.disk_usage(a.output).free>count*32,'PROTECTED_PROGRAM_PHYSICAL_MEMORY')
    if a.mode=='full':
        checkpoints=(p['updates']+training.POLICY['checkpoint_steps']-1)//training.POLICY['checkpoint_steps']
        projected=checkpoints*count*16+sum(v['bytes'] for v in p['evaluation'].values())*64+len(ids)*64
        need(shutil.disk_usage(a.output).free>projected,'PROTECTED_PROGRAM_PROJECTED_DISK')
    return dict(plan=p,ids=ids,rows=rows,parent=parent,config=config,implementation=implementation)


def examples(s,model,side):
    return [SurfaceExample(r['occurrence'],r['uid'],r['side'],bytes(r['raw'].tolist()),measured_native(r,model))
            for r in s['rows'] if r['side']==side]


def initial_preservation(model,selected):
    model.eval();result=[]
    with torch.no_grad():
        for b in batches(selected,model.config,model.specs,'gathered',training.POLICY['rows']):
            b=b.to(next(model.parameters()).device)
            parent=model.parent(b.prefix,b.native,offsets=b.offsets)['logits']
            output=model(b.prefix,b.native,offsets=b.offsets)['logits']
            need(torch.equal(parent,output),'PROTECTED_PROGRAM_INITIAL_CHANGED')
            result.append(dict(records=b.records,shape=list(output.shape),logits=fingerprint(output)))
    return result


def evaluation(model,selected):
    model.eval()
    with torch.no_grad():teacher,profile=evaluate(model,selected,'gathered',training.POLICY['evaluation_rows'])
    free=[observe(render_atom(model,e.native,capture_logits=True),32) for e in selected]
    return dict(teacher=teacher,free=free),profile


def binding(a,s,device,size):
    return dict(plan=a.plan_pin,policy=training.POLICY,scope='full' if a.mode=='full' else 'proof',size=size,
                code=digest(s['implementation']),parent=s['parent'].receipt['model'],
                parent_checkpoint=s['parent'].receipt['checkpoint'],parent_optimizer=s['parent'].receipt['optimizer'],
                device=str(device),torch=str(torch.__version__),cuda=torch.version.cuda,config=asdict(s['config']),
                runtime=lineage.runtime(device))


def controls(a,model,opt,s,b,ref,size):
    out={};bad=deepcopy(ref);bad['cursor']-=1
    out['cursor']=rejected(lambda:training.restore(bad,model,opt,b,size=size),'PROTECTED_CORPUS_CURSOR')
    bad=deepcopy(ref);bad['ledger']='0'*64
    out['ledger']=rejected(lambda:training.restore(bad,model,opt,b,size=size),'PROTECTED_STATE_BINDING')
    bad_binding=deepcopy(b);bad_binding['plan']='0'*64
    out['cohort']=rejected(lambda:training.restore(ref,model,opt,bad_binding,size=size),'PROTECTED_STATE_BINDING')
    bad_binding=deepcopy(b);bad_binding['policy']['rows']=8
    out['policy']=rejected(lambda:training.restore(ref,model,opt,bad_binding,size=size),'PROTECTED_CORPUS_POLICY')
    out['nontrain']=rejected(lambda:training.update(model,opt,examples(s,model,'query'),ref['step']),'PROTECTED_CORPUS_TRAIN_ONLY')
    badroot=a.output/'corrupt-checkpoint';badroot.mkdir();path=badroot/'manifest.json'
    raw=(Path(ref['root'])/'manifest.json').read_bytes()
    with path.open('xb') as f:f.write(raw+b' ');f.flush();os.fsync(f.fileno())
    bad=deepcopy(ref);bad['root']=str(badroot)
    out['physical_manifest_corruption']=rejected(lambda:training.restore(bad,model,opt,b,size=size),'PROTECTED_STATE_MANIFEST_PIN')
    need(fingerprint(dict(correction=model.correction.state_dict(),optimizer=opt.state_dict()))==ref['state'],
         'PROTECTED_PROGRAM_CONTROLS_CHANGED_STATE')
    return out


def execute(a,device):
    began=tick=time.perf_counter();c0=time.process_time();s=load(a,device);parent=s['parent'];source=None
    model,opt=training.create(parent.model,s['config']);size=len(s['ids']) if a.mode=='full' else 384
    train=examples(s,model,'train');held=examples(s,model,'development')+examples(s,model,'query')
    b=binding(a,s,device,size);ledger=digest(b);cursor=step=0;refs=[];records=[];files={};resume=None
    if a.mode=='full':
        need(not a.cpu_proof and a.qualification and a.qualification_pin,'PROTECTED_PROGRAM_GPU_QUALIFICATION')
        need(file_sha(a.qualification/'report.json')==a.qualification_pin,'PROTECTED_PROGRAM_QUALIFICATION_PIN')
        q=json.loads((a.qualification/'report.json').read_bytes())
        need(q['complete'] and q['plan']==a.plan_pin and q['device']==str(device)=='cuda:0'
             and q['all_update_state_scores_exact'] and q['midpoint_resume_exact'] and q['implementation']==s['implementation']
             and q['updates']==3 and q['train_atoms']==384 and q['runtime']==b['runtime'],
             'PROTECTED_PROGRAM_QUALIFICATION_SCOPE')
        source=original.sources(a.root);need(source['adapter'].binding==s['plan']['binding'],'PROTECTED_PROGRAM_SOURCE')
    if a.resume:
        need(a.mode=='full' and a.resume_pin and file_sha(a.resume)==a.resume_pin,'PROTECTED_PROGRAM_RESUME_PIN')
        ref=json.loads(a.resume.read_bytes());lineage.origin(ref,b)
        cursor,step,ledger=training.restore(ref,model,opt,b,size=size)
        need(fingerprint(model.correction.state_dict())==ref['after'],'PROTECTED_PROGRAM_RESUME_CORRECTION')
        resume=dict(path=str(a.resume.resolve()),pin=a.resume_pin,reference=ref)
        need(cursor<size,'PROTECTED_PROGRAM_ALREADY_COMPLETE')
    origin=dict(schema='protected-corpus-origin-v1',program=b,resume=resume,
                start=dict(cursor=cursor,step=step,ledger=ledger),initial_correction=fingerprint(model.correction.state_dict()))
    origin_path=a.output/'origin.json';origin_ref=dict(path=str(origin_path),pin=write_committed(origin_path,origin))
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    if not a.resume:
        files['initial-preservation.pt']=save(a.output/'initial-preservation.pt',initial_preservation(model,train[:128]))
    before,profile=evaluation(model,held);files['initial-evaluation.pt']=save(a.output/'initial-evaluation.pt',before)
    phases['initial_evaluation']=time.perf_counter()-tick
    phases.update(read=0.,update=0.,persistence=0.);write_committed(a.output/'binding.json',b)
    try:
        with (a.output/'steps.jsonl').open('x') as log:
            while cursor<size:
                tick=time.perf_counter();ids=s['ids'][cursor:cursor+128].tolist()
                batch=source['adapter'].fetch(ids,side='train',fit=True) if source else train[cursor:cursor+128]
                need([e.occurrence for e in batch]==ids,'PROTECTED_PROGRAM_BATCH_ORDER')
                phases['read']+=time.perf_counter()-tick;tick=time.perf_counter()
                row,profile,scores=training.update(model,opt,batch,step)
                phases['update']+=time.perf_counter()-tick;tick=time.perf_counter()
                cursor+=len(batch);step+=1;row.update(cursor=cursor,previous=ledger);ledger=digest(row)
                record=dict(row=row,pin=ledger,profile=profile);log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
                if a.mode=='proof':
                    records.append(record);files[f'scores-{step}.pt']=save(a.output/f'scores-{step}.pt',scores)
                if (a.mode=='proof' and step>=2) or step%training.POLICY['checkpoint_steps']==0 or cursor==size:
                    os.fsync(log.fileno());ref=training.checkpoint(a.output/f'checkpoint-{step:06d}',model,opt,b,cursor,step,ledger,size)
                    ref.update(origin=origin_ref,after=row['after'],optimizer=row['optimizer'])
                    write_committed(a.output/f'checkpoint-{step:06d}.json',ref);refs.append(ref)
                    if a.mode=='proof' and step==2:
                        del model,opt;gc.collect();model,opt=training.create(parent.model,s['config'])
                        need(training.restore(ref,model,opt,b,size=size)==(cursor,step,ledger),'PROTECTED_PROGRAM_RESUME')
                phases['persistence']+=time.perf_counter()-tick
                if a.mode=='proof' or step%128==0:print(json.dumps(dict(step=step,cursor=cursor,objective=row['objective'])),flush=True)
        tick=time.perf_counter();after,profile=evaluation(model,held)
        files['final-evaluation.pt']=save(a.output/'final-evaluation.pt',after)
        full_evaluation=None
        if source:
            # Same original document-separated population and paired controls; no re-selection.
            full_evaluation=original.evaluation(source,model,{side:v['documents'] for side,v in s['plan']['evaluation'].items()},
                                                s['ids'],a.output/'full-final-evaluation')
        phases['final_evaluation']=time.perf_counter()-tick;tick=time.perf_counter()
        negatives=controls(a,model,opt,s,b,refs[-1],size) if a.mode=='proof' else {}
        parent.fresh();parent.close()
        need(all(file_sha(Path(__file__).resolve().parents[1]/n)==h for n,h in s['implementation'].items()),'PROTECTED_PROGRAM_CODE_CHANGED')
        phases['controls_freshness']=time.perf_counter()-tick
        report=dict(complete=True,plan=a.plan_pin,binding=b,implementation=s['implementation'],files=files,checkpoints=refs,
                    origin=origin_ref,resume=resume,runtime=b['runtime'],
                    cursor=cursor,steps=step,ledger=ledger,records=records,controls=negatives,parent=s['plan']['parent'],
                    source_census=s['plan']['census'],full_evaluation=full_evaluation,device=str(device),
                    full_training=a.mode=='full',cpu_proof=a.cpu_proof,phases=phases,
                    wall=time.perf_counter()-began,cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                    capability_demonstrated=False,promoted=False,acceptance='author-checked')
        need(.9<=sum(phases.values())/report['wall']<=1.000001,'PROTECTED_PROGRAM_PHASE_COVERAGE')
        pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,wall=report['wall'])),flush=True)
    finally:
        if source:original.close(source)


def cold(a,device):
    began=tick=time.perf_counter();c0=time.process_time()
    need(file_sha(a.input/'report.json')==a.input_pin,'PROTECTED_PROGRAM_COLD_REPORT')
    r=json.loads((a.input/'report.json').read_bytes());need(r['complete'] and not r['full_training'] and r['device']==str(device),
                                                        'PROTECTED_PROGRAM_COLD_SCOPE')
    s=load(a,device);need(r['implementation']==s['implementation'],'PROTECTED_PROGRAM_COLD_IMPLEMENTATION')
    model,opt=training.create(s['parent'].model,s['config']);train=examples(s,model,'train')
    held=examples(s,model,'development')+examples(s,model,'query');b=binding(a,s,device,384)
    need(b==r['binding'],'PROTECTED_PROGRAM_COLD_BINDING');ledger=digest(b);cursor=0
    o=lineage.origin(r['checkpoints'][-1],b)
    need(r['origin']==r['checkpoints'][-1]['origin'] and r['resume'] is None and o['resume'] is None
         and o['initial_correction']==fingerprint(model.correction.state_dict()),'PROTECTED_PROGRAM_COLD_ORIGIN')
    def saved(name):
        path=a.input/name;ref=r['files'][name];need(file_sha(path)==ref['sha256'],'PROTECTED_PROGRAM_COLD_FILE')
        v=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(v)==ref['state'],'PROTECTED_PROGRAM_COLD_STATE');return v
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    need(fingerprint(initial_preservation(model,train[:128]))==fingerprint(saved('initial-preservation.pt')),'PROTECTED_PROGRAM_COLD_INITIAL')
    before,_=evaluation(model,held);need(fingerprint(before)==fingerprint(saved('initial-evaluation.pt')),'PROTECTED_PROGRAM_COLD_EVAL')
    phases['initial_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    for step in range(3):
        row,_,scores=training.update(model,opt,train[cursor:cursor+128],step);cursor+=128;row.update(cursor=cursor,previous=ledger);ledger=digest(row)
        need(row==r['records'][step]['row'] and ledger==r['records'][step]['pin']
             and fingerprint(scores)==fingerprint(saved(f'scores-{step+1}.pt')),'PROTECTED_PROGRAM_COLD_UPDATE')
        if step>=1:
            ref=r['checkpoints'][step-1]
            need(ref['state']==fingerprint(dict(correction=model.correction.state_dict(),optimizer=opt.state_dict())),'PROTECTED_PROGRAM_COLD_CHECKPOINT')
            payload,m=state.read(ref['root'],ref['pin'],binding=training.progress(b,cursor,step+1,ledger,384),parent=model.parent_pin)
            need(fingerprint(payload)==ref['state'],'PROTECTED_PROGRAM_COLD_SHARDS');del payload
            if step==1:
                del model,opt;gc.collect();model,opt=training.create(s['parent'].model,s['config'])
                need(training.restore(ref,model,opt,b,size=384)==(cursor,2,ledger),'PROTECTED_PROGRAM_COLD_RESTART')
    phases['updates_and_restart']=time.perf_counter()-tick;tick=time.perf_counter()
    after,_=evaluation(model,held);need(fingerprint(after)==fingerprint(saved('final-evaluation.pt')),'PROTECTED_PROGRAM_COLD_FINAL')
    s['parent'].fresh();s['parent'].close();phases['final_replay']=time.perf_counter()-tick
    result=dict(complete=True,input_pin=a.input_pin,plan=a.plan_pin,device=str(device),implementation=s['implementation'],
                runtime=b['runtime'],origin=r['origin'],
                all_update_state_scores_exact=True,midpoint_resume_exact=True,initial_parent_exact=True,all_evaluation_exact=True,
                updates=3,train_atoms=384,phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked',promoted=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,wall=result['wall'])),flush=True)


def main(a):
    a.output.mkdir(exist_ok=False);leased=False
    need(not a.cpu_proof or a.mode!='full','PROTECTED_PROGRAM_FULL_GPU_REQUIRED')
    if a.mode=='full':
        need(a.qualification and a.qualification_pin and file_sha(a.qualification/'report.json')==a.qualification_pin,
             'PROTECTED_PROGRAM_GPU_QUALIFICATION')
        q=json.loads((a.qualification/'report.json').read_bytes())
        need(q['complete'] and q['device']=='cuda:0' and q['plan']==a.plan_pin
             and q['all_update_state_scores_exact'] and q['midpoint_resume_exact'],'PROTECTED_PROGRAM_QUALIFICATION_SCOPE')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if a.cpu_proof:device=torch.device('cpu')
        else:
            (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
            take_lease('29','protected-corpus-'+a.mode);leased=True
            need(torch.cuda.is_available(),'PROTECTED_PROGRAM_CUDA_REQUIRED');device=torch.device('cuda:0')
        (cold if a.mode=='cold' else execute)(a,device)
        if a.cpu_proof:need(not torch.cuda.is_initialized(),'PROTECTED_PROGRAM_CPU_ONLY')
    finally:
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['proof','cold','full'],required=True)
    for n in ('root','plan','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-pin',required=True);p.add_argument('--cpu-proof',action='store_true')
    for n in ('input','qualification','resume'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:main(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

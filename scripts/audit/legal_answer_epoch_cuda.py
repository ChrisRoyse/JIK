"""Qualify complete source-fed answer updates against the original CUDA learner."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.legal_complete_answer_consumer import prepare
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import answer_full_history as learner
from model.guarded_transient_answer import guarded_transient_attention
from model.legal_answer_epoch import AnswerEpoch, persist
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
BASE = Path(__file__).resolve().parents[1]
DOCUMENTS = (15639, 27195, 32791, 105748, 106288)
ARCHITECTURES = ('shared', 'own_slot', 'whole_proposals')
PROOFS = dict(
    cpu=('legal-answer-epoch-proof-v3-20261024', '50bed7838cb795e65c937550595789944aec3d087b81a1f8bd933450611637cf'),
    cold=('legal-answer-epoch-proof-cold-v3-20261024', 'cb6170cd7dc18cadc66bb6fb19f53395f062761ad4bc018fbf3d162eff48d3c6'),
    readback=('legal-answer-epoch-readback-v2-20261024', 'ce196b9dee4b67df9c6c6eeae8e8a4509acc6e571074a23d6fd4dcab1277a13e'),
    inputs=('legal-complete-answer-consumer-v2-20261024', '5ea64522dde1aa415c84bcacdac263c36f15e8177f5b48dfaf37c9e44be65ea2'),
    inputs_cold=('legal-complete-answer-consumer-cold-v2-20261024', 'c2b6c3534393239073db8e81e852f7ddc4e6deab8e1e067f37b9415632491268'),
    initial=('open-window-answer-cpu-v2-20261024', 'a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339'))
NEW = ('audit/legal_answer_epoch_cuda.py', 'audit/legal_answer_epoch_queue.py', 'audit/surface_device_queue.py')


def read(path, pin):
    need(file_sha(path) == pin, 'ANSWER_CUDA_PROOF_PIN')
    return json.loads(Path(path).read_bytes())


def authenticate(root):
    reports = {n:read(root/d/'report.json', h) for n,(d,h) in PROOFS.items()}
    first, cold, back = (reports[n] for n in ('cpu','cold','readback'))
    need(first['complete'] and cold['complete'] and not first['gpu_used'] and not cold['gpu_used']
         and first['optimizer_updates'] == cold['optimizer_updates'] == 15
         and first['decisions'] == cold['decisions'] == 198
         and first['code'] == cold['code'] and cold['previous_pin'] == PROOFS['cpu'][1]
         and cold['cold_reproduced'], 'ANSWER_CUDA_CPU_SCOPE')
    need(back['complete'] and back['producer'] == PROOFS['cpu'][1] and back['cold'] == PROOFS['cold'][1]
         and back['updates_compared'] == 30 and back['decisions'] == 396
         and back['checkpoints_exact'] and back['source_fed_updates_exact'] and back['cold_exact'],
         'ANSWER_CUDA_CPU_READBACK')
    for arch in ARCHITECTURES:
        a,b = first['results'][arch],cold['results'][arch]
        need(a['updates'] == b['updates'] and a['all_updates_exact'] and b['all_updates_exact']
             and a['all_checkpoint_restores_exact'] and b['all_checkpoint_restores_exact'], 'ANSWER_CUDA_CPU_UPDATES')
    src, sc = reports['inputs'], reports['inputs_cold']
    need(src['complete'] and sc['complete'] and sc['previous_pin'] == PROOFS['inputs'][1]
         and src['binding'] == sc['binding'] and src['rows'] == sc['rows'], 'ANSWER_CUDA_INPUT_COLD')
    physical = {root/d/'report.json':h for d,h in PROOFS.values()}
    for label in ('cpu','cold'):
        physical.update({root/PROOFS[label][0]/n:h for n,h in reports[label]['files'].items()})
    for label in ('inputs','inputs_cold'):
        for doc in DOCUMENTS:
            name = f'document-{doc}.pt'
            physical[root/PROOFS[label][0]/name] = reports[label]['files'][name]
    for arch in ARCHITECTURES:
        name = arch+'/update-5.pt'
        physical[root/PROOFS['initial'][0]/name] = reports['initial']['files'][name]
    for path,h in physical.items(): need(file_sha(path) == h, 'ANSWER_CUDA_PHYSICAL_AUTHORITY')
    code = dict(first['code'], **{n:file_sha(BASE/n) for n in NEW})
    stamps = {p:(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in physical}
    def fresh():
        need(all(file_sha(BASE/n) == h for n,h in code.items()), 'ANSWER_CUDA_CODE_CHANGED')
        for p,s in stamps.items():
            t=p.stat(); need((t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns) == s, 'ANSWER_CUDA_AUTHORITY_CHANGED')
    fresh(); return reports,code,fresh


def numerical(value):
    return dict(records=value['record']['records'], **{n:value[n] for n in ('gradients','early','model','optimizer')})


def negatives(epoch, output):
    reference=persist(output/'checkpoint-0.pt',epoch.snapshot()); before=fingerprint(epoch.snapshot()); result={}
    def reject(name, fn, code):
        try: fn()
        except (ValueError,RuntimeError) as exc:
            need(code in str(exc), 'ANSWER_CUDA_WRONG_REFUSAL:'+str(exc)); result[name]=str(exc)
        else: raise ValueError('ANSWER_CUDA_NEGATIVE_ACCEPTED:'+name)
        need(fingerprint(epoch.snapshot()) == before, 'ANSWER_CUDA_REFUSAL_MUTATED_STATE')
    path=Path(reference['path'])
    reject('pin',lambda:epoch.restore(path,dict(reference,pin='0'*64),cursor=0,chain=epoch.chain),
           'ANSWER_EPOCH_CHECKPOINT_PIN')
    reject('cursor',lambda:epoch.restore(path,reference,cursor=1,chain=epoch.chain),'ANSWER_EPOCH_CHECKPOINT_CURSOR')
    reject('next_document',lambda:epoch.update(DOCUMENTS[1]),'ANSWER_EPOCH_NEXT_DOCUMENT')
    held=next(i for i,r in enumerate(epoch.corpus.rows) if int(r['side']) != 0)
    reject('held_out',lambda:epoch.corpus.example(held,fitting=True),'LEGAL_ANSWER_NONTRAIN_FIT')
    for name in ('binding','moments'):
        bad=torch.load(path,map_location='cpu',weights_only=True)
        if name == 'binding': bad['binding']['architecture']='wrong'
        else: bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
        wrong=output/('wrong-'+name+'.pt'); receipt=persist(wrong,bad)
        error='ANSWER_EPOCH_CHECKPOINT_BINDING' if name=='binding' else 'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT'
        reject(name,lambda:epoch.restore(wrong,receipt,cursor=0,chain=epoch.chain),error)
    epoch.restore(path,reference,cursor=0,chain=epoch.chain)
    return result


def run(a):
    began=tick=time.perf_counter(); a.output.mkdir(exist_ok=False); corpus=None; leased=False; phases={}
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == VISIBLE, 'ANSWER_CUDA_IMPORT_CHANGED_VISIBILITY')
    need(not torch.cuda.is_initialized(), 'ANSWER_CUDA_PREMATURE_INITIALIZATION')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try:
        reports,code,fresh=authenticate(a.root)
        corpus,labels,continuation,bundle,source_code=prepare(a.root)
        need(all(code.get(n) == h for n,h in source_code.items()), 'ANSWER_CUDA_SOURCE_CODE')
        target,target_pin=extension(bundle); items={}; census=[]
        for doc in DOCUMENTS:
            name=f'document-{doc}.pt'
            saved=torch.load(a.root/PROOFS['inputs'][0]/name,map_location='cpu',weights_only=True)
            old=torch.load(a.root/PROOFS['inputs_cold'][0]/name,map_location='cpu',weights_only=True)
            need(fingerprint(saved) == fingerprint(old), 'ANSWER_CUDA_INPUT_EXACT_COLD')
            item,request,_=corpus.example(doc,fitting=True)
            need(fingerprint(item) == fingerprint(saved['item'])
                 and json.loads(request.descriptor) == saved['descriptor']
                 and torch.equal(request.features,saved['features']), 'ANSWER_CUDA_REAL_SOURCE_EXACT')
            items[doc]=saved; census.append(dict(document=doc,atoms=len(item['context'])+len(item['answers']),
                answers=len(item['answers']),decisions=len(item['answers'])+1,request=request.pin,native=item['native_pin']))
        need(sum(r['decisions'] for r in census) == 66, 'ANSWER_CUDA_CENSUS')
        source=dict(code=code,proofs=PROOFS,source=corpus.binding,target=target_pin,census=census,
                    documents=DOCUMENTS,initial_step=5,policy=bundle['policy'])
        write_committed(a.output/'source.json',source)
        if a.mode == 'cuda':
            proof=read(a.inspection/'report.json',a.inspection_pin)
            need(proof['complete'] and proof['mode']=='inspect' and proof['source']==json.loads((a.output/'source.json').read_bytes())
                 and proof['optimizer_updates']==0 and not proof['gpu_used'], 'ANSWER_CUDA_INSPECTION')
            for n,h in proof['files'].items(): need(file_sha(a.inspection/n)==h, 'ANSWER_CUDA_INSPECTION_BYTES')
            (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
            need(VISIBLE != '' and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
                 'ANSWER_CUDA_RESIDENT_OR_HIDDEN')
            take_lease('29','complete-answer-epoch-cuda'); leased=True
            need(torch.cuda.is_available(), 'ANSWER_CUDA_REQUIRED')
            device=torch.device('cuda:0'); target.to(device); torch.cuda.reset_peak_memory_stats(device)
        else: device=torch.device('cpu')
        policy=dict(bundle['policy'],device=str(device)); prior=None
        binding=dict(source=source,device=str(device),torch=str(torch.__version__),cuda=torch.version.cuda,
                     precision='float32_tf32_off',history='all_prior_positions_with_gradients')
        # JSON normalizes tuple/list representation once for cold-process comparison.
        write_committed(a.output/'binding.json',binding); binding=json.loads((a.output/'binding.json').read_bytes())
        if a.previous:
            prior=read(a.previous/'report.json',a.previous_pin)
            need(prior['complete'] and prior['mode']=='cuda' and prior['binding']==binding
                 and prior['all_updates_exact'] and prior['optimizer_updates']==30, 'ANSWER_CUDA_COLD_BINDING')
            for n,h in prior['files'].items(): need(file_sha(a.previous/n)==h, 'ANSWER_CUDA_PREVIOUS_BYTES')
        phases['authentication']=time.perf_counter()-tick; results={}; controls={}
        for arch in ARCHITECTURES:
            tick=time.perf_counter(); out=a.output/arch; out.mkdir()
            initial=torch.load(a.root/PROOFS['initial'][0]/arch/'update-5.pt',map_location='cpu',weights_only=True)
            epoch=AnswerEpoch(corpus,bundle,target,target_pin,arch,initial,5,policy,code,documents=DOCUMENTS)
            controls[arch]=negatives(epoch,out); updates=[]; timings=dict(source_update=0.,reference_update=0.,read=0.)
            if a.mode=='cuda':
                reference,opt=epoch.make(initial,5)
                for index,doc in enumerate(DOCUMENTS):
                    value,entry=epoch.update(doc); saved=items[doc]; request=NativeRequest(saved['descriptor'])
                    corpus.validate(saved['item'],request,fitting=True)
                    torch.cuda.synchronize(device); start=time.perf_counter()
                    with guarded_transient_attention(reference,heads=True):
                        row,gradients,early=learner.update(reference,opt,saved['item'],index+5,policy,
                            request=request,target=target,full=True)
                    torch.cuda.synchronize(device); timings['reference_update']+=time.perf_counter()-start
                    expected=dict(record=row,gradients=gradients,early=early,model=cpu(reference.state_dict()),optimizer=cpu(opt.state_dict()))
                    need(fingerprint(numerical(value)) == fingerprint(numerical(expected)), 'ANSWER_CUDA_ORIGINAL_UPDATE')
                    need(value['record']['branch_gradients']==row['branch_gradients']
                         and value['record']['storage']['remaining_saved_storage_bytes']==row['storage']['remaining_saved_storage_bytes']==0,
                         'ANSWER_CUDA_BRANCH_OR_STORAGE')
                    persist(out/f'update-{index+6}.pt',value); persist(out/f'reference-{index+6}.pt',expected)
                    checkpoint=persist(out/f'checkpoint-{index+1}.pt',epoch.snapshot())
                    epoch.restore(checkpoint['path'],checkpoint,cursor=index+1,chain=entry['chain'])
                    update=dict(document=doc,step=index+6,numerical=fingerprint(numerical(value)),chain=entry['chain'],
                                checkpoint_state=checkpoint['state'],counts=entry['counts'])
                    if prior:
                        need(update==prior['results'][arch]['updates'][index], 'ANSWER_CUDA_COLD_UPDATE')
                        old=torch.load(a.previous/arch/f'update-{index+6}.pt',map_location='cpu',weights_only=True)
                        need(fingerprint(numerical(old))==update['numerical'], 'ANSWER_CUDA_COLD_PHYSICAL')
                    write_committed(out/f'step-{index+1}.json',entry); updates.append(update)
                    timings['source_update']+=entry['update_seconds']; timings['read']+=entry['read_seconds']
                    del value,expected,gradients,early
                del reference,opt
            results[arch]=dict(updates=updates,counts=epoch.counts(epoch.cursor),chain=epoch.chain,
                binding=epoch.binding,parameters=sum(p.numel() for p in epoch.model.parameters()),timings=timings)
            phases[arch]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=arch,updates=len(updates),timings=timings)),flush=True)
            del epoch,initial; gc.collect()
        tick=time.perf_counter(); corpus.close(); corpus.cache.close(); corpus.reader.close(); labels.fresh(); continuation.fresh(); fresh()
        need(fingerprint(target.state_dict())==bundle['target_fingerprint'], 'ANSWER_CUDA_TARGET_CHANGED')
        if prior: need(controls==prior['controls'], 'ANSWER_CUDA_COLD_CONTROLS')
        if a.mode=='inspect': need(not torch.cuda.is_initialized(), 'ANSWER_CUDA_INSPECTION_INITIALIZED_GPU')
        phases['final_readback']=time.perf_counter()-tick; seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001, 'ANSWER_CUDA_PHASES')
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        report=dict(complete=True,mode=a.mode,source=source,binding=binding,code=code,files=files,results=results,
            controls=controls,optimizer_updates=30 if leased else 0,decisions=396 if leased else 0,
            all_updates_exact=leased,checkpoint_restores_exact=True,cold_reproduced=prior is not None,previous_pin=a.previous_pin,
            gpu_used=leased,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if leased else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if leased else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,phases=phases,seconds=seconds,
            native_encoder_calls=0,promoted=False,full_population_trained=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if corpus is not None and not corpus.closed: corpus.close(); corpus.cache.close(); corpus.reader.close()
        if leased: release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=['inspect','cuda'],required=True)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--output',type=Path,required=True)
    for name in ('inspection','previous'): p.add_argument('--'+name,type=Path)
    for name in ('inspection-pin','previous-pin'): p.add_argument('--'+name)
    a=p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

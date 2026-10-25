"""Matched real-source original/captured backward updates and physical continuation."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_epoch_cuda as source
from claims.process_conditions import write_committed
from model import answer_full_history as learner
from model import captured_answer_backward as candidate
from model.legal_answer_epoch import AnswerEpoch,persist
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha,need

NEW=('model/captured_answer_backward.py','audit/captured_answer_backward.py')


def reject(fn,code):
    try:fn()
    except (ValueError,RuntimeError) as exc:
        need(code in str(exc),'CAPTURE_PROOF_WRONG_REFUSAL:'+str(exc));return str(exc)
    raise ValueError('CAPTURE_PROOF_NEGATIVE_ACCEPTED')


def controls(value):
    result={};before=learner.backward
    def nested():
        with candidate.captured_backward():
            with candidate.captured_backward():pass
    result['nested']=reject(nested,'CAPTURE_BACKWARD_ALREADY_INSTALLED')
    def changed():
        with candidate.captured_backward():learner.backward=before
    result['ownership']=reject(changed,'CAPTURE_BACKWARD_OWNERSHIP_CHANGED')
    def interrupted():
        with candidate.captured_backward():raise ValueError('CAPTURE_PROOF_INTENTIONAL_EXCEPTION')
    result['exception']=reject(interrupted,'CAPTURE_PROOF_INTENTIONAL_EXCEPTION')
    need(learner.backward is before is candidate.ORIGINAL and not candidate.ACTIVE,'CAPTURE_PROOF_NOT_RESTORED')
    result['scope_restored']=True
    first={n:r['value'].clone().requires_grad_(True) for n,r in value['early'].items()}
    ends={n:r['end_gradient'] for n,r in value['early'].items()}
    grads={n:r['gradient'] for n,r in value['early'].items()};name=next(iter(grads))
    need(fingerprint(candidate.credit(first,ends,grads,first))==fingerprint(value['early']),'CAPTURE_PROOF_CREDIT_RECONSTRUCT')
    bad=dict(grads);bad.pop(name)
    result['missing_context']=reject(lambda:candidate.credit(first,ends,bad,first),'CAPTURE_BACKWARD_MISSING_CONTEXT')
    bad=dict(grads);bad[name]=torch.zeros_like(grads[name])
    result['zero_context']=reject(lambda:candidate.credit(first,ends,bad,first),'FULL_HISTORY_DISCONNECTED_CONTEXT')
    bad=dict(grads);bad[name]=torch.full_like(grads[name],float('nan'))
    result['nonfinite_context']=reject(lambda:candidate.credit(first,ends,bad,first),'FULL_HISTORY_DISCONNECTED_CONTEXT')
    bad=dict(ends);bad[name]=torch.zeros_like(ends[name])
    result['zero_end']=reject(lambda:candidate.credit(first,bad,grads,first),'FULL_HISTORY_DISCONNECTED_CONTEXT')
    return result


def run(a):
    need(a.cpu_proof and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CAPTURE_PROOF_EXPLICIT_CPU')
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(exist_ok=False);phases={};corpus=None
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    try:
        reports,code,fresh=source.authenticate(a.root)
        corpus,labels,continuation,bundle,inherited=source.prepare(a.root)
        need(all(code.get(n)==h for n,h in inherited.items()),'CAPTURE_PROOF_SOURCE_CODE')
        code.update({n:file_sha(source.BASE/n) for n in NEW});target,target_pin=extension(bundle)
        frozen=fingerprint(target.state_dict());policy=dict(bundle['policy'],device='cpu')
        binding=dict(source=corpus.binding,code=code,policy=policy,target=target_pin,
                     documents=source.DOCUMENTS,initial=source.PROOFS['initial'],device='cpu',torch=str(torch.__version__))
        write_committed(a.output/'binding.json',binding);binding=json.loads((a.output/'binding.json').read_bytes())
        prior=source.read(a.previous/'report.json',a.previous_pin) if a.previous else None
        if prior:need(prior['complete'] and prior['binding']==binding,'CAPTURE_PROOF_COLD_BINDING')
        phases['authentication']=time.perf_counter()-tick;results={};negative={};census=None
        for arch_index,arch in enumerate(source.ARCHITECTURES):
            tick=time.perf_counter();out=a.output/arch;out.mkdir()
            initial=torch.load(a.root/source.PROOFS['initial'][0]/arch/'update-5.pt',map_location='cpu',weights_only=True)
            epochs={n:AnswerEpoch(corpus,bundle,target,target_pin,arch,initial,5,policy,code,documents=source.DOCUMENTS)
                    for n in ('original','captured')}
            for name in epochs:(out/name).mkdir()
            rows=[]
            for index,doc in enumerate(source.DOCUMENTS):
                order=('original','captured') if (index+arch_index+bool(prior))%2==0 else ('captured','original')
                values={};entries={};elapsed={}
                for name in order:
                    start=time.perf_counter()
                    with candidate.captured_backward() if name=='captured' else nullcontext():
                        values[name],entries[name]=epochs[name].update(doc)
                    elapsed[name]=time.perf_counter()-start
                    need(learner.backward is candidate.ORIGINAL and not candidate.ACTIVE,'CAPTURE_PROOF_CONTEXT_LEAK')
                    persist(out/name/f'update-{index+6}.pt',values[name])
                pins={n:fingerprint(source.numerical(v)) for n,v in values.items()}
                need(pins['original']==pins['captured'],'CAPTURE_PROOF_NUMERICAL_UPDATE')
                # Independently sealed CPU updates remain the external numeric reference.
                path=a.root/source.PROOFS['cpu'][0]/arch/f'update-{index+6}.pt'
                need(file_sha(path)==reports['cpu']['files'][str(path.relative_to(a.root/source.PROOFS['cpu'][0]))],
                     'CAPTURE_PROOF_PRIOR_PAYLOAD')
                expected=torch.load(path,map_location='cpu',weights_only=True)
                need(pins['original']==fingerprint(source.numerical(expected)),'CAPTURE_PROOF_QUALIFIED_REFERENCE')
                need(entries['original']['chain']==entries['captured']['chain'],'CAPTURE_PROOF_NUMERICAL_CHAIN')
                for value in values.values():
                    need(value['record']['storage']['remaining_saved_storage_bytes']==0,'CAPTURE_PROOF_GRAPH_RETAINED')
                row=dict(document=doc,step=index+6,numerical=pins['original'],chain=entries['original']['chain'],
                    counts=entries['original']['counts'],order=order,seconds=elapsed,
                    phases={n:v['record']['phases'] for n,v in values.items()},
                    read_seconds={n:e['read_seconds'] for n,e in entries.items()},
                    update_seconds={n:e['update_seconds'] for n,e in entries.items()},
                    storage={n:v['record']['storage'] for n,v in values.items()})
                # Reopen both actual optimizer checkpoints before continuation.
                if index in (1,4):
                    states={}
                    for name,epoch in epochs.items():
                        ref=persist(out/name/f'checkpoint-{index+1}.pt',epoch.snapshot())
                        epoch.restore(ref['path'],ref,cursor=index+1,chain=entries[name]['chain']);states[name]=ref['state']
                    need(states['original']==states['captured'],'CAPTURE_PROOF_RESTORED_STATE');row['restored_state']=states['original']
                if prior:
                    previous=prior['results'][arch]['updates'][index]
                    for field in ('document','step','numerical','chain','counts','restored_state'):
                        need(row.get(field)==previous.get(field),'CAPTURE_PROOF_COLD_NUMERICAL')
                if index==4:negative[arch]=controls(values['captured'])
                rows.append(row);write_committed(out/f'step-{index+6}.json',row)
                del values,entries,expected
            census=epochs['original'].counts(5)
            need(census['decisions']==66 and census['atoms']==71,'CAPTURE_PROOF_CENSUS')
            results[arch]=dict(updates=rows,counts=census,
                original_seconds=sum(r['update_seconds']['original'] for r in rows),
                captured_seconds=sum(r['update_seconds']['captured'] for r in rows),
                original_probe_seconds=sum(r['phases']['original']['credit_probe'] for r in rows),
                captured_probe_seconds=sum(r['phases']['captured']['credit_probe'] for r in rows))
            phases[arch]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=arch,updates=10,original_seconds=results[arch]['original_seconds'],
                captured_seconds=results[arch]['captured_seconds'])),flush=True)
            del epochs,initial;gc.collect()
        tick=time.perf_counter();corpus.close();corpus.cache.close();corpus.reader.close();labels.fresh();continuation.fresh();fresh()
        need(fingerprint(target.state_dict())==frozen and not torch.cuda.is_initialized(),'CAPTURE_PROOF_FROZEN_DEVICE')
        if prior:need(negative==prior['controls'],'CAPTURE_PROOF_COLD_CONTROLS')
        for n,h in code.items():
            need(file_sha(source.BASE/n)==h,'CAPTURE_PROOF_CODE_CHANGED')
            path=a.output/'source'/n;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes((source.BASE/n).read_bytes())
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'CAPTURE_PROOF_PHASE_COVERAGE')
        report=dict(complete=True,binding=binding,code=code,results=results,controls=negative,files=files,
            optimizer_updates=30,decisions=396,all_updates_exact=True,restored_continuations_exact=True,
            previous_pin=a.previous_pin,cold_reproduced=prior is not None,phases=phases,seconds=seconds,
            cpu_seconds=time.process_time()-process,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_used=False,frozen_encoder_calls=0,promoted=False,cuda_qualified=False,long_answer_qualified=False,
            full_population_trained=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if corpus is not None and not corpus.closed:corpus.close();corpus.cache.close();corpus.reader.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

"""Independent physical comparison of original and externally resumed placement updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as source
from audit.legal_grouped_placement_readback import check_row
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint


def h(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def run(a):
    began=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
    seed=rd.read(a.seed/'report.json',a.seed_pin); final=rd.read(a.final/'report.json',a.final_pin)
    rd.manifest(a.seed,seed);rd.manifest(a.final,final)
    rd.require(seed['complete'] and final['complete'] and seed['mode']=='seed' and final['mode']=='continue'
        and final['previous_pin']==a.seed_pin and seed['pid']!=final['pid']
        and seed['binding']==final['binding'] and seed['execution']==final['execution'],'RESUME_READBACK_BINDING')
    _,bundle=source.source_bundle(a.root)
    data=source.load_data(argparse.Namespace(scope='small',root=a.root),bundle)
    train=[i for i in data['items'] if i['side']=='train']; labels=data['labels']; totals=dict(tensors=0,bytes=0)
    results={}; evaluations=0
    def exact(left,right):
        counts=rd.exact(left,right)
        for key in totals:totals[key]+=counts[key]
    for name in source.runtime.original.ARCHITECTURES:
        initial=rd.load(a.seed,name+'/checkpoint-0.pt'); branch=initial['binding']; chain=h(branch)
        rows=0; windows={n:0 for n in labels.roster}; model=fingerprint(initial['model'])
        original_lines=(a.seed/name/'uninterrupted/steps.jsonl').read_bytes().splitlines(keepends=True)
        actual_lines=(a.final/name/'steps.jsonl').read_bytes().splitlines(keepends=True)
        rd.require(actual_lines==original_lines and len(actual_lines)==len(train),'RESUME_READBACK_LEDGER_BYTES')
        for cursor,(raw,item) in enumerate(zip(actual_lines,train)):
            r=json.loads(raw); body={k:v for k,v in r.items() if k!='hash'};u=r['update']
            rd.require(r['parent']==chain and r['cursor']==cursor and r['hash']==h(body)
                and u['step']==cursor+1 and u['document']==item['document'] and u['targets']==len(item['codes'])
                and u['noise_seed']==branch['policy']['noise_seed']+cursor and u['before']==model,
                'RESUME_READBACK_CHAIN')
            exact(u,bundle['cpu_reference'][name]['updates'][cursor]['record'])
            left=rd.load(a.seed,name+f'/uninterrupted/update-{cursor+1}.pt')
            right=rd.load(a.seed if cursor<2 else a.final,name+f'/update-{cursor+1}.pt')
            exact(left,right); exact(right['record'],r)
            rd.require(fingerprint(right['model'])==u['after'] and fingerprint(right['optimizer'])==u['optimizer'],
                       'RESUME_READBACK_STATE')
            for key,value in right['optimizer']['state'].items():
                rd.require(int(value['step'])==cursor+1 and torch.isfinite(value['exp_avg']).all()
                    and torch.isfinite(value['exp_avg_sq']).all(),'RESUME_READBACK_MOMENTS')
            chain=r['hash'];model=u['after'];rows+=len(item['codes'])
            for lane in windows:windows[lane]+=sum(len(f[lane]) for f in item['frames'])
        saved=rd.load(a.final,name+'/checkpoint-5.pt');exact(saved,rd.load(a.seed,name+'/uninterrupted/checkpoint-5.pt'))
        rd.require(saved['chain']==chain and saved['step']==5 and saved['rows']==rows==66
            and final['results'][name]['counts']['windows']==windows,'RESUME_READBACK_FINAL_COVERAGE')
        exact(rd.load(a.final,name+'/restored.pt'),rd.load(a.seed,name+'/checkpoint-2.pt'))
        # Recompute the persisted final metrics against source readings, separately
        # from the numerical training comparison and the producer's success flags.
        checked=[]
        for item in data['items']:
            proposals=tuple(labels.roster) if name=='whole_proposals' else ('assembled',)
            for lo in range(0,len(item['codes']),8):
                stop=min(lo+8,len(item['codes']))
                for proposal in proposals:
                    rel=name+f"/final/document-{item['document']}-rows-{lo}-{stop}-{proposal}.pt"
                    actual=rd.load(a.final,rel)
                    expected=rd.load(a.seed,name+'/uninterrupted/'+rel.split('/',1)[1])
                    exact(actual,expected)
                    for offset,payload in enumerate(actual,lo):
                        row=check_row(payload,item,labels)
                        rd.require((row['offset'],row['document'],row['proposal'])==(offset,item['document'],proposal),
                                   'RESUME_READBACK_EVALUATION_IDENTITY')
                        checked.append(row); evaluations+=1
        exact(checked,rd.load(a.final,name+'/metrics.pt'))
        rd.require(fingerprint(checked)==final['results'][name]['metrics'],'RESUME_READBACK_EVALUATION_STATE')
        results[name]=dict(updates=5,targets=rows,windows=windows,chain=chain,metrics=len(checked))
    data['close']();rd.require(not torch.cuda.is_initialized(),'RESUME_READBACK_GPU')
    # The reader must detect a real single-bit tensor change, including signed zero.
    x=torch.tensor([0.],dtype=torch.float64);y=torch.tensor([-0.],dtype=torch.float64)
    try:rd.exact(x,y)
    except ValueError as error:
        rd.require('CAPTURE_READBACK_BYTES' in str(error),'RESUME_READBACK_WRONG_REFUSAL');negative=True
    else:raise AssertionError('RESUME_READBACK_NEGATIVE_ACCEPTED')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    report=dict(complete=True,seed=a.seed_pin,final=a.final_pin,results=results,comparison=totals,
        evaluation_rows=evaluations,separate_process=True,every_update_and_final_evaluation_exact=True,
        signed_zero_control=negative,gpu_used=False,cuda_continuation_qualified=False,
        seconds=time.perf_counter()-began,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),comparison=totals,evaluation_rows=evaluations)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('seed','final','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('seed','final'):p.add_argument('--'+n+'-pin',required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

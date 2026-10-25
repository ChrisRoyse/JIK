"""Independent full schedule derivation and cold replay of every paired proof update."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.placement_source_exposure import read,save,sha


def require(ok, code):
    if not ok: raise ValueError('MNLM_ORDER_COLD_'+code)


def expected(rows, mode, seed):
    counts=Counter(r['source'] for r in rows); positions=Counter(); tagged=[]
    for r in rows:
        s=r['source']; rank=positions[s]; positions[s]+=1
        tagged.append((Fraction(2*rank+1,2*counts[s]),s,r))
    if mode=='interleaved': tagged.sort(key=lambda x:(x[0],x[1]))
    return [dict(step=i,**r,noise_seed=seed+r['original_step']-1) for i,(_,_,r) in enumerate(tagged,1)]


def check_rows(actual, wanted):
    require(len(actual)==len(wanted),'ROW_COUNT')
    require(all(a==b for a,b in zip(actual,wanted)),'ROW_CONTENT_OR_ORDER')


def rejection(fn):
    try: fn()
    except ValueError as error:
        require(str(error).startswith('MNLM_ORDER_COLD_'),'NEGATIVE_REASON'); return str(error)
    raise AssertionError('MNLM_ORDER_COLD_NEGATIVE_ACCEPTED')


def run(a):
    start=time.perf_counter(); cpu0=time.process_time(); a.output.mkdir(exist_ok=False)
    report=read(a.proof/'report.json',a.pin)
    require(report['complete'] and report['device']=='cpu' and not report['gpu_qualified'],'SCOPE')
    for path,pin in report['inputs'].items(): require(sha(path)==pin,'INPUT_PIN')
    for name,pin in report['files'].items(): require(sha(a.proof/name)==pin,'OUTPUT_PIN')
    root=Path(__file__).resolve().parents[1]
    for name,pin in report['code'].items():
        require(sha(root/name)==sha(a.proof/'source'/name)==pin,'CODE_PIN')
    results=read(a.proof/'results.json',report['files']['results.json'])
    # Reconstruct membership from the original sealed plan and original physical metadata.
    boundary_path=next(Path(p) for p in report['inputs'] if p.endswith('/seal/boundary.json'))
    seal=read(boundary_path,report['inputs'][str(boundary_path)])
    plan=seal['binding']['source']['plan']['train_documents']; selected=set(plan)
    require(len(plan)==len(selected)==91540,'ORIGINAL_MEMBERSHIP')
    metadata_path=next(Path(p) for p in report['inputs'] if p.endswith('/documents.jsonl'))
    mapping={}; h=hashlib.sha256()
    with metadata_path.open('rb') as stream:
        for address,raw in enumerate(stream):
            h.update(raw)
            if address in selected:
                r=json.loads(raw); mapping[address]=(r['source_path'],r['atom_start'],r['atom_end'])
    require(h.hexdigest()==report['inputs'][str(metadata_path)],'METADATA_CHANGED')
    rows=[dict(document=d,source=mapping[d][0],start=mapping[d][1],stop=mapping[d][2],
               targets=mapping[d][2]-mapping[d][1]-1,original_step=i,side='train') for i,d in enumerate(plan,1)]
    require(sum(r['targets'] for r in rows)==6455636,'TARGETS')
    seed=seal['binding']['policy']['noise_seed']; require(seed==report['seed'],'FULL_SEED')
    schedules={}; controls={}
    for mode in ('original','interleaved'):
        with (a.proof/(mode+'.jsonl')).open() as stream: actual=[json.loads(s) for s in stream]
        wanted=expected(rows,mode,seed); check_rows(actual,wanted)
        counts=Counter(); targets=Counter(); first={}; last={}; longest=Counter(); previous=None; run=0; transitions=0
        for row in actual:
            source=row['source']; counts[source]+=1; targets[source]+=row['targets']
            first.setdefault(source,row['step']); last[source]=row['step']
            if source==previous: run+=1
            else: transitions+=previous is not None; run=1; previous=source
            longest[source]=max(longest[source],run)
        summary=dict(documents=len(actual),targets=sum(targets.values()),transitions=transitions,
            sources={s:dict(documents=n,targets=targets[s],longest_run=longest[s],first=first[s],last=last[s]) for s,n in sorted(counts.items())})
        require(summary==results['schedules'][mode],'SUMMARY'); schedules[mode]=summary
        if mode=='interleaved':
            for field,value in (('source','wrong-source'),('document',-1),('targets',-1),('noise_seed',-1),('start',-1)):
                bad=list(actual); bad[0]=dict(bad[0],**{field:value})
                controls['changed_'+field]=rejection(lambda:check_rows(bad,wanted))
            controls['missing_document']=rejection(lambda:check_rows(actual[:-1],wanted))
            bad=list(actual); bad[0]=bad[1]
            controls['duplicate_document']=rejection(lambda:check_rows(bad,wanted))
            controls['reversed_schedule']=rejection(lambda:check_rows(actual[::-1],wanted))
    phases=dict(authentication_and_schedules=time.perf_counter()-start)
    tick=time.perf_counter()
    import torch
    from audit.legal_grouped_placement_training import source_bundle,small_metadata
    from audit.legal_placement_runtime_readback import make
    from model.native_rq_packed import PackedNativeRQ
    from model.native_population_training import cpu,fingerprint
    from model.native_placement_efficient import update as independent_update
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    source,bundle=source_bundle(a.root); small_metadata(a.root,bundle['items'])
    target=PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True); target.eval().requires_grad_(False)
    frozen=fingerprint(target.state_dict()); policy=dict(bundle['policy'],device='cpu')
    require(policy==report['policy'],'POLICY')
    fitting=[i for i in bundle['items'] if i['side']=='train']; lookup={i['document']:i for i in fitting}
    small=[dict(document=i['document'],source=i['source_collection'],start=i['start'],stop=i['stop'],
                targets=len(i['codes']),original_step=s,side='train') for s,i in enumerate(fitting,1)]
    require(small==results['small_rows'],'SMALL_SOURCE_JOIN')
    phases['bundle']=time.perf_counter()-tick; tick=time.perf_counter(); states=0; updates=0; replay=[]
    for architecture,arms in results['results'].items():
        for mode,arm in arms.items():
            schedule=expected(small,mode,policy['noise_seed']); directory=a.proof/architecture/mode
            with (directory/'updates.jsonl').open() as stream: records=[json.loads(s) for s in stream]
            require(arm['order']==[r['document'] for r in schedule] and len(records)==5,'SMALL_SCHEDULE')
            initial=torch.load(directory/'checkpoint-0.pt',map_location='cpu',weights_only=True)
            m,o=make(bundle,target,architecture,'cpu')
            require(fingerprint((m.state_dict(),o.state_dict()))==fingerprint((initial['model'],initial['optimizer'])),'INITIAL')
            for step,row in enumerate(schedule):
                # Fresh objects and a persisted prior checkpoint for each independent transition.
                prior=torch.load(directory/f'checkpoint-{step}.pt',map_location='cpu',weights_only=True)
                saved=torch.load(directory/f'checkpoint-{step+1}.pt',map_location='cpu',weights_only=True)
                for index,value in ((step,prior),(step+1,saved)):
                    cp=arm['checkpoints'][index]
                    require(fingerprint(value)==cp['state'] and value['schedule']==schedule and value['policy']==policy
                            and value['step']==index and value['target']==frozen,'CHECKPOINT_BINDING'); states+=1
                m,o=make(bundle,target,architecture,'cpu'); m.load_state_dict(prior['model'],strict=True); o.load_state_dict(prior['optimizer'])
                item=lookup[row['document']]
                # Reproduce RNG directly, independently of the scheduling adapter.
                noise=torch.randn(len(item['codes']),policy['samples'],policy['noise_width'],
                    generator=torch.Generator(device='cpu').manual_seed(policy['noise_seed']+row['original_step']-1))
                require(torch.equal(noise,saved['noise']),'DOCUMENT_NOISE')
                call_policy=dict(policy,noise_seed=policy['noise_seed']+row['original_step']-1-step)
                rec,grad,sample=independent_update(m,o,item,step,call_policy,target=target)
                require(rec==records[step]==saved['record'],'UPDATE_RECORD')
                actual=cpu(dict(model=m.state_dict(),optimizer=o.state_dict(),step=step+1,schedule=schedule,
                                policy=policy,target=frozen,record=rec,gradients=grad,noise=sample))
                require(fingerprint(actual)==fingerprint(saved),'REPLAY_STATE'); updates+=1
                if mode=='original': require(rec==bundle['cpu_reference'][architecture]['updates'][step]['record'],'ORIGINAL_REFERENCE')
            require(fingerprint((m.state_dict(),o.state_dict()))==arm['final_state'],'FINAL_STATE')
            replay.append(dict(architecture=architecture,mode=mode,updates=5,targets=66,final_state=arm['final_state']))
        require(arms['original']['samples']==arms['interleaved']['samples'],'PAIRED_INPUT_SAMPLES')
    require(fingerprint(target.state_dict())==frozen,'TARGET_CHANGED')
    phases['cold_replay']=time.perf_counter()-tick
    controls['wrong_proof_pin']=rejection(lambda:require(sha(a.proof/'report.json')=='0'*64,'REPORT_PIN'))
    final=dict(schema='placement-source-order-cold-v1',complete=True,previous_pin=a.pin,
               schedules=schedules,replayed= replay,updates=updates,successor_targets=396,
               restored_checkpoint_states=states,negative_controls=controls,
               inputs_unchanged=all(sha(p)==pin for p,pin in report['inputs'].items()),
               phases=phases,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               device='cpu',gpu_qualified=False,accuracy_gain_measured=False,acceptance='author-checked',
               code_sha256=sha(__file__))
    require(final['inputs_unchanged'],'INPUT_CHANGED')
    save(a.output/'report.json',final)
    print(json.dumps(dict(pin=sha(a.output/'report.json'),updates=updates,seconds=final['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','proof','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True); a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():save(a.output/'failure.json',dict(error=repr(error)))
        raise

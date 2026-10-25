"""Cold source joins, optimizer replay and unmasked rendering for prefix masking."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import decoder_prefix_training as trial
from audit import legal_surface_inversion as original
from audit.decoder_prefix_cohort import code,SEED
from claims.process_conditions import write_committed
from model import decoder_prefix_mask as masking
from model import surface_inversion_training as inversion
from model.answer_surface_stream import render_atom
from model.legal_native_training_stream import CertifiedNativeFrames
from model.native_population_training import cpu,fingerprint
from model.surface_device_runtime import batches
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need


def sources(root,s):
    source=original.sources(root);data=source['data'];checked=[];native=None
    try:
        directory,pin=original.statistics.DEFAULTS['audit']
        native=CertifiedNativeFrames(data,root/directory,pin)
        # Recompute the score-blind document/atom selection from immutable arrays.
        excluded={int(data.components[d]) for docs in original.select(data).values() for d in docs}
        excluded.update(int(data.components[m['document']]) for m in source['metadata'])
        need(sorted(excluded)==s['report']['excluded_components'],'PREFIX_COLD_EXCLUSIONS')
        seen=set(excluded);expected=[]
        for side,bit,count,per in [('train',0,128,8),('development',1,16,4),('query',2,16,4)]:
            ranked=[]
            for d in np.flatnonzero(data.doc_sides==bit).tolist():
                a,z=map(int,data.documents[d]);c=int(data.components[d])
                if z-a>=per and c not in seen:
                    key=hashlib.sha256(canonical(dict(seed=SEED,source=data.source.pin,component=c,document=d))).hexdigest()
                    ranked.append((key,d,c,a,z))
            selected=0
            for _,d,c,a,z in sorted(ranked):
                if c in seen:continue
                seen.add(c);selected+=1
                ids=sorted(range(a,z),key=lambda i:hashlib.sha256(canonical(dict(seed=SEED,document=d,occurrence=i))).hexdigest())[:per]
                expected.extend(dict(side=side,document=d,component=c,occurrence=i) for i in sorted(ids))
                if selected==count:break
            need(selected==count,'PREFIX_COLD_SELECTION_SIZE')
        need(expected==s['report']['selection'],'PREFIX_COLD_SELECTION')
        for start in range(0,len(s['rows']),128):
            rows=s['rows'][start:start+128];uids=[int(data.mapping[r['occurrence']]) for r in rows]
            frames=native.frames(uids);raws=source['surfaces'].read(uids)
            for r,uid,frame,raw in zip(rows,uids,frames,raws):
                bit={'train':0,'development':1,'query':2}[r['side']];doc=r['document'];a,z=map(int,data.documents[doc])
                need(a<=r['occurrence']<z and int(data.sides[r['occurrence']])==bit and int(data.doc_sides[doc])==bit
                     and int(data.components[doc])==r['component'] and uid==r['uid'],'PREFIX_COLD_SOURCE_SPLIT')
                need(bytes(r['raw'].tolist())==raw and fingerprint(frame)==fingerprint(r['native']), 'PREFIX_COLD_SOURCE_VALUES')
                e=source['adapter'].wrap(r['occurrence'],uid,raw,frame,side=r['side'])
                need(next(iter(e.native.values())).rows[0]==r['row']
                     and bool(source['continuation'].presence[uid]&1)==r['train_present'],'PREFIX_COLD_IDENTITY')
                checked.append(e)
        native.close();native=None
        need(source['adapter'].binding==s['report']['binding'],'PREFIX_COLD_BINDING')
    finally:
        if native is not None:native.close()
        original.close(source)
    train=[e for e in checked if e.side=='train'];evals=[e for e in checked if e.side!='train']
    donors=[train[d] for r,d in zip(s['rows'],s['report']['donor_train_indices']) if r['side']!='train']
    s.update(train=train,evals=evals,donors=donors)
    # The full CUDA decoder is an input checkpoint, not a replayed 51,150-step history.
    path=root/s['report']['trained_sources']['full'][0]/'report.json'
    need(file_sha(path)==s['report']['trained_sources']['full'][1],'PREFIX_COLD_INITIAL_REPORT')
    full=json.loads(path.read_bytes());ref=full['checkpoints'][-1]
    need(file_sha(ref['path'])==ref['pin'],'PREFIX_COLD_INITIAL_PIN')
    state=torch.load(ref['path'],map_location='cpu',weights_only=True)
    need(state['step']==51150 and fingerprint(state)==ref['state']
         and fingerprint(state['model'])==fingerprint(s['initial']['after'])
         and fingerprint(state['optimizer'])==fingerprint(s['initial']['optimizer']),'PREFIX_COLD_INITIAL_STATE')
    return dict(atoms=len(checked),windows=sum(len(v) for r in s['rows'] for v in r['native'].values()),
                bytes=sum(len(e.raw) for e in checked),selection_exact=True,initial_checkpoint_exact=True)


def masks(examples,model,traces,step):
    count=eligible=0;number=0
    for number,b in enumerate(batches(examples,model.config,model.specs,'gathered',128),1):
        trace=traces[number-1];want=torch.zeros_like(b.prefix,dtype=torch.bool);allowed=want.clone()
        for j,r in enumerate(b.records):
            stream=hashlib.shake_256(canonical(dict(seed=SEED,step=step,occurrence=r['occurrence'],offset=r['offset']))).digest(8*b.prefix.shape[1])
            for col in range(b.prefix.shape[1]):
                ok=col<r['prefix_length'] and int(b.prefix[j,col])<256
                allowed[j,col]=ok
                want[j,col]=ok and struct.unpack_from('<Q',stream,col*8)[0]<int(.4*(1<<64))
        need(trace['records']==b.records and torch.equal(want,trace['mask']) and torch.equal(allowed,trace['eligible']),
             'PREFIX_COLD_MASK')
        count+=int(want.sum());eligible+=int(allowed.sum())
    need(number==len(traces),'PREFIX_COLD_MASK_COUNT')
    return dict(masked=count,eligible=eligible)


def comparisons(evaluations,rows):
    result={}
    for arm in ('ordinary','masked'):
        ref=evaluations['initial'];current=evaluations[arm];docs=defaultdict(lambda:[0.,0.,0])
        for i,r in enumerate(rows):
            diff=float(current['paired']['correct'][i]['losses'].double().sum()-ref['paired']['correct'][i]['losses'].double().sum())
            before=float(ref['paired']['shuffled'][i]['losses'].double().sum()-ref['paired']['correct'][i]['losses'].double().sum())
            after=float(current['paired']['shuffled'][i]['losses'].double().sum()-current['paired']['correct'][i]['losses'].double().sum())
            v=docs[(r['side'],r['component'])];v[0]+=diff;v[1]+=after-before;v[2]+=len(r['raw'])+1
        result[arm]=[dict(side=k[0],component=k[1],correct_nll_delta=v[0]/v[2],margin_delta=v[1]/v[2],positions=v[2])
                     for k,v in sorted(docs.items())]
    return result


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'PREFIX_COLD_REPORT_PIN');report=json.loads((a.input/'report.json').read_bytes())
    need(report['complete'] and report['cohort']==trial.PIN and report['policy']==dict(mask=.4,epochs=1,atoms=128,steps=8,
         initial_step=51150,observer=96,training_context='true_native'),'PREFIX_COLD_POLICY')
    base=Path(__file__).resolve().parents[1]
    need(all(file_sha(base/n)==p for n,p in report['code'].items()),'PREFIX_COLD_CODE')
    s=trial.load(a.root);source=sources(a.root,s);phases=dict(source=time.perf_counter()-tick);tick=time.perf_counter()
    def saved(name):
        ref=report['files'][name];path=a.input/name;need(file_sha(path)==ref['sha256'],'PREFIX_COLD_PAYLOAD_PIN')
        value=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(value)==ref['state'],'PREFIX_COLD_PAYLOAD_STATE')
        return value
    total=0;restarts=0;mask_counts=[];evaluations={};summaries={}
    for arm in trial.ARMS:
        model,opt=trial.new(s);ledger=digest(dict(cohort=trial.PIN,arm=arm));binding=dict(cohort=trial.PIN,arm=arm)
        records=report['records'][arm];need(len(records['updates'])==(0 if arm=='initial' else 8),'PREFIX_COLD_STEPS')
        for number,entry in enumerate(records['updates']):
            examples=s['train'][number*128:(number+1)*128];step=51150+number;prior=saved(entry['path'])
            if arm=='ordinary':row,profile=inversion.update(model,opt,examples,step);payload={}
            else:
                row,profile,payload=masking.update(model,opt,examples,step,rate=.4)
                mask_counts.append(masks(examples,model,prior['payload']['traces'],step))
            after,moments=inversion.check_state(model,opt,step+1);gradients=cpu({n:p.grad for n,p in model.named_parameters()})
            ledger=digest(dict(parent=ledger,update=row))
            need(row==entry['row']==prior['row'] and ledger==entry['ledger']==prior['ledger']
                 and fingerprint(after)==fingerprint(prior['after']) and fingerprint(moments)==fingerprint(prior['optimizer'])
                 and fingerprint(gradients)==fingerprint(prior['gradients']) and fingerprint(payload)==fingerprint(prior['payload']),
                 'PREFIX_COLD_UPDATE')
            total+=1
            if number in (3,7):
                ref=records['checkpoints'][0 if number==3 else 1]
                fresh,fo=trial.new(s);cursor,st,le=inversion.restore(ref,fresh,fo,binding,size=1024)
                need(cursor==(number+1)*128 and st==step+1 and le==ledger
                     and fingerprint(fresh.state_dict())==fingerprint(after) and fingerprint(fo.state_dict())==fingerprint(moments),
                     'PREFIX_COLD_RESTART')
                if number==3:model,opt=fresh,fo;restarts+=1
        paired,_=inversion.paired(model,s['evals'],s['donors']);expected=saved(arm+'-evaluation.pt')
        need(fingerprint(paired)==fingerprint(expected['paired']),'PREFIX_COLD_TEACHER')
        model.eval();free=[]
        for i,e in enumerate(s['evals']):
            observed=trial.observe(render_atom(model,e.native,cached=False,capture_logits=True),96)
            need(fingerprint(observed)==fingerprint(expected['free'][i]),'PREFIX_COLD_FREE_DIRECT')
            free.append(observed)
            if (i+1)%32==0:print(json.dumps(dict(cold=arm,free_atoms=i+1)),flush=True)
        summary=dict(teacher=trial.summarize(paired,s['evalrows']),free=trial.free_summary(free,s['evalrows']))
        need(summary==report['summary'][arm] and not model.tokens._forward_hooks,'PREFIX_COLD_SUMMARY')
        evaluations[arm]=expected;summaries[arm]=summary
    phases['cold_updates_and_evaluation']=time.perf_counter()-tick
    s['fresh']();need(not torch.cuda.is_initialized(),'PREFIX_COLD_CPU_ONLY')
    result=dict(complete=True,input_pin=a.pin,source=source,updates=total,midpoint_restarts=restarts,mask_counts=mask_counts,
        summary=summaries,component_comparisons=comparisons(evaluations,s['evalrows']),code=code(),
        source_exact=True,all_steps_exact=True,all_teacher_positions_exact=True,all_free_direct_logits_exact=True,
        full_original_training_replayed=False,phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked',promoted=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,updates=total,restarts=restarts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

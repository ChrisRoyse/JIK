"""Matched reference/half/full-feedback learning with preserved broad Adam state."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import breadth_answer_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_generated_history as learning
from model import breadth_answer_transfer as transfer
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,legal
from text.constellation_corpus import file_sha,need

PARENT=('broad-answer-transfer-v2-20261024','bae9d16b3e25e91a3c7562087eb96580193af05fd7fb32ff5b6a0c95b190f5e1')
COLD=('broad-answer-transfer-cold-v2-20261024','5729fb0f9cd4db41f95081c0dc1cddd06f9a45d4232b15877df9e5424edc4a87')
NEW=('model/answer_generated_history.py','audit/generated_answer_training.py','audit/generated_answer_readback.py')
MODES={'reference':0.,'half':.5,'generated':1.}
EPOCHS=4


def load(root):
    s=source.load(root);reports=[]
    for name,pin in (PARENT,COLD):
        path=root/name/'report.json';need(file_sha(path)==pin,'FEEDBACK_PARENT_REPORT');reports.append(json.loads(path.read_bytes()))
    r,c=reports;need(c['source']['pin']==PARENT[1] and c['all_updates_exact'] and c['all_trajectories_exact']
        and r['complete'] and c['complete'] and r['code']==s['code'],'FEEDBACK_PARENT_COLD')
    for n,h in r['files'].items():need(file_sha(root/PARENT[0]/n)==h,'FEEDBACK_PARENT_BYTES')
    s['saved']={arm:torch.load(root/PARENT[0]/arm/'update-5.pt',map_location='cpu',weights_only=True)['state']
                for arm in ('initial','breadth')}
    s['code'].update({n:file_sha(Path(__file__).resolve().parents[1]/n) for n in NEW})
    s['items_ready']={k:source.prior.prior.item(s,k) for k in sorted(s['items'])};s['parent_report']=r
    return s


def make(s,arm):
    _,model=source.models(s,0 if arm=='initial' else 91540);saved=s['saved'][arm]
    opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
    transfer.restore(model,opt,saved,saved['binding'],s['policy'],pin=fingerprint(saved))
    return model,opt


def binding(s,arm,mode):
    need(arm in s['saved'] and mode in MODES,'FEEDBACK_RUN_SCOPE')
    return dict(schema='generated-answer-recipe-v1',parent=fingerprint(s['saved'][arm]),parent_step=5,arm=arm,mode=mode,
        probability=MODES[mode],epochs=EPOCHS,train=s['train'],policy=s['policy'],code=s['code'],
        source_native={k:s['items'][k]['native_pin'] for k in s['train']},source_labels={k:s['items'][k]['label_pin'] for k in s['train']})


def checkpoint(s,model,opt,run,ledger):
    saved=s['saved'][run['arm']];state=transfer.checkpoint(model,opt,saved['binding'],5+len(ledger),s['policy'])
    return dict(schema='generated-answer-checkpoint-v1',recipe=run,ledger=ledger,state=state)


def restore(s,model,opt,value,run,*,pin):
    need(fingerprint(value)==pin and value['schema']=='generated-answer-checkpoint-v1' and value['recipe']==run,'FEEDBACK_RESTORE_RECIPE')
    ledger=value['ledger'];state=value['state'];parent=s['saved'][run['arm']]
    need(state['cursor']==5+len(ledger) and [r['cursor'] for r in ledger]==list(range(6,6+len(ledger))), 'FEEDBACK_RESTORE_CURSOR')
    need(not ledger or (ledger[-1]['after']==fingerprint(state['model']) and ledger[-1]['optimizer']==fingerprint(state['optimizer'])),
         'FEEDBACK_RESTORE_STATE')
    transfer.restore(model,opt,state,parent['binding'],s['policy'],pin=fingerprint(state));return ledger


def controls(s):
    m,o=make(s,'breadth');item,req=s['items_ready'][s['train'][0]];b=s['saved']['breadth']['binding'];out={}
    seal=fingerprint((m.state_dict(),o.state_dict()))
    def run(value=item,request=req,p=.5,policy=None):
        return learning.update(m,o,value,5,policy or s['policy'],b,request=request,target=s['target'],probability=p)
    held=next(k for k,v in s['items'].items() if v['side']=='query')
    out['query_fit']=rejected(lambda:run(value=s['items_ready'][held][0]),'CONDITIONED_NONTRAIN')
    out['probability']=rejected(lambda:run(p=.7),'FEEDBACK_PROBABILITY')
    m.forecaster.architecture='own_slot';out['architecture']=rejected(lambda:run(),'FEEDBACK_ARCHITECTURE');m.forecaster.architecture='shared'
    bad=deepcopy(item);next(iter(bad['answers'][0].values())).add_(1)
    out['native']=rejected(lambda:run(value=bad),'CONDITIONED_NATIVE_PIN')
    bad=deepcopy(item);bad['codes'][0,0]=(bad['codes'][0,0]+1)%4
    out['label']=rejected(lambda:run(value=bad),'CONDITIONED_LABEL_PIN')
    out['request']=rejected(lambda:run(request=NativeRequest(dict(legal(2),operation='respond_to_source_question'))),'FEEDBACK_REQUEST')
    out['memory']=rejected(lambda:run(policy=dict(s['policy'],working_bytes=1)),'GRADIENT_STORAGE_MEMORY')
    recipe=binding(s,'breadth','half');cp=checkpoint(s,m,o,recipe,[])
    changed=deepcopy(cp);changed['recipe']['probability']=1.
    out['recipe']=rejected(lambda:restore(s,m,o,changed,recipe,pin=fingerprint(changed)),'FEEDBACK_RESTORE_RECIPE')
    changed=deepcopy(cp);changed['state']['optimizer']['state']={}
    out['checkpoint']=rejected(lambda:restore(s,m,o,changed,recipe,pin=fingerprint(cp)),'FEEDBACK_RESTORE_RECIPE')
    need(fingerprint((m.state_dict(),o.state_dict()))==seal and all(p.grad is None for p in m.parameters()),'FEEDBACK_CONTROL_STATE')
    # Deliberately permute real future targets only in this non-fitting negative intervention.
    bad=deepcopy(item);bad['answers']=bad['answers'][1:]+bad['answers'][:1]
    bad['codes']=bad['codes'].roll(-1,0);bad['counts']={n:v.roll(-1,0) for n,v in bad['counts'].items()}
    bad['native_pin']=fingerprint((bad['context'],bad['answers']));bad['label_pin']=fingerprint((bad['codes'],bad['counts']))
    with torch.no_grad():
        a,_,left=learning.forward(m,item,5,s['policy'],req,s['target'],1.)
        z,_,right=learning.forward(m,bad,5,s['policy'],req,s['target'],1.)
    need(fingerprint([r['feedback'] for r in left])==fingerprint([r['feedback'] for r in right])
         and float(a)!=float(z),'FEEDBACK_FUTURE_TARGET_LEAK')
    out['future_target_independence']=dict(trace=fingerprint([r['feedback'] for r in left]),loss=float(a),altered_loss=float(z))
    out['zero_feedback_equivalence']={}
    for arm in ('initial','breadth'):
        a,oa=make(s,arm);z,oz=make(s,arm);bound=s['saved'][arm]['binding']
        old,_,_=transfer.update(a,oa,item,5,s['policy'],bound,request=req,target=s['target'])
        new,_=learning.update(z,oz,item,5,s['policy'],bound,request=req,target=s['target'],probability=0.)
        need((old['gradients'],old['after'],old['optimizer'])==(new['gradients'],new['after'],new['optimizer']),
             'FEEDBACK_ZERO_EQUIVALENCE')
        out['zero_feedback_equivalence'][arm]={k:new[k] for k in ('gradients','after','optimizer')}
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    s=load(a.root);write_committed(a.output/'source.json',dict(parent=PARENT,cold=COLD,code=s['code'],census=s['census'],
        epochs=EPOCHS,modes=MODES,train=s['train'],initial_evaluations={arm:s['parent_report']['files'][arm+'/final-evaluation.pt'] for arm in s['saved']}))
    phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter();negative=controls(s)
    write_committed(a.output/'controls.json',negative);phases['controls']=time.perf_counter()-tick
    for arm in ('initial','breadth'):
        for mode,probability in MODES.items():
            tick=time.perf_counter();out=a.output/(arm+'-'+mode);out.mkdir();m,o=make(s,arm);m.train()
            bound=binding(s,arm,mode);ledger=[];persist(out/'initial.pt',checkpoint(s,m,o,bound,ledger));evaluations=[]
            generated=overrides=decisions=peak=0
            for epoch in range(EPOCHS):
                for doc in s['train']:
                    item,request=s['items_ready'][doc];cursor=5+len(ledger)
                    record,early=learning.update(m,o,item,cursor,s['policy'],s['saved'][arm]['binding'],
                        request=request,target=s['target'],probability=probability)
                    persist(out/f'update-{cursor+1}.pt',dict(record=record,early=early))
                    decisions+=record['decisions'];peak=max(peak,record['storage']['peak_saved_storage_bytes'])
                    need(record['storage']['remaining_saved_storage_bytes']==0,'FEEDBACK_GRAPH_RETAINED')
                    generated+=sum(v['feedback'] is not None and v['feedback']['source']=='generated_unverified' for v in record['records'])
                    overrides+=sum(v['feedback'] is not None and v['feedback']['training_continuation_override'] for v in record['records'])
                    ledger.append({k:record[k] for k in ('cursor','document','after','optimizer','gradients','early')})
                state=checkpoint(s,m,o,bound,ledger);persist(out/f'epoch-{epoch+1}.pt',state)
                if epoch==0:
                    m,o=make(s,arm);ledger=restore(s,m,o,state,bound,pin=fingerprint(state));m.train()
                if epoch in (0,1,3):
                    evaluated=source.prior.evaluate(s,m);persist(out/f'evaluation-{epoch+1}.pt',evaluated);evaluations.append(len(evaluated));m.train()
            write_committed(out/'ledger.json',ledger)
            results[arm+'/'+mode]=dict(recipe=bound,updates=len(ledger),final=fingerprint(m.state_dict()),optimizer=fingerprint(o.state_dict()),
                evaluations=evaluations,final_cursor=25,generated_inputs=generated,training_overrides=overrides,
                decisions=decisions,peak_saved_storage=peak,all_saved_storage_released=True)
            phases[arm+'_'+mode]=time.perf_counter()-tick;print(json.dumps(dict(arm=arm,mode=mode,updates=len(ledger))),flush=True)
    tick=time.perf_counter();s['fresh']();need(not torch.cuda.is_initialized(),'FEEDBACK_CPU_SCOPE')
    files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'FEEDBACK_PHASES')
    total_updates=sum(v['updates'] for v in results.values());total_decisions=sum(v['decisions'] for v in results.values())
    rollouts=sum(sum(v['evaluations']) for v in results.values())
    need((total_updates,total_decisions,rollouts)==(120,1584,720),'FEEDBACK_TOTAL_CENSUS')
    report=dict(complete=True,acceptance='author-checked',code=s['code'],files=files,results=results,controls=negative,
        updates=total_updates,decisions=total_decisions,new_rollouts=rollouts,phases=phases,seconds=seconds,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

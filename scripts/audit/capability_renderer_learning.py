"""True native-slot surface reconstruction with carried AdamW state and a matched control."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_conditioning import fingerprint, score, select_donors, summarize
from audit.capability_kernel_learning import system, shuffled, branch_evidence, save_checkpoint, parameter_hash
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, pinned, predict
from model.stream_surface import whole_surface_loss


def read(path): return json.loads(Path(path).read_bytes())


def restore(config, corpus, parent_ref):
    parent=pinned(parent_ref); path=Path(parent_ref['path']).parent/'final.pt'
    need(parent['source']['config']==sha(Path(config['_path']).read_bytes()),'RENDERER_PARENT_CONFIG')
    for name,pin in parent['source']['code'].items():
        need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'RENDERER_PARENT_CODE')
    need(sha(path.read_bytes())==parent['final_checkpoint'],'RENDERER_PARENT_CHECKPOINT')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=system(config,corpus); model.load_state_dict(saved['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],foreach=False)
    optimizer.load_state_dict(saved['optimizer'])
    need(parameter_hash(model)==parent['final_parameters'] and fingerprint(optimizer.state_dict())==saved['optimizer_hash']
         and saved['updates']==parent['optimizer_updates'],'RENDERER_PARENT_STATE')
    model.requires_grad_(False).eval(); model['decoder'].requires_grad_(True)
    return model,optimizer,parent,saved


def frozen(model,optimizer):
    return fingerprint(dict(weights={n:x for n,x in model.state_dict().items() if not n.startswith('decoder.')},
        moments={n:optimizer.state.get(p,{}) for n,p in model.named_parameters() if not n.startswith('decoder.')}))


def train_step(model,optimizer,corpus,key,condition):
    need(key in corpus.train and condition in corpus.train,'RENDERER_TRAIN_SPLIT')
    need(corpus.tasks[key]['kind']==corpus.tasks[condition]['kind'],'RENDERER_OPERATION')
    raw,_=corpus.target(key,model['world'].config,'cuda',fitting=True)
    _,native=corpus.target(condition,model['world'].config,'cuda',fitting=True)
    before=parameter_hash(model); optimizer.zero_grad(set_to_none=True)
    loss=whole_surface_loss(model['decoder'],raw,native,backward=True)
    need(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decoder.')),'RENDERER_FROZEN_GRADIENT')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'RENDERER_GRADIENT')
    gradients=branch_evidence(model); optimizer.step()
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'RENDERER_PARAMETER')
    after=parameter_hash(model); need(before!=after,'RENDERER_NO_UPDATE')
    return dict(task=key,condition=condition,family=corpus.tasks[key]['family'],before=before,after=after,
                surface=loss,gradients=gradients,native_hash=fingerprint({n:x.values for n,x in native.items()}))


@torch.no_grad()
def evaluate(model,corpus,donors,arm,destination,parent_rows):
    records=[]; world=model['world']; decoder=model['decoder']
    with (destination/'queries.jsonl').open('x') as out:
        for key in corpus.query:
            task=corpus.tasks[key]; raw,target=corpus.target(key,world.config,'cuda',fitting=False)
            _,wrong=corpus.target(donors[key],world.config,'cuda',fitting=True)
            native=corpus.batch(key,'context',world.config,'cuda')
            prediction=predict(world,model['request'],native,key=key,operation=task['kind'])['predicted_constellation']
            row=dict(task=key,arm=arm,family=task['family'],group=task['split_group'],donor=donors[key],
                     source_bytes=len(raw),source_sha256=sha(raw),scores={})
            for name,slots in dict(predicted=prediction.slots,target=target,wrong_target=wrong).items():
                row['scores'][name]=score(decoder,raw,slots)
                need(row['scores'][name]['condition_sha256']==parent_rows[key]['scores'][name]['condition_sha256'],
                     'RENDERER_CONDITION_DRIFT')
            row['parent']={n:parent_rows[key]['scores'][n]['loss'] for n in row['scores']}
            out.write(json.dumps(row,sort_keys=True)+'\n'); out.flush(); records.append(row)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--parent-pin',required=True); p.add_argument('--conditioning',type=Path,required=True)
    p.add_argument('--conditioning-pin',required=True); p.add_argument('--arm',choices=('correct','shuffled'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); need(not a.output.exists(),'RENDERER_EXISTS'); a.output.mkdir()
    started=tick=time.perf_counter(); phases={}
    try:
        config=read(a.config); config['_path']=str(a.config); corpus=KernelLearningCorpus(config)
        parent_ref=dict(path=str(a.parent),pin=a.parent_pin); parent=pinned(parent_ref)
        previous=pinned(dict(path=str(a.conditioning),pin=a.conditioning_pin))
        need(previous['source']['config']['pin']==sha(a.config.read_bytes()),'RENDERER_CONDITIONING_CONFIG')
        prior_path=a.conditioning.parent/'queries.jsonl'
        need(sha(prior_path.read_bytes())==previous['query_records_sha256'],'RENDERER_CONDITIONING_ROWS')
        parent_rows={r['task']:r for r in map(json.loads,prior_path.read_bytes().splitlines()) if r['arm']=='correct'}
        mapping=shuffled(corpus); donors=select_donors(corpus)
        first={f:next(i for i,k in enumerate(corpus.train) if corpus.tasks[k]['family']==f) for f in ('coding','conversation','reasoning')}
        boundaries=sorted({0,len(corpus.train)-1,*first.values()})
        source=dict(config=dict(path=str(a.config),pin=sha(a.config.read_bytes())),parent=parent_ref,
                    conditioning=dict(path=str(a.conditioning),pin=a.conditioning_pin),
                    code={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in
                          ('audit/capability_renderer_learning.py','audit/capability_conditioning.py','audit/capability_kernel_learning.py')})
        plan=dict(source=source,train=corpus.train,queries=corpus.query,shuffled=mapping,donors=donors,
                  first_by_family=first,boundaries=boundaries,arm=a.arm,families=dict(Counter(corpus.tasks[k]['family'] for k in corpus.train)))
        write_committed(a.output/'plan.json',plan)
        phases['authenticate']=time.perf_counter()-tick; tick=time.perf_counter()
        gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-renderer-'+a.arm); configure()
        model,optimizer,parent,saved=restore(config,corpus,parent_ref)
        frozen_before=frozen(model,optimizer); base=saved['updates']; initial=parameter_hash(model)
        chain=save_checkpoint(a.output/'initial.pt',model,optimizer,base,source,parent['final_checkpoint'])
        phases['restore']=time.perf_counter()-tick; tick=time.perf_counter()
        with (a.output/'steps.jsonl').open('x') as out:
            for index,key in enumerate(corpus.train):
                if index in boundaries and index:
                    save_checkpoint(a.output/f'before-{index}.pt',model,optimizer,base+index,source,chain)
                at=time.perf_counter(); cpu=time.process_time()
                row=train_step(model,optimizer,corpus,key,key if a.arm=='correct' else mapping[key])
                row.update(index=index,previous=chain,seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                chain=digest(row); out.write(json.dumps(dict(row,sha256=chain),sort_keys=True)+'\n'); out.flush()
                if index in boundaries:
                    save_checkpoint(a.output/f'after-{index}.pt',model,optimizer,base+index+1,source,chain)
                print(json.dumps(dict(arm=a.arm,step=index,loss=row['surface']['loss'])),flush=True)
        phases['learn']=time.perf_counter()-tick; tick=time.perf_counter()
        need(frozen(model,optimizer)==frozen_before,'RENDERER_FROZEN_DRIFT')
        final=save_checkpoint(a.output/'final.pt',model,optimizer,base+len(corpus.train),source,chain)
        records=evaluate(model,corpus,donors,a.arm,a.output,parent_rows)
        need(frozen(model,optimizer)==frozen_before,'RENDERER_EVAL_DRIFT')
        phases['evaluate']=time.perf_counter()-tick; tick=time.perf_counter()
        need(sha(Path(config['journal']['path']).read_bytes())==config['journal']['pin'],'RENDERER_JOURNAL')
        result=dict(schema='capability-renderer-learning-v1',source=source,arm=a.arm,initial_parameters=initial,
            final_parameters=parameter_hash(model),initial_optimizer=saved['optimizer_hash'],parent_updates=base,
            new_updates=len(corpus.train),total_update_events=base+len(corpus.train),final_checkpoint=final,
            frozen_weights_and_moments_unchanged=True,frozen_identity=frozen_before,
            steps_sha256=sha((a.output/'steps.jsonl').read_bytes()),queries_sha256=sha((a.output/'queries.jsonl').read_bytes()),
            families=summarize(records),gpu_before=gpu,scientific_admission=False,learned_capability=False,serving=False,
            generated_surface_evaluated=False,exposure='development_exposed_group_holdout',
            number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick; wall=time.perf_counter()-started
        need(sum(phases.values())/wall>=.9,'RENDERER_PHASES'); result.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',result))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc())); raise


if __name__=='__main__':run_cli(main)

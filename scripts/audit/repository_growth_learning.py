"""Exercise all connected cohorts through a real candidate and exact rollback gate."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.git_source_memory import open_memory
from audit.repository_growth_memory import authenticate
from audit.repository_relation_learning import retained
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli,take_lease
from model.repository_growth_data import GrowthData,roster
from model import repository_relation_checkpoint as checkpoint
from model.loop_training_checkpoint import fingerprint,tensor_fingerprint
from model.repository_retention import compare,measurements
from senses.contract import SenseError
from senses.trajectory_population import read_pinned
import torch


def read(path):return json.loads(path.read_bytes())


def views(data,model):
    return {group:data.evaluate(model,group) for group in data.order}


def decision(data,before,after,old_loss,new_loss):
    old_keys=set(data.stage(data.order[-2],data.parent)['keys'])
    old={g+'|'+k:r for g,rows in before.items() for k,r in rows.items() if k in old_keys}
    new={g+'|'+k:r for g,rows in after.items() for k,r in rows.items() if k in old_keys}
    regressions=compare(measurements(old),measurements(new))
    return dict(retained_checks=len(measurements(old)),regressions=regressions,
        objective_before=old_loss,objective_after=new_loss,
        accepted=not regressions and new_loss<old_loss,serving=False,scientific_admission=False)


def objective(data,model):
    last=data.order[-1]
    keys=[k for k,e in data.examples.items() if e['partition']==last]
    with torch.no_grad():return sum(float(data.losses(model,k,last)) for k in keys)/len(keys)


def proposal(data,model,optimizer):
    last=data.order[-1];keys=data.stage(last,model)['keys']
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for key in keys:
        loss=data.losses(model,key,last)/len(keys)
        check(bool(torch.isfinite(loss)),'GROWTH_LEARNING_LOSS_FINITE')
        loss.backward();losses.append(float(loss.detach()))
    parameters=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(parameters,checkpoint.RECIPE['gradient_clip'])
    check(bool(torch.isfinite(norm)),'GROWTH_LEARNING_GRADIENT_FINITE')
    optimizer.step();torch.cuda.synchronize(data.device)
    return dict(examples=len(keys),loss=sum(losses),gradient_norm=float(norm),
                fingerprint=fingerprint(model,optimizer))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('learn','cold'),required=True)
    a=p.parse_args();started=time.perf_counter()
    config=read(a.config);root=Path(config['root']);source=Path(config['source'])
    proof=read_pinned(Path(config['proof']),config['proof_pin'])
    check(sha((source/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((source/'snapshot-final.json').read_bytes())==proof['snapshot'] and proof['cold_replay_equal'],
          'GROWTH_LEARNING_SOURCE')
    recipes=authenticate(read(source/'config.json'))
    if a.stage=='learn':
        check(not root.exists(),'GROWTH_LEARNING_EXISTS');root.mkdir()
        write_committed(root/'config.json',config)
    else:check(read(root/'config.json')==config,'GROWTH_LEARNING_CONFIG')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-growth-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure();authenticated=time.perf_counter()
    session,sessions=open_memory(recipes,source,read(source/'snapshot-final.json'))
    original=digest(session.snapshot());restored=time.perf_counter()
    print(json.dumps(dict(stage=a.stage,restored=session.journal.count)),flush=True)
    data=GrowthData(session,config['populations'])
    model,optimizer=checkpoint.start(data)
    legacy=retained(session,model)
    check(legacy==retained(session,data.parent),'GROWTH_LEARNING_LEGACY')
    controls=[]
    def refuse(name,code,fn):
        try:fn()
        except (SenseError,ValueError) as error:check(code in str(error),'GROWTH_LEARNING_CONTROL:'+str(error))
        else:raise AssertionError('GROWTH_LEARNING_NEGATIVE_ACCEPTED:'+name)
        check(digest(session.snapshot())==original,'GROWTH_LEARNING_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))
    refuse('missing_population','GROWTH_DATA_MISSING_POPULATION',lambda:roster(session,config['populations'][:-1]))
    refuse('duplicate_population','GROWTH_DATA_DUPLICATE_POPULATION',
           lambda:roster(session,config['populations']+[config['populations'][0]]))
    refuse('reserved_group','GROWTH_DATA_PARTITION',lambda:data.stage('reserved_evaluation',model))
    future=next(k for k,e in data.examples.items() if e['partition']==data.order[-1])
    refuse('future_group','RELATION_FUTURE_PARTITION',lambda:data.forward(model,future,data.order[0]))
    first=next(iter(data.examples));entry=data.examples[first]
    provider=data.providers[entry['partition']]
    cutoff=provider.population.fields[first]['input_available_by']
    refuse('future_echo','REPOSITORY_NATIVE_FUTURE',lambda:provider.materialize(first,'echo',cutoff))
    refuse('unadmitted_assertion','REPOSITORY_NATIVE_UNADMITTED',
           lambda:provider.materialize(first,'input',cutoff,scientific_admission=True))
    changed=copy.deepcopy(entry['batch']);changed.seal=changed.stamps()
    changed[data.specs[0].name].values[0,0,0]+=1
    refuse('changed_input','RELATION_INPUT_TENSOR_CHANGED',lambda:changed.checked(data.specs))
    for group in data.order:data.stage(group,model)
    materialized=time.perf_counter()
    census=dict(examples=len(data.examples),projects=len({e['task'] for e in data.examples.values()}),
        groups=[dict(id=g,examples=len(data.stage(g,model)['keys']),regions=len(data.stage(g,model)['regions']))
                for g in data.order],native_target_absent=sum(e['absent'] for e in data.examples.values()),
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        max_input_windows={s.name:max(int((~e['batch'][s.name].padding[0]).sum()) for e in data.examples.values())
                           for s in data.specs})
    print(json.dumps(dict(materialized=census)),flush=True)
    if a.stage=='learn':
        initial=checkpoint.save(root/'initial',model,optimizer,data,0,None)
        before=views(data,model);old_loss=objective(data,model)
        initial_parameters=tensor_fingerprint(dict(model.named_parameters()))
        step=proposal(data,model,optimizer)
        after=views(data,model);new_loss=objective(data,model)
        verdict=decision(data,before,after,old_loss,new_loss)
        proposed=checkpoint.save(root/'proposal',model,optimizer,data,1,initial)
        check(retained(session,model)==legacy,'GROWTH_LEARNING_LEGACY_CHANGED')
        proposed_parameters=tensor_fingerprint(dict(model.named_parameters()))
        changes=[n for n,v in proposed_parameters.items() if v!=initial_parameters[n]]
        check(changes and all(n.startswith('query._repository.') for n in changes),'GROWTH_LEARNING_SCOPE')
        # Proposed state stays inspectable; a failed gate always restores the predecessor.
        if not verdict['accepted']:model,optimizer,_=checkpoint.load(root/'initial',data,initial)
        final=checkpoint.save(root/'final',model,optimizer,data,int(verdict['accepted']),proposed)
        accepted_views=after if verdict['accepted'] else before
        check(views(data,model)==accepted_views,'GROWTH_LEARNING_ROLLBACK')
        result=dict(binding=data.binding,census=census,controls=controls,before=before,after=after,
            decision=verdict,proposal=step,checkpoints=dict(initial=initial,proposal=proposed,final=final),
            final_fingerprint=fingerprint(model,optimizer),legacy=legacy,changed_parameters=changes,
            withheld_action=data.evaluate(model,data.order[-1],intervention=True),
            driver_sha256=sha(Path(__file__).read_bytes()),serving=False,scientific_admission=False,
            learned_capability=False,number_status='measured',acceptance='author-checked')
        write_committed(root/'result.json',result)
    else:
        result=read(root/'result.json')
        check(result['binding']==data.binding and result['census']==census and result['controls']==controls,
              'GROWTH_LEARNING_COLD_BINDING')
        model,optimizer,_=checkpoint.load(root/'initial',data,result['checkpoints']['initial'])
        check(views(data,model)==result['before'],'GROWTH_LEARNING_COLD_INITIAL')
        old_loss=objective(data,model)
        check(proposal(data,model,optimizer)==result['proposal'],'GROWTH_LEARNING_COLD_PROPOSAL')
        check(views(data,model)==result['after'],'GROWTH_LEARNING_COLD_OUTPUT')
        check(decision(data,result['before'],result['after'],old_loss,objective(data,model))==result['decision'],
              'GROWTH_LEARNING_COLD_DECISION')
        model,optimizer,_=checkpoint.load(root/'final',data,result['checkpoints']['final'])
        expected=result['after'] if result['decision']['accepted'] else result['before']
        check(views(data,model)==expected and fingerprint(model,optimizer)==result['final_fingerprint'],
              'GROWTH_LEARNING_COLD_FINAL')
        check(retained(session,model)==result['legacy'],'GROWTH_LEARNING_COLD_LEGACY')
    refuse('wrong_checkpoint','RELATION_CHECKPOINT_PIN',
           lambda:checkpoint.load(root/'initial',data,'0'*64))
    data.fresh();check(digest(session.snapshot())==original,'GROWTH_LEARNING_MEMORY_CHANGED')
    for s in sessions:s.close()
    check(sha((source/'experience.sqlite3').read_bytes())==proof['journal'],'GROWTH_LEARNING_JOURNAL_CHANGED')
    ended=time.perf_counter()
    report=dict(stage=a.stage,result=sha((root/'result.json').read_bytes()),controls=controls,
        journal=proof['journal'],snapshot=proof['snapshot'],phases=dict(authentication=authenticated-started,
        replay=restored-authenticated,materialization=materialized-restored,proposal_readback=ended-materialized),
        wall_seconds=ended-started,acceptance='author-checked',serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-readback.json'),report)
    print(json.dumps(dict(stage=a.stage,result=report['result'],accepted=result['decision']['accepted'],
        regressions=len(result['decision']['regressions']),wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':run_cli(main)

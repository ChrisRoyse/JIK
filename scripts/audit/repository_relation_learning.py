"""Author-check two bounded updates from persisted repository experience, without promotion."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_cohort_open import authenticate,open_memory
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli,take_lease
from model.constellation_world import ConstellationWorld
from model.loop_training_checkpoint import fingerprint,tensor_fingerprint
from model.repository_relation_data import RelationData
from model import repository_relation_checkpoint as checkpoint
from model.tool_context_regions import attach
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def retained(session,model):
    """Replay the active predictor on every prior source-bound learning example."""
    results = {}
    with torch.no_grad():
        for key,e in session.attempt_data.examples.items():
            original = e['view']
            with torch.random.fork_rng(devices=[]):
                view = ConstellationWorld(model.core,original.references(),world_source_id=original.world_source_id)
            view.to(model.core.initial.device).eval()
            attach(view,model)
            output = view(e['native'],(view.bind_state(e['state']),),allowed=e['allowed'],phase=2,
                event_ids=[digest(dict(retained_probe=key))],evidence_origin=['source_observation'],
                history=e.get('history',getattr(original,'context_history',None)),advance_phase=False)
            results[key] = tensor_fingerprint(dict(regions=output['regions'],slots=output['slots'],
                identity_regions=output['identity_regions'],identity_slots=output['identity_slots'],
                verdict=output['predicted_verdict'],states=[s.hidden for s in output['states']]))
    return results


def step(data,model,optimizer,partition,index):
    began = time.perf_counter()
    keys = data.stage(partition,model)['keys']
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for key in keys:
        loss = data.losses(model,key,partition)/len(keys)
        check(bool(torch.isfinite(loss)), 'RELATION_LOSS_FINITE')
        loss.backward()
        losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    check(bool(torch.isfinite(norm)), 'RELATION_GRADIENT_FINITE')
    optimizer.step()
    torch.cuda.synchronize(model.core.initial.device)
    row = dict(step=index,partition=partition,loss=sum(losses),gradient_norm=float(norm),
               wall_seconds=time.perf_counter()-began)
    print(row,flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('learn','cold'),required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = read(args.config)
    root,parent = Path(config['root']),Path(config['source'])
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    check(sha((parent/'experience.sqlite3').read_bytes()) == proof['journal'] and
          sha((parent/'snapshot-final.json').read_bytes()) == proof['snapshot'] and proof['cold_replay_equal'],
          'RELATION_MEMORY_PARENT')
    for path,pin in proof['immutable_after'].items():
        check(sha(Path(path).read_bytes())==pin, 'RELATION_ANCESTOR_PIN')
    recipes = authenticate(read(parent/'config.json'))
    if args.stage=='learn':
        check(not root.exists(),'RELATION_OUTPUT_EXISTS')
        root.mkdir()
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json')==config,'RELATION_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-relation-'+args.stage)
    (root/('gpu-'+args.stage+'.txt')).open('x').write(gpu)
    configure()
    authenticated = time.perf_counter()
    session,sessions = open_memory(recipes,parent,read(parent/'snapshot-final.json'))
    before_snapshot = digest(session.snapshot())
    restored = time.perf_counter()
    print(dict(stage=args.stage,restored=session.journal.count),flush=True)
    data = RelationData(session,config['native'],config['native_proof'],config['native_proof_pin'])
    model,optimizer = checkpoint.start(data)
    parent_retained = retained(session,data.parent)
    check(retained(session,model)==parent_retained,'RELATION_ZERO_UPDATE_RETENTION')
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'RELATION_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('RELATION_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    first = next(iter(data.examples))
    e = data.examples[first]
    changed = copy.deepcopy(e['batch']);changed.seal=changed.stamps()
    changed.description['fields'][data.specs[0].name][0]['role']='recorded_patch'
    refuse('changed_role','RELATION_INPUT_BINDING',lambda:changed.checked(data.specs))
    changed = copy.deepcopy(e['batch']);changed.seal=changed.stamps()
    changed[data.specs[0].name].values[0,0,0] += 1
    refuse('changed_native','RELATION_INPUT_TENSOR_CHANGED',lambda:changed.checked(data.specs))
    refuse('reserved_partition','RELATION_PARTITION',lambda:data.stage('evaluation',model))
    provider = data.providers[e['partition']]
    cutoff = provider.population.fields[first]['input_available_by']
    refuse('future_echo','REPOSITORY_NATIVE_FUTURE',lambda:provider.materialize(first,'echo',cutoff))
    refuse('unadmitted_assertion','REPOSITORY_NATIVE_UNADMITTED',
           lambda:provider.materialize(first,'input',cutoff,scientific_admission=True))
    materialized = time.perf_counter()
    results = dict(binding=data.binding,code=checkpoint.code_pins(),recipe=checkpoint.RECIPE,
        parent_retained=parent_retained,controls=controls,stages={},checkpoints={},steps=[])
    phase_times = dict(authentication=authenticated-started,lineage_restore=restored-authenticated,
                       materialization_and_migration=materialized-restored)
    if args.stage=='learn':
        cursor = time.perf_counter()
        initial = checkpoint.save(root/'initial',model,optimizer,data,0,None)
        results['checkpoints']['initial']=initial
        results['parent_parameters']=tensor_fingerprint(dict(model.named_parameters()))
        for j,partition in enumerate(('learn_1','learn_2')):
            stage = data.stage(partition,model)
            if j==0:
                future = next(k for k,e in data.examples.items() if e['partition']=='learn_2')
                refuse('future_partition','RELATION_FUTURE_PARTITION',lambda:data.forward(model,future,partition))
            before = data.evaluate(model,partition)
            with torch.no_grad():
                for key in stage['keys']:
                    out = data.forward(model,key,partition)
                    check(all(not bool(t.any()) for t in out['repository_residuals'].values()) if j==0 else True,
                          'RELATION_INITIAL_NONZERO')
            row = dict(description=stage['description'],world=stage['view'].world_id,
                       before=before,absence=[k for k in stage['keys'] if data.examples[k]['absent']])
            for k in range(1,9):
                results['steps'].append(step(data,model,optimizer,partition,j*8+k))
                if k==4:
                    name=partition+'-mid'
                    results['checkpoints'][name]=checkpoint.save(root/name,model,optimizer,data,j*8+k,
                        initial if j==0 else results['checkpoints']['learn_1-final'])
            row['after']=data.evaluate(model,partition)
            row['withheld_action']=data.evaluate(model,partition,intervention=True)
            row['retained_previous']=data.evaluate(model,'learn_1') if j else None
            check(retained(session,model)==parent_retained,'RELATION_OLD_BEHAVIOR_DRIFT')
            name=partition+'-final'
            results['checkpoints'][name]=checkpoint.save(root/name,model,optimizer,data,(j+1)*8,
                                                        results['checkpoints'][partition+'-mid'])
            results['stages'][partition]=row
            write_committed(root/(partition+'-result.json'),row)
        results['final_parameters']=tensor_fingerprint(dict(model.named_parameters()))
        changed_names=[n for n,p in results['final_parameters'].items() if p!=results['parent_parameters'][n]]
        check(changed_names and all(n.startswith('query._repository.') for n in changed_names),
              'RELATION_UPDATE_SCOPE')
        results['changed_parameters']=changed_names
        results['final_fingerprint']=fingerprint(model,optimizer)
        rollback,_,_ = checkpoint.load(root/'initial',data,initial)
        check(data.evaluate(rollback,'learn_1')==results['stages']['learn_1']['before'] and
              retained(session,rollback)==parent_retained,'RELATION_ROLLBACK')
        refuse('wrong_checkpoint_pin','RELATION_CHECKPOINT_PIN',lambda:checkpoint.load(root/'initial',data,'0'*64))
        phase_times['learning_evaluation_persistence']=time.perf_counter()-cursor
        write_committed(root/'result.json',results)
    else:
        expected = read(root/'result.json')
        check(expected['binding']==data.binding and expected['code']==checkpoint.code_pins(),
              'RELATION_COLD_BINDING')
        cursor=time.perf_counter()
        for j,partition in enumerate(('learn_1','learn_2')):
            model,optimizer,_ = checkpoint.load(root/(partition+'-mid'),data,
                                                expected['checkpoints'][partition+'-mid'])
            for k in range(5,9):
                step(data,model,optimizer,partition,j*8+k)
            final_manifest=read(root/(partition+'-final')/'manifest.json')
            check(fingerprint(model,optimizer)==final_manifest['fingerprint'],'RELATION_RESUMED_STEP')
            check(data.evaluate(model,partition)==expected['stages'][partition]['after'],'RELATION_COLD_SCORES')
            check(retained(session,model)==parent_retained,'RELATION_COLD_RETENTION')
        loaded,_,_ = checkpoint.load(root/'learn_2-final',data,expected['checkpoints']['learn_2-final'])
        check(data.evaluate(loaded,'learn_1')==expected['stages']['learn_2']['retained_previous'],
              'RELATION_COLD_PREVIOUS')
        rollback,_,_ = checkpoint.load(root/'initial',data,expected['checkpoints']['initial'])
        check(data.evaluate(rollback,'learn_1')==expected['stages']['learn_1']['before'],'RELATION_COLD_ROLLBACK')
        phase_times['restart_evaluation_rollback']=time.perf_counter()-cursor
        results=dict(resumed_both_increments=True,final_load_equal=True,rollback_equal=True,
                     parent_retained_equal=True,result=sha((root/'result.json').read_bytes()))
    data.fresh()
    check(digest(session.snapshot())==before_snapshot and sha((parent/'experience.sqlite3').read_bytes())==proof['journal'],
          'RELATION_SOURCE_MUTATED')
    for s in sessions:s.close()
    ended=time.perf_counter()
    phase_times['final_source_readback']=ended-started-sum(phase_times.values())
    report=dict(stage=args.stage,wall_seconds=ended-started,phases=phase_times,controls=controls,
        source_unchanged=True,serving=False,scientific_admission=False,learned_repair=False,
        useful_new_capability_demonstrated=False,acceptance='author-checked',number_status='measured',
        result=sha((root/'result.json').read_bytes()),cold=results if args.stage=='cold' else None)
    write_committed(root/(args.stage+'-readback.json'),report)
    print(report,flush=True)


if __name__=='__main__':
    run_cli(main)

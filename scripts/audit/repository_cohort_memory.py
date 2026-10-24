"""Append two grounded intake partitions, preserve prior state, and replay cold."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_cohort_open import authenticate,prepare,open_memory
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from lens.lenscommon import run_cli,take_lease
from senses.contract import SenseError
from senses.trajectory_population import read_pinned


def retained(old,final):
    check(all(final[k] == v for k,v in old.items() if k not in ('head','events','memory')),
          'COHORT_MEMORY_OLD_MODEL_OR_CONTEXT')
    check(all(final['memory']['states'][k] == v for k,v in old['memory']['states'].items()),
          'COHORT_MEMORY_OLD_EPISODE')
    check(final['memory']['grounded_source_statements'] == old['memory']['grounded_source_statements'] and
          final['memory']['epistemic_distinction_learned'] == old['memory']['epistemic_distinction_learned'],
          'COHORT_MEMORY_OLD_GROUNDING')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('intake','cold'),required=True)
    a = p.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,parent = Path(config['root']),Path(config['source'])
    recipes = authenticate(config)
    proof = read_pinned(Path(config['native_proof']),config['native_proof_pin'])
    check(proof['native'] == config['native'] and proof['source_actions'] > 0 and
          not proof['scientific_admission'] and not proof['learned_repair'],'COHORT_MEMORY_NATIVE_PROOF')
    old = read(parent/'snapshot-final.json')
    if a.stage == 'intake':
        prepare(config,recipes,root)
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'COHORT_MEMORY_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-cohort-memory-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = old if a.stage == 'intake' else read(root/'snapshot-final.json')
    authenticated = time.perf_counter()
    session,sessions = open_memory(recipes,root,saved)
    opened = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls = []
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'COHORT_MEMORY_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('COHORT_MEMORY_CONTROL_ACCEPTED:'+name)
        check(before == digest(session.snapshot()),'COHORT_MEMORY_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))
    if a.stage == 'intake':
        spec = config['native']
        refuse('second_partition_without_prefix','COHORT_MEMORY_PREFIX',
               lambda:session.observe_cohort(dict(**spec,partition='learn_2')))
        refuse('reserved_partition','COHORT_MEMORY_PARTITION',
               lambda:session.observe_cohort(dict(**spec,partition='evaluation')))
        pop = CohortPopulation(spec['population'],spec['population_pin'])
        ob = pop.observations
        first = next(k for k,e in ob.entries.items() if e['outcome']==0)
        for label,edit in (
            ('forged_success',lambda e:e['content'].update(outcome=1)),
            ('invented_why',lambda e:e.update(why='the patch proves the general rule')),
            ('historical_claim',lambda e:e.update(historical_occurrence_established=True))):
            ep = copy.deepcopy(ob.episode(first));edit(ep)
            refuse(label,'COHORT_OBSERVATION_EPISODE',lambda:ob.assess(ep))
        for partition in ('learn_1','learn_2'):
            chosen = dict(**spec,partition=partition)
            body = session.observe_cohort(chosen)
            payload = session.journal.payload(body['payload'])
            keys = {k for k,v in pop.fields.items() if v['partition']==partition}
            check(set(payload['bindings']) == keys and all(
                payload['bindings'][k][s] == proof['receipts'][k][s] for k in keys for s in ('input','echo')),
                'COHORT_MEMORY_NATIVE_JOIN')
            refuse('duplicate_'+partition,'COHORT_MEMORY_DUPLICATE',lambda:session.observe_cohort(chosen))
            write_committed(root/(partition+'-intake.json'),dict(event=body,payload=payload))
            write_committed(root/(partition+'-snapshot.json'),session.snapshot())
            retained(old,session.snapshot())
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    retained(old,final)
    check(final == read(root/'snapshot-final.json'),'COHORT_MEMORY_COLD_REPLAY')
    added = set(final['memory']['states'])-set(old['memory']['states'])
    expected = {ep for v in final['repository_cohort_intakes'].values() for ep in v['episodes'].values()}
    check(added == expected and len(added) == proof['source_actions'] and
          final['events'] == old['events']+2 and
          set(final)-set(old) == {'repository_cohort_intakes'},'COHORT_MEMORY_COVERAGE')
    result = dict(stage=a.stage,head=session.head,events=session.journal.count,added=len(added),
        source_head=old['head'],source_events=old['events'],controls=controls,
        old_contexts_and_models_unchanged=True,old_episodes_unchanged=True,
        native_measured=True,learned_repair=False,serving=False,scientific_admission=False,
        snapshot=sha((root/'snapshot-final.json').read_bytes()),replay_equal=True,acceptance='author-checked')
    for s in sessions:
        s.close()
    finished = time.perf_counter()
    result.update(wall_seconds=finished-started,phases=dict(authentication=authenticated-started,
        existing_lineage_replay=opened-authenticated,intake_and_readback=finished-opened))
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    run_cli(main)

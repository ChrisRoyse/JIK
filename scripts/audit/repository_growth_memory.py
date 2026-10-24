"""Continue connected source memory with another pair of grounded repair partitions."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import git_source_memory as source
from audit.fsv_identity_loop import configure
from audit.repository_memory_readback import read
from audit.state_cycle_open import prepare
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from lens.lenscommon import run_cli,take_lease
from senses.contract import SenseError
from senses.trajectory_population import read_pinned


def authenticate(config):
    check(config['schema'] == 'repository-growth-memory-v1','GROWTH_MEMORY_SCHEMA')
    parent = Path(config['source'])
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    for name,pin in config['parent_files'].items():
        check(sha((parent/name).read_bytes()) == pin,'GROWTH_MEMORY_PARENT')
    check(proof['journal'] == config['parent_files']['experience.sqlite3'] and
          proof['snapshot'] == config['parent_files']['snapshot-final.json'] and
          proof['cold_replay_equal'],'GROWTH_MEMORY_PARENT_PROOF')
    for name,pin in proof['immutable_after'].items():
        check(sha(Path(name).read_bytes()) == pin,'GROWTH_MEMORY_ANCESTOR')
    if config['parent_kind'] == 'git_source_memory':
        check(proof['schema'] == 'git-source-memory-readback-v1','GROWTH_MEMORY_PARENT_KIND')
        return source.authenticate(read(parent/'config.json'))
    check(config['parent_kind'] == 'repository_growth_memory' and
          proof['schema'] == 'repository-growth-memory-readback-v1','GROWTH_MEMORY_PARENT_KIND')
    return authenticate(read(parent/'config.json'))


def retained(old,final):
    excepted = ('head','events','memory','repository_cohort_intakes')
    check(all(final[k] == v for k,v in old.items() if k not in excepted),'GROWTH_MEMORY_OLD_MODEL')
    check(all(final['repository_cohort_intakes'][k] == v
              for k,v in old.get('repository_cohort_intakes',{}).items()),'GROWTH_MEMORY_OLD_INTAKES')
    check(all(final['memory']['states'][k] == v for k,v in old['memory']['states'].items()) and
          final['memory']['grounded_source_statements'] == old['memory']['grounded_source_statements'] and
          final['memory']['epistemic_distinction_learned'] == old['memory']['epistemic_distinction_learned'],
          'GROWTH_MEMORY_OLD_EVIDENCE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('intake','cold'),required=True)
    a = p.parse_args()
    start = time.perf_counter()
    config = read(a.config)
    root,parent = Path(config['root']),Path(config['source'])
    recipes = authenticate(config)
    proof = read_pinned(Path(config['native_proof']),config['native_proof_pin'])
    check(proof['native'] == config['native'] and proof['source_actions'] > 0 and
          not proof['scientific_admission'] and not proof['learned_repair'],'GROWTH_MEMORY_NATIVE_PROOF')
    old = read(parent/'snapshot-final.json')
    if a.stage == 'intake':
        prepare(config,recipes,root)
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'GROWTH_MEMORY_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-growth-memory-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    authenticated = time.perf_counter()
    session,sessions = source.open_memory(recipes,root,
        old if a.stage == 'intake' else read(root/'snapshot-final.json'))
    opened = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls = []
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'GROWTH_MEMORY_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('GROWTH_MEMORY_CONTROL_ACCEPTED:'+name)
        check(before == digest(session.snapshot()),'GROWTH_MEMORY_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code,before=before,after=before))
    spec = config['native']
    if a.stage == 'intake':
        refuse('missing_prefix','COHORT_MEMORY_PREFIX',
               lambda:session.observe_cohort(dict(**spec,partition='learn_2')))
        refuse('reserved_training','COHORT_MEMORY_PARTITION',
               lambda:session.observe_cohort(dict(**spec,partition='reserved_evaluation')))
        pop = CohortPopulation(spec['population'],spec['population_pin'])
        check(all(any(f['partition'] == p for f in pop.fields.values())
                  for p in ('learn_1','learn_2')),'GROWTH_MEMORY_EMPTY_PARTITION')
        failed = next(k for k,v in pop.observations.entries.items() if v['outcome'] == 0)
        forged = copy.deepcopy(pop.observations.episode(failed))
        forged['content']['outcome'] = 1
        refuse('forged_success','COHORT_OBSERVATION_EPISODE',lambda:pop.observations.assess(forged))
        for partition in ('learn_1','learn_2'):
            chosen = dict(**spec,partition=partition)
            body = session.observe_cohort(chosen)
            payload = session.journal.payload(body['payload'])
            keys = {k for k,v in pop.fields.items() if v['partition'] == partition}
            check(set(payload['bindings']) == keys and all(payload['bindings'][k][s] == proof['receipts'][k][s]
                for k in keys for s in ('input','echo')),'GROWTH_MEMORY_NATIVE_JOIN')
            refuse('duplicate_'+partition,'COHORT_MEMORY_DUPLICATE',lambda:session.observe_cohort(chosen))
            write_committed(root/(partition+'-intake.json'),dict(event=body,payload=payload))
            write_committed(root/(partition+'-snapshot.json'),session.snapshot())
            retained(old,session.snapshot())
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    retained(old,final)
    check(final == read(root/'snapshot-final.json'),'GROWTH_MEMORY_COLD')
    added = set(final['memory']['states'])-set(old['memory']['states'])
    keys = {digest(dict(**spec,partition=p)) for p in ('learn_1','learn_2')}
    intakes = final['repository_cohort_intakes']
    check(set(intakes)-set(old.get('repository_cohort_intakes',{})) == keys and
          added == {e for k in keys for e in intakes[k]['episodes'].values()} and
          len(added) == proof['source_actions'] and final['events'] == old['events']+2,
          'GROWTH_MEMORY_COVERAGE')
    for s in sessions:
        s.close()
    finish = time.perf_counter()
    result = dict(stage=a.stage,head=final['head'],events=final['events'],added=len(added),
        source_head=old['head'],source_events=old['events'],controls=controls,
        old_contexts_and_models_unchanged=True,old_intakes_unchanged=True,
        native_measured=True,learned_repair=False,scientific_admission=False,serving=False,
        snapshot=sha((root/'snapshot-final.json').read_bytes()),
        journal=sha((root/'experience.sqlite3').read_bytes()),cold_replay_equal=True,
        phases=dict(authentication=authenticated-start,replay=opened-authenticated,intake=finish-opened),
        wall_seconds=finish-start,acceptance='author-checked')
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    run_cli(main)

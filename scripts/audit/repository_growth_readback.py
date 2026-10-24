"""Independently reconstruct exclusion provenance and check the actual selected cohort."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.process_conditions import write_committed
from claims.repository_cohort_input import RepositoryCohort
from senses.trajectory_population import read_pinned


def check_catalogue(catalogue,recipe):
    original = RepositoryCohort(**recipe['original'])
    c = original.config
    before = read_pinned(Path(c['contracts']),c['contracts_pin'])
    entries = copy.deepcopy(before['contracts'])
    projects = set()
    protected = set()
    for binding in recipe['prior']:
        cohort = RepositoryCohort(**binding)
        for row in cohort.cases:
            project = row['repo'].casefold()
            check(project not in projects,'GROWTH_READER_PRIOR_OVERLAP')
            projects.add(project)
            if row['partition'] in ('development','reserved_evaluation'):
                protected.add(project)
            entries.append(dict(kind='previous_cohort_project_exclusion',repo=row['repo'],
                task=row['task'],partition=row['partition'],
                source=dict(root=str(cohort.root),pin=cohort.pin,index=row['index'])))
    check(catalogue == dict(schema='repository-selection-exclusions-v1',
        benchmark=before['benchmark'],contracts=entries,execution_contract=False,
        original=dict(path=c['contracts'],sha256=c['contracts_pin']),prior=recipe['prior']),
        'GROWTH_READER_CATALOGUE')
    return projects,protected


def run(root,pin,cohort_root,cohort_pin,proof_path,proof_pin,output):
    start = time.perf_counter()
    r = read_pinned(root/'report.json',pin)
    check(r['schema'] == 'repository-growth-v1' and
          json.loads((root/'COMMITTED.json').read_bytes()) == dict(sha256=pin),
          'GROWTH_READER_COMMITMENT')
    check(set(r['files']) == {'recipe.json','config.json','exclusions.json'},'GROWTH_READER_FILES')
    before = {}
    for name,entry in r['files'].items():
        raw = (root/name).read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw)) == entry,'GROWTH_READER_FILE')
        before[str(root/name)] = sha(raw)
    recipe = json.loads((root/'recipe.json').read_bytes())
    catalogue = json.loads((root/'exclusions.json').read_bytes())
    projects,protected = check_catalogue(catalogue,recipe)
    cohort = RepositoryCohort(cohort_root,cohort_pin)
    config = json.loads((root/'config.json').read_bytes())
    check(cohort.config == config and config['contracts'] == str(root/'exclusions.json') and
          config['contracts_pin'] == r['files']['exclusions.json']['sha256'] and
          all(config[k] == recipe[k] for k in ('journal','proof','proof_pin')),
          'GROWTH_READER_CONFIG')
    proof = read_pinned(proof_path,proof_pin)
    check(proof['schema'] == 'repository-cohort-readback-v1' and proof['cohort'] == cohort_pin and
          len(proof['controls']) == 7,'GROWTH_READER_SELECTION_PROOF')
    check(projects <= set(cohort.exposure['projects']) and
          not projects & {c['repo'].casefold() for c in cohort.cases},'GROWTH_READER_EXCLUSION')
    check(len(projects) == r['prior_projects'] and len(protected) == r['protected_prior_projects'],
          'GROWTH_READER_COUNTS')
    for binding in recipe['prior']:
        old = RepositoryCohort(**binding)
        before.update({str(old.root/n):v['sha256'] for n,v in old.report['files'].items()})
    controls = []
    for name in ('missing_previous_project','changed_reserved_role','exclusion_as_execution'):
        bad = copy.deepcopy(catalogue)
        if name == 'missing_previous_project':
            bad['contracts'].pop()
        elif name == 'changed_reserved_role':
            bad['contracts'][-1]['partition'] = 'learn_1'
        else:
            bad['execution_contract'] = True
        try:
            check_catalogue(bad,recipe)
        except ValueError as exc:
            check('GROWTH_READER_CATALOGUE' in str(exc),'GROWTH_READER_CONTROL_CODE')
        else:
            raise AssertionError('GROWTH_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code='GROWTH_READER_CATALOGUE'))
    check(before == {n:sha(Path(n).read_bytes()) for n in before},'GROWTH_READER_MUTATED')
    result = dict(schema='repository-growth-readback-v1',growth=pin,cohort=cohort_pin,
        selection_proof=proof_pin,prior_projects=len(projects),protected_prior_projects=len(protected),
        new_projects=len(cohort.cases),new_learning_projects=len(cohort.training_cases('learn_1'))+
            len(cohort.training_cases('learn_2')),controls=controls,unchanged=before,
        wall_seconds=time.perf_counter()-start,outcomes_read=False,scientific_admission=False,
        acceptance='author-checked',reader_code=sha(Path(__file__).read_bytes()))
    result_pin = write_committed(output,result)
    print(json.dumps(dict(pin=result_pin,prior_projects=len(projects),protected=len(protected),
        new_projects=len(cohort.cases),controls=len(controls))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort','proof','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('pin','cohort-pin','proof-pin'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    run(a.root,a.pin,a.cohort,a.cohort_pin,a.proof,a.proof_pin,a.output)

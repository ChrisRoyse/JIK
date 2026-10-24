"""Ground successful reference repairs without relabelling them as candidate attempts."""
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_cohort_input import RepositoryCohort
from claims.repository_cohort_observation import CohortObservations
from oracle.bound_rows import BoundRows
from oracle.prepare_swe import patch_paths
from oracle.replay_swe import recount
from oracle.repository_cohort_actions import open_witness
from senses.trajectory_population import read_pinned


def action_paths(raw,protected):
    paths=patch_paths(raw.decode())
    check(paths and not paths & protected,'DEMONSTRATION_ORACLE_PATCH')
    return sorted(paths)


class DemonstrationObservations(CohortObservations):
    def __init__(self,path,pin,proof,proof_pin):
        self.path,self.pin=Path(path),pin
        self.binding=dict(path=str(path),pin=pin,proof=str(proof),proof_pin=proof_pin)
        frontier=read_pinned(self.path,pin);parent=read_pinned(Path(proof),proof_pin)
        check(frontier['schema']=='repository-reference-frontier-v1' and
              parent['schema']=='repository-breadth-result-v1' and
              parent['reference_frontier']['pin']==pin and not frontier['already_in_learning_data'] and
              not frontier['heldout_evaluation'] and not frontier['learned_capability'],
              'DEMONSTRATION_SOURCE_PROOF')
        scripts=Path(__file__).resolve().parents[1]
        code={p.name:sha(p.read_bytes()) for p in (scripts/'oracle').glob('*.py')}
        self.entries,self.surfaces,self.files,self.gaps={},{},{},[];self.code=code
        sources={};order=[]
        for wave in parent['waves']:
            cohort=RepositoryCohort(wave['cohort']['root'],wave['cohort']['pin'])
            cases=cohort.training_cases('learn_1')+cohort.training_cases('learn_2')
            action_proof=read_pinned(Path(wave['actions_readback']['path']),wave['actions_readback']['pin'])
            actions=read_pinned(Path(action_proof['source']['path']),action_proof['source']['sha256'])
            scope=read_pinned(Path(actions['witness']['path']),actions['witness']['sha256'])
            check(scope['cohort']==dict(path=wave['cohort']['root'],sha256=wave['cohort']['pin']),
                  'DEMONSTRATION_COHORT')
            for case,item in zip(cases,scope['results'],strict=True):
                check(case['task']==item['task'] and case['partition']==item['partition'],
                      'DEMONSTRATION_TASK')
                witness=read_pinned(Path(item['path']),item['sha256'])
                if witness['status']=='observed':
                    check(case['task'] not in sources,'DEMONSTRATION_DUPLICATE_TASK')
                    sources[case['task']]=(case,item,witness,Path(scope['root']))
                    order.append(case['task'])
        check([r['task'] for r in frontier['references']]==order and
              len(order)==frontier['passing_reference_controls'],'DEMONSTRATION_DENOMINATOR')
        self.references={}
        roots={v[3] for v in sources.values()}
        check(len(roots)==1,'DEMONSTRATION_SOURCE_ROOT')
        rows=BoundRows(next(iter(roots)),[(v[0]['benchmark'],['problem_statement','patch','test_patch'])
                                          for v in sources.values()])
        for ref in frontier['references']:
            case,item,witness,root=sources[ref['task']]
            check(ref['witness']==item and ref['partition']==case['partition'] and
                  ref['role']=='reference_control_not_model_attempt' and ref['outcome']==1,
                  'DEMONSTRATION_REFERENCE_ROLE')
            plan,prepared,pp,spec,before,_,original,_=open_witness(root,item,witness,code)
            check(ref['prepared']==dict(path=str(prepared),pin=pp),'DEMONSTRATION_PREPARED')
            source=rows.read(case['benchmark'],['problem_statement','patch','test_patch'])
            protected=set(plan['allowed_paths']['test'])|set(spec['test_modules'])
            check(patch_paths(source['test_patch'])<=protected,'DEMONSTRATION_TEST_CONTRACT')
            raw=self.file(prepared/'reference.patch')
            check(raw==source['patch'].encode() and
                  ref['patch']==dict(path=str(prepared/'reference.patch'),pin=sha(raw)),
                  'DEMONSTRATION_PATCH')
            paths=action_paths(raw,protected)
            capture=Path(witness['execution'])/'reference'
            record=json.loads(self.file(capture/'execution.json'))
            check(ref['execution']==dict(path=str(capture/'execution.json'),
                                         pin=sha((capture/'execution.json').read_bytes())),
                  'DEMONSTRATION_EXECUTION')
            after=json.loads(self.file(capture/'after.json'))
            tests=recount(record,spec['required_tests'])
            check(record['container_id']==after['Id'] and tests['outcome']==1 and
                  len(spec['required_tests'])==ref['required_tests'],'DEMONSTRATION_OUTCOME')
            inputs=[dict(role='task_report',raw=source['problem_statement'].encode(),
                         source=dict(benchmark=case['benchmark'],field='problem_statement'))]
            missing=[];condition=before['reference']
            for name in paths:
                if name in original:
                    inputs.append(dict(role='original_source',raw=original[name],source=dict(
                        task=case['task'],base_commit=case['base_commit'],path=name,sha256=sha(original[name]))))
                else:missing.append(dict(path=name,side='original',state='absent'))
                if name in condition:
                    value=self.file(prepared/'reference'/name)
                    check(sha(value)==condition[name],'DEMONSTRATION_SOURCE')
                    inputs.append(dict(role='candidate_source',raw=value,
                        source=dict(path=str(prepared/'reference'/name),sha256=sha(value))))
                else:missing.append(dict(path=name,side='candidate',state='absent'))
            inputs.append(dict(role='recorded_patch',raw=raw,
                source=dict(path=str(prepared/'reference.patch'),sha256=sha(raw))))
            outputs=[]
            for stream in ('stdout','stderr'):
                filename=capture/(stream+'.bin');value=self.file(filename);surface=value
                if stream=='stdout':
                    surface,marker,envelope=value.partition(b'MNLM_EXECUTION_RESULT=')
                    check(marker and envelope.count(b'MNLM_EXECUTION_RESULT=')==0 and
                          json.loads(envelope)==record['result'],'DEMONSTRATION_ENVELOPE')
                outputs.append(dict(role=stream,raw=surface,source=dict(path=str(filename),sha256=sha(value)),
                                    source_span=[0,len(surface)]))
            key=case['task']+'/reference'
            self.entries[key]=dict(task=case['task'],partition=case['partition'],ordinal='reference',
                action_origin='supplied_reference_repair',independent_candidate_attempt=False,
                container=record['container_id'],image=spec['image'],tests=tests,outcome=tests['outcome'],
                source_snapshot=digest(condition),runtime=digest(spec['runtime']),
                required_tests=digest(spec['required_tests']),action_sha256=sha(raw),
                started=after['State']['StartedAt'],finished=after['State']['FinishedAt'])
            self.surfaces[key]=dict(input=inputs,echo=outputs,missing=missing,excluded_oracle_paths=sorted(protected))
            self.references[key]=ref
        self.receipts=[{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in rows.receipts]
        check(len({e['container'] for e in self.entries.values()})==len(self.entries) and
              sum(len(e['tests']['states']) for e in self.entries.values())==frontier['required_test_checks'],
              'DEMONSTRATION_EXECUTION_COVERAGE')

    def episode(self,key,evidence=()):
        body=super().episode(key,evidence);body.pop('id')
        body['who']['action_origin']='supplied_reference_repair'
        body.update(reference_demonstration=True,independent_candidate_attempt=False)
        return dict(id=digest(body),**body)

    def assess(self,episode):
        body=super().assess(episode);body.pop('id')
        body.update(checker='original-reference-demonstration-v1',reference_demonstration=True,
                    independent_candidate_attempt=False)
        return dict(id=digest(body),**body)

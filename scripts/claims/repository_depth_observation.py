"""Retain newly executed depth actions as scoped occurrences with causal surfaces."""
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_depth import DepthCohort
from claims.repository_cohort_observation import CohortObservations
from oracle.bound_rows import BoundRows
from oracle.prepare_swe import patch_paths
from oracle.replay_swe import recount
from oracle.repository_cohort_actions import open_witness
from oracle.repository_cohort_readback import read_action
from senses.trajectory_population import read_pinned


class DepthObservations(CohortObservations):
    def __init__(self,path,pin,proof,proof_pin):
        self.path,self.pin=Path(path),pin
        self.binding=dict(path=str(path),pin=pin,proof=str(proof),proof_pin=proof_pin)
        report=read_pinned(self.path,pin);checked=read_pinned(Path(proof),proof_pin)
        scripts=Path(__file__).resolve().parents[1]
        check(report['schema']=='repository-depth-actions-v1' and
              checked['schema']=='repository-depth-actions-readback-v1' and
              checked['source']==dict(path=str(path),sha256=pin) and checked['admitted'] is False,
              'DEPTH_OBSERVATION_PROOF')
        check(report['code']==sha((scripts/'claims/repository_depth_actions.py').read_bytes()) and
              checked['code']==sha((scripts/'audit/repository_depth_actions.py').read_bytes()),
              'DEPTH_OBSERVATION_CODE')
        self.cohort=DepthCohort(**report['selection']);root=Path(self.cohort.report['source_root'])
        check(checked['selection']==report['selection'],'DEPTH_OBSERVATION_SELECTION')
        cases=self.cohort.cases
        rows=BoundRows(root,[(c['benchmark'],['problem_statement']) for c in cases])
        code={p.name:sha(p.read_bytes()) for p in (scripts/'oracle').glob('*.py')}
        check(code==report['oracle_code'],'DEPTH_OBSERVATION_ORACLE_CODE')
        self.entries,self.surfaces,self.files,self.gaps={},{},{},[];self.code=code
        for case,later,cold in zip(cases,report['results'],checked['results'],strict=True):
            check(case['task']==later['task']==cold['task'] and
                  case['partition']==later['partition']==cold['partition'],'DEPTH_OBSERVATION_TASK')
            first=self.cohort.sources[case['task']]['item']
            witness=read_pinned(Path(first['path']),first['sha256'])
            plan,prepared,prep_pin,spec,before,modes,original,shared=open_witness(root,first,witness,code)
            check([a['ordinal'] for a in later['actions']]==list(range(case['prior_attempts'],len(case['attempts']))),
                  'DEPTH_OBSERVATION_COVERAGE')
            problem=rows.read(case['benchmark'],['problem_statement'])['problem_statement'].encode()
            protected=set(plan['allowed_paths']['test'])
            for item,recheck in zip(later['actions'],cold['actions'],strict=True):
                check(item['ordinal']==recheck['ordinal'] and item['status']==recheck['status'],
                      'DEPTH_OBSERVATION_ACTION')
                if item['status']=='gap':
                    gap=read_pinned(Path(item['path']),item['sha256'])
                    check(gap['task']==case['task'] and gap['attempt']==case['attempts'][item['ordinal']] and
                          gap['error']==recheck['error'],'DEPTH_OBSERVATION_GAP')
                    self.gaps.append(dict(task=case['task'],ordinal=item['ordinal'],source=item,error=gap['error']))
                    continue
                observed=read_action(item,case,self.cohort,spec,prepared,shared)
                check(observed==recheck,'DEPTH_OBSERVATION_RECOUNT')
                directory=Path(item['path']).parent;action=read_pinned(Path(item['path']),item['sha256'])
                self.retain(case,item['ordinal'],directory,action,spec,original,protected,problem)
        self.receipts=[{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in rows.receipts]
        check(len(self.entries)+len(self.gaps)==self.cohort.report['selected_actions'] and
              len({e['container'] for e in self.entries.values()})==len(self.entries),'DEPTH_OBSERVATION_DENOMINATOR')

    def retain(self,case,ordinal,directory,action,spec,original,protected,problem):
        capture=directory/'execution';raw=self.file(directory/'agent.patch');attempt=case['attempts'][ordinal]
        check(sha(raw)==attempt['sha256'] and raw==(self.cohort.root/attempt['path']).read_bytes(),
              'DEPTH_OBSERVATION_PATCH')
        after=json.loads(self.file(capture/'after.json'));record=json.loads(self.file(capture/'execution.json'))
        check(after['Id']==record['container_id'],'DEPTH_OBSERVATION_CONTAINER')
        inputs=[dict(role='task_report',raw=problem,
                     source=dict(benchmark=case['benchmark'],field='problem_statement'))]
        missing=[];condition=action['before']
        for name in sorted(patch_paths(raw.decode())-protected):
            if name in original:
                inputs.append(dict(role='original_source',raw=original[name],source=dict(task=case['task'],
                    base_commit=case['base_commit'],path=name,sha256=sha(original[name]))))
            else:missing.append(dict(path=name,side='original',state='absent'))
            if name in condition:
                value=self.file(directory/'agent'/name)
                check(sha(value)==condition[name],'DEPTH_OBSERVATION_SOURCE')
                inputs.append(dict(role='candidate_source',raw=value,
                    source=dict(path=str(directory/'agent'/name),sha256=sha(value))))
            else:missing.append(dict(path=name,side='candidate',state='absent'))
        inputs.append(dict(role='recorded_patch',raw=raw,source=dict(path=str(directory/'agent.patch'),sha256=sha(raw))))
        outputs=[]
        for stream in ('stdout','stderr'):
            filename=capture/(stream+'.bin');value=self.file(filename);surface=value
            if stream=='stdout':
                surface,marker,envelope=value.partition(b'MNLM_EXECUTION_RESULT=')
                check(marker and envelope.count(b'MNLM_EXECUTION_RESULT=')==0 and
                      json.loads(envelope)==record['result'],'DEPTH_OBSERVATION_ENVELOPE')
            outputs.append(dict(role=stream,raw=surface,source=dict(path=str(filename),sha256=sha(value)),
                                source_span=[0,len(surface)]))
        key=case['task']+'/'+str(ordinal)
        self.entries[key]=dict(task=case['task'],partition=case['partition'],ordinal=ordinal,
            container=record['container_id'],image=spec['image'],tests=recount(record,spec['required_tests'],
            allow_uncollectable=True),outcome=action['result']['outcome'],source_snapshot=digest(condition),
            runtime=digest(spec['runtime']),required_tests=digest(spec['required_tests']),action_sha256=attempt['sha256'],
            started=after['State']['StartedAt'],finished=after['State']['FinishedAt'])
        self.surfaces[key]=dict(input=inputs,echo=outputs,missing=missing,excluded_oracle_paths=sorted(protected))

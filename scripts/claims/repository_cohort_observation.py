"""Original recorded actions and observed consequences, never hidden-oracle model inputs."""
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_cohort_input import RepositoryCohort
from claims.repository_observation import SCOPE
from oracle.bound_rows import BoundRows
from oracle.prepare_swe import patch_paths
from oracle.replay_swe import recount
from oracle.repository_cohort_actions import open_witness
from oracle.repository_cohort_readback import read_action
from senses.trajectory_population import read_pinned


class CohortObservations:
    def __init__(self,path,pin,proof,proof_pin):
        self.path,self.pin = Path(path),pin
        self.binding = dict(path=str(path),pin=pin,proof=str(proof),proof_pin=proof_pin)
        report = read_pinned(self.path,pin)
        checked = read_pinned(Path(proof),proof_pin)
        check(checked['schema'] == 'repository-cohort-actions-readback-v1' and
              checked['source'] == dict(path=str(path),sha256=pin) and
              checked['admitted'] is False,'COHORT_OBSERVATION_PROOF')
        scope = read_pinned(Path(report['witness']['path']),report['witness']['sha256'])
        self.cohort = RepositoryCohort(scope['cohort']['path'],scope['cohort']['sha256'])
        cases = self.cohort.training_cases('learn_1')+self.cohort.training_cases('learn_2')
        rows = BoundRows(Path(scope['root']),[(c['benchmark'],['problem_statement']) for c in cases])
        code_root = Path(__file__).resolve().parents[1]/'oracle'
        code = {p.name:sha(p.read_bytes()) for p in code_root.glob('*.py')}
        check(code['repository_cohort_actions.py'] == report['code_sha256'] and
              code['repository_cohort_readback.py'] == checked['code_sha256'],'COHORT_OBSERVATION_CODE')
        self.entries,self.surfaces,self.files,self.gaps = {},{},{},[]
        self.code = code
        for case,first,later,cold in zip(cases,scope['results'],report['results'],checked['results'],strict=True):
            check(case['task'] == first['task'] == later['task'] == cold['task'] and
                  case['partition'] == later['partition'] == cold['partition'],'COHORT_OBSERVATION_PARTITION')
            witness = read_pinned(Path(first['path']),first['sha256'])
            if witness['status'] != 'observed':
                check(later['status'] == cold['status'] == 'gap','COHORT_OBSERVATION_GAP')
                self.gaps.append(dict(task=case['task'],partition=case['partition'],error=witness['error']))
                continue
            plan,prepared,prep_pin,spec,before,modes,original,shared = open_witness(
                Path(scope['root']),first,witness,code)
            execution = Path(witness['execution'])
            items = [(0,prepared/'agent.patch',prepared/'agent',execution/'agent',
                      witness['agent_outcome'],spec['conditions']['agent'])]
            check([a['ordinal'] for a in later['actions']] == [1,2,3],'COHORT_OBSERVATION_COVERAGE')
            for item in later['actions']:
                if item['status'] == 'gap':
                    self.gaps.append(read_pinned(Path(item['path']),item['sha256']))
                    continue
                observed = read_action(item,case,self.cohort,spec,prepared,shared)
                directory = Path(item['path']).parent
                action = read_pinned(Path(item['path']),item['sha256'])
                items.append((item['ordinal'],directory/'agent.patch',directory/'agent',
                              directory/'execution',observed['outcome'],action['before']))
            problem = rows.read(case['benchmark'],['problem_statement'])['problem_statement'].encode()
            protected = set(plan['allowed_paths']['test'])
            for ordinal,patch,tree,capture,outcome,condition in items:
                key = case['task']+'/'+str(ordinal)
                attempt = case['attempts'][ordinal]
                raw = self.file(patch)
                check(sha(raw) == attempt['sha256'] and raw == (self.cohort.root/attempt['path']).read_bytes(),
                      'COHORT_OBSERVATION_ACTION')
                after = json.loads(self.file(capture/'after.json'))
                record = json.loads(self.file(capture/'execution.json'))
                check(after['Id'] == record['container_id'],'COHORT_OBSERVATION_CONTAINER')
                source = dict(benchmark=case['benchmark'],field='problem_statement')
                inputs = [dict(role='task_report',raw=problem,source=source)]
                paths = sorted(patch_paths(raw.decode())-protected)
                missing = []
                for name in paths:
                    if name in original:
                        inputs.append(dict(role='original_source',raw=original[name],
                            source=dict(task=case['task'],base_commit=case['base_commit'],path=name,
                                        sha256=sha(original[name]))))
                    else:
                        missing.append(dict(path=name,side='original',state='absent'))
                    if name in condition:
                        value = self.file(tree/name)
                        check(sha(value) == condition[name],'COHORT_OBSERVATION_SOURCE')
                        inputs.append(dict(role='candidate_source',raw=value,
                            source=dict(path=str(tree/name),sha256=sha(value))))
                    else:
                        missing.append(dict(path=name,side='candidate',state='absent'))
                inputs.append(dict(role='recorded_patch',raw=raw,source=dict(path=str(patch),sha256=sha(raw))))
                outputs = []
                for stream in ('stdout','stderr'):
                    filename = capture/(stream+'.bin')
                    value = self.file(filename)
                    surface = value
                    if stream == 'stdout':
                        surface,marker,envelope = value.partition(b'MNLM_EXECUTION_RESULT=')
                        check(marker and envelope.count(b'MNLM_EXECUTION_RESULT=') == 0 and
                              json.loads(envelope) == record['result'],'COHORT_OBSERVATION_ENVELOPE')
                    outputs.append(dict(role=stream,raw=surface,
                        source=dict(path=str(filename),sha256=sha(value)),source_span=[0,len(surface)]))
                entry = dict(task=case['task'],partition=case['partition'],ordinal=ordinal,
                    container=record['container_id'],image=spec['image'],
                    tests=recount(record,spec['required_tests'],allow_uncollectable=True),outcome=outcome,
                    source_snapshot=digest(condition),runtime=digest(spec['runtime']),
                    required_tests=digest(spec['required_tests']),action_sha256=attempt['sha256'],
                    started=after['State']['StartedAt'],finished=after['State']['FinishedAt'])
                self.entries[key] = entry
                self.surfaces[key] = dict(input=inputs,echo=outputs,missing=missing,
                    excluded_oracle_paths=sorted(protected))
        self.receipts = [{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in rows.receipts]
        check(len({e['container'] for e in self.entries.values()}) == len(self.entries),'COHORT_OBSERVATION_REUSE')

    def file(self,path):
        check(not path.is_symlink(),'COHORT_OBSERVATION_SYMLINK')
        raw = path.read_bytes()
        entry = dict(bytes=len(raw),sha256=sha(raw))
        check(str(path) not in self.files or self.files[str(path)] == entry,'COHORT_OBSERVATION_DRIFT')
        self.files[str(path)] = entry
        return raw

    def episode(self,key,evidence=()):
        check(key in self.entries,'COHORT_OBSERVATION_KEY')
        e = self.entries[key]
        body = dict(claim_kind='event_occurrence',content=dict(outcome=e['outcome'],tests=e['tests']),
            source=dict(observation_set=self.pin,observation=key,task=e['task'],required_tests=e['required_tests']),
            evidence=list(evidence),event_time=e['finished'],
            who=dict(actor='isolated_repository_test_process',action_origin='recorded_source_attempt'),
            what=dict(condition=key,source_snapshot=e['source_snapshot'],runtime=e['runtime'],action=e['action_sha256']),
            where=dict(image=e['image'],container=e['container']),
            when=dict(clock='container_runtime_utc',started=e['started'],finished=e['finished']),
            why=None,why_status='unresolved',scope=SCOPE,independent_occurrence=True,
            native_measured=False,model_generated_action=False,historical_occurrence_established=False)
        return dict(id=digest(body),**body)

    def validate_episode(self,episode):
        key = episode.get('source',{}).get('observation')
        check(key in self.entries and episode == self.episode(key,episode.get('evidence',())) and
              not episode['evidence'],'COHORT_OBSERVATION_EPISODE')

    def assess(self,episode):
        self.validate_episode(episode)
        body = dict(checker='original-repository-cohort-occurrence-v1',episode=episode['id'],
            claim_kind='event_occurrence',scope=SCOPE,source=episode['source'],
            world_validity='scoped_observed_event',native_measured=False,
            model_generated_action=False,historical_occurrence_established=False)
        return dict(id=digest(body),**body)

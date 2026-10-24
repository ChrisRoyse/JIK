"""Scoped repository-test occurrences; retained repairs are not model-generated acts."""
from datetime import datetime
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from oracle.container_run import validate_config
from oracle.replay_swe import recount, verify_prepared
from oracle.test_contract import selectors
from senses.trajectory_population import publish, read_pinned

SCOPE = 'observed_original_task_tests_in_pinned_runtime'


def seal(run_root, summary_pin, output):
    run_root = Path(run_root).resolve()
    report = read_pinned(run_root/'summary.json', summary_pin)
    check(report['schema'] == 'action-repository-run-v1' and
          report['result']['status'] == 'observed' and not report['admitted'], 'REPOSITORY_RUN_SCOPE')
    files = {}
    for p in sorted(run_root.rglob('*')):
        check(not p.is_symlink(), 'REPOSITORY_SYMLINK')
        if p.is_file():
            raw = p.read_bytes()
            files[p.relative_to(run_root).as_posix()] = dict(bytes=len(raw), sha256=sha(raw))
    root = Path(__file__).resolve().parents[1]
    code = {n:sha((root/n).read_bytes()) for n in ('claims/repository_observation.py',
        'oracle/replay_swe.py','oracle/container_run.py','oracle/prepare_swe.py','oracle/test_contract.py')}
    return publish(output, dict(schema='repository-observation-seal-v1', root=str(run_root),
        summary_pin=summary_pin, files=files, code=code, native_measured=False,
        model_generated_action=False, historical_occurrence_established=False, admitted=False))


class RepositoryObservations:
    def __init__(self, path, pin):
        self.path, self.pin = Path(path), pin
        self.manifest = read_pinned(self.path, pin)
        m = self.manifest
        check(m['schema'] == 'repository-observation-seal-v1' and
              all(m[k] is False for k in ('native_measured','model_generated_action',
                  'historical_occurrence_established','admitted')), 'REPOSITORY_OBSERVATION_SCOPE')
        self.root = Path(m['root'])
        code_root = Path(__file__).resolve().parents[1]
        check(all(sha((code_root/n).read_bytes()) == p for n,p in m['code'].items()), 'REPOSITORY_CODE')
        for n in m['files']:
            self.file(n)
        summary = json.loads(self.file('summary.json'))
        check(sha(self.file('summary.json')) == m['summary_pin'], 'REPOSITORY_SUMMARY_PIN')
        self.result = summary['result']
        self.prepared = Path(self.result['prepared'])
        self.execution = Path(self.result['execution'])
        pp = self.prepared.relative_to(self.root).as_posix()+'/prepared.json'
        ep = self.execution.relative_to(self.root).as_posix()+'/report.json'
        self.spec, _ = verify_prepared(self.prepared, m['files'][pp]['sha256'])
        self.report = json.loads(self.file(ep))
        check(self.report['prepared_sha256'] == m['files'][pp]['sha256'] and
              self.report['task'] == self.spec['task'] == self.result['task'] and
              self.report['admitted'] is False, 'REPOSITORY_PREPARED_JOIN')
        self.prefix = self.execution.relative_to(self.root).as_posix()
        self.entries = {}
        for name in ('reference','base','agent'):
            self.entries[name] = self.capture(name, name != 'reference')
        check(self.entries['reference']['tests']['outcome'] == 1 and
              all(self.entries['base']['tests']['states'][n] ==
                  ('failed' if role == 'FAIL_TO_PASS' else 'passed')
                  for n,role in self.spec['required_tests'].items()), 'REPOSITORY_TASK_WITNESS')
        check(len({v['container'] for v in self.entries.values()}) == 3, 'REPOSITORY_OCCURRENCE_REUSED')

    def file(self, name):
        check(name in self.manifest['files'] and not Path(name).is_absolute() and
              '..' not in Path(name).parts and '\\' not in name, 'REPOSITORY_FILE')
        p = self.root/name
        check(not p.is_symlink() and p.resolve().is_relative_to(self.root.resolve()), 'REPOSITORY_FILE')
        raw = p.read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw)) == self.manifest['files'][name], 'REPOSITORY_FILE_PIN')
        return raw

    def capture(self, name, witness):
        prefix = self.prefix+'/'+name+'/'
        e = json.loads(self.file(prefix+'execution.json'))
        before, after = (json.loads(self.file(prefix+s+'.json')) for s in ('before','after'))
        check(e['mode'] == 'tests' and e['status'] == 'observed' and not e['oom_killed'] and
              e['container_id'] == before['Id'] == after['Id'] and not after['State']['Running'] and
              after['State']['ExitCode'] == e['exitcode'] and
              datetime.fromisoformat(after['State']['FinishedAt'].replace('Z','+00:00')) >
              datetime.fromisoformat(after['State']['StartedAt'].replace('Z','+00:00')),
              'REPOSITORY_OCCURRENCE')
        mounts = {'/work':str((self.prepared/name).resolve()),
                  '/runtime':str((self.prepared/'runtime').resolve()),
                  '/harness.py':str(Path(__file__).resolve().parents[1]/'oracle/pytest_inside.py')}
        label = before['Config']['Labels']['mnlm.oracle.run']
        for state in (before,after):
            validate_config(state,self.spec['image'],label,mounts,work_tmpfs=True)
            check(state['Config']['Entrypoint'] == ['python'] and state['Config']['Cmd'] ==
                  ['-I','-B','/harness.py','tests',*selectors(self.spec['required_tests'])],
                  'REPOSITORY_COMMAND')
        markers = [json.loads(line.split('=',1)[1]) for line in
            self.file(prefix+'stdout.bin').decode().splitlines() if line.startswith('MNLM_EXECUTION_RESULT=')]
        check(markers == [e['result']] and e['image'] == self.spec['image'], 'REPOSITORY_CAPTURE')
        self.file(prefix+'stderr.bin')
        tests = recount(e,self.spec['required_tests'],allow_uncollectable=witness)
        check(tests == self.report['conditions'][name], 'REPOSITORY_RECOUNT')
        return dict(container=e['container_id'],image=e['image'],tests=tests,
            started=after['State']['StartedAt'],finished=after['State']['FinishedAt'],
            source_snapshot=digest(self.spec['conditions'][name]),runtime=digest(self.spec['runtime']))

    def episode(self, name, evidence=()):
        check(name in self.entries, 'REPOSITORY_UNKNOWN_OCCURRENCE')
        observed = self.capture(name,name != 'reference')
        check(observed == self.entries[name], 'REPOSITORY_OBSERVATION_CHANGED')
        body = dict(claim_kind='event_occurrence', content=observed['tests'],
            source=dict(observation_set=self.pin,observation=name,task=self.spec['task'],
                        required_tests=digest(self.spec['required_tests'])),
            evidence=list(evidence),event_time=observed['finished'],
            who=dict(actor='isolated_repository_test_process',action_origin='recorded_source_attempt'
                     if name == 'agent' else 'benchmark_'+name),
            what=dict(condition=name,source_snapshot=observed['source_snapshot'],runtime=observed['runtime']),
            where=dict(image=observed['image'],container=observed['container']),
            when=dict(clock='container_runtime_utc',started=observed['started'],finished=observed['finished']),
            why=None,why_status='unresolved',scope=SCOPE,independent_occurrence=True,
            native_measured=False,model_generated_action=False,historical_occurrence_established=False)
        return dict(id=digest(body),**body)

    def validate_episode(self, episode):
        name = episode.get('source',{}).get('observation')
        check(name in self.entries and episode == self.episode(name,episode.get('evidence',())) and
              len(set(episode['evidence'])) == len(episode['evidence']), 'REPOSITORY_EPISODE')

    def assess(self, episode):
        self.validate_episode(episode)
        body = dict(checker='original-repository-test-occurrence-v1',episode=episode['id'],
            claim_kind='event_occurrence',scope=SCOPE,source=episode['source'],
            world_validity='scoped_observed_event',native_measured=False,
            model_generated_action=False,historical_occurrence_established=False)
        return dict(id=digest(body),**body)


class RepositoryEvidence:
    def __init__(self, base, observations):
        self.base, self.observations = base, observations

    def __getattr__(self, name):
        return getattr(self.base,name)

    def checker(self, episode):
        return (self.observations if episode.get('source',{}).get('observation_set') ==
                self.observations.pin else self.base)

    def validate_episode(self, episode):
        self.checker(episode).validate_episode(episode)

    def assess(self, episode):
        return self.checker(episode).assess(episode)

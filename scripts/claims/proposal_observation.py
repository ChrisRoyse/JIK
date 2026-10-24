"""Read new proposed-program occurrences without a fabricated historical target."""
import base64
from datetime import datetime
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims import proposal_forecast
from oracle.prepare_proposal import PreparedProposal
from oracle.container_run import validate_config


class ProposalObservations:
    def __init__(self, root, pin, forecast_root, forecast_pin):
        self.root, self.pin = Path(root), pin
        self.manifest = self._manifest()
        m = self.manifest
        check(m['schema'] == 'proposal-observations-v1' and m['forecast'] == forecast_pin
              and all(m[k] is False for k in ('historical_occurrence_established',
                  'task_correctness_established','scientific_admission')), 'PROPOSAL_OBSERVATION_SCOPE')
        self.seal, self.forecasts = proposal_forecast.read(forecast_root, forecast_pin)
        check(m['prepared'] == self.forecasts['conditions'], 'PROPOSAL_OBSERVATION_PREPARED')
        self.prepared = PreparedProposal(m['prepared']['root'],m['prepared']['pin'])
        check(m['source'] == self.prepared.source.reference() == self.forecasts['source'],
              'PROPOSAL_OBSERVATION_SOURCE')
        self.entries = m['observations']
        check(len(self.entries) == 2 and all(k == digest(v) for k,v in self.entries.items()),
              'PROPOSAL_OBSERVATION_IDS')
        check({v['condition'] for v in self.entries.values()} == set(self.forecasts['predictions']),
              'PROPOSAL_OBSERVATION_ROSTER')
        for name in m['files']:
            self.file(name)
        probe, _ = self.capture('probe', 'probe', 'without_dependency')
        check(probe['result']['probe']['uid'] == 65534
              and all(v is True for k,v in probe['result']['probe'].items() if k != 'uid'),
              'PROPOSAL_OBSERVATION_ISOLATION')
        ids = [self.read(k)['container'] for k in self.entries]
        check(len(set(ids+[probe['container_id']])) == 3, 'PROPOSAL_OCCURRENCE_REUSED')

    def _manifest(self):
        raw = (self.root/'manifest.json').read_bytes()
        check(sha(raw) == self.pin == (self.root/'COMMITTED').read_text(), 'PROPOSAL_OBSERVATION_PIN')
        value = json.loads(raw)
        check(raw == canonical(value), 'PROPOSAL_OBSERVATION_CANONICAL')
        return value

    def file(self, name):
        check(isinstance(name,str) and name and '\\' not in name and ':' not in name
              and not Path(name).is_absolute() and '..' not in Path(name).parts,
              'PROPOSAL_OBSERVATION_PATH')
        path = self.root/name
        check(not path.is_symlink() and path.resolve().is_relative_to(self.root.resolve()),
              'PROPOSAL_OBSERVATION_PATH')
        raw = path.read_bytes()
        check(self.manifest['files'][name] == dict(bytes=len(raw),sha256=sha(raw)),
              'PROPOSAL_OBSERVATION_BYTES')
        return raw

    def capture(self, path, mode, name):
        execution = json.loads(self.file(path+'/execution.json'))
        before, after = (json.loads(self.file(path+'/'+s+'.json')) for s in ('before','after'))
        check(execution['mode'] == mode and execution['status'] == 'observed'
              and execution['exitcode'] == 0 and not execution['oom_killed']
              and execution['container_id'] == before['Id'] == after['Id']
              and not after['State']['Running'] and after['State']['ExitCode'] == 0
              and datetime.fromisoformat(after['State']['FinishedAt'].replace('Z','+00:00')) >
                  datetime.fromisoformat(after['State']['StartedAt'].replace('Z','+00:00')),
              'PROPOSAL_OBSERVATION_EXECUTION')
        harness = 'process_inside.py' if mode == 'observe-script' else 'pytest_inside.py'
        mounts = {'/work':str((self.prepared.root/name/'work').resolve()),
            '/runtime':str((self.prepared.root/name/'runtime').resolve()),
            '/harness.py':str(Path(__file__).resolve().parents[1]/'oracle'/harness)}
        label = before['Config']['Labels']['mnlm.oracle.run']
        for state in (before,after):
            validate_config(state,self.prepared.spec['image'],label,mounts)
            check(state['Config']['Entrypoint'] == ['python']
                  and state['Config']['Cmd'] == ['-I','-B','/harness.py',mode],
                  'PROPOSAL_OBSERVATION_COMMAND')
        check(execution['image'] == self.prepared.spec['image'], 'PROPOSAL_OBSERVATION_IMAGE')
        outer = self.file(path+'/stdout.bin').decode('utf-8')
        markers = [line.split('=',1)[1] for line in outer.splitlines()
                   if line.startswith('MNLM_EXECUTION_RESULT=')]
        check(len(markers) == 1 and json.loads(markers[0]) == execution['result'],
              'PROPOSAL_OBSERVATION_ENVELOPE')
        self.file(path+'/stderr.bin')
        return execution, after

    def read(self, key):
        check(self._manifest() == self.manifest and key in self.entries, 'PROPOSAL_OBSERVATION_ENTRY')
        entry = self.entries[key]
        name = entry['name']
        check(name in self.prepared.spec['conditions'] and entry['execution'] == name+'/execution'
              and entry['condition'] == self.prepared.conditions.commitment['routing'][name]
              and entry['proposal'] == self.prepared.source.reference(), 'PROPOSAL_OBSERVATION_JOIN')
        execution, after = self.capture(entry['execution'],'observe-script',name)
        check(entry['occurrence'] == execution['container_id'], 'PROPOSAL_OBSERVATION_OCCURRENCE')
        result = execution['result']
        check(result['schema'] == 'observed-python-process-v1' and type(result['returncode']) is int,
              'PROPOSAL_OBSERVATION_RESULT')
        captured = {}
        for channel in ('stdout','stderr'):
            value = result[channel]
            raw = base64.b64decode(value['base64'],validate=True)
            check(len(raw) == value['bytes'] and sha(raw) == value['sha256'], 'PROPOSAL_OBSERVATION_CAPTURE')
            captured[channel] = raw.decode('utf-8',errors='strict')
        observed = dict(result=result,captured=captured,image=execution['image'],container=execution['container_id'],
            started=after['State']['StartedAt'],finished=after['State']['FinishedAt'])
        expected = dict(program_sha256=self.prepared.spec['program_sha256'],
                        wheels=self.prepared.spec['conditions'][name]['wheels'])
        observed['conditions'] = self.prepared.conditions.match(entry['condition'],expected,observed)
        proposal_forecast.match(self.seal,self.forecasts,entry['condition'],observed)
        return observed

    def condition_inputs(self):
        return self.prepared.conditions

    def for_observation(self, key):
        check(key in self.entries,'PROPOSAL_OBSERVATION_UNKNOWN')
        return self

    def episode(self, key, evidence=()):
        observed = self.read(key)
        body = dict(claim_kind='event_occurrence',content=observed['captured'],
            source=dict(observation_set=self.pin,observation=key,proposal=self.entries[key]['proposal']),
            evidence=list(evidence),event_time=observed['finished'],
            who=dict(actor='isolated_python_process',python=observed['result']['python']),
            what=dict(program_sha256=observed['result']['program_sha256'],returncode=observed['result']['returncode']),
            where=dict(image=observed['image'],container=observed['container']),
            when=dict(clock='container_runtime_utc',started=observed['started'],finished=observed['finished']),
            execution_conditions=observed['conditions'],why=None,why_status='unresolved',
            independent_occurrence=True,scope='observed_process_in_pinned_runtime',native_measured=False)
        return dict(id=digest(body),**body)

    def validate_episode(self, episode):
        key = episode.get('source',{}).get('observation')
        check(key in self.entries and episode == self.episode(key,episode.get('evidence',()))
              and len(set(episode['evidence'])) == len(episode['evidence']), 'PROPOSAL_OBSERVATION_EPISODE')

    def assess(self, episode):
        self.validate_episode(episode)
        body = dict(checker='proposed-process-observation-v1',episode=episode['id'],claim_kind='event_occurrence',
            scope=episode['scope'],source=episode['source'],world_validity='scoped_observed_event',
            historical_occurrence_established=False,native_measured=False)
        return dict(id=digest(body),**body)

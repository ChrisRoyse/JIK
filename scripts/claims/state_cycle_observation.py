"""Fresh source adapter over the unchanged physical process observation checks."""
from pathlib import Path

from claims.codata_source import check
from claims.episodic_memory import digest
from claims import state_cycle_forecast
from claims.proposal_observation import ProposalObservations
from oracle.prepare_attempt import PreparedAttempt


class StateCycleObservations(ProposalObservations):
    def __init__(self, root, pin, forecast_root, forecast_pin):
        self.root, self.pin = Path(root), pin
        self.manifest = self._manifest()
        m = self.manifest
        check(m['schema'] == 'state-cycle-observations-v1' and m['forecast'] == forecast_pin
              and all(m[k] is False for k in ('historical_occurrence_established',
                  'task_correctness_established','scientific_admission')), 'PROPOSAL_OBSERVATION_SCOPE')
        self.seal, self.forecasts = state_cycle_forecast.read(forecast_root, forecast_pin)
        from claims.codata_source import sha
        check(sha((self.root.parent/'forecast-restart.json').read_bytes()) == m['forecast_restart'],
              'ATTEMPT_OBSERVATION_RESTART_PIN')
        check(m['prepared'] == self.forecasts['conditions'], 'PROPOSAL_OBSERVATION_PREPARED')
        self.prepared = PreparedAttempt(m['prepared']['root'],m['prepared']['pin'])
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

"""Add native source observations to the existing connected experience lineage."""
import copy
from pathlib import Path

from claims.episodic_memory import digest
from claims.repository_observation import RepositoryEvidence
from model.git_source_input import SourceNative
from model.repository_cohort_memory import CohortMemoryLoop
from oracle.git_source_slice import digest as sha, require


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n: sha((root/n).read_bytes()) for n in
            ('model/git_source_memory.py', 'model/git_source_input.py', 'claims/git_source_population.py')}


class SourceMemoryLoop(CohortMemoryLoop):
    def __init__(self, *args, **kwargs):
        self.git_source_intakes = {}
        super().__init__(*args, **kwargs)

    def observe_git_source(self, spec, expected=None):
        self._fresh()
        key = digest(spec)
        require(key not in self.git_source_intakes, 'DUPLICATE_INTAKE')
        native = SourceNative(**spec)
        require(native.specs == self.model.specs, 'NATIVE_INSTRUMENT')
        old, staged = self.memory, copy.copy(self.memory)
        for name in ('episodes', 'state', 'children', 'events'):
            setattr(staged, name, copy.deepcopy(getattr(old, name)))
        staged.checker = RepositoryEvidence(old.checker, native.population)
        self.memory = staged
        try:
            records, bindings = [], {}
            for name in native.population.occurrences:
                ep = native.population.episode(name)
                require(ep['id'] not in self.memory.episodes, 'REUSED_SOURCE_EPISODE')
                receipt = native.receipt(name, self.model.config, self.device, self.model.specs)
                records.extend(self._intake(ep))
                bindings[name] = dict(episode=ep['id'], native=receipt)
            native.fresh()
            payload = dict(bindings=bindings, memory_records=records,
                independent_lineages=1, independent_actions=0, native_measured=True,
                source_scope='pinned_artifact_contents_only', learned_skill=False,
                scientific_admission=False, serving=False)
            body = dict(kind='git_source_intake', spec=spec, prior_head=self.head,
                        payload=digest(payload), code=code_pins())
            self._echo_publish(body, payload, expected)
        except BaseException:
            self.memory = old
            raise
        self.git_source_intakes[key] = dict(event=body, bindings=bindings)
        return body

    def _replay(self, body):
        if body.get('kind') == 'git_source_intake':
            require(body['code'] == code_pins(), 'MEMORY_CODE')
            self.observe_git_source(body['spec'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.git_source_intakes:
            value['git_source_intakes'] = copy.deepcopy(self.git_source_intakes)
        return value

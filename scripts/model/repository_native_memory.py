"""Attach native surfaces to existing occurrences without promoting learned competence."""
import copy
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.repository_input import RepositoryNative
from model.repository_memory import RepositoryMemoryLoop


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in
        ('model/repository_native_memory.py','model/repository_input.py',
         'claims/repository_population.py','model/repository_memory.py')}


class RepositoryNativeLoop(RepositoryMemoryLoop):
    def __init__(self,*args,**kwargs):
        self.repository_native_bindings = {}
        super().__init__(*args,**kwargs)

    def bind_repository_native(self,spec,expected=None):
        self._fresh()
        key = digest(spec)
        check(key not in self.repository_native_bindings,'REPOSITORY_NATIVE_DUPLICATE')
        native = RepositoryNative(**spec)
        pin = native.population.observations.pin
        check(pin in self.repository_occurrences,'REPOSITORY_NATIVE_OCCURRENCE_MISSING')
        bindings = {}
        for condition,episode in sorted(self.repository_occurrences[pin]['episodes'].items()):
            ep = self.memory.episodes[episode]
            check(self.memory.state[episode]['active'],'REPOSITORY_NATIVE_INACTIVE_EPISODE')
            native.population.observations.validate_episode(ep)
            fields = native.population.fields[condition]
            check(fields['container'] == ep['where']['container'] and
                  fields['input_available_by'] == ep['when']['started'] and
                  fields['echo_available_after'] == ep['when']['finished'],
                  'REPOSITORY_NATIVE_CLOCK_JOIN')
            bindings[condition] = dict(episode=episode,
                input=native.receipt(condition,'input',fields['input_available_by'],
                    self.model.config,self.device,self.model.specs),
                echo=native.receipt(condition,'echo',fields['echo_available_after'],
                    self.model.config,self.device,self.model.specs))
        payload = dict(bindings=bindings,observations=pin,native_measured=True,
            native_consumer='typed_constellation',learned_repair=False,
            scientific_admission=False,serving=False)
        body = dict(kind='repository_native_binding',spec=spec,prior_head=self.head,
                    payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.repository_native_bindings[key] = dict(event=body,bindings=bindings)
        return body

    def _replay(self,body):
        if body.get('kind') == 'repository_native_binding':
            check(body['code'] == code_pins(),'REPOSITORY_NATIVE_BINDING_CODE')
            self.bind_repository_native(body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.repository_native_bindings:
            result['repository_native_bindings'] = copy.deepcopy(self.repository_native_bindings)
        return result

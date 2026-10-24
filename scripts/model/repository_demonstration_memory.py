"""Admit supplied repair demonstrations without claiming independent model attempts."""
import copy
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_observation import RepositoryEvidence
from model.repository_demonstration_input import DemonstrationNative
from model.repository_depth_memory import DepthMemoryLoop


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'model/repository_demonstration_memory.py','model/repository_demonstration_input.py',
        'claims/repository_demonstration_observation.py','claims/repository_demonstration_population.py')}


class DemonstrationMemoryLoop(DepthMemoryLoop):
    def __init__(self,*args,**kwargs):
        self.repository_demonstration_intakes = {}
        super().__init__(*args,**kwargs)

    def observe_demonstration(self,spec,expected=None):
        self._fresh()
        key = digest(spec)
        check(key not in self.repository_demonstration_intakes,'DEMONSTRATION_MEMORY_DUPLICATE')
        check(spec['partition'] in ('learn_1','learn_2'),'DEMONSTRATION_MEMORY_PARTITION')
        prefix = {**spec,'partition':'learn_1'}
        check(spec['partition'] == 'learn_1' or digest(prefix) in self.repository_demonstration_intakes,
              'DEMONSTRATION_MEMORY_PREFIX')
        native = DemonstrationNative(**spec)
        observations = native.population.observations
        fields = native.population.fields
        selected = sorted((k for k,v in fields.items() if v['partition'] == spec['partition']),
                          key=lambda k:(fields[k]['input_available_by'],k))
        check(selected,'DEMONSTRATION_MEMORY_EMPTY')
        bindings,episodes = {},{}
        for condition in selected:
            ep = observations.episode(condition)
            observations.validate_episode(ep)
            f = fields[condition]
            check(f['container'] == ep['where']['container'] and
                  f['input_available_by'] == ep['when']['started'] and
                  f['echo_available_after'] == ep['when']['finished'],'DEMONSTRATION_MEMORY_CLOCK_JOIN')
            episodes[condition] = ep
            bindings[condition] = dict(episode=ep['id'],**{
                side:native.receipt(condition,side,f[clock],self.model.config,self.device,self.model.specs)
                for side,clock in (('input','input_available_by'),('echo','echo_available_after'))})
        old,staged = self.memory,copy.copy(self.memory)
        for name in ('episodes','state','children','events'):
            setattr(staged,name,copy.deepcopy(getattr(old,name)))
        staged.checker = RepositoryEvidence(old.checker,observations)
        self.memory = staged
        try:
            records = []
            for ep in episodes.values():
                check(ep['id'] not in self.memory.episodes,'DEMONSTRATION_MEMORY_REUSED_OCCURRENCE')
                records.extend(self._intake(ep))
            payload = dict(episodes={k:v['id'] for k,v in episodes.items()},bindings=bindings,
                memory_records=records,observations=observations.pin,partition=spec['partition'],
                native_measured=True,native_consumer='typed_constellation',independent_actions=False,reference_demonstrations=True,
                learned_repair=False,scientific_admission=False,serving=False)
            body = dict(kind='repository_demonstration_intake',spec=spec,prior_head=self.head,
                        payload=digest(payload),code=code_pins())
            self._echo_publish(body,payload,expected)
        except BaseException:
            self.memory = old
            raise
        self.repository_demonstration_intakes[key] = dict(event=body,episodes=payload['episodes'],bindings=bindings)
        return body

    def _replay(self,body):
        if body.get('kind') == 'repository_demonstration_intake':
            check(body['code'] == code_pins(),'DEMONSTRATION_MEMORY_CODE')
            self.observe_demonstration(body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.repository_demonstration_intakes:
            result['repository_demonstration_intakes'] = copy.deepcopy(self.repository_demonstration_intakes)
        return result

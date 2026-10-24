"""Append independently observed task outcomes to the existing intrinsic-loop memory."""
import copy
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_observation import RepositoryEvidence,RepositoryObservations
from model.state_cycle import StateCycleLoop


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in
            ('claims/repository_observation.py','model/repository_memory.py')}


class RepositoryMemoryLoop(StateCycleLoop):
    def __init__(self,*args,**kwargs):
        self.repository_occurrences = {}
        super().__init__(*args,**kwargs)

    def observe_repository(self,path,pin,expected=None):
        self._fresh()
        check(pin not in self.repository_occurrences,'REPOSITORY_MEMORY_DUPLICATE')
        observations = RepositoryObservations(path,pin)
        old,staged = self.memory,copy.copy(self.memory)
        for name in ('episodes','state','children','events'):
            setattr(staged,name,copy.deepcopy(getattr(old,name)))
        staged.checker = RepositoryEvidence(old.checker,observations)
        self.memory = staged
        try:
            records,episodes = [],{}
            for name in sorted(observations.entries,key=lambda n:observations.entries[n]['started']):
                # Evaluation lineage is not additional independent evidence of the repair.
                links = tuple(episodes.values()) if name == 'agent' else ()
                ep = observations.episode(name,evidence=links)
                records.extend(self._intake(ep))
                episodes[name] = ep['id']
            payload = dict(episodes=episodes,memory_records=records,scope='original_task_test_occurrences',
                native_measured=False,model_generated_action=False,learned_skill=False,
                scientific_admission=False,serving=False)
            body = dict(kind='repository_occurrences',path=str(path),pin=pin,prior_head=self.head,
                        payload=digest(payload),code=code_pins())
            self._echo_publish(body,payload,expected)
        except BaseException:
            self.memory = old
            raise
        self.repository_occurrences[pin] = dict(episodes=episodes,event=body)
        return body

    def _replay(self,body):
        if body.get('kind') == 'repository_occurrences':
            check(body['code'] == code_pins(),'REPOSITORY_MEMORY_CODE')
            self.observe_repository(body['path'],body['pin'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.repository_occurrences:
            value['repository_occurrences'] = copy.deepcopy(self.repository_occurrences)
        return value

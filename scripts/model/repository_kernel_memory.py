"""Active repository routing and model context in the existing experience journal."""
import copy
from pathlib import Path

import torch

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_observation import SCOPE
from model.loop_session import output_digest
from model.native_episode_session import tensor
from model.repository_demonstration_memory import DemonstrationMemoryLoop
from model.repository_kernel_view import RepositoryKernelView
from model.repository_relation_world import RepositoryWorld
from model.repository_relation_checkpoint import frozen
from model.tool_growth import parameters


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'model/repository_kernel_memory.py','model/repository_kernel_view.py',
        'model/repository_streamed_world.py','model/repository_relation_world.py',
        'model/repository_relation_input.py')}


class RepositoryKernelMemoryLoop(DemonstrationMemoryLoop):
    def __init__(self,*args,repository_population,**kwargs):
        self.repository_population = repository_population
        self.repository_runtime = None
        self.repository_views,self.repository_states,self.repository_history = {},{},{}
        self.repository_dirty = set()
        self.repository_base = None
        super().__init__(*args,**kwargs)

    def _repository_model(self):
        if self.repository_base is None:
            self.repository_base = RepositoryWorld.from_world(self.model).eval()
            check(frozen(self.repository_base)==frozen(self.model),'REPOSITORY_RUNTIME_PARENT_DRIFT')
            self.repository_parameters = parameters(self.repository_base)
            self.repository_parent = (self.active_checkpoint,parameters(self.model))
        check((self.active_checkpoint,parameters(self.model))==self.repository_parent,
              'REPOSITORY_RUNTIME_PARENT_CHANGED')
        check(parameters(self.repository_base)==self.repository_parameters,'REPOSITORY_RUNTIME_WEIGHT_DRIFT')
        return self.repository_base

    def _repository_checked(self,episode):
        state = self.memory.state.get(episode)
        check(state is not None and state['active'] and state['grounded_scope']==SCOPE
              and state['receipt']==self.memory.checker.assess(self.memory.episodes[episode]),
              'REPOSITORY_RUNTIME_EVIDENCE')

    def activate_repository_kernel(self,spec,*,expected=None):
        self._fresh()
        p = self.repository_population
        p.fresh()
        check(set(spec)=={'path','pin','source'},'REPOSITORY_RUNTIME_SPEC')
        check(self.repository_runtime is None or self.repository_runtime.spec['pin']!=spec['pin'],
              'REPOSITORY_RUNTIME_DUPLICATE_KERNEL')
        model = self._repository_model()
        if spec['pin'] not in self.repository_views:
            self.repository_views[spec['pin']] = RepositoryKernelView(model,p,spec)
        runtime = self.repository_views[spec['pin']]
        check(runtime.spec == spec,'REPOSITORY_RUNTIME_KERNEL_ALIAS')
        # Every source reference must already be present and eligible in this same memory.
        for key in runtime.ids:
            self._repository_checked(p.records[key]['episode'])
        previous = self.repository_runtime
        removed = sorted(set(previous.ids)-set(runtime.ids)) if previous else []
        dirty = sorted(self.repository_states) if removed else []
        payload = dict(previous=None if previous is None else previous.spec,active=spec,
            regions=runtime.regions,bank=runtime.view.world_id,removed=removed,dirty=dirty,
            parameters=self.repository_parameters,parent_model=self.repository_parent[0],standard=model.standard_id)
        body = dict(kind='repository_kernel_activate',spec=copy.deepcopy(spec),prior_head=self.head,
                    payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.repository_runtime = runtime
        self.repository_states = {c:runtime.view.bind_state(s) for c,s in self.repository_states.items()}
        self.repository_dirty.update(dirty)
        return body

    def read_repository(self,context,key,*,kernel,cutoff,expected=None):
        self._fresh()
        runtime = self.repository_runtime
        check(runtime is not None and kernel==runtime.spec['pin'],'REPOSITORY_RUNTIME_STALE_KERNEL')
        check(isinstance(context,str) and context.startswith('repository:') and context not in self.contexts,
              'REPOSITORY_RUNTIME_CONTEXT')
        check(context not in self.repository_dirty,'REPOSITORY_RUNTIME_REVIEW_REQUIRED')
        check(key in self.repository_population.records and key not in runtime.kernel.value['withdrawn'],
              'REPOSITORY_RUNTIME_WITHDRAWN')
        record = self.repository_population.records[key]
        self._repository_checked(record['episode'])
        for atom in runtime.ids:self._repository_checked(self.repository_population.records[atom]['episode'])
        state = self.repository_states.get(context)
        if state is None:state=runtime.view.start(context)
        request = dict(context=context,key=key,kernel=kernel,cutoff=cutoff,
            observation=len(self.repository_history.get(context,[])),prior_head=self.head,
            previous_state=tensor(state.hidden)['sha256'],parent_model=self.active_checkpoint)
        self._repository_model()
        with torch.no_grad():out,trace=runtime.forward(key,state,cutoff=cutoff,event=digest(request))
        new = out['states'][0]
        payload = dict(trace=trace,hidden=tensor(new.hidden),sequence=new.sequence,last_event=new.last_event,
            predicted_regions=list(out['predicted_constellation'].regions),output=output_digest(out),
            region_scores={n:tensor(v) for n,v in out['regions'].items()},
            identity_output={n:tensor(v)['sha256'] for n,v in out['identity_regions'].items()},
            scientific_admission=False,learned_capability=False,serving=False)
        body = dict(kind='repository_kernel_observe',request=request,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.repository_states[context] = new
        self.repository_history.setdefault(context,[]).append(request)
        self.journal.cache.clear()
        return body

    def reset_repository_context(self,context,*,reason,expected=None):
        self._fresh()
        self._repository_model()
        check(context in self.repository_dirty and isinstance(reason,str) and reason.strip(),
              'REPOSITORY_RUNTIME_RESET_REVIEW')
        state = self.repository_runtime.view.start(context)
        payload = dict(context=context,reason=reason,previous=tensor(self.repository_states[context].hidden)['sha256'],
                       reset=tensor(state.hidden)['sha256'],kernel=self.repository_runtime.spec['pin'])
        body = dict(kind='repository_kernel_reset',context=context,reason=reason,prior_head=self.head,
                    payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.repository_states[context]=state;self.repository_history[context]=[]
        self.repository_dirty.remove(context)
        return body

    def _replay(self,body):
        kind=body.get('kind')
        if kind.startswith('repository_kernel_'):
            check(body['code']==code_pins(),'REPOSITORY_RUNTIME_CODE')
        if kind=='repository_kernel_activate':self.activate_repository_kernel(body['spec'],expected=body)
        elif kind=='repository_kernel_observe':
            r=body['request']
            self.read_repository(r['context'],r['key'],kernel=r['kernel'],cutoff=r['cutoff'],expected=body)
        elif kind=='repository_kernel_reset':
            self.reset_repository_context(body['context'],reason=body['reason'],expected=body)
        else:super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.repository_runtime is not None:
            result['repository_runtime'] = dict(active=self.repository_runtime.spec,bank=self.repository_runtime.view.world_id,
                model_parameters=self.repository_parameters,parent_model=self.repository_parent[0],standard=self.repository_base.standard_id,
                dirty=sorted(self.repository_dirty),contexts={c:dict(world=s.world_id,sequence=s.sequence,
                    hidden=tensor(s.hidden)['sha256'],last_event=s.last_event,history=self.repository_history[c])
                    for c,s in sorted(self.repository_states.items())},scientific_admission=False,serving=False)
        return result

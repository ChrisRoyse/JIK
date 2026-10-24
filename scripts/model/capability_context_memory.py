"""Journaled context revisions replace training views, retaining original episodes and withdrawals."""
import copy
import json
from pathlib import Path

from claims.capability_evidence import CapabilityEvidence, execution_bridges
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from kernel.capability_population import TaskPopulation
from model.capability_context_input import ContextNative, ContextStore
from model.capability_growth_memory import GrowthCapabilityMemoryLoop
from model.capability_live_population import JoinedTasks


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'model/capability_context_memory.py','model/capability_context_input.py',
        'claims/capability_context_population.py','claims/repository_context.py',
        'text/capability_context_plan.py','kernel/capability_context_registry.py')}


class ContextCapabilityMemoryLoop(GrowthCapabilityMemoryLoop):
    def __init__(self,*args,**kwargs):
        self.context_revisions = {}
        super().__init__(*args,**kwargs)

    def _delta_contract(self,spec):
        if spec.get('adapter',{}).get('schema') != 'capability-context-native-v1':
            return super()._delta_contract(spec)
        need(set(spec)=={'population','population_pin','plan','plan_pin','native','native_pin','adapter'},
             'CONTEXT_MEMORY_SPEC')
        adapter=spec['adapter']
        need(set(adapter)=={'schema','code','registry','native_proof'} and adapter['code']==code_pins(),
             'CONTEXT_MEMORY_CODE')
        registry=adapter['registry']; raw=Path(registry['path']).read_bytes()
        need(sha(raw)==registry['pin'],'CONTEXT_MEMORY_REGISTRY_PIN')
        value=json.loads(raw)
        need(value['schema']=='capability-global-split-registry-v1' and value['complete']
             and value['historical_membership_unchanged'] and spec['population_pin'] in value['splits']
             and value['population_paths'][spec['population_pin']]==spec['population'],'CONTEXT_MEMORY_REGISTRY')
        for evidence in self.capability_sources.values():
            pin=evidence.population.pin
            need(pin in value['splits'] and evidence.native.pin in self.task_splits,
                 'CONTEXT_MEMORY_UNREGISTERED_HISTORY')
            old=self.task_splits[evidence.native.pin]; new=value['splits'][pin]
            need(all(old[k]==new[k] for k in ('groups','excluded','train','queries')),
                 'CONTEXT_MEMORY_SPLIT_MOVED')
        proof=adapter['native_proof']; raw=(Path(proof['path'])/'report.json').read_bytes()
        need(sha(raw)==proof['pin'],'CONTEXT_MEMORY_PROOF_PIN'); report=json.loads(raw)
        need(report['schema']=='capability-context-native-fsv-v1' and report['complete']
             and report['all_native_windows_preserved'] and report['plan']==spec['plan_pin']
             and report['native']==spec['native_pin'] and report['registry']==registry['pin'],
             'CONTEXT_MEMORY_PROOF_JOIN')
        return value,report['populations'][spec['population_pin']]['binding']

    def _capability_source(self,spec):
        if spec.get('adapter',{}).get('schema')!='capability-context-native-v1':
            return super()._capability_source(spec)
        _,binding=self._delta_contract(spec); identity=digest(spec)
        if identity in self.capability_sources:
            evidence=self.capability_sources[identity]; evidence.fresh(); return evidence
        args={k:spec[k] for k in ('plan','plan_pin','native','native_pin')}; key=digest(args)
        if key not in self.delta_stores: self.delta_stores[key]=ContextStore(**args)
        native=ContextNative(spec['population'],spec['population_pin'],store=self.delta_stores[key])
        need(native.binding==binding and tuple(native.specs)==tuple(sorted(self.model.specs,key=lambda s:s.name)),
             'CONTEXT_MEMORY_NATIVE_BINDING')
        native.fresh(); original=execution_bridges(self,native.population.parent)
        bridges={}
        for key,task in native.population.tasks.items():
            parent=task['provenance']['parent_task']
            need(parent in self.capability_tasks and self.capability_tasks[parent]['population']==native.population.parent.pin,
                 'CONTEXT_MEMORY_PARENT_TASK')
            bridges[key]=[self.capability_tasks[parent]['episode']]+original[parent]
        return CapabilityEvidence(self.memory.checker,native,bridges)

    def observe_context(self,spec,*,expected=None):
        """One durable intake; an ineligible predecessor cannot be resurrected by richer context."""
        self._fresh(); evidence=self._capability_source(spec); p=evidence.population
        need(evidence.native.pin in self.task_splits and self.live_intent is None,'CONTEXT_MEMORY_INTAKE_STATE')
        selected, excluded={},{}
        for key,task in p.tasks.items():
            parent=task['provenance']['parent_task']; old=self.capability_tasks[parent]
            ids=[old['episode']]+self.memory.episodes[old['episode']]['evidence']
            bad=[ep for ep in ids if not self.memory.state[ep]['active']
                 or self.memory.state[ep]['grounded_scope'] is None]
            if bad: excluded[key]=dict(parent_task=parent,unavailable_episodes=bad)
            else: selected[key]=task
        need(selected and not set(selected)&self.capability_tasks.keys(),'CONTEXT_MEMORY_DUPLICATE')
        need(set(self.task_splits[evidence.native.pin]['queries'])<=set(selected),'CONTEXT_MEMORY_QUERY_UNAVAILABLE')
        fields=sorted({f for t in selected.values() for side in ('context','targets','consequences') for f in t[side]})
        field_set=set(fields); links=[e for e in p.edges if e['origin'] in field_set and e['target'] in field_set]
        old,staged=self.memory,copy.copy(self.memory)
        for name in ('episodes','state','children','events'): setattr(staged,name,copy.deepcopy(getattr(old,name)))
        staged.checker=evidence; self.memory=staged
        try:
            records,receipts,tasks=[],{},{}
            for field in fields:
                records.extend(self._intake(evidence.field_episode(field)))
                receipts[field]=evidence.field_receipt(field)
            for key,task in selected.items():
                for ep in evidence.bridges[key]: self._capability_support(ep)
                ep=evidence.task_episode(key); records.extend(self._intake(ep))
                tasks[key]=dict(episode=ep['id'],family=p.family,kind=task['kind'],split_group=task['split_group'],
                    population=p.pin,binding=evidence.native.pin,
                    selections={s:p.selection(key,side=s) for s in ('context','targets','consequences')},
                    related_executions=evidence.bridges[key][1:],
                    partition='development_exposed',scientific_admission=False,training_ready=False,serving=False)
            payload=dict(tasks=tasks,fields=receipts,relationships=links,memory_records=records,
                native_binding=evidence.native.binding,source_spec=spec,excluded=excluded,
                predecessor_population=p.parent.pin,scientific_admission=False,learned_capability=False,serving=False)
            body=dict(kind='context_revision_intake',spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins())
            self._echo_publish(body,payload,expected)
        except BaseException:
            self.memory=old; raise
        self.capability_sources[digest(spec)]=evidence
        self.capability_intakes[digest(body)]=dict(body=body,tasks=list(tasks),fields=fields)
        self.capability_tasks.update(tasks)
        self.capability_fields.update({p.pin+':'+k:v for k,v in receipts.items()})
        self.capability_relations.update({digest(e):e for e in links})
        self.journal.cache.clear(); self._after_task_change(expected)
        return body

    def select_context_revision(self,population,*,expected=None):
        self._fresh()
        matches=[e for e in self.capability_sources.values() if e.population.pin==population]
        need(len(matches)==1 and isinstance(matches[0].native,ContextNative),'CONTEXT_MEMORY_REVISION_SOURCE')
        evidence=matches[0]; parent=evidence.population.parent.pin
        need(parent not in self.context_revisions,'CONTEXT_MEMORY_REVISION_ALREADY_SELECTED')
        admitted={k for k,t in self.capability_tasks.items() if t['population']==population}
        eligible=set()
        for k,t in evidence.population.tasks.items():
            ep=self.capability_tasks[t['provenance']['parent_task']]['episode']
            parents=[ep]+self.memory.episodes[ep]['evidence']
            if all(self.memory.state[p]['active'] and self.memory.state[p]['grounded_scope'] is not None
                   for p in parents): eligible.add(k)
        need(admitted==eligible,'CONTEXT_MEMORY_REVISION_COVERAGE')
        payload=dict(parent=parent,successor=population,tasks=sorted(admitted),
            parent_to_successor={evidence.population.tasks[k]['provenance']['parent_task']:k for k in sorted(admitted)},
            policy='replace active training view; retain original evidence and dependency links')
        body=dict(kind='context_revision_select',prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected); self.context_revisions[parent]=payload
        self.task_views.clear(); self._after_task_change(expected)
        return body

    def _task_population(self):
        if not self.context_revisions: return super()._task_population()
        key=(self.memory.head,digest(self.task_splits),digest(self.context_revisions))
        if key not in self.task_views:
            pops={}
            for evidence in self.capability_sources.values():
                native=evidence.native
                if native.population.pin in self.context_revisions: continue
                p=TaskPopulation(native,self.capability_tasks,self.capability_fields,self.capability_relations,
                                 self.memory.state,self.task_splits[native.pin])
                if p.records: pops[native.pin]=p
            self.task_views={key:JoinedTasks(pops)}
        return self.task_views[key]

    def _replay(self,body):
        kind=body.get('kind','')
        if kind.startswith('context_revision_'):
            need(body['code']==code_pins(),'CONTEXT_MEMORY_REPLAY_CODE')
            if kind=='context_revision_intake': self.observe_context(body['spec'],expected=body)
            elif kind=='context_revision_select':
                self.select_context_revision(self.journal.payload(body['payload'])['successor'],expected=body)
            else: need(False,'CONTEXT_MEMORY_EVENT_KIND')
        else: super()._replay(body)

    def snapshot(self):
        value=super().snapshot()
        if self.context_revisions: value['context_revisions']=copy.deepcopy(self.context_revisions)
        return value

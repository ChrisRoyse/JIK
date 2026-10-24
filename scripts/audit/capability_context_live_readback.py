"""Independent SQL/native readback of context supersession and inherited withdrawals."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.capability_growth_readback import GrowthReader
from audit.repository_memory_readback import chain
from claims.capability_context_population import ContextPopulation
from claims.capability_population import need,sha
from claims.episodic_memory import canonical,digest
from claims.process_conditions import write_committed
from senses.trajectory_population import read_pinned


def read(path): return json.loads(Path(path).read_bytes())


class ContextReader(GrowthReader):
    def __init__(self,config,proofs):
        self.revisions={}
        parent=Path(config['source'])
        need(sha((parent/'experience.sqlite3').read_bytes())==config['source_pin'],'CONTEXT_READER_PARENT')
        super().__init__(read(parent/'config.json'),proofs)
        result=self.run_growth(parent/'experience.sqlite3')
        need(result['pending'] is None,'CONTEXT_READER_PENDING_PARENT')
        self.contract,self.prefix=chain(parent/'experience.sqlite3')
        self.context_config=config
        self.context_code={n:sha((self.code_root/n).read_bytes()) for n in (
            'model/capability_context_memory.py','model/capability_context_input.py',
            'claims/capability_context_population.py','claims/repository_context.py',
            'text/capability_context_plan.py','kernel/capability_context_registry.py')}
        spec=config['native_source']; self.context_population=ContextPopulation(spec['population'],spec['population_pin'])
        self.populations[self.context_population.pin]=self.context_population
        proof=config['native_proof']; root=Path(proof['path'])
        report=read_pinned(root/'report.json',proof['pin'])
        need(report['complete'] and report['registry']==config['registry']['pin'],'CONTEXT_READER_PROOF')
        for name,info in report['files'].items():
            raw=(root/name).read_bytes(); need(dict(bytes=len(raw),sha256=sha(raw))==info,'CONTEXT_READER_PROOF_FILE')
        self.receipts.update(read(root/'field-receipts.json'))
        self.bindings.update({pin:v['binding'] for pin,v in report['populations'].items()})
        self.context_registry=read_pinned(Path(config['registry']['path']),config['registry']['pin'])
        self.context_native_root=Path(spec['native']); self.context_intakes=[]
        kernel_code=copy.deepcopy(self.growth_code['kernel']['kernel'])
        for old,new in (('kernel/capability_growth_kernel.py','kernel/capability_retained_kernel.py'),
                        ('kernel/capability_updates.py','kernel/capability_retained_updates.py')):
            kernel_code['capability'].pop(old)
            kernel_code['capability'][new]=sha((self.code_root/new).read_bytes())
        self.retained_code=dict(kernel=dict(kernel=kernel_code,
            checkpoint=sha((self.code_root/'kernel/capability_retained_state.py').read_bytes())),
            live=sha((self.code_root/'model/capability_retained_memory.py').read_bytes()))
        self.retained_enabled=False

    def field_array(self,pin,field,name):
        if not hasattr(self,'context_population') or pin!=self.context_population.pin:
            return super().field_array(pin,field,name)
        old=self.native_root
        try:
            self.native_root=self.context_native_root
            return super().field_array(pin,field,name)
        finally: self.native_root=old

    def source(self):
        value=super().source()
        if not self.revisions: return value
        families={k:v for k,v in value['families'].items() if v['files']['population'] not in self.revisions}
        return dict(family='shared',files={k:v['files'] for k,v in families.items()},
            groups=sorted({g for v in families.values() for g in v['groups']}),
            ids=sorted(k for v in families.values() for k in v['ids']),families=families)

    def context_register(self,payload):
        spec=dict(**self.context_config['native_source'],adapter=dict(schema='capability-context-native-v1',
            code=self.context_code,registry=self.context_config['registry'],native_proof=self.context_config['native_proof']))
        pin=spec['population_pin']; binding=digest(self.bindings[pin]); split=self.context_registry['splits'][pin]
        need(payload==dict(binding=binding,spec=spec,split=split) and binding not in self.task_splits,
             'CONTEXT_READER_REGISTRATION')
        self.registrations[pin]=spec; self.task_splits[binding]=split

    def context_intake(self,body,payload):
        pop=self.context_population; pin=pop.pin; binding=self.bindings[pin]
        need(body['spec']==payload['source_spec']==self.registrations[pin]
             and payload['native_binding']==binding and payload['predecessor_population']==pop.parent.pin,
             'CONTEXT_READER_INTAKE_BINDING')
        selected,excluded={},{}
        for key,t in pop.tasks.items():
            old=self.tasks[t['provenance']['parent_task']]; ep=old['episode']
            bad=[k for k in [ep]+self.episodes[ep]['evidence']
                 if not self.states[k]['active'] or self.states[k]['grounded_scope'] is None]
            if bad: excluded[key]=dict(parent_task=t['provenance']['parent_task'],unavailable_episodes=bad)
            else: selected[key]=t
        fields={f for t in selected.values() for side in ('context','targets','consequences') for f in t[side]}
        need(payload['excluded']==excluded and set(payload['tasks'])==set(selected)
             and set(payload['fields'])==fields,'CONTEXT_READER_WITHDRAWALS')
        expected={}
        for key in sorted(fields):
            f=pop.fields[key]
            item=dict(claim_kind='source_statement',content=f['text'],
                source=dict(capability_population=pin,field=key,source=f['source']),evidence=[],event_time=f['clock'],
                role=f['role'],family=pop.family)
            eid=digest(item); expected[eid]=dict(id=eid,**item)
            receipt=payload['fields'][key]; slots={}
            for name,value in self.receipts[pin][key].items():
                slots[name]=dict(instrument=binding['instruments'][name],shape=[value['windows'],value['dimensions']],
                    dtype='float32-le',sha256=value['sha256'],description=value['description'])
            need(receipt==dict(episode=eid,binding=digest(binding),slots=slots),'CONTEXT_READER_FIELD_RECEIPT')
        for key,t in selected.items():
            old=self.tasks[t['provenance']['parent_task']]
            selections={side:dict(population=pin,task=key,side=side,
                fields=[dict(id=f,role=pop.fields[f]['role'],parts=pop.fields[f]['parts']) for f in t[side]],
                scientific_admission=False,training_ready=False) for side in ('context','targets','consequences')}
            dependencies=[payload['fields'][f]['episode'] for side in ('context','targets','consequences') for f in t[side]]
            dependencies += [old['episode']]+old['related_executions']
            item=dict(claim_kind='source_statement',content=canonical(t).decode(),
                source=dict(capability_population=pin,task=key),evidence=dependencies,event_time=None,
                family=pop.family,role='attested_task_connections',clock_claim='source_fields_only')
            eid=digest(item); expected[eid]=dict(id=eid,**item)
            need(payload['tasks'][key]==dict(episode=eid,family=pop.family,kind=t['kind'],split_group=t['split_group'],
                population=pin,binding=digest(binding),selections=selections,related_executions=old['related_executions'],
                partition='development_exposed',scientific_admission=False,training_ready=False,serving=False),
                'CONTEXT_READER_TASK_DEPENDENCIES')
        need(payload['relationships']==[e for e in pop.edges if e['origin'] in fields and e['target'] in fields],
             'CONTEXT_READER_RELATIONSHIPS')
        actual=set()
        for record in payload['memory_records']:
            if record['event']['op']!='intake': self.memory_record(record); continue
            need(record['sequence']==self.memory_sequence and record['previous']==self.memory_head
                 and record['sha256']==digest({k:v for k,v in record.items() if k!='sha256'}),'CONTEXT_READER_MEMORY_CHAIN')
            ep=record['event']['episode']; key=ep['id']
            need(key not in self.states and expected[key]==ep and all(k in self.states for k in ep['evidence']),
                 'CONTEXT_READER_EPISODE')
            self.episodes[key]=ep; self.states[key]=dict(active=True,grounded_scope=None,review='unassessed',receipt=None)
            for dep in ep['evidence']: self.children.setdefault(dep,set()).add(key)
            self.memory_head,self.memory_sequence=record['sha256'],self.memory_sequence+1; actual.add(key)
        need(actual==set(expected),'CONTEXT_READER_MISSING_EPISODE')
        self.tasks.update(payload['tasks']); self.fields.update({pin+':'+k:v for k,v in payload['fields'].items()})
        self.relationships.update({digest(e):e for e in payload['relationships']})
        for key in selected: self.add_slots(pin,key)
        split=self.task_splits[digest(binding)]
        self.source_templates[pin]=dict(family=pop.family,files=dict(population=pin,native=binding,split=digest(split)),
            groups=sorted(set(split['groups'])-set(split['excluded'])),ids=[],statuses={},connections={},
            query_contexts={k:digest(dict(slots=self.slots[k])) for k in split['queries']})
        self.context_intakes.append(dict(population=pin,tasks=len(selected),excluded=excluded,fields=len(fields)))

    def run_context(self,journal):
        contract,rows=chain(journal)
        need(contract==self.contract and rows[:len(self.prefix)]==self.prefix,'CONTEXT_READER_PREFIX')
        previous=self.prefix[-1][2]
        for body,payload,event in rows[len(self.prefix):]:
            kind=body['kind']; need(body['prior_head']==previous,'CONTEXT_READER_PRIOR_HEAD')
            if kind=='capability_delta_output_root':
                need(body['code']==self.expected_code and payload['previous']==self.spec['root']
                     and payload['memory']==self.memory_head and payload['root']==str(Path(self.context_config['root'])/'task-kernels'),
                     'CONTEXT_READER_OUTPUT_ROOT')
                self.spec=dict(self.spec,root=payload['root'])
            elif kind=='task_retained_enable':
                need(not self.retained_enabled and body['code']==self.retained_code
                     and payload==dict(root=str(Path(self.context_config['root'])/'task-kernels'),
                        previous_root=self.spec['root'],previous=self.active,superseded_intent=self.intent,
                        memory=self.memory_head),'CONTEXT_READER_RETAINED_ENABLE')
                self.spec=dict(self.spec,root=payload['root']); self.intent=None; self.retained_enabled=True
            elif kind.startswith('context_revision_'):
                need(body['code']==self.context_code,'CONTEXT_READER_CODE')
                if kind=='context_revision_intake': self.context_intake(body,payload)
                else:
                    pop=self.context_population
                    ids=sorted(k for k,t in self.tasks.items() if t['population']==pop.pin)
                    need(kind=='context_revision_select' and payload==dict(parent=pop.parent.pin,successor=pop.pin,tasks=ids,
                        parent_to_successor={pop.tasks[k]['provenance']['parent_task']:k for k in ids},
                        policy='replace active training view; retain original evidence and dependency links'),
                        'CONTEXT_READER_SELECTION')
                    self.revisions[pop.parent.pin]=payload
            else:
                prefix='task_retained_' if self.retained_enabled else 'task_growth_'
                expected=self.retained_code if self.retained_enabled else self.growth_code
                need(body['code']==expected,'CONTEXT_READER_GROWTH_CODE')
                if kind==prefix+'register': self.context_register(payload)
                elif kind==prefix+'begin':
                    need(self.intent is None and payload['source']==self.source() and payload['previous']==self.active
                         and payload['memory']==self.memory_head,'CONTEXT_READER_BEGIN')
                    self.intent=dict(id=digest(body),**payload)
                else:
                    need(kind==prefix+'publish','CONTEXT_READER_EVENT_KIND')
                    kernel=read(Path(payload['kernel']['path'])/'manifest.json')
                    updater=read(Path(payload['state']['path'])/'manifest.json')
                    need(kernel['code']==expected['kernel']['kernel'] and updater['code']==expected['kernel'],
                         'CONTEXT_READER_PRODUCERS')
                    if self.retained_enabled:
                        need(kernel['schema']=='connected-task-retained-kernel-v1'
                             and updater['schema']=='capability-retained-updater-state-v1',
                             'CONTEXT_READER_RETAINED_SCHEMA')
                        for name,calibration in kernel['calibration'].items():
                            entries=set(kernel['graphs'][name]['entries'])
                            for row in calibration['rows']:
                                route=row['route'];hits={k for k,_ in route['hits']}
                                need(set(route['kernel_entries'])==entries
                                     and set(route['entries'])==entries|set(route['native_window_entries'])
                                     and set(route['entries'])<=set(kernel['ids'])
                                     and route['work_limit'] is None,'CONTEXT_READER_ROUTE_SCOPE')
                                exact=set(row['exact']);targets=set(row['expected_targets'])
                                need(row['recall']==len(exact&hits)/kernel['plan']['top_k']
                                     and bool(targets) and row['target_recall']==
                                     len(targets&set(row['returned_targets']))/len(targets),
                                     'CONTEXT_READER_RECALL_ARITHMETIC')
                            for family,value in calibration['families'].items():
                                family_rows=[r for r in calibration['rows'] if self.tasks[r['query']]['family']==family]
                                need(len(family_rows)==value['queries'] and value['recall']==sum(r['recall'] for r in family_rows)/len(family_rows)
                                     and value['target_recall']==sum(r['target_recall'] for r in family_rows)/len(family_rows),
                                     'CONTEXT_READER_RECALL_DENOMINATOR')
                    self.origin=dict(slots=self.slots,connections=self.links); self.publication(payload)
            previous=event
        return dict(head=previous,events=len(rows),memory_head=self.memory_head,active=self.active,pending=self.intent,
            revisions=self.revisions,intakes=self.context_intakes,publications=self.publications,
            association_readbacks=self.association_readbacks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','proofs','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); start=time.perf_counter(); config=read(a.config); root=Path(config['root'])
    reader=ContextReader(config,read(a.proofs)); result=reader.run_context(root/'experience.sqlite3')
    saved=read(root/'snapshot-final.json')
    need(result['head']==saved['head'] and result['events']==saved['events']
         and result['memory_head']==saved['memory']['head'] and reader.states==saved['memory']['states']
         and reader.tasks==saved['capability_memory']['tasks'] and reader.fields==saved['capability_memory']['fields']
         and result['active']==saved['task_kernel']['checkpoint'] and result['pending'] is None
         and result['revisions']==saved['context_revisions'],'CONTEXT_READER_FINAL_STATE')
    for stage in ('update','cold'):
        report=read(root/(stage+'-report.json'))
        need(report['complete'] and report['head']==result['head'] and report['events']==result['events']
             and report['optimizer_steps']==0 and report['parameters_unchanged'],'CONTEXT_READER_STAGE')
    output=dict(**result,journal=sha((root/'experience.sqlite3').read_bytes()),elapsed=time.perf_counter()-start,
        native_arrays_read=len(reader.field_arrays),acceptance='author-checked',number_status='measured',
        original_episode_dependencies_retained=True,withdrawals_preserved=True,duplicated_training_experiences=False,
        scientific_admission=False,learned_capability=False,generated_surface_guard_complete=False)
    pin=write_committed(a.output,output)
    print(json.dumps(dict(pin=pin,events=result['events'],associations=result['association_readbacks'])),flush=True)


if __name__=='__main__': main()

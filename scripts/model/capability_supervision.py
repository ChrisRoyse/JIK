"""Purpose-bound targets over an authenticated immutable capability snapshot.

Evidence labels select a training target; they never enter the predictor input.
Source imitation, observed execution and local arithmetic have different scopes.
"""
from collections import Counter

from claims.capability_population import need
from claims.episodic_memory import digest
from claims.repository_cohort_population import CohortPopulation
from claims.repository_depth_population import DepthPopulation
from claims.repository_demonstration_population import DemonstrationPopulation

PURPOSES=('recorded_continuation','successful_patch','annotated_arithmetic','source_answer','grounded_correct_response')
POPULATIONS={'repository_cohort_intake':CohortPopulation,'repository_depth_intake':DepthPopulation,
             'repository_demonstration_intake':DemonstrationPopulation}


class CapabilitySupervision:
    def __init__(self,corpus):
        self.corpus=corpus;self.executions={};self.populations={}
        for source in corpus.sources.values():
            if source.population.family!='coding':continue
            parent=source.population.parent
            for record in parent.report['source']['populations']:
                ref=record['native'];pin=ref['population_pin']
                if pin not in self.populations:
                    pop=POPULATIONS[record['kind']](ref['population'],pin);self.populations[pin]=pop
                    for condition,entry in pop.observations.entries.items():
                        key=pin,condition;need(key not in self.executions,'SUPERVISION_EXECUTION_ALIAS')
                        self.executions[key]=entry

    def evidence(self,key):
        c=self.corpus;need(key in c.train or key in c.query,'SUPERVISION_TASK')
        task=c.tasks[key];pop=c.sources[task['population']].population;source=pop.tasks[key]
        target=pop.fields[source['targets'][0]]
        result=dict(task=key,population=pop.pin,binding=task['binding'],family=task['family'],operation=task['kind'],
            split='train' if key in c.train else 'query',group=task['split_group'],target=target['id'],target_sha256=target['text_sha256'],
            scope='attested_source_only',outcome=None,observations=[],checks=source['checks'],
            scientific_admission=False,world_correctness_established=False)
        if task['family']=='coding':
            original=pop.parent.tasks[source['provenance']['parent_task']];p=original['provenance']
            values=[p] if original['kind']=='reproduce_recorded_patch' else [p['failure'],p['repair']]
            for value in values:
                address=value['population'],value['condition']
                need(address in self.executions,'SUPERVISION_MISSING_EXECUTION')
                entry=self.executions[address]
                need(entry==value['execution_evidence'] and entry['outcome']==value['observed_outcome'],'SUPERVISION_EXECUTION_BINDING')
                result['observations'].append(dict(population=address[0],condition=address[1],receipt=digest(entry),outcome=entry['outcome']))
            witness=values[-1]
            need(target['source']['condition']==witness['condition'] and target['source']['population_pin']==witness['population'],
                 'SUPERVISION_TARGET_EXECUTION')
            result.update(scope='retained_execution_test_scope',outcome=witness['observed_outcome'],parent_kind=original['kind'])
        elif target['role']=='source_answer':result['scope']='question_author_answer_only'
        elif task['kind'] in ('continue_recorded_steps','solve_recorded_problem'):
            result.update(scope='annotated_equalities_only',equalities=dict(Counter(x['status'] for x in source['checks'])))
        return result

    def decide(self,key,purpose,*,fitting=False):
        need(purpose in PURPOSES,'SUPERVISION_PURPOSE')
        need(not fitting or key in self.corpus.train,'SUPERVISION_FIT_QUERY')
        evidence=self.evidence(key);allowed=False;reason='SUPERVISION_SCOPE_UNESTABLISHED'
        if purpose=='recorded_continuation':allowed=True
        elif purpose=='successful_patch':
            allowed=evidence['scope']=='retained_execution_test_scope' and evidence['outcome']==1
            if evidence['outcome']==0:reason='SUPERVISION_FAILED_ACTION'
        elif purpose=='annotated_arithmetic':
            checks=evidence['checks'];allowed=evidence['scope']=='annotated_equalities_only' and bool(checks) and all(x['status']=='pass' for x in checks)
        elif purpose=='source_answer':allowed=evidence['scope']=='question_author_answer_only'
        return dict(purpose=purpose,allowed=allowed,refusal=None if allowed else reason,evidence=digest(evidence),
            scope=evidence['scope'],target=evidence['target'],task=key,split=evidence['split'],fitting=fitting)

    def training_target(self,key,purpose,config,device):
        decision=self.decide(key,purpose,fitting=True);need(decision['allowed'],decision['refusal'])
        raw,slots=self.corpus.target(key,config,device,fitting=True)
        return decision,raw,slots

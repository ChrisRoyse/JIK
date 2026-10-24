"""Two exposed learning partitions, source-bound native targets and frozen region banks."""
import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation import ordered_distance
from model.constellation_learning import identity_pin
from model.native import NativeInput
from model.repository_cohort_input import CohortNative
from model.repository_relation_input import bind, withhold_action
from model.repository_relation_world import RepositoryWorld
from model.tool_context_regions import attach
from senses.trajectory_population import read_pinned


def normalized(slot):
    a = slot.values[0,~slot.padding[0]].detach().cpu().numpy().astype(np.float64)
    return a/np.linalg.norm(a, axis=1, keepdims=True)


def distances(left, right, config):
    a,b = normalized(left),normalized(right)
    check(len(a) > 0 and len(b) > 0 and len(a)*len(b) <= config.max_attention_elements,
          'RELATION_DISTANCE_BUDGET')
    return ordered_distance(np.clip(a@b.T, -1, 1))


class RelationData:
    def __init__(self, session, native, proof_path, proof_pin):
        self.session, self.parent = session, session.model
        self.specs, self.device = self.parent.specs, self.parent.core.initial.device
        proof = read_pinned(Path(proof_path), proof_pin)
        check(proof['native'] == native, 'RELATION_NATIVE_PROOF')
        self.providers, self.examples = {}, {}
        intakes = session.snapshot()['repository_cohort_intakes']
        saved = {k:v for intake in intakes.values() for k,v in intake['bindings'].items()}
        check({k:{s:v[s] for s in ('input','echo')} for k,v in saved.items()} == proof['receipts'],
              'RELATION_MEMORY_NATIVE_JOIN')
        for partition in ('learn_1','learn_2'):
            provider = CohortNative(**native, partition=partition)
            self.providers[partition] = provider
            for key,source in sorted(provider.population.fields.items()):
                if source['partition'] != partition:
                    continue
                episode = provider.population.observations.episode(key)
                check(saved[key]['episode'] == episode['id'] and
                      session.memory.episodes[episode['id']] == episode, 'RELATION_MEMORY_EPISODE_JOIN')
                batch = bind(provider, key, self.parent, saved[key]['input'])
                target,desc = provider.observed(key, 'echo', source['echo_available_after'],
                                               self.parent.config, self.device, self.specs)
                check(desc == saved[key]['echo']['description'], 'RELATION_TARGET_DESCRIPTION')
                for s in self.specs:
                    slot = target.slots[s.name]
                    a = slot.values[0,~slot.padding[0]].detach().cpu().numpy().astype('<f4')
                    check(dict(instrument=s.identity,shape=list(a.shape),sha256=sha(a.tobytes()),
                               dtype='float32-le',absent=bool(slot.absent[0])) == saved[key]['echo']['slots'][s.name],
                          'RELATION_TARGET_NATIVE_JOIN')
                absent = all(bool(v.absent[0]) for v in target.slots.values())
                check(absent == (desc.get('absence') == 'empty_retained_streams'), 'RELATION_TARGET_ABSENCE')
                entry = provider.population.observations.entries[key]
                region = None if absent else digest(saved[key]['echo']['slots'])
                self.examples[key] = dict(batch=batch, baseline=withhold_action(batch,self.specs),
                    target=target.slots, region=region, partition=partition, outcome=entry['outcome'],
                    task=source['task'], receipt=saved[key], absent=absent)
            provider.fresh()
        self.binding = dict(schema='repository-relation-learning-v1',head=session.head,
            active_model=session.snapshot()['active_model'],native=native,proof=dict(path=proof_path,pin=proof_pin),
            episodes={k:v['episode'] for k,v in saved.items()},
            examples={k:dict(input=v['batch'].pin,region=v['region'],outcome=v['outcome'],
                             partition=v['partition'],absent=v['absent']) for k,v in self.examples.items()},
            scope='exposed_original_actions_not_heldout_or_prospective',scientific_admission=False,serving=False)
        self.binding_pin = digest(self.binding)
        self.stages = {}

    def stage(self, partition, model):
        check(partition in ('learn_1','learn_2'), 'RELATION_PARTITION')
        if partition in self.stages:
            return self.stages[partition]
        keys = [k for k,e in self.examples.items() if e['partition']=='learn_1' or partition=='learn_2']
        regions = list(dict.fromkeys(self.examples[k]['region'] for k in keys if not self.examples[k]['absent']))
        representatives = [next(self.examples[k] for k in keys if self.examples[k]['region']==r) for r in regions]
        refs = {}
        for s in self.specs:
            width = max(e['target'][s.name].values.shape[1] for e in representatives)
            values = torch.zeros((len(regions),width,s.dims),device=self.device)
            padding = torch.ones(values.shape[:2],dtype=torch.bool,device=self.device)
            for i,e in enumerate(representatives):
                a = e['target'][s.name].values[0,~e['target'][s.name].padding[0]]
                values[i,:len(a)],padding[i,:len(a)] = a,False
            refs[s.name] = NativeInput(s.identity,tuple(regions),values,padding,
                                       torch.zeros(len(regions),dtype=torch.bool,device=self.device))
        desc = dict(binding=self.binding_pin,partition=partition,regions=regions,keys=keys)
        view = RepositoryWorld.from_world(model,references=refs,world_source_id=digest(desc))
        labels = []
        for r in regions:
            seen = {self.examples[k]['outcome'] for k in keys if self.examples[k]['region']==r}
            labels.append(next(iter(seen)) if len(seen)==1 else 2)
        costs,persistence = {},{}
        for key in keys:
            e = self.examples[key]
            if e['absent']:
                continue
            costs[key],persistence[key] = {},{}
            for s in self.specs:
                lane = [distances(e['target'][s.name],r['target'][s.name],model.config) for r in representatives]
                costs[key][s.name] = torch.tensor(lane,dtype=torch.float32,device=self.device)
                persistence[key][s.name] = distances(e['target'][s.name],e['batch'][s.name],model.config)
        self.stages[partition] = dict(view=view,keys=keys,regions=regions,description=desc,costs=costs,
            persistence=persistence,labels=torch.tensor(labels,device=self.device),
            allowed=torch.ones(len(regions),dtype=torch.bool,device=self.device))
        return self.stages[partition]

    def forward(self, model, key, partition, *, intervention=False):
        stage,e = self.stage(partition,model),self.examples[key]
        check(key in stage['keys'], 'RELATION_FUTURE_PARTITION')
        view = attach(stage['view'],model)
        batch = e['baseline'] if intervention else e['batch']
        # This reads a retained experience. It does not invent a fresh Act or Echo occurrence.
        state = view.start('repository:'+e['task'])
        return view(batch,(state,),allowed=stage['allowed'],phase=0,event_ids=[batch.pin],
                    evidence_origin=['source_observation'],advance_phase=False)

    def losses(self, model, key, partition):
        stage,e = self.stage(partition,model),self.examples[key]
        out = self.forward(model,key,partition)
        lanes = dict(out['regions'],common=out['constellation_regions']/len(self.specs))
        terms = []
        truth = F.one_hot(torch.tensor(e['outcome'],device=self.device),3).float()
        for name,scores in lanes.items():
            probs = F.softmax(scores[0],dim=-1)
            anchor = torch.stack([probs[stage['labels']==i].sum() for i in range(3)])
            term = ((anchor-truth)**2).sum()
            if not e['absent']:
                target = torch.tensor([stage['regions'].index(e['region'])],device=self.device)
                term = term+F.cross_entropy(scores,target)
                if name != 'common':
                    term = term+(probs*stage['costs'][key][name]).sum()
            terms.append(term)
        return torch.stack(terms).mean()

    def evaluate(self, model, partition, *, intervention=False):
        stage = self.stage(partition,model)
        results = {}
        with torch.no_grad():
            for key in stage['keys']:
                e = self.examples[key]
                output = self.forward(model,key,partition,intervention=intervention)
                lanes = dict(output['regions'],common=output['constellation_regions']/len(self.specs))
                row = dict(outcome=e['outcome'],native_target_absent=e['absent'],task=e['task'],
                    region=output['predicted_constellation'].regions[0],identity=identity_pin(output),lanes={})
                for name,scores in lanes.items():
                    probabilities = F.softmax(scores[0],dim=-1)
                    anchor = [float(probabilities[stage['labels']==i].sum()) for i in range(3)]
                    value = dict(probabilities=probabilities.tolist(),anchor=anchor,
                                 brier=sum((v-float(i==e['outcome']))**2 for i,v in enumerate(anchor)))
                    if not e['absent']:
                        index = stage['regions'].index(e['region'])
                        value.update(target_probability=float(probabilities[index]),correct=int(scores.argmax())==index)
                        if name != 'common':
                            value.update(native_expected_loss=float((probabilities*stage['costs'][key][name]).sum()),
                                native_placement_loss=distances(e['target'][name],output['predicted_constellation'].slots[name],model.config),
                                persistence_loss=stage['persistence'][key][name])
                    row['lanes'][name] = value
                results[key] = row
        return results

    def fresh(self):
        check(self.session.head == self.binding['head'] and digest(self.binding) == self.binding_pin,
              'RELATION_BINDING_CHANGED')
        for p in self.providers.values():
            p.fresh()

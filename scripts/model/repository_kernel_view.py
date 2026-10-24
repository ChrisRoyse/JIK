"""Make the checked repository kernel's references visible to the actual model."""
import json
from pathlib import Path

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from kernel.repository_kernel import RepositoryKernel
from model.native import NativeInput
from model.repository_relation_input import assemble
from model.repository_streamed_world import StreamedRepositoryWorld


class RepositoryKernelView:
    def __init__(self,model,population,spec):
        self.population,self.spec = population,dict(spec)
        self.kernel = RepositoryKernel(population,spec['path'],spec['pin'],expected_source=spec['source'])
        check(tuple(model.specs) == tuple(population.specs[n] for n in sorted(population.specs)),
              'REPOSITORY_RUNTIME_INSTRUMENTS')
        self.ids = tuple(self.kernel.value['ids'])
        self.regions,self.representatives,self.echo_by_condition = [],[],{}
        self.region_by_atom = {}
        for key in self.ids:
            record = population.records[key]
            if record['side'] != 'echo' or record['absence']:continue
            region = digest(record['receipt']['slots'])
            self.region_by_atom[key] = region
            check(record['condition'] not in self.echo_by_condition,'REPOSITORY_RUNTIME_CONDITION')
            self.echo_by_condition[record['condition']] = key
            if region not in self.regions:
                self.regions.append(region)
                self.representatives.append(dict(target=self.slots(model,key)))
        self.region_seal = digest(dict(regions=self.regions,echoes=self.echo_by_condition,atoms=self.region_by_atom))
        self.view = StreamedRepositoryWorld(model,self.regions,self.representatives,
            world_source_id=digest(dict(kernel=spec['pin'],regions=self.regions))).eval()
        self.fields = {}
        for group in population.groups:
            path = Path(group['spec']['population'])/'fields.json'
            if str(path) not in self.fields:self.fields[str(path)] = json.loads(path.read_bytes())
        self.condition_fields = {key:value for fields in self.fields.values() for key,value in fields.items()}

    def slots(self,model,key):
        record = self.population.records[key]
        result = {}
        for s in model.specs:
            values = torch.from_numpy(self.population.values[key][s.name].copy()).unsqueeze(0).to(model.core.initial.device)
            result[s.name] = NativeInput(s.identity,(key,),values,
                torch.zeros(values.shape[:2],dtype=torch.bool,device=values.device),
                torch.tensor([record['absence']],dtype=torch.bool,device=values.device))
        return result

    def batch(self,key):
        check(key in self.population.records and self.population.records[key]['side'] == 'input',
              'REPOSITORY_RUNTIME_INPUT_SIDE')
        record = self.population.records[key]
        return assemble(self.slots(self.view,key),record['receipt']['description'],
                        self.condition_fields[record['condition']]['missing']).checked(self.view.specs)

    def route(self,key,cutoff):
        check(digest(dict(regions=self.regions,echoes=self.echo_by_condition,atoms=self.region_by_atom)) == self.region_seal,
              'REPOSITORY_RUNTIME_REGION_CHANGED')
        trace = self.kernel.route(key,source=self.spec['source'],cutoff=cutoff)
        support = set()
        for atom in trace['sources']:
            record = self.population.records[atom]
            echo = self.echo_by_condition.get(record['condition'])
            if echo is not None:support.add(echo)
        check(support,'REPOSITORY_RUNTIME_NO_ECHO_SUPPORT')
        chosen = {self.region_by_atom[k] for k in support}
        allowed = torch.tensor([r in chosen for r in self.regions],dtype=torch.bool,
                               device=self.view.core.initial.device)
        return allowed,dict(trace=trace,echo_support=sorted(support),regions=[r for r in self.regions if r in chosen])

    def forward(self,key,state,*,cutoff,event):
        batch = self.batch(key)
        allowed,routing = self.route(key,cutoff)
        out = self.view(batch,(self.view.bind_state(state),),allowed=allowed,phase=state.sequence%5,
            event_ids=[event],evidence_origin=['source_observation'],advance_phase=False)
        check(out['phase']=='Observation' and out['states'][0].sequence==state.sequence,'REPOSITORY_RUNTIME_PHASE')
        for name,scores in out['regions'].items():
            check(torch.equal(torch.isfinite(scores[0]),allowed),'REPOSITORY_RUNTIME_MASK_CONSUMED')
        check(out['predicted_constellation'].regions[0] in routing['regions'],'REPOSITORY_RUNTIME_SELECTED_REGION')
        return out,dict(routing=routing,allowed=allowed.tolist(),bank=self.view.world_id,
            input=batch.pin,native_slots=self.population.records[key]['receipt']['slots'])

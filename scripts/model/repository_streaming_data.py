"""Complete teaching history with bounded typed-region scoring for growing banks."""
from pathlib import Path
import torch

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.repository_relation_data import RelationData,distances
from model.repository_streamed_world import StreamedRepositoryWorld
from model.repository_teaching_data import TeachingData,roster,code_pins as teaching_pins


def code_pins():
    root=Path(__file__).parent
    return dict(teaching_pins(),**{'model/'+name:sha((root/name).read_bytes()) for name in
                ('repository_streaming_data.py','repository_streamed_world.py')})


class StreamingTeachingData(TeachingData):
    def __init__(self,session,populations):
        super().__init__(session,populations)
        self.binding.update(schema='repository-streaming-teaching-v1',adapter_code=code_pins())
        self.binding_pin=digest(self.binding)

    def stage(self,partition,model):
        check(partition in self.order,'GROWTH_DATA_PARTITION')
        if partition in self.stages:return self.stages[partition]
        visible=set(self.order[:self.order.index(partition)+1])
        keys=[k for k,e in self.examples.items() if e['partition'] in visible]
        regions=list(dict.fromkeys(self.examples[k]['region'] for k in keys if not self.examples[k]['absent']))
        representatives=[next(self.examples[k] for k in keys if self.examples[k]['region']==r) for r in regions]
        check(regions,'GROWTH_DATA_EMPTY_BANK')
        desc=dict(binding=self.binding_pin,partition=partition,regions=regions,keys=keys)
        view=StreamedRepositoryWorld(model,regions,representatives,world_source_id=digest(desc))
        labels=[]
        for region in regions:
            outcomes={self.examples[k]['outcome'] for k in keys if self.examples[k]['region']==region}
            labels.append(next(iter(outcomes)) if len(outcomes)==1 else 2)
        costs,persistence={},{}
        for key in keys:
            e=self.examples[key]
            if e['absent']:continue
            costs[key],persistence[key]={},{}
            for s in self.specs:
                lane=[distances(e['target'][s.name],r['target'][s.name],model.config) for r in representatives]
                costs[key][s.name]=torch.tensor(lane,dtype=torch.float32,device=self.device)
                persistence[key][s.name]=distances(e['target'][s.name],e['batch'][s.name],model.config)
        self.stages[partition]=dict(view=view,keys=keys,regions=regions,description=desc,costs=costs,
            persistence=persistence,labels=torch.tensor(labels,device=self.device),
            allowed=torch.ones(len(regions),dtype=torch.bool,device=self.device))
        return self.stages[partition]

    def fresh(self):
        check(self.binding['adapter_code']==code_pins(),'EXPERIENCE_DATA_CODE_CHANGED')
        check(roster(self.session,self.binding['populations'])[0]==self.groups,'EXPERIENCE_DATA_ORDER_CHANGED')
        for stage in self.stages.values():stage['view'].check_reference()
        RelationData.fresh(self)

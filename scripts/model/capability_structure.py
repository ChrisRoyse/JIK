"""Per-instrument field context with a zero-initialized, parameter-isolated extension."""
from dataclasses import replace

import torch
from torch import nn

from claims.capability_population import need
from claims.episodic_memory import digest
from model.attention import Attention, Linear
from model.capability_structure_input import FIELD_FEATURES, WINDOW_FEATURES, StructuredInput
from model.stream_attention import blocks, observe

ARMS=('boundaries','structured','mismatched')


class FieldContext(nn.Module):
    def __init__(self,world):
        super().__init__();self.names=tuple(s.name for s in world.specs);self.config=world.config
        self.windows=nn.ModuleDict({n:Attention(world.config) for n in self.names})
        self.fields=nn.ModuleDict({n:Attention(world.config) for n in self.names})
        self.position=nn.ModuleDict({n:Linear(WINDOW_FEATURES,world.config.width,bias=False) for n in self.names})
        self.metadata=nn.ModuleDict({n:Linear(FIELD_FEATURES,world.config.width,bias=False) for n in self.names})
        self.out=nn.ModuleDict({n:Linear(world.config.width,world.config.width,bias=False) for n in self.names})
        for layer in self.out.values():nn.init.zeros_(layer.weight)

    def forward(self,world,state,item,*,mode):
        need(isinstance(item,StructuredInput) and mode in ARMS,'FIELD_CONTEXT_INPUT')
        need(state.context=='capability:'+item.task,'FIELD_CONTEXT_CONTEXT');result=[]
        # Immutable inputs were authenticated and shape-checked when materialized.
        # Keep metadata on CPU; no corpus hash or synchronization in the model loop.
        metadata=item.field_metadata
        if mode=='boundaries':metadata=torch.zeros_like(metadata)
        elif mode=='mismatched':metadata=metadata.roll(1,0)
        for lane,name in enumerate(self.names):
            slot=item.native[name];q=state.hidden[lane:lane+1,None,:];field_states=[]
            for index,(lo,hi) in enumerate(item.spans[name]):
                positions=item.windows[name][index]
                if mode=='boundaries':positions=torch.zeros_like(positions)
                elif mode=='mismatched':positions=positions.flip(0)
                # Stream metadata on the native device, preserving each field's full native span.
                positions=positions.to(slot.values.device)[None]
                derived=observe(self.windows[name],world.core.projectors[name],q,
                    blocks(slot.values[:,lo:hi],slot.padding[:,lo:hi],self.config,positions=positions),
                    counts=(hi-lo,),config=self.config,positioner=self.position[name])
                field_states.append(derived)
            bank=torch.cat(field_states,dim=1)+self.metadata[name](metadata.to(q.device))[None]
            result.append(self.out[name](self.fields[name](q,bank))[:,0])
        return torch.stack(result,dim=1)


def attach_context(model,optimizer,seed):
    """Keep all inherited weights/moments; newly allocated parameters start without moments."""
    need('context' not in model,'FIELD_CONTEXT_EXISTS');model.requires_grad_(False)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);extension=FieldContext(model['world']).to('cuda').eval()
    model['context']=extension
    need(sum(p.numel() for p in model.parameters())<=model['world'].config.max_parameters,'FIELD_CONTEXT_PARAMETER_BUDGET')
    optimizer.add_param_group(dict(params=list(extension.parameters())))
    need(all(p not in optimizer.state for p in extension.parameters()),'FIELD_CONTEXT_NEW_MOMENTS')
    return extension


def predict_structured(model,item,operation,*,mode='structured'):
    world=model['world'];state=model['request'](world.start('capability:'+item.task),operation=operation)
    residual=model['context'](world,state,item,mode=mode)
    state=replace(state,hidden=state.hidden+residual[0])
    output=world(item.native,(state,),allowed=torch.ones(len(world.reference_rows[world.specs[0].name]),
        dtype=torch.bool,device=world.core.initial.device),phase=0,
        event_ids=[digest(dict(key=item.task,operation=operation))],evidence_origin=['source_observation'],advance_phase=False)
    return dict(output,structural_context=residual,context_kind='source-bound-field-context-v1')

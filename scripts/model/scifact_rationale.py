"""Typed multiscale relation head with a canonical complete-evidence pointer."""
import numpy as np
import torch
from torch import nn
from model.scifact_readout import NativeStore,RelationReadout,check

CONFIG=dict(width=64,heads=4,batch=32,steps=200,learning_rate=.001,weight_decay=.01,seeds=[25,26,27],
            arms={"relation-only":0.,"rationale":1.,"shifted-rationale":1.},stop=42,precision="fp32")


class RationaleStore(NativeStore):
    def __init__(self,ctx,truth,device):
        super().__init__(ctx,device);self.valid=torch.from_numpy(truth["valid"]).to(device)
        self.members=[torch.from_numpy(x).to(device) for x in truth["members"]]

    def batch(self,index):
        result=super().batch(index);rows=torch.as_tensor(index,device=self.device,dtype=torch.long);docs=self.documents[rows]
        return [{**m,"membership":x[docs]} for m,x in zip(result,self.members)],self.valid[docs]


class Pointer(nn.Module):
    def __init__(self):
        super().__init__();self.keys=nn.Linear(64,64);self.state=nn.Linear(64,64);self.energy=nn.Linear(64,1)
        self.stop=nn.Parameter(torch.randn(1,1,64)*.02);self.start=nn.Parameter(torch.randn(1,1,64)*.02)

    def raw(self,hidden,previous):
        batch,sentences,_=hidden.shape;values=torch.cat([hidden,self.stop.expand(batch,1,-1)],dim=1)
        prior=values.gather(1,previous.clamp_min(0).unsqueeze(-1).expand(-1,-1,64))
        prior=torch.where((previous>=0).unsqueeze(-1),prior,0.)+self.start
        return self.energy(torch.tanh(self.keys(values).unsqueeze(1)+self.state(prior).unsqueeze(2))).squeeze(-1)

    def normalized(self,raw,valid,previous):
        positions=torch.arange(valid.shape[1],device=valid.device)
        legal=valid.unsqueeze(1)&(positions>previous.unsqueeze(-1))
        legal=torch.cat([legal,torch.ones((*legal.shape[:2],1),device=valid.device,dtype=torch.bool)],dim=-1)
        return raw.masked_fill(~legal,-torch.inf).log_softmax(-1)

    def paths(self,hidden,valid,choices,steps):
        batch,alternatives,length=choices.shape
        previous=torch.cat([torch.full_like(choices[:,:,:1],-1),choices[:,:,:-1]],dim=-1).reshape(batch,-1)
        raw=self.raw(hidden,previous);lp=self.normalized(raw,valid,previous)
        value=lp.gather(2,choices.reshape(batch,-1,1)).squeeze(-1).reshape(batch,alternatives,length)
        return torch.where(steps,value,0.).sum(-1),raw.reshape(batch,alternatives,length,-1)

    def loss(self,hidden,valid,targets):
        active=targets["present"].any(1)
        if not bool(active.any()):return hidden.sum()*0
        values,_=self.paths(hidden[active],valid[active],targets["choices"][active],targets["steps"][active])
        return -values.masked_fill(~targets["present"][active],-torch.inf).logsumexp(1).mean()

    @torch.inference_mode()
    def decode(self,hidden,valid):
        batch,width=valid.shape;previous=torch.full((batch,1),-1,device=hidden.device,dtype=torch.long)
        paths=torch.full((batch,width+1),width,device=hidden.device,dtype=torch.long);steps=torch.zeros_like(paths,dtype=torch.bool)
        traces=torch.zeros((batch,width+1,width+1),device=hidden.device);active=torch.ones(batch,device=hidden.device,dtype=torch.bool)
        for t in range(width+1):
            raw=self.raw(hidden,previous);lp=self.normalized(raw,valid,previous);choice=lp[:,0].argmax(-1)
            paths[:,t]=torch.where(active,choice,width);steps[:,t]=active;traces[:,t]=torch.where(active[:,None],raw[:,0],0.)
            previous=paths[:,t:t+1];active=active&(choice!=width)
            if not bool(active.any()):break
        check(not bool(active.any()),"POINTER_UNTERMINATED")
        return dict(paths=paths,path_steps=steps,greedy_logits=traces)


class RationaleReadout(RelationReadout):
    def __init__(self,dimensions):
        super().__init__(dimensions);self.pointer=Pointer()

    def forward(self,memories,valid):
        check(len(memories)==14 and valid.dtype==torch.bool and valid.ndim==2 and valid.shape[1]==42,"RATIONALE_INPUT")
        check(all(set(m)=={"query","evidence","mask","levels","geometry","membership"} for m in memories),"INPUT_FIELDS")
        batch,sentences=valid.shape;task=self.query.expand(batch,1,-1);doc=[];local=[]
        for k,m in enumerate(memories):
            q=self.projections[k](m["query"]).unsqueeze(1)
            evidence=self.projections[k](m["evidence"])+self.scale(m["levels"])+self.geometry(m["geometry"])
            read,_=self.attention[k](q,evidence,evidence,key_padding_mask=~m["mask"],need_weights=False)
            hidden=q+read;doc.append(self.relations[k](hidden)+self.instrument[k])
            member=m["membership"];check(member.shape==(batch,sentences,evidence.shape[1]) and member.dtype==torch.bool,"MEMBERSHIP_SHAPE")
            check(torch.equal(member.any(-1),valid) and not bool((member&~m["mask"][:,None,:]).any()),"MEMBERSHIP_COVERAGE")
            safe=member.clone();safe[:,:,0]|=~valid
            mask=(~safe)[:,None,:,:].expand(-1,4,-1,-1).reshape(batch*4,sentences,-1)
            view,_=self.attention[k](hidden.expand(-1,sentences,-1),evidence,evidence,attn_mask=mask,need_weights=False)
            local.append(self.relations[k](hidden+view)+self.instrument[k])
        bound,_=self.binding(task,torch.cat(doc,dim=1),torch.cat(doc,dim=1),need_weights=False)
        classification=self.head((task+bound).squeeze(1))
        # Only learned typed branch tokens bind; native instrument arrays stay separate.
        tokens=torch.stack(local,dim=2).reshape(batch*sentences,14,64);query=self.query.expand(batch*sentences,1,-1)
        selected,_=self.binding(query,tokens,tokens,need_weights=False)
        sentence_hidden=(query+selected).reshape(batch,sentences,64).masked_fill(~valid[:,:,None],0.)
        return classification,sentence_hidden


@torch.inference_mode()
def evaluate(model,store,rows,truth):
    outputs={k:[] for k in ("probabilities","paths","path_steps","greedy_logits","teacher_logits")}
    for begin in range(0,len(rows),32):
        ids=np.arange(begin,min(len(rows),begin+32));memories,valid=store.batch(ids);logits,hidden=model(memories,valid)
        values=model.pointer.decode(hidden,valid);values["probabilities"]=logits.softmax(-1)
        t={k:torch.from_numpy(v[ids]).to(store.device) for k,v in truth["targets"].items()}
        _,values["teacher_logits"]=model.pointer.paths(hidden,valid,t["choices"],t["steps"])
        for name,v in values.items():outputs[name].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in outputs.items()}

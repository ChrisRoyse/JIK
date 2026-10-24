"""From-scratch typed-sense choice scorer with no option-position or target input."""
import numpy as np
import torch
from torch import nn
from model.qasc_context import check

CONFIG=dict(width=64,heads=4,steps=400,batch=32,learning_rate=.001,weight_decay=.01,seeds=[25,26,27],precision='fp32',bootstrap=500)
ARMS=('both-facts','fact1-only','fact2-only','question-only','facts-only')

class Store:
    def __init__(self,ctx,device):
        self.native=[torch.from_numpy(x).to(device) for x in ctx['native']]
        self.maps=torch.from_numpy(ctx['maps']).to(device);self.device=device
    def batch(self,ids,donors=None):
        ids=torch.as_tensor(ids,device=self.device);m=self.maps[ids]
        factmap=m[:,1:3] if donors is None else self.maps[torch.as_tensor(donors,device=self.device)][:,1:3]
        return [dict(question=x[m[:,0]],facts=x[factmap],options=x[m[:,3:]]) for x in self.native]

class Combination(nn.Module):
    def __init__(self):
        super().__init__();w=CONFIG['width']
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(768,w),nn.LayerNorm(w)) for _ in range(6)])
        self.attend=nn.ModuleList([nn.ModuleList([nn.MultiheadAttention(w,4,batch_first=True,dropout=0.) for _ in range(2)]) for _ in range(6)])
        self.relate=nn.ModuleList([nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w),nn.LayerNorm(w)) for _ in range(6)])
        self.roles=nn.Embedding(3,w);self.instrument=nn.Parameter(torch.randn(6,w)*.02)
        self.task=nn.Parameter(torch.randn(1,1,w)*.02);self.binding=nn.MultiheadAttention(w,4,batch_first=True,dropout=0.)
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,1))
    def forward(self,memories,scope):
        check(scope in ARMS and len(memories)==6,'FORWARD_SCOPE')
        check(all(set(m)=={'question','facts','options'} for m in memories),'FORWARD_FIELDS')
        batch=len(memories[0]['question']);typed=[]
        allowed=[scope!='facts-only',scope not in ('fact2-only','question-only'),scope not in ('fact1-only','question-only')]
        for k,m in enumerate(memories):
            check(m['question'].shape==(batch,768) and m['facts'].shape==(batch,2,768) and m['options'].shape==(batch,8,768),'FORWARD_SHAPE')
            q=self.project[k](m['question']).unsqueeze(1)+self.roles.weight[0]
            facts=self.project[k](m['facts'])+self.roles.weight[1]
            memory=torch.cat([q,facts],1);mask=~torch.tensor(allowed,device=memory.device).expand(batch,-1)
            h=self.project[k](m['options'])+self.roles.weight[2]
            for attention in self.attend[k]:
                read,_=attention(h,memory,memory,key_padding_mask=mask,need_weights=False);h=h+read
            typed.append(self.relate[k](h)+self.instrument[k])
        # Stack typed learned tokens along the instrument axis; never pool native slots.
        tokens=torch.stack(typed,dim=2).reshape(batch*8,6,CONFIG['width']);query=self.task.expand(batch*8,1,-1)
        bound,_=self.binding(query,tokens,tokens,need_weights=False)
        return self.head(query+bound).reshape(batch,8)

def schedule(rows,seed):
    train=np.array([i for i,r in enumerate(rows) if r['partition']=='train'],np.int64);rng=np.random.default_rng(seed)
    return np.asarray([rng.choice(train,CONFIG['batch'],replace=False) for _ in range(CONFIG['steps'])])

def donors(rows):
    result=np.zeros(len(rows),np.int64)
    for split in ('train','calibration','evaluation'):
        groups={}
        for i,r in enumerate(rows):
            if r['partition']==split:groups.setdefault(r['group'],[]).append(i)
        keys=sorted(groups);check(len(keys)>1,'DONOR_GROUPS')
        for j,g in enumerate(keys):
            for i in groups[g]:result[i]=groups[keys[(j+1)%len(keys)]][0]
    return result

@torch.inference_mode()
def evaluate(model,store,n,scope,donor=None):
    values=[]
    for start in range(0,n,32):
        ids=np.arange(start,min(n,start+32));values.append(model(store.batch(ids,None if donor is None else donor[ids]),scope).softmax(1).cpu().numpy())
    return np.vstack(values)

@torch.inference_mode()
def controls(model,store):
    mem=store.batch(np.arange(32));base=model(mem,'both-facts');perm=torch.arange(7,-1,-1,device=store.device)
    changed=[{**m,'options':m['options'][:,perm]} for m in mem]
    difference=float((model(changed,'both-facts')-base[:,perm]).abs().max());check(difference<2e-5,'CHOICE_POSITION')
    changed=[{**m,'facts':m['facts'].flip(1)} for m in mem]
    swap=float((model(changed,'both-facts')-base).abs().max());check(swap<2e-5,'FACT_ORDER')
    original=model(mem,'question-only');changed=[{**m,'facts':m['facts']+17} for m in mem]
    check(torch.equal(original,model(changed,'question-only')),'DROPPED_FACT_LEAKAGE')
    original=model(mem,'facts-only');changed=[{**m,'question':m['question']+17} for m in mem]
    check(torch.equal(original,model(changed,'facts-only')),'DROPPED_QUESTION_LEAKAGE')
    bad=[{**m,'answer_index':0} for m in mem]
    try:model(bad,'both-facts')
    except ValueError as e:check('FORWARD_FIELDS' in str(e),'TARGET_ERROR')
    else:raise ValueError('MNLM_QASC_TARGET_ACCEPTED')
    return dict(choice_permutation_max_error=difference,fact_swap_max_error=swap,removed_inputs_bitwise_irrelevant=True,private_target_refused=True)

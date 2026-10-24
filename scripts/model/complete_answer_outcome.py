"""Purpose routing and typed native execution prediction; never an execution oracle."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from claims.episodic_memory import digest
from model.capability_experience_learning import PURPOSE
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need

ROLES = ('task_report', 'original_source', 'base_documentation', 'base_api_source',
         'public_base_test', 'candidate')
PRIOR = ('recorded_patch', 'candidate_source', 'stdout')
ARMS = ('actual', 'shuffled', 'context_only', 'candidate_only')


def routes(supervisor):
    result = {}
    for key in supervisor.view.source.train + supervisor.view.source.query:
        e = supervisor.evidence(key); fitting = e['split'] == 'train'
        d = supervisor.decide(key, PURPOSE[e['operation']], fitting=fitting)
        result[key] = dict(task=key, split=e['split'], evidence=d['evidence'], generation=d,
            outcome=e['original_evidence']['outcome'] if e['original_evidence']['scope']=='retained_execution_test_scope' else None,
            scope=e['original_evidence']['scope'], recorded_observation_retained=True,
            scientific_admission=False, serving=False)
    return result


class OutcomeInputs:
    """IDs join immutable fields; only native readings and declared roles reach the head."""
    def __init__(self, rows, fields, roster, *, device='cpu'):
        self.rows = deepcopy(rows); self.seal = digest(self.rows); self.roster = tuple(roster)
        self.train = tuple(k for k,r in rows.items() if r['split']=='train')
        self.query = tuple(k for k,r in rows.items() if r['split']=='query')
        need(self.train and self.query and not {rows[k]['group'] for k in self.train} &
             {rows[k]['group'] for k in self.query}, 'OUTCOME_GROUP_SPLIT')
        self.fields = {f:{n:x.detach().clone().to(device) for n,x in v.items()} for f,v in fields.items()}
        need(all(tuple(sorted(v))==tuple(sorted(self.roster)) and all(x.ndim==2 and len(x)>0
             and x.dtype==torch.float32 and bool(x.isfinite().all()) for x in v.values())
             for v in self.fields.values()), 'OUTCOME_NATIVE_LAYOUT')
        for row in rows.values():
            need(row['outcome'] in (0,1) and row['scope']=='retained_execution_test_scope'
                 and row['context'] and len(row['candidate'])==1, 'OUTCOME_ROW_SCOPE')
            need(all(r['role'] in ROLES and r['field'] in fields for r in row['context']+row['candidate'])
                 and all(r['role']!='candidate' for r in row['context'])
                 and row['candidate'][0]['role']=='candidate', 'OUTCOME_ROW_FIELDS')
        self.native = fingerprint(self.fields)
        self.versions = {(f,n):x._version for f,v in self.fields.items() for n,x in v.items()}

    def fresh(self, *, full=False):
        need(digest(self.rows)==self.seal, 'OUTCOME_CHANGED_BINDING')
        need(all(x._version==self.versions[f,n] for f,v in self.fields.items() for n,x in v.items()),
             'OUTCOME_CHANGED_NATIVE')
        if full: need(fingerprint(self.fields)==self.native, 'OUTCOME_CHANGED_NATIVE')

    def batch(self, keys, arm, *, fitting):
        self.fresh(); need(arm in ARMS and type(fitting) is bool, 'OUTCOME_ARM')
        need(keys and len(set(keys))==len(keys) and all(k in (self.train if fitting else self.query) for k in keys),
             'OUTCOME_FIT_QUERY' if fitting else 'OUTCOME_QUERY_ROSTER')
        sequences = []
        for key in keys:
            row=self.rows[key]
            seq = (row['context'] if arm=='context_only' else row['candidate'] if arm=='candidate_only'
                   else row['context']+row['candidate'])
            sequences.append(tuple((r['field'], ROLES.index(r['role'])) for r in seq))
        used=sorted({f for seq in sequences for f,_ in seq})
        return dict(sequences=tuple(sequences), fields={f:self.fields[f] for f in used}, roster=self.roster)

    def labels(self, keys, arm, *, fitting):
        self.batch(keys,arm,fitting=fitting)
        selected=list(keys)
        if arm=='shuffled' and fitting:
            need(tuple(keys)==self.train, 'OUTCOME_SHUFFLE_ROSTER')
            selected=sorted(keys,key=lambda k:digest(['outcome-label-permutation-v1',k]))
            need(any(self.rows[k]['outcome']!=self.rows[t]['outcome'] for k,t in zip(keys,selected)),
                 'OUTCOME_UNCHANGED_NULL')
        return torch.tensor([self.rows[k]['outcome'] for k in selected],dtype=torch.float32,
                            device=next(iter(next(iter(self.fields.values())).values())).device)


class NativeExecutionOutcome(nn.Module):
    def __init__(self, dimensions, width=16):
        super().__init__(); self.roster=tuple(dimensions)
        self.lanes=nn.ModuleDict({n:nn.ModuleDict(dict(project=nn.Linear(d,width),
            windows=nn.GRU(width,width,batch_first=True), roles=nn.Embedding(len(ROLES),width),
            fields=nn.GRU(width,width,batch_first=True), outcome=nn.Linear(width,1))) for n,d in dimensions.items()})

    def forward(self, batch):
        need(batch['roster']==self.roster, 'OUTCOME_INSTRUMENTS')
        names=tuple(batch['fields']); index={f:i for i,f in enumerate(names)}; outputs={}
        for name,lane in self.lanes.items():
            # Compute each distinct field once, preserving all windows and their order.
            packed=pack_sequence([batch['fields'][f][name] for f in names],enforce_sorted=False)
            packed=PackedSequence(torch.tanh(lane['project'](packed.data)),packed.batch_sizes,
                                  packed.sorted_indices,packed.unsorted_indices)
            _,hidden=lane['windows'](packed); fields=hidden[0]
            sequences=[]
            for seq in batch['sequences']:
                ids=torch.tensor([index[f] for f,_ in seq],device=fields.device)
                roles=torch.tensor([r for _,r in seq],device=fields.device)
                sequences.append(fields[ids]+lane['roles'](roles))
            _,hidden=lane['fields'](pack_sequence(sequences,enforce_sorted=False))
            outputs[name]=lane['outcome'](hidden[0]).squeeze(-1)
        return outputs


def make(dimensions, seed, *, device='cpu'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model=NativeExecutionOutcome(dimensions).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01,foreach=False)
    return model,optimizer


def validate(model, optimizer, step):
    need(type(step) is int and step>=0 and len(optimizer.param_groups)==1
         and {id(p) for p in optimizer.param_groups[0]['params']}=={id(p) for p in model.parameters()}, 'OUTCOME_OPTIMIZER_OWNER')
    group=optimizer.param_groups[0]
    need(group['lr']==0.003 and group['weight_decay']==0.01 and group['betas']==(0.9,0.999)
         and group['eps']==1e-8 and group['foreach'] is False, 'OUTCOME_OPTIMIZER_POLICY')
    need((step==0 and not optimizer.state) or (step>0 and len(optimizer.state)==len(list(model.parameters()))
         and all(int(v['step'])==step and bool(v['exp_avg'].isfinite().all())
                 and bool(v['exp_avg_sq'].isfinite().all()) for v in optimizer.state.values())), 'OUTCOME_OPTIMIZER_STATE')


def update(model, optimizer, inputs, arm, step):
    validate(model,optimizer,step)
    keys=inputs.train; batch=inputs.batch(keys,arm,fitting=True); labels=inputs.labels(keys,arm,fitting=True)
    optimizer.zero_grad(set_to_none=True); before=fingerprint(model.state_dict())
    logits=model(batch); stacked=torch.stack(list(logits.values()),1)
    loss=F.binary_cross_entropy_with_logits(stacked,labels[:,None].expand_as(stacked))
    gradient=torch.autograd.grad(loss,stacked,retain_graph=True)[0]
    expected=(stacked.detach().sigmoid()-labels[:,None])/stacked.numel()
    need(torch.equal(gradient,expected), 'OUTCOME_LABEL_GRADIENT')
    need(bool((gradient[labels==0]>0).all()) and bool((gradient[labels==1]<0).all()), 'OUTCOME_NEGATIVE_DIRECTION')
    loss.backward(); gradients=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()), 'OUTCOME_GRADIENTS')
    for name in model.roster:
        for branch in ('project','windows','roles','fields','outcome'):
            need(any(bool(g.abs().any()) for n,g in gradients.items() if n.startswith('lanes.'+name+'.'+branch+'.')),
                 'OUTCOME_DISCONNECTED:'+name+':'+branch)
    optimizer.step(); inputs.fresh()
    validate(model,optimizer,step+1)
    return dict(step=step, before=before, after=fingerprint(model.state_dict()), loss=float(loss.detach()),
        labels=fingerprint(labels), logits=cpu(logits), gradients=fingerprint(gradients),
        optimizer=fingerprint(optimizer.state_dict()), negative_examples=int((labels==0).sum()),
        positive_examples=int((labels==1).sum()), negative_gradient_direction_checked=True)


def predict(model, inputs, arm):
    with torch.no_grad():
        return {split:cpu(model(inputs.batch(keys,arm,fitting=split=='train')))
                for split,keys in (('train',inputs.train),('query',inputs.query))}

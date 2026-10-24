"""Decision head restricted to packed native windows of selected source sentences."""
import numpy as np
import torch
from torch import nn
from model.scifact_readout import RelationReadout,CONFIG as BASE

CONFIG={**BASE,"arms":["all-sentences","selected","claim-only","evidence-only","shuffled"],
        "absence":"learned-null-not-a-reading","selector":"frozen-prior-greedy-source-indices"}
ARMS=[(s,a) for s in (25,26,27) for a in CONFIG["arms"] if a!="shuffled" or s==25]


def check(ok,code):
    if not ok:raise ValueError("MNLM_SELECTED_"+code)


def selections(ctx,truth,inspected):
    docs=np.array([r["document_index"] for r in ctx["rows"]]);all_=truth["valid"][docs].copy()
    out={"all":all_,"none":np.zeros_like(all_)}
    for seed in (25,26,27):
        chosen=np.zeros_like(all_)
        for i,a in enumerate(inspected["agreements"][f"{seed}/rationale"]):chosen[i,a["selected"]]=True
        check(not (chosen&~all_).any(),"SELECTION_RANGE");out[str(seed)]=chosen
    return out


def plans(ctx,truth,selection):
    """Only gather addresses/geometry; no labels or selector latent state."""
    docs=np.array([r["document_index"] for r in ctx["rows"]]);out={}
    for name,chosen in selection.items():
        check(chosen.dtype==bool and chosen.shape==(len(docs),42),"SELECTION_SHAPE")
        check(not (chosen&~truth["valid"][docs]).any(),"SELECTION_RANGE");lanes=[]
        for mapping,membership in zip(ctx["maps"],truth["members"]):
            allowed=np.zeros((len(docs),mapping["indices"].shape[1]),bool)
            for i,d in enumerate(docs):allowed[i]=membership[d,chosen[i]].any(0)
            width=max(1,int(allowed.sum(1).max()));indices=np.zeros((len(docs),width),np.int64)
            mask=np.zeros(indices.shape,bool);geometry=np.zeros((*indices.shape,3),np.float32)
            for i,d in enumerate(docs):
                cols=np.flatnonzero(allowed[i]);n=len(cols)
                check((mapping["levels"][d,cols]==1).all(),"PARENT_WINDOW")
                indices[i,:n]=mapping["indices"][d,cols];mask[i,:n]=True;geometry[i,:n]=mapping["geometry"][d,cols]
            lanes.append(dict(indices=indices,mask=mask,geometry=geometry))
        out[name]=lanes
    return out


def donors(rows):
    result=np.zeros(len(rows),np.int64)
    for split in ("train","calibration","evaluation"):
        groups={}
        for i,r in enumerate(rows):
            if r["partition"]==split:groups.setdefault(r["group"],[]).append(i)
        ordered=sorted(groups);check(len(ordered)>1,"DONOR_GROUPS")
        for j,g in enumerate(ordered):
            for i in groups[g]:result[i]=groups[ordered[(j+1)%len(ordered)]][0]
    return result


def arm_plan(seed,name):return "all" if name=="all-sentences" else "none" if name=="claim-only" else str(seed)


class SelectedStore:
    def __init__(self,ctx,gathers,device):
        self.native=[torch.from_numpy(v).to(device) for v in ctx["native"]]
        self.query=[torch.from_numpy(m["query"]).to(device) for m in ctx["maps"]]
        self.plans={n:[{k:torch.from_numpy(v).to(device) for k,v in lane.items()} for lane in ls] for n,ls in gathers.items()}
        self.device=device

    def batch(self,rows,plan,query_rows=None,evidence_rows=None):
        q=torch.as_tensor(rows if query_rows is None else query_rows,device=self.device,dtype=torch.long)
        e=torch.as_tensor(rows if evidence_rows is None else evidence_rows,device=self.device,dtype=torch.long);out=[]
        for x,query,m in zip(self.native,self.query,self.plans[plan]):
            mask=m["mask"][e];values=x[m["indices"][e]].masked_fill(~mask.unsqueeze(-1),0.)
            out.append(dict(query=x[query[q]],evidence=values,mask=mask,geometry=m["geometry"][e],levels=mask.long()))
        return out


class SelectedReadout(RelationReadout):
    def __init__(self,dimensions):
        super().__init__(dimensions);self.absence=nn.Parameter(torch.zeros(14,1,64))

    def forward(self,memories,scope="joint"):
        check(len(memories)==14 and scope in ("joint","evidence"),"INPUT")
        check(all(set(m)=={"query","evidence","mask","geometry","levels"} for m in memories),"INPUT_FIELDS")
        task=self.query.expand(len(memories[0]["query"]),1,-1);typed=[]
        for k,m in enumerate(memories):
            check(m["mask"].dtype==torch.bool and bool((m["levels"][m["mask"]]==1).all()),"NON_SENTENCE")
            check(bool((m["evidence"][~m["mask"]]==0).all()) and bool((m["geometry"][~m["mask"]]==0).all()),"PADDING")
            q=task if scope=="evidence" else self.projections[k](m["query"]).unsqueeze(1)
            native=self.projections[k](m["evidence"])+self.scale(m["levels"])+self.geometry(m["geometry"])
            # NULL only participates when no selected observation exists. It is not an admitted sense.
            absent=~m["mask"].any(1,keepdim=True);mask=torch.cat([m["mask"],absent],1)
            memory=torch.cat([native,self.absence[k:k+1].expand(len(q),1,-1)],1)
            read,_=self.attention[k](q,memory,memory,key_padding_mask=~mask,need_weights=False)
            typed.append(self.relations[k](q+read)+self.instrument[k])
        joined=torch.cat(typed,1);bound,_=self.binding(task,joined,joined,need_weights=False)
        return self.head((task+bound).squeeze(1))


@torch.inference_mode()
def evaluate(model,store,rows,plan,scope="joint",donor=None,mode=None):
    out=[]
    for start in range(0,len(rows),32):
        ids=np.arange(start,min(len(rows),start+32));q=donor[ids] if mode=="query" else None;e=donor[ids] if mode=="evidence" else None
        out.append(model(store.batch(ids,plan,q,e),scope).softmax(1).cpu().numpy())
    return np.vstack(out)


@torch.inference_mode()
def controls(model,store,ctx,plan):
    ids=np.arange(32);original=store.batch(ids,plan);before=model(original);rejected=False
    try:model([{**m,"gold":None} for m in original])
    except ValueError as e:rejected="INPUT_FIELDS" in str(e)
    check(rejected,"GOLD_ALLOWED");excluded_changed=selected_changed=0
    for kind in ("excluded","selected"):
        restore=[]
        try:
            for k,(x,m) in enumerate(zip(store.native,store.plans[plan])):
                used=set(store.query[k][ids].cpu().tolist())|set(m["indices"][ids][m["mask"][ids]].cpu().tolist())
                if kind=="excluded":
                    docids=[r["document_index"] for r in ctx["rows"][:32]];mp=ctx["maps"][k]
                    candidates=sorted(set(mp["indices"][docids][mp["mask"][docids]].tolist())-used)
                else:candidates=sorted(set(m["indices"][ids][m["mask"][ids]].cpu().tolist())-set(store.query[k][ids].cpu().tolist()))
                if not candidates:continue
                index=candidates[0];restore.append((x,index,x[index].clone()));x[index]=x[(index+1)%len(x)]
            packed=store.batch(ids,plan);changed=model(packed)
            if kind=="excluded":
                check(len(restore)>0 and all(torch.equal(m[f],n[f]) for m,n in zip(original,packed) for f in m),"EXCLUDED_INPUT_LEAK")
                check(torch.equal(before,changed),"EXCLUDED_OUTPUT_LEAK");excluded_changed=len(restore)
            else:
                check(len(restore)>0 and not torch.equal(before,changed),"IGNORED_SELECTED");selected_changed=len(restore)
        finally:
            for x,index,value in restore:x[index]=value
        check(torch.equal(model(store.batch(ids,plan)),before),"RESTORE")
    return dict(gold_rejected=rejected,excluded_native_mutations=excluded_changed,selected_native_mutations=selected_changed,
                excluded_packed_and_output_exact=True,selected_output_changed=True,restored_output_exact=True)

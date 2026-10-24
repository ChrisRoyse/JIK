"""Research relation head over separate native memories; no text or labels in forward."""
import hashlib
import numpy as np
import torch
from torch import nn

CONFIG=dict(width=64,heads=4,classes=3,batch=32,steps=200,learning_rate=.001,weight_decay=.01,
            seeds=[25,26,27],precision="fp32",confidence_thresholds=[.5,.75,.9,.95,.99])
ARMS={"joint-all":(14,"joint"),"joint-public":(6,"joint"),"claim-only":(14,"claim"),
      "evidence-only":(14,"evidence"),"sentences-only":(14,"sentences"),"permuted":(14,"joint")}


def check(ok,code):
    if not ok:raise ValueError("MNLM_RELATION_MODEL_"+code)


def array_sha(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(memoryview(x).cast('B'));return h.hexdigest()


class NativeStore:
    """One native array per instrument; view/source indices are gather addresses only."""
    def __init__(self,context,device):
        self.native=[torch.from_numpy(x).to(device) for x in context["native"]]
        self.maps=[{k:torch.from_numpy(v).to(device) for k,v in m.items()} for m in context["maps"]]
        self.documents=torch.tensor([r["document_index"] for r in context["rows"]],device=device,dtype=torch.long)
        self.device=device

    def batch(self,index):
        rows=torch.as_tensor(index,device=self.device,dtype=torch.long);docs=self.documents[rows];result=[]
        for x,m in zip(self.native,self.maps):
            result.append(dict(query=x[m["query"][rows]],evidence=x[m["indices"][docs]],mask=m["mask"][docs],
                               levels=m["levels"][docs],geometry=m["geometry"][docs]))
        return result


class RelationReadout(nn.Module):
    def __init__(self,dimensions):
        super().__init__();w=CONFIG["width"];h=CONFIG["heads"]
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(d,w),nn.LayerNorm(w)) for d in dimensions])
        self.attention=nn.ModuleList([nn.MultiheadAttention(w,h,batch_first=True,dropout=0.) for _ in dimensions])
        self.relations=nn.ModuleList([nn.Sequential(nn.Linear(w,w*2),nn.GELU(),nn.Linear(w*2,w),nn.LayerNorm(w)) for _ in dimensions])
        self.scale=nn.Embedding(4,w);self.geometry=nn.Linear(3,w,bias=False)
        self.instrument=nn.Parameter(torch.randn(len(dimensions),w)*.02);self.query=nn.Parameter(torch.randn(1,1,w)*.02)
        self.binding=nn.MultiheadAttention(w,h,batch_first=True,dropout=0.);self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,3))

    def forward(self,memories,active,scope):
        check(len(memories)==len(self.projections)==14 and active in (6,14) and scope in ("joint","claim","evidence","sentences"),"INPUT")
        check(all(set(m)=={"query","evidence","mask","levels","geometry"} for m in memories),"INPUT_FIELDS")
        batch=len(memories[0]["query"]);task=self.query.expand(batch,1,-1);typed=[]
        for k in range(active):
            m=memories[k]
            q=task if scope=="evidence" else self.projections[k](m["query"]).unsqueeze(1)
            if scope=="claim":hidden=q
            else:
                mask=m["mask"] if scope!="sentences" else m["mask"]&(m["levels"]==1)
                check(bool(mask.any(1).all()),"EMPTY_EVIDENCE")
                evidence=self.projections[k](m["evidence"])+self.scale(m["levels"])+self.geometry(m["geometry"])
                read,_=self.attention[k](q,evidence,evidence,key_padding_mask=~mask,need_weights=False)
                hidden=q+read
            typed.append(self.relations[k](hidden)+self.instrument[k])
        # Typed learned branches bind; native coordinates remain in their own slots.
        bound,_=self.binding(task,torch.cat(typed,dim=1),torch.cat(typed,dim=1),need_weights=False)
        return self.head((task+bound).squeeze(1))


def schedule(rows,seed):
    train=np.asarray([i for i,r in enumerate(rows) if r["partition"]=="train"],np.int64)
    rng=np.random.default_rng(seed)
    return np.asarray([rng.choice(train,CONFIG["batch"],replace=False) for _ in range(CONFIG["steps"])])


def metrics(prob,rows,labels):
    out={};p=np.asarray(prob,dtype=np.float64);y=np.asarray(labels)
    for split in ("train","calibration","evaluation"):
        ids=np.asarray([i for i,r in enumerate(rows) if r["partition"]==split]);part=p[ids];truth=y[ids];pred=part.argmax(1)
        confusion=np.bincount(truth*3+pred,minlength=9).reshape(3,3);tp=np.diag(confusion);den=confusion.sum(0)+confusion.sum(1)
        confidence=part.max(1);curves={}
        for threshold in CONFIG["confidence_thresholds"]:
            kept=confidence>=threshold;n=int(kept.sum());wrong=int(np.sum(kept&(pred!=truth)))
            curves[str(threshold)]=dict(accepted=n,wrong=wrong,coverage=n/len(ids),error_rate=wrong/n if n else None,
                                       production_assertions=0)
        out[split]=dict(rows=len(ids),components=len({rows[i]["group"] for i in ids}),confusion=confusion.tolist(),
                       accuracy=float((pred==truth).mean()),macro_f1=float(np.divide(2*tp,den,out=np.zeros(3,float),where=den>0).mean()),
                       nll_nats=float(-np.log(part[np.arange(len(ids)),truth]).mean()),
                       brier=float(((part-np.eye(3)[truth])**2).sum(1).mean()),confidence_diagnostics=curves)
    return out


@torch.inference_mode()
def probabilities(model,store,rows,active,scope):
    values=[]
    for start in range(0,len(rows),CONFIG["batch"]):
        values.append(model(store.batch(range(start,min(len(rows),start+CONFIG["batch"]))),active,scope).softmax(1).cpu().numpy())
    return np.vstack(values)

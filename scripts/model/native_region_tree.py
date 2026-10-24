"""Frozen prototype hierarchy over typed complete native windows; no vector pooling."""
from collections import deque
import json
from pathlib import Path

import numpy as np
from model.constellation import ordered_distance
from text.constellation_corpus import canonical,digest,file_sha,need
from text.constellation_input import stamp
from claims.process_conditions import write_committed

POLICY=dict(branching=4,depth=3,min_split_inputs=32,landmark_candidates=128,tile=512,
            distance='mean_of_per_instrument_ordered_cosine_edit',seeding='hash_order_then_farthest',
            arithmetic='float64_einsum_fixed_coordinate_reduction')


def normal(values):
    need(values.ndim==2 and len(values)>0 and values.dtype==np.float32 and np.isfinite(values).all(),
         'REGION_TREE_NATIVE_ARRAY')
    x=values.astype(np.float64);norm=np.sqrt(np.einsum('ij,ij->i',x,x))
    need(bool((norm>0).all()),'REGION_TREE_NATIVE_ZERO')
    return x/norm[:,None]


def distances(slots,prototypes,roster):
    """Only same-instrument coordinates interact; rows retain every ordered window."""
    need(tuple(slots)==tuple(roster) and prototypes and all(tuple(p)==tuple(roster) for p in prototypes),
         'REGION_TREE_ROSTER')
    count=len(slots[roster[0]]);need(count>0,'REGION_TREE_EMPTY')
    result={}
    for name in roster:
        rows=[normal(v) for v in slots[name]];refs=[normal(p[name]) for p in prototypes]
        need(len(rows)==count and len({x.shape[1] for x in rows+refs})==1,'REGION_TREE_DIMENSIONS')
        out=np.empty((count,len(refs)),dtype=np.float64)
        single=[i for i,x in enumerate(rows) if len(x)==1];one=[j for j,x in enumerate(refs) if len(x)==1]
        if single and one:
            q=np.stack([rows[i][0] for i in single]);r=np.stack([refs[j][0] for j in one])
            out[np.ix_(single,one)]=1-np.clip(np.einsum('id,jd->ij',q,r,optimize=False),-1,1)
        for i,row in enumerate(rows):
            for j,ref in enumerate(refs):
                if len(row)==len(ref)==1:continue
                cosine=np.clip(np.einsum('id,jd->ij',row,ref,optimize=False),-1,1)
                out[i,j]=ordered_distance(cosine)
        need(np.isfinite(out).all(),'REGION_TREE_NONFINITE');result[name]=out
    return result


def common(per_instrument,roster):
    # Decision evidence only. Native representations remain in separate arrays.
    return sum(per_instrument[n] for n in roster)/len(roster)


def split_landmarks(read,uids,roster,policy):
    pool=uids[:policy['landmark_candidates']];slots=read(pool.tolist())
    candidates=[{n:slots[n][i] for n in roster} for i in range(len(pool))]
    selected=[0];best=np.full(len(pool),np.inf)
    while len(selected)<policy['branching']:
        current=common(distances(slots,[candidates[selected[-1]]],roster),roster)[:,0]
        best=np.minimum(best,current);best[selected]=0
        # Byte-identical readings cannot acquire separate identities from a tie.
        for j,candidate in enumerate(candidates):
            if any(all(np.array_equal(candidate[n],candidates[i][n]) for n in roster) for i in selected):best[j]=0
        winner=int(best.argmax())
        if best[winner]<=0:break
        selected.append(winner)
    return pool[selected],[candidates[i] for i in selected]


def fit(read,train_uids,roster,policy=POLICY):
    need(train_uids.dtype==np.int64 and train_uids.ndim==1 and len(train_uids)>0
         and bool((train_uids[1:]>train_uids[:-1]).all()),'REGION_TREE_FIT_MEMBERS')
    need(policy==POLICY,'REGION_TREE_POLICY')
    nodes=[dict(parent=-1,depth=0,prototype=int(train_uids[0]),support=len(train_uids),children=[],
                members_sha256=digest(train_uids.tobytes()))]
    first=read([int(train_uids[0])]);prototypes={int(train_uids[0]):{n:first[n][0] for n in roster}}
    queue=deque([(0,train_uids)]);counts=dict(routed_fit_inputs=0,candidate_nodes=0,empty_landmarks=0)
    while queue:
        parent,members=queue.popleft();node=nodes[parent]
        if node['depth']==policy['depth'] or len(members)<policy['min_split_inputs']:continue
        uids,refs=split_landmarks(read,members,roster,policy);counts['candidate_nodes']+=1
        if len(refs)<2:node['terminal_reason']='native_readings_not_separated';continue
        labels=np.empty(len(members),dtype='<i2')
        for lo in range(0,len(members),policy['tile']):
            hi=min(lo+policy['tile'],len(members))
            labels[lo:hi]=common(distances(read(members[lo:hi].tolist()),refs,roster),roster).argmin(1)
        counts['routed_fit_inputs']+=len(members)
        retained=[j for j in range(len(refs)) if bool((labels==j).any())]
        counts['empty_landmarks']+=len(refs)-len(retained)
        if len(retained)<2:node['terminal_reason']='one_nonempty_child';continue
        for j in retained:
            subset=members[labels==j];child=len(nodes);uid=int(uids[j]);prototypes[uid]=refs[j]
            nodes.append(dict(parent=parent,depth=node['depth']+1,prototype=uid,support=len(subset),children=[],
                              members_sha256=digest(subset.tobytes())))
            node['children'].append(child);queue.append((child,subset))
    return nodes,prototypes,counts


def encode(read,uids,nodes,prototypes,roster,policy=POLICY):
    count=len(uids);need(count>0,'REGION_TREE_QUERY_EMPTY')
    codes=np.full((count,policy['depth']),-1,dtype='<i2')
    paths=np.full((count,policy['depth']+1),-1,dtype='<i4');paths[:,0]=0
    queue=deque([(0,np.arange(count,dtype='<i8'))])
    while queue:
        node_id,indices=queue.popleft();node=nodes[node_id];children=node['children']
        if not children:continue
        refs=[prototypes[nodes[c]['prototype']] for c in children];labels=np.empty(len(indices),dtype='<i2')
        for lo in range(0,len(indices),policy['tile']):
            hi=min(lo+policy['tile'],len(indices));positions=indices[lo:hi]
            labels[lo:hi]=common(distances(read(uids[positions].tolist()),refs,roster),roster).argmin(1)
        codes[indices,node['depth']]=labels;paths[indices,node['depth']+1]=np.asarray(children)[labels]
        for label,child in enumerate(children):
            selected=indices[labels==label]
            if len(selected):queue.append((child,selected))
    return codes,paths


def publish(root,nodes,prototypes,roster,binding,policy=POLICY):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    uids=np.array(sorted(prototypes),dtype='<i8');np.save(root/'prototype-uids.npy',uids,allow_pickle=False)
    for lane,name in enumerate(roster):
        rows=[prototypes[int(uid)][name] for uid in uids]
        offsets=np.r_[np.int64(0),np.cumsum([len(v) for v in rows],dtype='<i8')]
        np.savez(root/f'instrument-{lane}.npz',values=np.concatenate(rows),offsets=offsets)
    write_committed(root/'nodes.json',nodes)
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(root.iterdir()) if p.is_file()}
    result=dict(schema='native-prototype-region-tree-v1',complete=True,roster=list(roster),binding=binding,
                policy=policy,files=files,scientific_admission=False,learned_rq_autoencoder=False)
    pin=write_committed(root/'report.json',result);write_committed(root/'COMMITTED',dict(report_sha256=pin));return pin


class NativeRegionTree:
    def __init__(self,root,pin,binding):
        self.root=Path(root);self.pin=pin
        raw=(self.root/'report.json').read_bytes();need(digest(raw)==pin,'REGION_TREE_REPORT_PIN')
        need(json.loads((self.root/'COMMITTED').read_bytes())==dict(report_sha256=pin),'REGION_TREE_COMMIT')
        self.report=r=json.loads(raw)
        need(r['schema']=='native-prototype-region-tree-v1' and r['complete'] and r['binding']==binding
             and r['policy']==POLICY,'REGION_TREE_BINDING')
        self.roster=tuple(r['roster']);self.stamps={self.root/n:stamp(self.root/n) for n in ('report.json','COMMITTED')}
        expected={'prototype-uids.npy','nodes.json'}|{f'instrument-{i}.npz' for i in range(len(self.roster))}
        need(set(r['files'])==expected and len(set(self.roster))==len(self.roster)>0,'REGION_TREE_FILES')
        for name,ref in r['files'].items():
            p=self.root/name;need(p.stat().st_size==ref['bytes'] and file_sha(p)==ref['sha256'],'REGION_TREE_FILE_PIN')
            self.stamps[p]=stamp(p)
        uids=np.load(self.root/'prototype-uids.npy',allow_pickle=False)
        need(uids.dtype==np.int64 and uids.ndim==1 and len(uids)>0 and np.all(uids[1:]>uids[:-1]),'REGION_TREE_UIDS')
        self.prototypes={int(uid):{} for uid in uids}
        for lane,name in enumerate(self.roster):
            with np.load(self.root/f'instrument-{lane}.npz',allow_pickle=False) as saved:
                values,offsets=saved['values'],saved['offsets']
                need(values.dtype==np.float32 and values.ndim==2 and np.isfinite(values).all()
                     and offsets.dtype==np.int64 and offsets.shape==(len(uids)+1,) and offsets[0]==0
                     and offsets[-1]==len(values) and np.all(np.diff(offsets)>0),'REGION_TREE_ARRAYS')
                for j,uid in enumerate(uids):
                    value=values[offsets[j]:offsets[j+1]].copy();value.flags.writeable=False
                    self.prototypes[int(uid)][name]=value
        self.nodes=json.loads((self.root/'nodes.json').read_bytes());self.validate()
        self.nodes_seal=digest(canonical(self.nodes));self.native_seal=self.native_stamps()

    def native_stamps(self):
        return {(uid,name):(id(v),v.__array_interface__['data'][0],v.shape,v.strides,str(v.dtype),v.flags.writeable)
                for uid,slots in self.prototypes.items() for name,v in slots.items()}

    def validate(self):
        parents=[]
        for i,node in enumerate(self.nodes):
            need(node['prototype'] in self.prototypes and 0<=node['depth']<=POLICY['depth'] and node['support']>0,
                 'REGION_TREE_NODE')
            children=node['children'];need(len(children)<=POLICY['branching'] and len(set(children))==len(children),
                                          'REGION_TREE_CHILDREN')
            for child in children:
                need(i<child<len(self.nodes) and self.nodes[child]['parent']==i
                     and self.nodes[child]['depth']==node['depth']+1,'REGION_TREE_PARENT')
                parents.append(child)
            if children:need(sum(self.nodes[c]['support'] for c in children)==node['support'],'REGION_TREE_SUPPORT')
        need(self.nodes[0]['parent']==-1 and self.nodes[0]['depth']==0
             and sorted(parents)==list(range(1,len(self.nodes))),'REGION_TREE_COVERAGE')

    def fresh(self):
        try:unchanged=all(stamp(p)==s for p,s in self.stamps.items())
        except OSError:unchanged=False
        need(unchanged,'REGION_TREE_CHANGED')
        need(digest(canonical(self.nodes))==self.nodes_seal and self.native_stamps()==self.native_seal,
             'REGION_TREE_MEMORY_CHANGED')

    def encode(self,read,uids):
        self.fresh();result=encode(read,uids,self.nodes,self.prototypes,self.roster);self.fresh();return result

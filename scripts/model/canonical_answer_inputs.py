"""Complete task-native inputs; public requests and private supervision stay separate."""
import gzip
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest, describe
from text.capability_answer_source import stamp
from text.constellation_corpus import canonical, digest, file_sha, need


def report(root, pin):
    path=Path(root)/'report.json'
    need(path.is_file() and file_sha(path)==pin, 'TASK_NATIVE_REPORT_PIN')
    return json.loads(path.read_bytes())


def lane_arrays(root, row, counts, dimension):
    need(set(row['files'])=={'reference.npz','traces.jsonl.gz'}, 'TASK_NATIVE_FILES')
    for name,ref in row['files'].items():
        path=root/name
        need(path.stat().st_size==ref['bytes'] and file_sha(path)==ref['sha256'], 'TASK_NATIVE_BYTES')
    with np.load(root/'reference.npz',allow_pickle=False) as archive:
        need(set(archive.files)=={'vectors','bounds'}, 'TASK_NATIVE_ARRAY_KEYS')
        values,bounds=archive['vectors'],archive['bounds']
    expected=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    need(values.dtype==np.float32 and values.shape==(int(expected[-1]),dimension)
         and bounds.dtype==np.int64 and np.array_equal(bounds,expected)
         and np.isfinite(values).all(), 'TASK_NATIVE_ARRAY_LAYOUT')
    with gzip.open(root/'traces.jsonl.gz','rt',encoding='utf-8') as stream:
        traces=[json.loads(line) for line in stream]
    need(len(traces)==len(counts) and [len(t['windows']) for t in traces]==counts,
         'TASK_NATIVE_TRACE_COVERAGE')
    return values,bounds,traces


class CanonicalAnswerInputs:
    """One authenticated resident store, no lens loads and no legacy-runtime reuse."""
    def __init__(self,schedule,source,source_pin,cold,cold_pin):
        self.schedule=schedule; self.plan=schedule.plan
        self.source,self.cold=Path(source),Path(cold)
        first=report(source,source_pin); second=report(cold,cold_pin); p=self.plan.report
        need(self.source.resolve()!=self.cold.resolve(), 'TASK_NATIVE_COLD_ALIAS')
        for r in (first,second):
            need(r['schema']=='answer-canonical-native-v1' and r['complete'] and not r['errors']
                 and r['plan']==dict(path=str(self.plan.root),pin=self.plan.pin)
                 and r['roster']==p['roster'] and r['inputs']==p['inputs']
                 and set(r['instruments'])==set(p['roster'])
                 and not r['scientific_admission'] and not r['training_ready'], 'TASK_NATIVE_SCOPE')
        need(first['cold_reference'] is None and not first['cold_all_inputs_exact']
             and second['cold_reference']==dict(path=str(self.source),pin=source_pin)
             and second['cold_all_inputs_exact'] and second['code']==first['code']==p['code'],
             'TASK_NATIVE_COLD_BINDING')
        self.roster=tuple(p['roster']);self.dimensions={s['name']:s['dims'] for s in p['specs']}
        self.index=MappingProxyType({r['id']:i for i,r in enumerate(self.plan.records)})
        # Allocate only after admitting the peak of one forward/cold lane pair,
        # all resident vectors and expanded token traces. This is physical memory.
        native_bytes=sum(self.plan.tokens[n]['total_windows']*self.dimensions[n]*4 for n in self.roster)
        token_count=sum(self.plan.tokens[n]['total_tokens'] for n in self.roster)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                           if x.startswith('MemAvailable:')))*1024
        projected=3*native_bytes+128*token_count+64*p['input_bytes']
        need(projected<=available//4, 'TASK_NATIVE_RESIDENT_MEMORY_REQUIRES_TILING')
        self.arrays={};self.offsets={};self.seals={};self.stamps={};self.lens_ids={}
        for root in (self.source,self.cold):self.stamps[str(root/'report.json')]=stamp(root/'report.json')
        for name in self.roster:
            rows=[]
            for root,r in ((self.source,first),(self.cold,second)):
                lane=report(root/name,r['instruments'][name]['pin']);rows.append(lane)
                need(lane['name']==name and lane['lens_id']==r['instruments'][name]['lens_id']
                     ==digest(canonical(p['contracts'][name])) and lane['contract']==p['contracts'][name]
                     and lane['inputs']==p['inputs'] and lane['all_windows_retained']
                     and lane['frozen_unchanged'] and not lane['scientific_admission']
                     and lane['windows']==self.plan.tokens[name]['total_windows'], 'TASK_NATIVE_INSTRUMENT')
                for f in ('report.json',*lane['files']):self.stamps[str(root/name/f)]=stamp(root/name/f)
            need(rows[0]['cohort']==32 and rows[1]['cohort']==8
                 and rows[1]['measured_inputs']==p['inputs'] and not rows[1]['reused_inputs']
                 and rows[1]['controls']['all_cold_inputs_exact'], 'TASK_NATIVE_COLD_COHORT')
            a,b,t=lane_arrays(self.source/name,rows[0],self.plan.tokens[name]['windows'],self.dimensions[name])
            x,y,u=lane_arrays(self.cold/name,rows[1],self.plan.tokens[name]['windows'],self.dimensions[name])
            need(np.array_equal(a,x) and np.array_equal(b,y) and t==u, 'TASK_NATIVE_COLD_DIFFERENCE')
            del x,y,t,u
            a.flags.writeable=False;b.flags.writeable=False
            self.arrays[name],self.offsets[name]=a,b
            self.seals[name]=tuple(digest(a[b[i]:b[i+1]].tobytes()) for i in range(p['inputs']))
            self.lens_ids[name]=rows[0]['lens_id']
        self.binding=dict(plan=self.plan.pin,source=source_pin,cold=cold_pin,lens_ids=self.lens_ids)
        self.pin=digest(canonical(self.binding));self.native_bytes=native_bytes
        self.stats=dict(constructed_stores=1,lens_loads=0,frame_reads=0,window_reads=0,
                        resident_bytes=native_bytes,projected_peak_bytes=projected)
        self.fresh()

    def fresh(self):
        need(digest(canonical(self.binding))==self.pin, 'TASK_NATIVE_BINDING_CHANGED')
        need(all(stamp(p)==s for p,s in self.stamps.items()), 'TASK_NATIVE_PUBLICATION_CHANGED')

    def frame(self,identity):
        self.fresh();need(identity in self.index, 'TASK_NATIVE_INPUT_MISSING');i=self.index[identity];out={}
        for name in self.roster:
            a,b=self.arrays[name],self.offsets[name];value=a[b[i]:b[i+1]]
            need(value.shape==(self.plan.tokens[name]['windows'][i],self.dimensions[name])
                 and digest(value.tobytes())==self.seals[name][i], 'TASK_NATIVE_CACHE_CHANGED')
            out[name]=torch.from_numpy(value.copy());self.stats['window_reads']+=len(value)
        self.stats['frame_reads']+=1;return out

    def prompt(self,key):
        public=self.schedule.request(key);request=NativeRequest(describe(self.schedule,key))
        context=tuple(self.frame(f['input']) for f in public['fields'])
        need(request.observed==len(context), 'TASK_NATIVE_OBSERVED_COUNT')
        return dict(context=context,request=request)

    def item(self,key,labels,*,fitting):
        # Split authorization precedes any private native reads or cache access.
        private=self.schedule.supervision(key,fitting=fitting)
        need(labels.store is self,'TASK_NATIVE_WRONG_LABEL_STORE');public=self.prompt(key)
        row=self.schedule.rows[key];ids=private['targets'];answers=tuple(self.frame(i) for i in ids)
        codes,counts=labels.get(ids)
        value=dict(document=key,side=row['split'],context=public['context'],answers=answers,codes=codes,counts=counts,
            provenance=dict(store=self.pin,labels=labels.pin,request=public['request'].pin,
                observed=len(public['context']),component=row['group'],family=row['family'],
                operation=row['request']['operation'],position_kind=row['request']['position_kind'],
                targets=list(ids),end=private['end'],scope='complete_recorded_task_answer'))
        value['native_pin']=fingerprint((value['context'],answers));value['label_pin']=fingerprint((codes,counts))
        value['binding']=fingerprint((key,value['side'],value['provenance']))
        need(len(private['decisions'])==len(answers)+1, 'TASK_NATIVE_ANSWER_END_COUNT')
        return value,public['request']


class AnswerLabelCache:
    """Compute each immutable code once; no uint8 count storage in the consumer."""
    def __init__(self,store,labels):
        self.store=store;self.labels=labels;self.ids=tuple(r['id'] for r in store.plan.records)
        self.index=MappingProxyType({key:i for i,key in enumerate(self.ids)});codes=[];counts=[]
        for lo in range(0,len(self.ids),128):
            frames=[store.frame(i) for i in self.ids[lo:lo+128]]
            c,n=labels.encode(frames,canonical=True,reference=True)
            codes.append(torch.from_numpy(c.astype(np.int64)));counts.append(torch.from_numpy(n.astype(np.int64)))
        self.codes=torch.cat(codes);self.counts=torch.cat(counts);self.seal=fingerprint((self.codes,self.counts))
        self.versions=(self.codes._version,self.counts._version)
        self.binding=dict(store=store.pin,codebook=labels.state,runtime=labels.runtime_state,
                          runtime_policy='canonical-singleton-fp64-with-independent-numpy-argmin')
        self.binding_seal=digest(canonical(self.binding))
        self.row_seals=tuple(fingerprint((self.codes[i],self.counts[i])) for i in range(len(self.ids)))
        self.pin=fingerprint((self.binding,self.ids,self.codes,self.counts))

    def fresh(self):
        self.store.fresh();self.labels.fresh()
        need(digest(canonical(self.binding))==self.binding_seal,'TASK_NATIVE_LABEL_BINDING_CHANGED')
        need((self.codes._version,self.counts._version)==self.versions, 'TASK_NATIVE_LABEL_CACHE_CHANGED')

    def get(self,identities):
        self.fresh();need(identities and all(i in self.index for i in identities), 'TASK_NATIVE_LABEL_INPUT')
        indices=[self.index[i] for i in identities]
        need(all(fingerprint((self.codes[i],self.counts[i]))==self.row_seals[i] for i in indices),
             'TASK_NATIVE_LABEL_BYTES_CHANGED')
        return self.codes[indices].clone(),{n:self.counts[indices,j].clone() for j,n in enumerate(self.store.roster)}

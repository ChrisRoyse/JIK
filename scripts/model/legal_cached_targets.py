"""Consume an exhaustive frozen-code cache by source address, with TRAIN-only fit joins."""
import json
from pathlib import Path

import numpy as np
import torch

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp


def pinned(path,pin):
    need(path.is_file() and file_sha(path)==pin,'CACHED_TARGET_REPORT_PIN')
    return json.loads(path.read_bytes())


class LegalCachedTargets:
    def __init__(self, root, pin, cold, cold_pin, data, expected_binding):
        self.root,self.cold,self.data=Path(root),Path(cold),data
        self.report=pinned(self.root/'report.json',pin);proof=pinned(self.cold/'report.json',cold_pin)
        r=self.report;b=r['binding']
        need(r['complete'] and r['mode']=='publish' and r['scope']=='full'
             and b==expected_binding and b['source']==data.binding
             and r['rows']==b['population']==data.native.config['inputs'],'CACHED_TARGET_AUTHORITY')
        need(proof['complete'] and proof['mode']=='cold' and proof['scope']=='full'
             and proof['previous_pin']==pin and proof['binding']==b
             and proof['result']['rows']==r['rows'] and proof['result']['all_rows_exact']
             and proof['result']['all_exports_exact']
             and proof['result']['all_rows_canonical_singleton_exact']
             and proof['result']['baseline_rows']==data.report['support']['train']['successors'],
             'CACHED_TARGET_COMPLETE_COLD')
        for directory,report,authority in [(self.root,r,pin),(self.cold,proof,cold_pin)]:
            need((directory/'COMMITTED').is_file() and (directory/'COMMITTED').read_text().strip()==authority,
                 'CACHED_TARGET_COMMIT')
            for name,digest in report['files'].items():
                path=directory/name
                need(path.is_file() and file_sha(path)==digest,'CACHED_TARGET_PHYSICAL_FILE')
        self.binding=dict(publication=pin,cold=cold_pin,source=data.binding,target=b['target_checkpoint'],
                          code=b['code'],roster=b['roster'],sizes=b['sizes'])
        self.roster=tuple(b['roster']);self.sizes=tuple(b['sizes']);self.rows=r['rows']
        self.paths=[self.root/'export'/(name+'.npy') for name in ['codes','counts']]
        self.maps=[np.load(path,mmap_mode='r',allow_pickle=False) for path in self.paths]
        need(all(a.dtype==np.uint8 and a.shape==(self.rows,width)
                 for a,width in zip(self.maps,[len(self.sizes),len(self.roster)])), 'CACHED_TARGET_MMAP_LAYOUT')
        self.stamps={path:stamp(path) for path in self.paths+
                     [self.root/'report.json',self.root/'COMMITTED',self.cold/'report.json',self.cold/'COMMITTED']}
        self.baselines=pinned(self.root/'baselines.json',r['files']['baselines.json'])
        need(self.baselines['rows']==proof['result']['baseline_rows'] and not self.baselines['fit_uses_held_out'],
             'CACHED_TARGET_BASELINE_BINDING')
        self.stamps[self.root/'baselines.json']=stamp(self.root/'baselines.json')
        self.calls=self.returned_rows=0
        self.fresh()

    def fresh(self):
        self.data.fresh()
        try:unchanged=all(stamp(path)==value for path,value in self.stamps.items())
        except OSError:unchanged=False
        need(unchanged,'CACHED_TARGET_CONSUMED_BYTES_CHANGED')

    def take(self,uids):
        need(isinstance(uids,(list,tuple)) and 0<len(uids)<=128
             and all(type(i) is int and 0<=i<self.rows for i in uids),'CACHED_TARGET_UID_SCOPE')
        self.fresh()
        indices=np.asarray(uids,dtype=np.int64)
        codes=np.array(self.maps[0][indices],dtype=np.int64,copy=True)
        counts=np.array(self.maps[1][indices],dtype=np.int64,copy=True)
        need(all(bool((codes[:,i]<size).all()) for i,size in enumerate(self.sizes))
             and bool((counts>0).all()),'CACHED_TARGET_LABEL_RANGE')
        self.fresh();self.calls+=1;self.returned_rows+=len(uids)
        return torch.from_numpy(codes),{name:torch.from_numpy(counts[:,i].copy()) for i,name in enumerate(self.roster)}

    def document(self,document,*,side,fit=False,expected_bounds=None):
        need(type(document) is int and 0<=document<len(self.data.documents),'CACHED_TARGET_DOCUMENT')
        start,stop=map(int,self.data.documents[document])
        need(expected_bounds is None or tuple(expected_bounds)==(start,stop),'CACHED_TARGET_DOCUMENT_BOUNDARY')
        indices=self.data.indices(list(range(start,stop)),side,fit=fit)
        self.data.validate_next(indices[:-1],indices[1:])
        uids=self.data.mapping[indices[1:]].tolist();codes=[];counts={n:[] for n in self.roster}
        for lo in range(0,len(uids),128):
            a,b=self.take(uids[lo:lo+128]);codes.append(a)
            for n in self.roster:counts[n].append(b[n])
        need(codes,'CACHED_TARGET_NO_SUCCESSOR')
        return dict(document=document,side=side,start=start,stop=stop,target_uids=uids,
                    codes=torch.cat(codes),counts={n:torch.cat(v) for n,v in counts.items()},
                    source=self.data.binding,cache=self.binding)

    def close(self):
        try:
            self.fresh()
            for path in self.paths:
                need(file_sha(path)==self.report['files'][str(path.relative_to(self.root))],
                     'CACHED_TARGET_FINAL_PHYSICAL_PIN')
        finally:
            for a in self.maps:a._mmap.close()

    def training_authority(self):
        raise ValueError('MNLM_CACHED_TARGET_EXPERIMENT_AUTHORITY_REQUIRED')

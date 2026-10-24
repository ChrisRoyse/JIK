"""Indexed explicit continuation baseline; no semantic or training authority."""
import json
from pathlib import Path

import numpy as np

from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_continuations import SCHEMA,SIDES,PAIR_DTYPE


class Continuations:
    def __init__(self,root,pin):
        self.root=Path(root);self.pin=pin
        need(file_sha(self.root/'report.json')==pin and json.loads((self.root/'COMMITTED').read_bytes())==
             dict(report_sha256=pin),'CONTINUATION_COMMIT')
        self.report=r=json.loads((self.root/'report.json').read_bytes())
        need(r['schema']==SCHEMA and r['complete'] and r['baseline_fit_side']=='train_only','CONTINUATION_SCOPE')
        names={'presence.npy','train-modal.npy'}|{side+'-pairs.npy' for side in SIDES}
        need(set(r['files'])==names,'CONTINUATION_FILES')
        self.stamps={self.root/n:stamp(self.root/n) for n in ('report.json','COMMITTED')}
        for name,ref in r['files'].items():
            path=self.root/name;need(path.stat().st_size==ref['bytes'] and file_sha(path)==ref['sha256'],'CONTINUATION_FILE_PIN')
            self.stamps[path]=stamp(path)
        self.presence=np.load(self.root/'presence.npy',mmap_mode='r',allow_pickle=False)
        self.modal=np.load(self.root/'train-modal.npy',mmap_mode='r',allow_pickle=False)
        self.u=r['counts']['unique_inputs']
        need(self.presence.shape==self.modal.shape==(self.u,) and self.presence.dtype==np.dtype('u1')
             and self.modal.dtype==np.dtype('<i8') and np.all((self.presence>0)&(self.presence<8))
             and np.all((self.modal>=-1)&(self.modal<self.u)),'CONTINUATION_ARRAY_SCOPE')
        self.pairs={side:np.load(self.root/(side+'-pairs.npy'),mmap_mode='r',allow_pickle=False) for side in SIDES}
        for side,value in self.pairs.items():
            need(value.ndim==1 and value.dtype==PAIR_DTYPE and np.all(value['count']>0)
                 and np.all(value['key'][1:]>value['key'][:-1])
                 and int(value['count'].sum(dtype='<u8'))==r['metrics'][side]['occurrences'],'CONTINUATION_PAIR_SCOPE')

    def fresh(self):
        need(all(stamp(p)==s for p,s in self.stamps.items()),'CONTINUATION_CHANGED')

    def lookup(self,uid):
        need(type(uid) is int and 0<=uid<self.u,'CONTINUATION_UID');self.fresh()
        target=int(self.modal[uid]);return None if target<0 else target

    def successors(self,uid,side):
        need(type(uid) is int and 0<=uid<self.u and side in SIDES,'CONTINUATION_QUERY_SCOPE');self.fresh()
        values=self.pairs[side];a,b=np.searchsorted(values['key'],np.asarray([uid<<32,(uid+1)<<32],dtype='<u8'))
        return values[int(a):int(b)]

    def training_authority(self):
        raise ValueError('MNLM_CONTINUATION_TRAINING_AUTHORITY_ABSENT')

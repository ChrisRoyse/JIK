"""Load the bounded identity development population through its real native slots."""
import json

import numpy as np
import torch

from audit.loop_population_readback import readback, check, sha
from audit.window_readback import readback as read_windows
from model.contracts import Instrument
from model.native import NativeInput
from senses.contract import canonical, digest

LANES=('all-mpnet-base-v2','codebert-mean','e5-base-v2','finbert-mean','pubmedbert-embeddings','specter2-base')


class LoopInputs:
    def __init__(self, population, native, source, *, audit_windows=True):
        self.population, self.native = population, native
        self.source_readback=readback(population,source)
        self.window_readback=read_windows(native,population) if audit_windows else None
        rep=json.loads((native/'report.json').read_bytes())
        check(rep['complete'] and not rep['errors'] and rep['calibration_report_sha256']==sha(population/'report.json')
              and tuple(sorted(rep['lenses']))==LANES, 'NATIVE_JOIN')
        self.refs=list(map(json.loads,(population/'references.jsonl').read_bytes().splitlines()))
        self.lookup={(r['record'],r['field']):r['unit'] for r in self.refs}
        self.specs=[];self.arrays={}
        for name in LANES:
            root=native/name;lane=json.loads((root/'report.json').read_bytes());contract=lane['contract']
            check(sha(root/'report.json')==rep['lenses'][name]['report_sha256'], 'LANE_PIN')
            check(sha(root/'slots.npz')==lane['files']['slots.npz']['sha256'], 'SLOT_PIN')
            with np.load(root/'slots.npz',allow_pickle=False) as z:
                x,ptr=z['vectors'].copy(),z['offsets'].copy()
            definition=dict(schema='loop-instrument-v1',name=name,channel='language',layout='multi',dims=x.shape[1],
                            frozen=True,weights_sha256=contract['weights_sha256'],training_lineage={'status':'upstream_not_independently_attested'},
                            measurement_contract=contract,lens_id=lane['lens_id'])
            self.specs.append(Instrument(name,digest(definition),'language','multi',x.shape[1],canonical(definition).decode()).validate())
            self.arrays[name]=(x,ptr)
        self.specs=tuple(self.specs)
        self.standard_units=[r['unit'] for r in self.refs if r['role']=='standard']
        check(len(self.standard_units)==10, 'STANDARDS')
        self.source_id=digest(dict(population=sha(population/'report.json'),native=sha(native/'report.json')))

    def batch(self, units, config, device):
        check(bool(units) and len(set(units))==len(units) and all(type(u) is int and 0<=u<len(self.refs) for u in units), 'UNITS')
        result={}
        for spec in self.specs:
            x,ptr=self.arrays[spec.name];width=max(int(ptr[u+1]-ptr[u]) for u in units)
            check(width<=config.max_native_tokens and len(units)*width*spec.dims<=config.max_native_elements, 'NATIVE_BUDGET')
            values=np.zeros((len(units),width,spec.dims),np.float32);padding=np.ones((len(units),width),bool)
            for j,u in enumerate(units):
                row=x[ptr[u]:ptr[u+1]];values[j,:len(row)]=row;padding[j,:len(row)]=False
            ids=tuple(str(u)+':'+self.refs[u]['text_sha256'] for u in units)
            result[spec.name]=NativeInput(spec.identity,ids,torch.from_numpy(values).to(device),
                    torch.from_numpy(padding).to(device),torch.zeros(len(units),dtype=torch.bool,device=device))
            result[spec.name].check(spec,config)
        return result

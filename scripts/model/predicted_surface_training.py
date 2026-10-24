"""Typed predicted-conditioning adaptation with source-bound continuation.

Targets are attested surface bytes. Predicted slots are frozen inputs, not new
observations or admitted meanings. Missing predictions never become true slots.
"""
from dataclasses import dataclass, replace
from copy import deepcopy
import hashlib

import torch

from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import native_versions
from model.native_population_training import fingerprint
from model import surface_inversion_training as inversion
from text.constellation_corpus import canonical, digest, need


@dataclass(frozen=True)
class ConditioningPair:
    observed: object
    predicted: object
    receipt: dict
    seal: str
    versions: tuple

    def fresh(self):
        o,p=self.observed,self.predicted
        need((o.occurrence,o.uid,o.side,o.raw)==(p.occurrence,p.uid,p.side,p.raw),
             'PREDICTED_SURFACE_TARGET_CHANGED')
        need(digest(canonical(dict(receipt=self.receipt,occurrence=o.occurrence,uid=o.uid,side=o.side,
                                   raw=hashlib.sha256(o.raw).hexdigest())))==self.seal,
             'PREDICTED_SURFACE_BINDING_CHANGED')
        need((native_versions(o.native),native_versions(p.native))==self.versions,
             'PREDICTED_SURFACE_NATIVE_CHANGED')


def paired(observed,event,specs,config,*,position,proposal,trace_pin,region_pin,region_sizes,
           source_sha,document,source_index,fitting=False):
    need(not fitting or observed.side=='train','PREDICTED_SURFACE_NONTRAIN')
    need(isinstance(observed.raw,bytes) and observed.raw
         and hashlib.sha256(observed.raw).hexdigest()==source_sha,'PREDICTED_SURFACE_SOURCE_BYTES')
    need(type(position) is int and position>=0 and type(source_index) is int and source_index>=0
         and type(document) is int and document>=0,'PREDICTED_SURFACE_SOURCE_POSITION')
    need(isinstance(event,dict) and event.get('proposal')==proposal,'PREDICTED_SURFACE_PROPOSAL')
    atom=predicted_atom(event,specs,config,region_pin=region_pin,region_sizes=region_sizes,
                        trace_pin=trace_pin,position=position)
    for spec in specs:observed.native[spec.name].check(spec,config)
    result=replace(observed,native=atom.slots)
    receipt=dict(document=document,source_index=source_index,position=position,proposal=proposal,
        trace_pin=trace_pin,region_pin=region_pin,regions=atom.regions,source_sha=source_sha,
        observed=fingerprint({n:x.values for n,x in observed.native.items()}),
        predicted=fingerprint({n:x.values for n,x in result.native.items()}),
        predicted_counts={n:len(x.values[0]) for n,x in result.native.items()},
        scientific_admission=False)
    seal=digest(canonical(dict(receipt=receipt,occurrence=observed.occurrence,uid=observed.uid,
                               side=observed.side,raw=source_sha)))
    pair=ConditioningPair(observed,result,receipt,seal,(native_versions(observed.native),native_versions(result.native)))
    pair.fresh();return pair


def views(pairs,*,predicted):
    need(type(predicted) is bool and pairs,'PREDICTED_SURFACE_MIXTURE')
    need(len({(p.observed.occurrence,p.receipt['position']) for p in pairs})==len(pairs),
         'PREDICTED_SURFACE_DUPLICATE_TARGET')
    output=[]
    for pair in pairs:
        pair.fresh();need(pair.observed.side=='train','PREDICTED_SURFACE_NONTRAIN')
        # Identical count, byte weighting, window scheduling and optimizer steps.
        output.extend([pair.observed,pair.predicted if predicted else pair.observed])
    return output


def update(model,optimizer,pairs,step,*,predicted):
    examples=views(pairs,predicted=predicted)
    row,profile=inversion.update(model,optimizer,examples,step)
    for pair in pairs:pair.fresh()
    row['conditioning']=dict(rule='half_true_half_predicted' if predicted else 'two_true_views',
                              pairs=[p.seal for p in pairs],unique_targets=len(pairs))
    return row,profile


def continuation(config,specs,checkpoint,device,*,step):
    need(set(checkpoint)>= {'after','optimizer'},'PREDICTED_SURFACE_INITIAL_STATE')
    need(type(step) is int and step>=0,'PREDICTED_SURFACE_INITIAL_STEP')
    model,optimizer=inversion.create(config,specs,checkpoint['after'],device)
    # CPU load_state_dict can retain the supplied moment tensors by reference.
    optimizer.load_state_dict(deepcopy(checkpoint['optimizer']))
    inversion.check_state(model,optimizer,step)
    need(fingerprint(model.state_dict())==fingerprint(checkpoint['after'])
         and fingerprint(optimizer.state_dict())==fingerprint(checkpoint['optimizer']),
         'PREDICTED_SURFACE_CONTINUATION')
    return model,optimizer

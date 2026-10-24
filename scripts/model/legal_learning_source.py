"""All legal occurrences as ordered streams and typed, target-withheld supervision.

Addresses, split labels and provenance are joins/audit fields, never model features.
Source succession is recorded text order; it is not an empirical-world truth oracle.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from text.constellation_corpus import need
from text.constellation_input import stamp
from text.legal_continuations import SIDES,bind_edges,split_input
from text.legal_native_batch import NativeBatchSession


@dataclass(frozen=True)
class SourceTile:
    document: int
    side: str
    start: int
    stop: int
    document_start: int
    document_stop: int

    @property
    def earlier_prefix(self):return (self.document_start,self.start)

    @property
    def requires_carried_state(self):return self.start!=self.document_start


@dataclass(frozen=True)
class TypedSupervision:
    inputs: object
    targets: object
    input_occurrences: tuple
    target_occurrences: tuple
    context_scope: str
    withheld: str | None = None
    control: str | None = None


class LegalLearningSource:
    def __init__(self,source,split,split_pin,cold,cold_pin,bank,bank_pin):
        self.source=source;self.split=Path(split);self.split_pin=split_pin
        self.report,self.sides,self.split_stamps=split_input(self.split,split_pin,Path(cold),cold_pin,source)
        self.native=NativeBatchSession(bank,bank_pin,source)
        self.mapping=source.arrays['mapping'];self.documents=source.arrays['documents']
        self.components=np.load(self.split/'component.npy',mmap_mode='r',allow_pickle=False)
        self.doc_sides=np.load(self.split/'document-side.npy',mmap_mode='r',allow_pickle=False)
        need(self.components.shape==self.doc_sides.shape==(len(self.documents),),'LEGAL_LEARNING_DOCUMENTS')
        bind_edges(self.documents,source.arrays['successors'],self.sides)
        self.roster=tuple(self.native.config['roster'])
        self.binding=dict(source=source.pin,split=split_pin,cold_split=cold_pin,bank=bank_pin,
                          exposure=self.report['exposure'],source_claims_are_truth=False,
                          scientific_admission=False,implicit_training_authority=False)

    def fresh(self):
        self.native.fresh()
        try:unchanged=all(stamp(p)==s for p,s in self.split_stamps.items())
        except OSError:unchanged=False
        need(unchanged,'LEGAL_LEARNING_SPLIT_CHANGED')

    def indices(self,occurrences,side,*,fit=False):
        need(type(fit) is bool and side in SIDES,'LEGAL_LEARNING_SIDE')
        need(not fit or side=='train','LEGAL_LEARNING_FIT_REQUIRES_TRAIN')
        need(isinstance(occurrences,(list,tuple)) and len(occurrences)>0
             and all(type(i) is int and 0<=i<len(self.mapping) for i in occurrences),'LEGAL_LEARNING_OCCURRENCES')
        self.fresh();indices=np.asarray(occurrences,dtype='<i8')
        need(bool((self.sides[indices]==SIDES.index(side)).all()),'LEGAL_LEARNING_CROSS_SPLIT')
        return indices

    def tiles(self,side,*,atoms_per_tile):
        """No repeated prefixes: carry prior state across contiguous tiles of a document."""
        need(side in SIDES and type(atoms_per_tile) is int and atoms_per_tile>0,'LEGAL_LEARNING_TILE_POLICY')
        self.fresh()
        for document,(lo,hi) in enumerate(self.documents):
            if self.doc_sides[document]!=SIDES.index(side):continue
            lo,hi=int(lo),int(hi)
            for start in range(lo,hi,atoms_per_tile):
                yield SourceTile(document,side,start,min(start+atoms_per_tile,hi),lo,hi)
        self.fresh()

    def validate_next(self,left,right):
        need(left.shape==right.shape and bool((right==left+1).all()) and bool((right<len(self.mapping)).all()),
             'LEGAL_LEARNING_NOT_IMMEDIATE_SUCCESSOR')
        document=np.searchsorted(self.documents[:,1],left,side='right')
        need(bool((right<self.documents[document,1]).all()),'LEGAL_LEARNING_CROSS_DOCUMENT_TARGET')
        need(bool((self.sides[left]==self.sides[right]).all()),'LEGAL_LEARNING_CROSS_SPLIT_TARGET')

    def validate_donors(self,occurrences,donors):
        need(occurrences.shape==donors.shape,'LEGAL_LEARNING_DONOR_SHAPE')
        need(bool((self.mapping[occurrences]!=self.mapping[donors]).all()),'LEGAL_LEARNING_DONOR_IDENTICAL_INPUT')
        a=np.searchsorted(self.documents[:,1],occurrences,side='right')
        b=np.searchsorted(self.documents[:,1],donors,side='right')
        need(bool((self.components[a]!=self.components[b]).all()),'LEGAL_LEARNING_DONOR_SAME_COMPONENT')

    def assert_withheld(self,inputs,targets,name):
        need(name in self.roster and set(inputs)==set(self.roster)-{name} and set(targets)=={name},
             'LEGAL_LEARNING_TARGET_VISIBLE')

    def _gather(self,indices,names,remaining):
        need(type(remaining) is int and remaining>0,'LEGAL_LEARNING_MEMORY_TILE')
        uids=self.mapping[indices].tolist();slots={}
        for name in names:
            batch=self.native.read_many(name,uids,memory_bytes=remaining)
            remaining-=batch.values.nbytes+batch.offsets.nbytes+sum(len(t) for t in batch.traces)
            need(remaining>=0,'LEGAL_LEARNING_MEMORY_TILE')
            slots[name]=batch
        return MappingProxyType(slots),remaining

    def observations(self,occurrences,*,side,fit=False,memory_bytes=64<<20):
        indices=self.indices(occurrences,side,fit=fit)
        result,_=self._gather(indices,self.roster,memory_bytes);self.fresh();return result

    def next_atoms(self,occurrences,*,side,fit=False,memory_bytes=64<<20):
        """One current atom per row. This method never claims a full earlier history."""
        left=self.indices(occurrences,side,fit=fit);right=left+1;self.validate_next(left,right)
        inputs,remaining=self._gather(left,self.roster,memory_bytes)
        targets,_=self._gather(right,self.roster,remaining);self.fresh()
        return TypedSupervision(inputs,targets,tuple(left.tolist()),tuple(right.tolist()),'single_source_atom')

    def reciprocal(self,occurrences,name,*,side,fit=False,donors=None,memory_bytes=64<<20):
        need(name in self.roster,'LEGAL_LEARNING_WITHHELD_INSTRUMENT')
        indices=self.indices(occurrences,side,fit=fit);target=indices;control=None
        if donors is not None:
            need(side=='train','LEGAL_LEARNING_SHUFFLE_REQUIRES_TRAIN')
            target=self.indices(donors,'train',fit=True);self.validate_donors(indices,target)
            control='different_input_and_case_component_train_donor'
        inputs,remaining=self._gather(indices,[n for n in self.roster if n!=name],memory_bytes)
        targets,_=self._gather(target,[name],remaining)
        self.assert_withheld(inputs,targets,name);self.fresh()
        return TypedSupervision(inputs,targets,tuple(indices.tolist()),tuple(target.tolist()),
                                'other_instruments_of_same_observation',name,control)

    def training_authority(self):
        raise ValueError('MNLM_LEGAL_LEARNING_EXPERIMENT_ADMISSION_REQUIRED')

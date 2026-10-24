"""The existing typed constellation consumer over a frozen training partition."""
from pathlib import Path
import copy

import numpy as np

from audit.window_readback import readback
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_cohort_population import CohortPopulation
from model.proposal_input import load
from model.repository_input import RepositoryNative,utc_ns
from model.constellation import Constellation
from model.native import from_arrays


class CohortNative(RepositoryNative):
    def __init__(self,population,population_pin,native,native_pin,partition):
        check(partition in ('learn_1','learn_2'),'COHORT_NATIVE_TRAINING_PARTITION')
        self.partition = partition
        self.population = CohortPopulation(population,population_pin)
        self.native = Path(native)
        self.readback = readback(self.native,self.population.root)
        self.arrays,specs,self.files = load(self.native,native_pin,population_pin)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.identity = dict(population=population_pin,native=native_pin,
                             observations=self.population.observations.pin,partition=partition)
        self.memory_pin = self.memory_hash()
        self.fields_pin = digest(self.population.fields)

    def materialize(self,condition,side,cutoff,*,fields=None,scientific_admission=False):
        check(condition in self.population.fields and
              self.population.fields[condition]['partition'] == self.partition,'COHORT_NATIVE_PARTITION')
        check(side in ('input','echo'),'REPOSITORY_NATIVE_SELECTION')
        source = self.population.fields[condition]
        if not any(f['parts'] for f in source[side]):
            self.fresh()
            check(scientific_admission is False,'REPOSITORY_NATIVE_UNADMITTED')
            available = source['input_available_by' if side == 'input' else 'echo_available_after']
            check(utc_ns(cutoff) >= utc_ns(available),'REPOSITORY_NATIVE_FUTURE')
            check(fields is None or fields == source[side],'REPOSITORY_NATIVE_CONDITION_FIELDS')
            check(all(f['empty'] and f['bytes'] == 0 and f['sha256'] == sha(b'') for f in source[side]),
                  'COHORT_NATIVE_ABSENCE_SOURCE')
            description = dict(schema='repository-native-selection-v1',pins=self.identity,
                condition=condition,side=side,cutoff=cutoff,available=available,
                fields={s.name:[dict(**copy.deepcopy(f),windows=[]) for f in source[side]] for s in self.specs},
                task=source['task'],container=source['container'],action=source['action'],
                scope='retained_repository_surfaces',historical_occurrence_established=False,
                scientific_admission=False,serving=False,numerical_identifiers=False,
                absence='empty_retained_streams')
            return {s.name:np.empty((0,s.dims),dtype=np.float32) for s in self.specs},description
        return super().materialize(condition,side,cutoff,fields=fields,scientific_admission=scientific_admission)

    def observed(self,condition,side,cutoff,config,device,specs):
        check(tuple(specs) == self.specs,'REPOSITORY_NATIVE_MODEL_INSTRUMENT')
        arrays,description = self.materialize(condition,side,cutoff)
        row,slots = digest(description),{}
        for spec in specs:
            values = arrays[spec.name]
            slots[spec.name] = from_arrays(spec,dict(tokens=values,
                offsets=np.asarray([0,len(values)],dtype=np.int64),absent=np.asarray([not len(values)])),
                (row,),config).to(device)
        return Constellation(slots,'observed').validate(specs,config),description

    def receipt(self,condition,side,cutoff,config,device,specs):
        native,description = self.observed(condition,side,cutoff,config,device,specs)
        slots = {}
        for spec in specs:
            slot = native.slots[spec.name]
            a = slot.values[0,~slot.padding[0]].detach().cpu().contiguous().numpy().astype('<f4')
            slots[spec.name] = dict(instrument=spec.identity,shape=list(a.shape),
                sha256=sha(a.tobytes()),dtype='float32-le',absent=bool(slot.absent[0]))
        return dict(description=description,slots=slots)

"""Native repository surfaces retain their condition, containment and causal cutoff."""
import calendar
import copy
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch

from audit.window_readback import readback
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.repository_population import RepositoryPopulation
from model.constellation import Constellation
from model.native import NativeInput
from model.proposal_input import load


def utc_ns(value):
    match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z',value)
    check(match is not None,'REPOSITORY_NATIVE_UTC_CLOCK')
    seconds = calendar.timegm(datetime.strptime(match[1],'%Y-%m-%dT%H:%M:%S').timetuple())
    return seconds*1_000_000_000+int((match[2] or '').ljust(9,'0'))


class RepositoryNative:
    def __init__(self,population,population_pin,native,native_pin):
        self.population = RepositoryPopulation(population,population_pin)
        self.native = Path(native)
        self.readback = readback(self.native,self.population.root)
        self.arrays,specs,self.files = load(self.native,native_pin,population_pin)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.identity = dict(population=population_pin,native=native_pin,
                             observations=self.population.observations.pin)
        self.memory_pin = self.memory_hash()
        self.fields_pin = digest(self.population.fields)

    def memory_hash(self):
        return digest({n:[sha(a.tobytes()) for a in v] for n,v in self.arrays.items()})

    def fresh(self):
        check(self.memory_hash() == self.memory_pin,'REPOSITORY_NATIVE_MEMORY_CHANGED')
        check(digest(self.population.fields) == self.fields_pin,'REPOSITORY_NATIVE_FIELDS_CHANGED')
        for name,pin in self.files.items():
            check(sha((self.native/name).read_bytes()) == pin,'REPOSITORY_NATIVE_BYTES_CHANGED')
        check(sha((self.population.root/'report.json').read_bytes()) == self.population.pin,
              'REPOSITORY_NATIVE_POPULATION_CHANGED')
        for name,value in self.population.report['files'].items():
            check(sha((self.population.root/name).read_bytes()) == value['sha256'],
                  'REPOSITORY_NATIVE_POPULATION_CHANGED')

    def materialize(self,condition,side,cutoff,*,fields=None,scientific_admission=False):
        self.fresh()
        check(scientific_admission is False,'REPOSITORY_NATIVE_UNADMITTED')
        check(condition in self.population.fields and side in ('input','echo'),
              'REPOSITORY_NATIVE_SELECTION')
        source = self.population.fields[condition]
        available = source['input_available_by' if side == 'input' else 'echo_available_after']
        check(utc_ns(cutoff) >= utc_ns(available),'REPOSITORY_NATIVE_FUTURE')
        selected = source[side] if fields is None else fields
        check(selected == source[side],'REPOSITORY_NATIVE_CONDITION_FIELDS')
        arrays,spans = {},{}
        for spec in self.specs:
            values,offsets = self.arrays[spec.name]
            pieces,descriptors,cursor = [],[],0
            for field in selected:
                parts = []
                for part in field['parts']:
                    unit = part['unit']
                    check(sha(self.population.rows[unit]['text'].encode()) == part['sha256'],
                          'REPOSITORY_NATIVE_SOURCE_SPAN')
                    piece = values[offsets[unit]:offsets[unit+1]]
                    check(len(piece) > 0 and piece.shape[1] == spec.dims,'REPOSITORY_NATIVE_WINDOW')
                    pieces.append(piece)
                    parts.append(dict(**part,window_start=cursor,window_end=cursor+len(piece)))
                    cursor += len(piece)
                descriptors.append(dict(**copy.deepcopy(field),windows=parts))
            check(pieces,'REPOSITORY_NATIVE_NO_SURFACE')
            arrays[spec.name] = np.concatenate(pieces)
            spans[spec.name] = descriptors
        description = dict(schema='repository-native-selection-v1',pins=self.identity,
            condition=condition,side=side,cutoff=cutoff,available=available,fields=spans,
            task=source['task'],container=source['container'],action=source['action'],
            scope='retained_repository_surfaces',historical_occurrence_established=False,
            scientific_admission=False,serving=False,numerical_identifiers=False)
        return arrays,description

    def observed(self,condition,side,cutoff,config,device,specs):
        check(tuple(specs) == self.specs,'REPOSITORY_NATIVE_MODEL_INSTRUMENT')
        arrays,description = self.materialize(condition,side,cutoff)
        slots,row = {},digest(description)
        for spec in self.specs:
            values = torch.from_numpy(arrays[spec.name]).unsqueeze(0).to(device)
            slots[spec.name] = NativeInput(spec.identity,(row,),values,
                torch.zeros(values.shape[:2],dtype=torch.bool,device=device),
                torch.zeros(1,dtype=torch.bool,device=device))
        return Constellation(slots,'observed').validate(specs,config),description

    def receipt(self,condition,side,cutoff,config,device,specs):
        native,description = self.observed(condition,side,cutoff,config,device,specs)
        slots = {}
        for spec in self.specs:
            a = native.slots[spec.name].values[0].detach().cpu().contiguous().numpy().astype('<f4')
            slots[spec.name] = dict(instrument=spec.identity,shape=list(a.shape),
                                   sha256=sha(a.tobytes()),dtype='float32-le')
        return dict(description=description,slots=slots)

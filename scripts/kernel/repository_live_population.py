"""Extend an authenticated native population from committed intake events only."""
import copy
from pathlib import Path

import numpy as np

from audit.window_readback import readback
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from kernel.repository_population import KINDS,RepositoryPopulation
from model.proposal_input import load
from model.repository_input import utc_ns


class LiveRepositoryPopulation(RepositoryPopulation):
    def __init__(self,journal,pin,live_journal):
        super().__init__(journal,pin)
        self.base=dict(path=str(self.path),sha256=pin,events=len(self.rows),head=self.rows[-1][2])
        self.base_groups=len(self.groups);self.live_path=str(Path(live_journal).resolve())
        self.loaded={};self.seen={digest(g['spec']) for g in self.groups}
        self.prefix_files={self.base_groups:dict(self.files)}

    def at(self,groups):
        check(groups in self.prefix_files,'LIVE_POPULATION_CHECKPOINT_PREFIX')
        result=copy.copy(self);result.groups=self.groups[:groups]
        ids={k for g in result.groups for k in g['ids']}
        result.records={k:self.records[k] for k in ids};result.values={k:self.values[k] for k in ids}
        result.files=dict(self.prefix_files[groups])
        return result

    def consume(self,body,payload,*,seq,previous,event):
        check(type(seq) is int and seq>=self.base['events'] and seq>self.groups[-1]['seq'],
              'LIVE_POPULATION_ORDER')
        check(body.get('kind') in KINDS and body['payload']==digest(payload) and
              body['prior_head']==previous and event==digest(dict(seq=seq,previous=previous,body=body)),
              'LIVE_POPULATION_EVENT')
        spec=body['spec'];kind=body['kind'];identity=digest({k:v for k,v in spec.items() if k!='partition'})
        check(spec['partition'] in ('learn_1','learn_2') and digest(spec) not in self.seen,
              'LIVE_POPULATION_PARTITION')
        check(spec['partition']=='learn_1' or digest(dict(**{k:v for k,v in spec.items() if k!='partition'},
              partition='learn_1')) in self.seen,'LIVE_POPULATION_PREFIX')
        self.fresh()
        if identity not in self.loaded:
            population=KINDS[kind](spec['population'],spec['population_pin']);root=Path(spec['native'])
            readback(root,population.root)
            arrays,specs,files=load(root,spec['native_pin'],spec['population_pin'])
            check(specs==self.specs,'LIVE_POPULATION_INSTRUMENT_CHANGE')
            sources={str(root/n):pin for n,pin in files.items()}
            sources[str(population.root/'report.json')]=spec['population_pin']
            sources.update({str(population.root/n):v['sha256'] for n,v in population.report['files'].items()})
            self.loaded[identity]=(population,arrays,sources)
        population,arrays,sources=self.loaded[identity]
        selected={k for k,f in population.fields.items() if f['partition']==spec['partition']}
        check(selected and set(payload['bindings'])==set(payload['episodes'])==selected,'LIVE_POPULATION_ROSTER')
        records,values={},{}
        for condition in sorted(selected):
            source=population.fields[condition]
            for side in ('input','echo'):
                receipt=payload['bindings'][condition][side];slots,spans={},{}
                for name,instrument in sorted(self.specs.items()):
                    x,ptr=arrays[name];pieces,fields,cursor=[],[],0
                    for field in source[side]:
                        windows=[]
                        for part in field['parts']:
                            unit=part['unit']
                            check(sha(population.rows[unit]['text'].encode())==part['sha256'],'LIVE_POPULATION_SPAN')
                            piece=x[ptr[unit]:ptr[unit+1]]
                            check(len(piece)>0,'LIVE_POPULATION_EMPTY_PART')
                            pieces.append(piece)
                            windows.append(dict(**part,window_start=cursor,window_end=cursor+len(piece)))
                            cursor+=len(piece)
                        fields.append(dict(**copy.deepcopy(field),windows=windows))
                    a=np.concatenate(pieces) if pieces else np.empty((0,instrument.dims),np.float32)
                    a.flags.writeable=False
                    check(receipt['slots'][name]==dict(instrument=instrument.identity,shape=list(a.shape),
                          sha256=sha(a.astype('<f4').tobytes()),dtype='float32-le',absent=not len(a)),
                          'LIVE_POPULATION_NATIVE_BYTES')
                    slots[name],spans[name]=a,fields
                available=source['input_available_by' if side=='input' else 'echo_available_after']
                description=receipt['description']
                check(set(receipt['slots'])==set(self.specs) and description['fields']==spans and
                      description['condition']==condition and description['side']==side and
                      description['task']==source['task'] and description['container']==source['container'] and
                      description['available']==available and utc_ns(description['cutoff'])>=utc_ns(available) and
                      description['pins']==dict(population=spec['population_pin'],native=spec['native_pin'],
                          observations=population.observations.pin,partition=spec['partition']) and
                      description['scientific_admission'] is False,'LIVE_POPULATION_BINDING')
                key=digest(dict(episode=payload['episodes'][condition],receipt=receipt))
                check(key not in self.records and key not in records,'LIVE_POPULATION_DUPLICATE')
                records[key]=dict(seq=seq,event=event,episode=payload['episodes'][condition],condition=condition,
                    side=side,task=source['task'],available=available,receipt=receipt,
                    absence=all(not len(a) for a in slots.values()))
                values[key]=slots
        self.groups.append(dict(seq=seq,event=event,spec=spec,kind=kind,ids=list(records)))
        self.records.update(records);self.values.update(values);self.files={**self.files,**sources}
        self.prefix_files[len(self.groups)]=dict(self.files)
        self.seen.add(digest(spec))

    def source(self,groups):
        if groups<=self.base_groups:return super().source(groups)
        check(groups<=len(self.groups),'LIVE_POPULATION_SOURCE_PREFIX')
        group=self.groups[groups-1]
        return dict(schema='repository-live-prefix-v1',journal=self.live_path,base=self.base,
                    event_count=group['seq']+1,head=group['event'],supported_intakes=groups)

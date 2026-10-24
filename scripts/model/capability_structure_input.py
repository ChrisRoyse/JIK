"""Source-bound field structure; explicit metadata never becomes a new sense."""
from dataclasses import dataclass
from fractions import Fraction
import json
import math

import numpy as np
import torch

from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest

ROLES = ('attested_speaker_act','base_api_source','base_documentation','candidate_source',
         'original_source','public_base_test','recorded_patch','stdout','task_report','problem',
         'recorded_step','source_passage','source_question','source_answer')
SPEAKER_ROLES = ('ID','ME','PM','UI')
NUMERIC = ('document_order_known','document_position','original_evidence','revised_evidence',
           'interval_known','start_from_latest_input','end_from_latest_input','duration',
           'sync_uncalibrated','receipt_only','base_before_patch')
FIELD_FEATURES = len(ROLES)+len(SPEAKER_ROLES)+len(NUMERIC)
WINDOW_FEATURES = 6


def logarithm(value):
    x=float(value)
    return math.copysign(math.log1p(abs(x)),x)


def field_features(descriptor):
    values=[]
    for field in descriptor['fields']:
        role=field['role']; need(role in ROLES,'STRUCTURE_ROLE')
        speaker=field['speaker_role']; need(speaker is None or speaker in SPEAKER_ROLES,'STRUCTURE_SPEAKER_ROLE')
        numeric=[float(field['document_position'] is not None),
                 logarithm(field['document_position'] or 0),float(field['revision']=='original'),
                 float(field['revision']=='revised'),0.,0.,0.,0.,0.,
                 float(field['time_basis']=='receipt_only'),float(field['time_basis']=='base_before_patch')]
        if field['time_basis']=='annotation_interval':
            clock=field['clock'];lo,hi=Fraction(clock['start']),Fraction(clock['end'])
            origin=Fraction(descriptor['latest_input_end'])
            need(lo<=hi<=origin and clock['physical_sync_calibrated'] is False,'STRUCTURE_INTERVAL')
            numeric[4:9]=[1.,logarithm(lo-origin),logarithm(hi-origin),logarithm(hi-lo),1.]
        values.append([float(role==r) for r in ROLES]+[float(speaker==r) for r in SPEAKER_ROLES]+numeric)
    result=np.asarray(values,dtype='<f4')
    need(result.shape==(len(descriptor['fields']),FIELD_FEATURES) and np.isfinite(result).all(),'STRUCTURE_FEATURES')
    return result


def window_features(field,lane):
    reading=field['readings'][lane];count=reading['windows'];result=np.zeros((count,WINDOW_FEATURES),dtype='<f4');cursor=0
    for part in reading['parts']:
        lo,hi=part['field_window_span'];start,end=part['source_byte_span']
        need(lo==cursor<hi<=count and 0<=start<end<=field['bytes'],'STRUCTURE_WINDOW_COVERAGE')
        for index in range(lo,hi):
            result[index]=[logarithm(index),logarithm(count),logarithm(start),logarithm(end),
                           logarithm(field['bytes']),logarithm(index-lo)]
        cursor=hi
    need(cursor==count,'STRUCTURE_WINDOW_COVERAGE')
    return result


@dataclass(frozen=True)
class StructuredInput:
    task: str
    descriptor: bytes
    native: dict
    field_metadata: torch.Tensor
    windows: dict
    spans: dict

    def validate(self,specs,config):
        value=json.loads(self.descriptor);need(value['task']==self.task,'STRUCTURE_TASK')
        expected=field_features(value)
        need(torch.equal(self.field_metadata.cpu(),torch.from_numpy(expected)),'STRUCTURE_METADATA_BINDING')
        need(set(self.native)==set(self.windows)==set(self.spans)=={s.name for s in specs},'STRUCTURE_ROSTER')
        for spec in specs:
            name=spec.name;slot=self.native[name];slot.check(spec,config)
            need(not bool(slot.padding.any()) and len(slot.rows)==1,'STRUCTURE_NATIVE')
            cursor=0
            for index,field in enumerate(value['fields']):
                lo,hi=self.spans[name][index];n=field['readings'][name]['windows']
                need(lo==cursor and hi==lo+n,'STRUCTURE_FIELD_COVERAGE')
                need(torch.equal(self.windows[name][index].cpu(),torch.from_numpy(window_features(field,name))),
                     'STRUCTURE_WINDOW_BINDING')
                cursor=hi
            need(cursor==slot.values.shape[1],'STRUCTURE_NATIVE_COVERAGE')
        return self


class StructuredContexts:
    """Compile immutable metadata once; authenticate source views outside model loops."""
    def __init__(self,corpus):
        self.corpus=corpus;self.descriptors={};self.materialized={}
        seen=set()
        for source in corpus.sources.values():
            if id(source.population) not in seen:source.population.fresh();seen.add(id(source.population))
        for key in (*corpus.train,*corpus.query):self.descriptors[key]=canonical(self.describe(key))
        self.identity=digest({k:sha(v) for k,v in self.descriptors.items()})

    def describe(self,key):
        corpus=self.corpus;task=corpus.tasks[key];source=corpus.sources[task['population']];population=source.population
        selection=population.selection(key,side='context')
        need(selection==task['selections']['context'],'STRUCTURE_SELECTION')
        fields=[population.fields[f['id']] for f in selection['fields']]
        interval=task['kind'] in ('respond_to_attested_act','continue_attested_dialogue')
        latest=max(Fraction(f['clock']['end']) for f in fields) if interval else None
        if interval:
            need(len({f['clock']['id'] for f in fields})==1,'STRUCTURE_CLOCK_JOIN')
            cutoff=Fraction(population.tasks[key]['provenance']['cutoff'])
            need(latest<=cutoff,'STRUCTURE_FUTURE_INTERVAL')
        result=[]
        for index,field in enumerate(fields):
            clock=field['clock'];role=field['role'];need(role in ROLES,'STRUCTURE_ROLE')
            if interval:time_basis='annotation_interval'
            elif clock is None:time_basis='unknown'
            elif set(clock)=={'available','id','input_clock_includes_patch'}:
                need(clock['input_clock_includes_patch'] is True,'STRUCTURE_RECEIPT');time_basis='receipt_only'
            else:
                need(set(clock)=={'id','ordering_evidence','relation','wall_time'} and clock['wall_time'] is None
                     and clock['relation']=='base snapshot before current patch','STRUCTURE_CLOCK');time_basis='base_before_patch'
            revision=('original' if index<3 else 'revised') if task['kind']=='revise_with_source_evidence' else None
            position=index if task['kind']=='continue_recorded_steps' else None
            readings={}
            for spec in corpus.specs:
                native=corpus.field(task['population'],field['id'],spec.name);_,description=source.field(field['id'],spec.name)
                need(description['role']==role and description['windows']==len(native),'STRUCTURE_NATIVE_BINDING')
                readings[spec.name]=dict(description,sha256=sha(native.astype('<f4',copy=False).tobytes()),instrument=spec.identity)
            result.append(dict(id=field['id'],role=role,clock=clock,time_basis=time_basis,document_position=position,
                revision=revision,source=field['source'],bytes=len(field['text'].encode()),text_sha256=field['text_sha256'],
                speaker_role=field['source']['speaker']['role'] if interval else None,readings=readings))
        return dict(schema='capability-field-structure-v1',task=key,operation=task['kind'],population=task['population'],
            binding=source.pin,fields=result,latest_input_end=str(latest) if latest is not None else None,
            target_cutoff_is_feature=False,source_identifiers_are_features=False,scientific_admission=False)

    def get(self,key,config,device,*,descriptor=None):
        need(key in self.descriptors,'STRUCTURE_TASK_SCOPE');expected=self.descriptors[key]
        need(descriptor is None or descriptor==expected,'STRUCTURE_DESCRIPTOR_BINDING')
        cache_key=(key,str(device))
        if cache_key not in self.materialized:
            value=json.loads(expected);native=self.corpus.batch(key,'context',config,device);spans={};windows={}
            for spec in self.corpus.specs:
                spans[spec.name]=[];windows[spec.name]=[];cursor=0
                for field in value['fields']:
                    n=field['readings'][spec.name]['windows'];spans[spec.name].append((cursor,cursor+n));cursor+=n
                    # Metadata stays on CPU for independent binding checks; each physical tile transfers with its native rows.
                    windows[spec.name].append(torch.from_numpy(window_features(field,spec.name)))
                need(cursor==native[spec.name].values.shape[1],'STRUCTURE_NATIVE_COVERAGE')
            item=StructuredInput(key,expected,native,torch.from_numpy(field_features(value)),windows,spans)
            self.materialized[cache_key]=item.validate(self.corpus.specs,config)
        return self.materialized[cache_key]

    def census(self):
        values=[json.loads(v) for v in self.descriptors.values()]
        return dict(tasks=len(values),fields=sum(len(v['fields']) for v in values),
            descriptor_bytes=sum(len(v) for v in self.descriptors.values()),
            native_bytes=sum(f['readings'][s.name]['windows']*s.dims*4 for v in values for f in v['fields'] for s in self.corpus.specs),
            field_feature_bytes=sum(len(v['fields'])*FIELD_FEATURES*4 for v in values),
            window_feature_bytes=sum(f['readings'][s.name]['windows']*WINDOW_FEATURES*4 for v in values for f in v['fields'] for s in self.corpus.specs))

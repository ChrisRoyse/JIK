"""Observed request structure; identifiers and private answer fields are not features."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import torch

from model.capability_kernel_learning import OPERATIONS
from model.capability_structure_input import FIELD_FEATURES, field_features
from text.constellation_corpus import canonical, digest, need

REQUESTS=(*OPERATIONS,'continue_attested_document')
WIDTH=len(REQUESTS)+FIELD_FEATURES+2


def describe(schedule,key):
    request=schedule.request(key);pop,task,_=schedule.targets.source.selected(key)
    fields=[pop.fields[f['field']] for f in request['fields']]
    interval=task['kind'] in ('respond_to_attested_act','continue_attested_dialogue')
    latest=max(Fraction(f['clock']['end']) for f in fields) if interval else None
    if interval:
        need(len({f['clock']['id'] for f in fields})==1 and latest<=Fraction(task['provenance']['cutoff']),
             'REQUEST_INPUT_CLOCK_SCOPE')
    rows=[]
    for i,f in enumerate(fields):
        clock=f['clock'];basis='annotation_interval' if interval else 'unknown'
        if not interval and clock is not None:
            if set(clock)=={'available','id','input_clock_includes_patch'}:
                need(clock['input_clock_includes_patch'] is True,'REQUEST_INPUT_RECEIPT');basis='receipt_only'
            else:
                need(set(clock)=={'id','ordering_evidence','relation','wall_time'} and clock['wall_time'] is None
                     and clock['relation']=='base snapshot before current patch','REQUEST_INPUT_CLOCK')
                basis='base_before_patch'
        rows.append(dict(role=f['role'],speaker_role=f['source']['speaker']['role'] if interval else None,
            document_position=i if task['kind']=='continue_recorded_steps' else None,
            revision=('original' if i<3 else 'revised') if task['kind']=='revise_with_source_evidence' else None,
            time_basis=basis,clock=deepcopy(clock)))
    return dict(schema='native-observed-request-v1',operation=request['operation'],fields=rows,
        latest_input_end=str(latest) if latest is not None else None,
        position_kind=request['position_kind'])


def legal(observed):
    return dict(schema='native-observed-request-v1',operation='continue_attested_document',
        fields=[dict(role='source_passage',speaker_role=None,document_position=i,revision=None,
                     time_basis='unknown',clock=None) for i in range(observed)],
        latest_input_end=None,position_kind='attested_document_order')


class NativeRequest:
    def __init__(self,descriptor):
        need(set(descriptor)=={'schema','operation','fields','latest_input_end','position_kind'}
             and descriptor['schema']=='native-observed-request-v1','REQUEST_INPUT_SCHEMA')
        need(descriptor['operation'] in REQUESTS and descriptor['fields'] and descriptor['position_kind'] in
             ('request_field_layout_not_event_chronology','attested_document_order'),'REQUEST_INPUT_OPERATION')
        for f in descriptor['fields']:
            need(set(f)=={'role','speaker_role','document_position','revision','time_basis','clock'},'REQUEST_INPUT_FIELD')
            need(f['document_position'] is None or (type(f['document_position']) is int and f['document_position']>=0),
                 'REQUEST_INPUT_POSITION')
            need(f['revision'] in (None,'original','revised'),'REQUEST_INPUT_REVISION')
            basis=f['time_basis'];clock=f['clock']
            need(basis in ('unknown','annotation_interval','receipt_only','base_before_patch'),'REQUEST_INPUT_TIME_BASIS')
            if basis=='unknown':need(clock is None,'REQUEST_INPUT_UNKNOWN_CLOCK')
            elif basis=='receipt_only':
                need(set(clock)=={'available','id','input_clock_includes_patch'} and clock['input_clock_includes_patch'] is True,
                     'REQUEST_INPUT_RECEIPT')
            elif basis=='base_before_patch':
                need(set(clock)=={'id','ordering_evidence','relation','wall_time'} and clock['wall_time'] is None
                     and clock['relation']=='base snapshot before current patch','REQUEST_INPUT_CLOCK')
            else:
                need(descriptor['latest_input_end'] is not None,'REQUEST_INPUT_INTERVAL_ORIGIN')
        intervals=[f['clock'] for f in descriptor['fields'] if f['time_basis']=='annotation_interval']
        if intervals:
            need(len(intervals)==len(descriptor['fields']) and len({v['id'] for v in intervals})==1
                 and Fraction(descriptor['latest_input_end'])==max(Fraction(v['end']) for v in intervals),
                 'REQUEST_INPUT_CLOCK_SCOPE')
        else:need(descriptor['latest_input_end'] is None,'REQUEST_INPUT_UNATTESTED_ORIGIN')
        features=field_features(descriptor);rows=np.zeros((len(features)+1,WIDTH),dtype='<f4')
        rows[:,REQUESTS.index(descriptor['operation'])]=1
        rows[:-1,len(REQUESTS):len(REQUESTS)+FIELD_FEATURES]=features
        rows[:-1,-2]=1;rows[-1,-1]=1
        self.descriptor=canonical(descriptor);self.pin=digest(self.descriptor)
        self.features=torch.from_numpy(rows);self.observed=len(features)
        self.seal=digest(self.features.numpy().tobytes())

    def materialize(self,device):
        need(digest(self.descriptor)==self.pin and digest(self.features.numpy().tobytes())==self.seal,
             'REQUEST_INPUT_MUTATED')
        return self.features.detach().to(device).clone()

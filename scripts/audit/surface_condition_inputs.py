"""Read only the retained native/source cohort consumed by conditioning diagnostics."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from audit.surface_device_authority import REFS as SOURCES
from audit.trained_answer_render import REFS as TRAINED
from model.native import NativeInput
from model.native_population_training import fingerprint
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp


def paired(target, donor):
    need(donor['side']=='train' and donor['document']!=target['document'] and donor['uid']!=target['uid'],
         'CONDITION_DONOR')


def check_reference(prediction, target):
    need(prediction['occurrence']==target['occurrence'] and prediction['uid']==target['uid']
         and prediction['side']==target['side'] and isinstance(prediction['reference'],torch.Tensor)
         and torch.equal(prediction['reference'],target['raw']), 'CONDITION_PREDICTED_SOURCE_JOIN')


def measured_native(row, model):
    need(set(row['native'])=={s.name for s in model.specs},'CONDITION_ROSTER')
    result={s.name:NativeInput(s.identity,(row['row'],),row['native'][s.name].clone()[None],
        torch.zeros((1,len(row['native'][s.name])),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        for s in model.specs}
    model.validate_inputs(result)
    return result


def check_ablation(original, ablated):
    left=original.state_dict();right=ablated.state_dict()
    gains={f'blocks.{i}.branches.{s.name}.gain' for i in range(len(original.blocks)) for s in original.specs}
    need(set(left)==set(right) and original.config==ablated.config and original.specs==ablated.specs,
         'CONDITION_ABLATION_ROSTER')
    need(all(torch.equal(right[n],torch.zeros_like(left[n]) if n in gains else left[n]) for n in left),
         'CONDITION_ABLATION_STATE')
    return sorted(gains)


def load(root, *, continuation=False):
    records={};stamps={};consumed={}
    def authenticate(path,pin):
        before=stamp(path);need(file_sha(path)==pin,'CONDITION_PIN')
        need(stamp(path)==before,'CONDITION_CHANGED');stamps[path]=before;consumed[str(path)]=pin
    def record(path,pin):
        authenticate(path,pin);value=json.loads(path.read_bytes())
        need(stamp(path)==stamps[path],'CONDITION_CHANGED');return value
    for key in ('source','source_readback','mixed','mixed_readback'):
        directory,pin=SOURCES[key];records[key]=record(root/directory/'report.json',pin)
    r=records['source'];q=records['source_readback'];m=records['mixed_readback']
    need(r['complete'] and q['complete'] and q['input_pin']==SOURCES['source'][1]
         and q['all_source_bytes_exact'] and q['all_native_arrays_exact'] and q['all_schedules_exact'],
         'CONDITION_SOURCE_READBACK')
    need(m['complete'] and m['input_pin']==SOURCES['mixed'][1] and m['all_cold_arm_states_exact'],
         'CONDITION_DONOR_READBACK')
    path=root/SOURCES['source'][0]/'observations.pt';authenticate(path,r['files'][path.name])
    payload=torch.load(path,map_location='cpu',weights_only=True)
    need(stamp(path)==stamps[path],'CONDITION_CHANGED')
    rows=[]
    for document in payload:
        need(len(document['observations'])==document['stop']-document['start'],'CONDITION_DOCUMENT')
        for i,row in enumerate(document['observations']):
            need(row['occurrence']==document['start']+i and row['raw'].dtype==torch.uint8 and len(row['raw'])>0,
                 'CONDITION_SOURCE_ROW')
            rows.append(dict(row,document=document['document'],side=document['side']))
    need(len(rows)==215 and len(payload)==15 and sum(len(v['raw']) for v in rows)==42523
         and len({v['occurrence'] for v in rows})==len(rows),'CONDITION_POPULATION')
    path=root/SOURCES['mixed'][0]/'results.json'
    selection=record(path,records['mixed']['files'][path.name]);donors=selection['donors']
    need(selection['metadata']==[{k:row[k] for k in ('document','side','train_present')} for row in rows]
         and len(donors)==len(rows),'CONDITION_SELECTION')
    for target,index in zip(rows,donors):
        need(type(index) is int and 0<=index<len(rows),'CONDITION_DONOR_INDEX');paired(target,rows[index])
    args=[]
    for key in ('full','plan','cold','final'):args.extend((root/TRAINED[key][0],TRAINED[key][1]))
    decoder=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True,retain_continuation=continuation);model=decoder.model
    need(r['config']==vars(model.config) and r['instruments']==[vars(s) for s in model.specs],'CONDITION_MODEL_ROSTER')
    ablated=deepcopy(model)
    with torch.no_grad():
        for block in ablated.blocks:
            for branch in block.branches.values():branch.gain.zero_()
    gains=check_ablation(model,ablated)
    input_pin=fingerprint(rows);ablation_pin=fingerprint(ablated.state_dict())
    def fresh():
        decoder.fresh();check_ablation(model,ablated)
        need(all(stamp(p)==s for p,s in stamps.items()) and fingerprint(rows)==input_pin,'CONDITION_CHANGED')
        need(fingerprint(ablated.state_dict())==ablation_pin,'CONDITION_ABLATION_CHANGED')
    return dict(rows=rows,donors=donors,decoder=decoder,ablated=ablated,gains=gains,ablation_pin=ablation_pin,
                input_pin=input_pin,fresh=fresh,records=records,consumed=consumed,args=args)

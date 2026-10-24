"""Bind the completed broad legal predictor to the unchanged native answer source."""
from pathlib import Path
import json

import torch
from audit import full_history_source as prior
from model.native_region_predictor import NativeRegionPredictor
from model.native_open_window_answer import answer_model
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need

ROOT=prior.ROOT
PARENT=('legal-efficient-predictor-breadth-v1-20261023','4921af55a18381dc978e25d593f946534a6e519d2d0b1a4623f640c0993c13f4')
COLD=('legal-efficient-predictor-breadth-cold-v1-20261023','9928a3993b4ea745ee50035a14210bb2d86fd34cdcdded10d947a43356b4450d')
NEW=('model/breadth_answer_transfer.py','audit/breadth_answer_source.py','audit/breadth_answer_training.py','audit/breadth_answer_readback.py')


def load(root):
    s=prior.load(root);base=Path(__file__).resolve().parents[1];files={};reports=[]
    for name,pin in (PARENT,COLD):
        path=root/name/'report.json';need(file_sha(path)==pin,'BROAD_ANSWER_REPORT');files[str(path)]=pin
        reports.append(json.loads(path.read_bytes()))
    r,c=reports
    need(r['complete'] and c['complete'] and c['previous_pin']==PARENT[1] and r['binding']==c['binding']
         and r['ledger_head']==c['ledger_head'] and r['updates']==c['updates']==91540,'BROAD_ANSWER_COLD')
    need(r['target_state']==s['bundle']['target_fingerprint'] and r['binding']['target']==s['bundle']['target_ref'],
         'BROAD_ANSWER_TARGET')
    parents={}
    for step in (0,91540):
        path=root/PARENT[0]/f'checkpoint-{step}.pt';ref=r['checkpoints'][str(step)]
        need(file_sha(path)==ref['pin'] and ref['state']==c['checkpoints'][str(step)]['state'],'BROAD_ANSWER_CHECKPOINT_PIN')
        value=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(value)==ref['state'] and value['step']==step and value['binding']==r['binding'],'BROAD_ANSWER_STATE')
        parents[step]=value;files[str(path)]=ref['pin']
    code=dict(r['execution']['code'],**s['code'])
    for n,h in code.items():need(file_sha(base/n)==h,'BROAD_ANSWER_CODE')
    code.update({n:file_sha(base/n) for n in NEW});s['code']=code
    original=s['fresh'];parent_seal=fingerprint(parents)
    def fresh():
        original();need(fingerprint(parents)==parent_seal,'BROAD_ANSWER_PARENT_CHANGED')
        for name,pin in files.items():need(file_sha(Path(name))==pin,'BROAD_ANSWER_FILE_CHANGED')
        for name,pin in code.items():need(file_sha(base/name)==pin,'BROAD_ANSWER_CODE_CHANGED')
    s.update(parents=parents,parent_files=files,fresh=fresh,pretraining_report=r)
    return s


def models(s,step):
    need(step in (0,91540),'BROAD_ANSWER_PARENT_STEP')
    source=NativeRegionPredictor(s['target'],target_pin=s['bundle']['target_ref']['pin'],seed=s['policy']['seed'])
    source.load_state_dict(s['parents'][step]['model'],strict=True)
    child=answer_model(s['target'],s['bundle'],'shared',target_pin=s['pin'])
    return source,child

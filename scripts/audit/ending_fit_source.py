"""Frozen real causal histories from the full-history answer experiment."""
import json
from pathlib import Path

import torch
from audit import full_history_source as prior
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,legal
from text.constellation_corpus import file_sha,need

ROOT=prior.ROOT
PARENT=('full-history-answer-v2-20261024','06727e721077db9c8554212f64d9b060769f5310ae0969d180eee622332fb408')
COLD=('full-history-answer-cold-v2-20261024','a5da10403e869b5d3a7e0d757bb71a3363b34498a3c0f27cc6666293fe27ff12')
NEW=('model/answer_ending_fit.py','audit/ending_fit_source.py','audit/ending_fit_training.py','audit/ending_fit_readback.py')


def load(root):
    s=prior.load(root);base=Path(__file__).resolve().parents[1];records={}
    for name,pin in (PARENT,COLD):
        p=root/name/'report.json';need(file_sha(p)==pin,'ENDING_SOURCE_PIN')
        records[name]=json.loads(p.read_bytes());need(records[name]['complete'],'ENDING_SOURCE_COMPLETE')
    report=records[PARENT[0]];need(records[COLD[0]]['source']['pin']==PARENT[1],'ENDING_SOURCE_COLD')
    for n,h in report['files'].items():need(file_sha(root/PARENT[0]/n)==h,'ENDING_SOURCE_BYTES')
    need(report['code']==s['code'],'ENDING_SOURCE_CODE')
    s['initial']={a:torch.load(root/PARENT[0]/a/'full/update-10.pt',weights_only=True,map_location='cpu')
                  for a in prior.prior.ARCHITECTURES}
    parent_seal=fingerprint(s['initial']);s['code'].update({n:file_sha(base/n) for n in NEW});old=s['fresh']
    def fresh():
        old();need(fingerprint(s['initial'])==parent_seal,'ENDING_PARENT_CHANGED')
        for n,h in s['code'].items():need(file_sha(base/n)==h,'ENDING_SOURCE_CODE_CHANGED')
        for n,h in report['files'].items():need(file_sha(root/PARENT[0]/n)==h,'ENDING_SOURCE_CHANGED')
    s['fresh']=fresh
    return s


def make(s, architecture):
    return prior.prior.make(s,architecture,step=10)[0]


def collect(s, model):
    model.eval();rows=[];request=NativeRequest(legal(2));before=fingerprint(model.state_dict())
    with torch.no_grad():
        for doc,d in sorted(s['items'].items()):
            stream=model.stream(request);cache=None
            for pos,frame in enumerate(d['frames']):
                hidden,cache=stream.step((frame,),context='ending-reference:'+str(doc),offset=pos,state=cache,detach_history=True)
                if pos<1:continue
                contexts=model.forecaster.contexts(hidden)
                rows.append(dict(document=doc,side=d['side'],component=d['component'],position=pos,
                    end=pos==len(d['frames'])-1,native=d['native_pin'],request=request.pin,
                    states={p:{n:cpu(h[0,i]) for i,n in enumerate(model.roster)} for p,(_,h) in contexts.items()},
                    original_logits={p:cpu(model.stop_logits(h,p)[0]) for p,(_,h) in contexts.items()}))
    need(fingerprint(model.state_dict())==before,'ENDING_CONTEXT_MODEL_CHANGED')
    need({side:sum(r['side']==side for r in rows) for side in ('train','development','query')}==
         dict(train=66,development=64,query=70),'ENDING_SOURCE_CENSUS')
    return rows

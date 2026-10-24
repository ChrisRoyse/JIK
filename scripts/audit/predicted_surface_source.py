"""Pinned real source joins for predicted-reading surface adaptation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from audit import surface_device_authority as authority
from audit.answer_surface_stream import REFS
from audit.legal_grouped_placement_training import source_bundle
from model.native_open_window_answer import answer_model
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.task_answer_training import extension
from model.predicted_surface_training import paired
from text.constellation_corpus import file_sha, need

TRAIN=(27195,106288)
EVALUATION=(22924,30520)
ARCHITECTURES=('shared','own_slot','whole_proposals')
NEW=('model/predicted_surface_training.py','audit/predicted_surface_source.py',
     'audit/predicted_surface_training.py','audit/predicted_surface_readback.py')


def load(root):
    records={k:authority.pinned(root/n,h) for k,(n,h) in REFS.items()}
    need(records['predictions_readback']['source']['pin']==REFS['predictions'][1]
         and records['decoder_readback']['input_pin']==REFS['decoder'][1],'ADAPTATION_PRIOR_READBACK')
    config,specs,examples,meta,_,_,_,code,surface_fresh=authority.load(root)
    source,bundle=source_bundle(root);target,pin=extension(bundle)
    need(pin==records['predictions']['extension_pin'],'ADAPTATION_REGION_RUNTIME')
    base=Path(__file__).resolve().parents[1]
    for record in [source,*records.values()]:
        for n,h in record.get('code',{}).items():
            need(file_sha(base/n)==h and (n not in code or code[n]==h),'ADAPTATION_PRIOR_CODE');code[n]=h
    for n in [*NEW,'model/surface_inversion_training.py','model/task_answer_training.py',
              'model/answer_surface_stream.py','audit/answer_surface_stream.py']:
        code[n]=file_sha(base/n)
    groups={}
    for i,m in enumerate(meta):groups.setdefault(m['document'],[]).append(i)
    items={d['document']:d for d in bundle['items']}
    for document in (*TRAIN,*EVALUATION):
        d=items[document];indices=groups[document]
        need(len(d['frames'])==len(indices),'ADAPTATION_DOCUMENT_JOIN')
        for position,i in enumerate(indices):
            e=examples[i]
            need(e.occurrence==d['start']+position and e.side==d['side']
                 and all(torch.equal(v,e.native[n].values[0]) for n,v in d['frames'][position].items()),
                 'ADAPTATION_NATIVE_JOIN')
    checkpoint=torch.load(root/REFS['decoder'][0]/'reference.pt',map_location='cpu',weights_only=True)
    initial_pin=fingerprint((checkpoint['after'],checkpoint['optimizer']))
    stamps={root/n/f:authority.stamp(root/n/f) for n,_ in REFS.values()
            for f in ['report.json',*json.loads((root/n/'report.json').read_bytes()).get('files',{})]}
    def fresh():
        surface_fresh()
        need(fingerprint((checkpoint['after'],checkpoint['optimizer']))==initial_pin,'ADAPTATION_INITIAL_MUTATED')
        for p,s in stamps.items():need(authority.stamp(p)==s,'ADAPTATION_SOURCE_CHANGED')
        for n,h in code.items():need(file_sha(base/n)==h,'ADAPTATION_CODE_CHANGED')
    return dict(root=root,records=records,config=config,specs=specs,examples=examples,metadata=meta,
        bundle=bundle,target=target,region_pin=pin,groups=groups,items=items,checkpoint=checkpoint,code=code,fresh=fresh)


def predictions(state):
    result=[];weights={};frozen=fingerprint(state['target'].state_dict())
    for arm in ARCHITECTURES:
        path=state['root']/REFS['predictions'][0]/arm/'update-5.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=answer_model(state['target'],state['bundle'],arm,target_pin=state['region_pin'])
        model.load_state_dict(checkpoint['model'],strict=True);model.eval().requires_grad_(False)
        initial=fingerprint(model.state_dict());weights[arm]=dict(path=str(path),pin=file_sha(path),state=initial)
        for proposal in model.proposals:
            for document in TRAIN:
                d=state['items'][document];need(d['side']=='train','ADAPTATION_PREDICT_TRAIN')
                request=NativeRequest(legal(2));seed=20261024+document
                gen=model.rollout(tuple(d['frames'][:2]),request=request,target=state['target'],
                                  proposal=proposal,context='request-proof',seed=seed)
                events=[]
                try:
                    for _ in range(len(d['frames'])-1):
                        event=next(gen);events.append(cpu(event))
                        if event['kind']=='end':break
                finally:gen.close()
                result.append(dict(architecture=arm,proposal=proposal,document=document,events=events,
                    seed=seed,request=request.pin,predictor=weights[arm]['pin'],input=fingerprint(d['frames'][:2])))
        need(fingerprint(model.state_dict())==initial,'ADAPTATION_PREDICTOR_CHANGED')
    need(fingerprint(state['target'].state_dict())==frozen,'ADAPTATION_FROZEN_CHANGED')
    return result,weights


def assemble(state,traces,trace_pin):
    groups={};receipts=[]
    def join(row,pin,fitting):
        doc=row['document'];key=row['architecture']+'/'+row['proposal'];indices=state['groups'][doc][2:]
        events=row['events'];need(len(events)>=len(indices),'ADAPTATION_MISSING_PREDICTION')
        values=[]
        for position,i in enumerate(indices):
            e=state['examples'][i]
            pair=paired(e,events[position],state['specs'],state['config'],position=position,proposal=row['proposal'],
                trace_pin=pin,region_pin=state['region_pin'],region_sizes=tuple(state['bundle']['sizes']),
                source_sha=hashlib.sha256(e.raw).hexdigest(),document=doc,source_index=i,fitting=fitting)
            receipts.append(dict(key=key,**pair.receipt,side=e.side,train_present=state['metadata'][i]['train_present']))
            values.append(pair)
        groups.setdefault(key,{})[doc]=values
    for r in traces:join(r,trace_pin,True)
    for arm in ARCHITECTURES:
        path=state['root']/REFS['predictions'][0]/arm/'evaluation.pt';pin=file_sha(path)
        for r in torch.load(path,map_location='cpu',weights_only=True):
            if r['mode']=='correct' and r['document'] in EVALUATION:join(dict(r,architecture=arm),pin,False)
    need(len(groups)==8 and all(set(g)==set((*TRAIN,*EVALUATION)) for g in groups.values()),'ADAPTATION_GROUPS')
    return groups,receipts


def evaluation_views(group):
    result={};metadata={}
    train=[p for doc in TRAIN for p in group[doc]]
    for document in EVALUATION:
        pairs=group[document];shuffled=[];donors=[]
        for position,p in enumerate(pairs):
            choices=[v for v in train if v.receipt['document']!=document and v.observed.uid!=p.observed.uid]
            need(choices,'ADAPTATION_NO_DONOR');donor=choices[position%len(choices)]
            shuffled.append(replace(p.predicted,native=donor.predicted.native));donors.append(donor.seal)
        result[document]=dict(true=[p.observed for p in pairs],predicted=[p.predicted for p in pairs],shuffled=shuffled)
        metadata[document]=dict(atoms=len(pairs),donors=donors,positions=sum(len(p.observed.raw)+1 for p in pairs))
    return result,metadata

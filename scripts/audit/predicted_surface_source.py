"""Pinned real source joins for predicted-reading surface adaptation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

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
RETAINED=('predicted-surface-adaptation-cpu-v4-20261024','74466130b7b0c9ae3ac8c983be78807c346011ba77fcf78754b62628aaf46202')
RETAINED_COLD=('predicted-surface-adaptation-readback-v4-20261024','9443189292c06ad544852da05d51ab4c52b2b493bb5222d1947e3292cf2af813')


def load(root, *, trained_decoder=False):
    need(type(trained_decoder) is bool,'ADAPTATION_DECODER_SELECTION')
    selected={k:v for k,v in REFS.items() if not trained_decoder or k.startswith('predictions')}
    records={k:authority.pinned(root/n,h) for k,(n,h) in selected.items()}
    need(records['predictions_readback']['source']['pin']==REFS['predictions'][1],'ADAPTATION_PRIOR_READBACK')
    decoder_receipt=None;retained=None
    if trained_decoder:
        from audit import surface_condition_inputs
        from model.legal_surface_inputs import SurfaceExample
        retained=surface_condition_inputs.load(root,continuation=True);decoder=retained['decoder']
        config,specs=decoder.config,decoder.specs;checkpoint,initial_step=decoder.export_continuation()
        decoder_receipt=dict(decoder.receipt)
        examples=[SurfaceExample(r['occurrence'],r['uid'],r['side'],bytes(r['raw'].tolist()),
                  surface_condition_inputs.measured_native(r,decoder.model)) for r in retained['rows']]
        meta=[{k:r[k] for k in ('document','side','train_present')} for r in retained['rows']]
        code={};surface_fresh=retained['fresh']
    else:
        need(records['decoder_readback']['input_pin']==REFS['decoder'][1],'ADAPTATION_PRIOR_READBACK')
        config,specs,examples,meta,_,_,_,code,surface_fresh=authority.load(root)
        checkpoint=torch.load(root/REFS['decoder'][0]/'reference.pt',map_location='cpu',weights_only=True)
        initial_step=1
    source,bundle=source_bundle(root);target,pin=extension(bundle)
    need(pin==records['predictions']['extension_pin'],'ADAPTATION_REGION_RUNTIME')
    base=Path(__file__).resolve().parents[1]
    for record in [source,*records.values()]:
        for n,h in record.get('code',{}).items():
            need(file_sha(base/n)==h and (n not in code or code[n]==h),'ADAPTATION_PRIOR_CODE');code[n]=h
    for n in [*NEW,'model/surface_inversion_training.py','model/task_answer_training.py',
              'model/answer_surface_stream.py','audit/answer_surface_stream.py',
              'model/trained_surface_decoder.py','audit/surface_condition_inputs.py',
              'audit/trained_answer_render.py']:
        code[n]=file_sha(base/n)
    # Both explicit CPU entry points have the same dependency closure. Include
    # the actual decoder/batching dependencies even when the old proof loader
    # (which previously supplied their pins) is intentionally bypassed.
    from audit import predicted_surface_training, predicted_surface_readback
    for module in list(sys.modules.values()):
        if not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        if path.is_relative_to(base):code[path.relative_to(base).as_posix()]=file_sha(path)
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
    initial_pin=fingerprint((checkpoint['after'],checkpoint['optimizer']))
    stamps={root/n/f:authority.stamp(root/n/f) for n,_ in selected.values()
            for f in ['report.json',*json.loads((root/n/'report.json').read_bytes()).get('files',{})]}
    def fresh():
        surface_fresh()
        need(fingerprint((checkpoint['after'],checkpoint['optimizer']))==initial_pin,'ADAPTATION_INITIAL_MUTATED')
        for p,s in stamps.items():need(authority.stamp(p)==s,'ADAPTATION_SOURCE_CHANGED')
        for n,h in code.items():need(file_sha(base/n)==h,'ADAPTATION_CODE_CHANGED')
    return dict(root=root,records=records,config=config,specs=specs,examples=examples,metadata=meta,
        bundle=bundle,target=target,region_pin=pin,groups=groups,items=items,checkpoint=checkpoint,code=code,fresh=fresh,
        initial_step=initial_step,decoder_receipt=decoder_receipt,retained=retained,stamps=stamps,trained_decoder=trained_decoder)


def retained_predictions(state):
    """Consume a previously replayed immutable trace, never rerun an unchanged predictor."""
    root=state['root'];path=root/RETAINED[0];cold=root/RETAINED_COLD[0]/'report.json'
    need(file_sha(path/'report.json')==RETAINED[1] and file_sha(cold)==RETAINED_COLD[1],
         'ADAPTATION_RETAINED_PIN')
    report=json.loads((path/'report.json').read_bytes());reader=json.loads(cold.read_bytes())
    need(report['complete'] and reader['complete'] and reader['source']['pin']==RETAINED[1]
         and reader['all_predictor_outputs_exact'] and reader['all_native_source_pairs_exact']
         and reader['predictor_decisions']==report['predictor_decisions']==88
         and report['source']=={k:list(v) for k,v in REFS.items()},'ADAPTATION_RETAINED_READBACK')
    for p in (path/'report.json',cold,*(path/n for n in ('train-predictions.pt','predictors.json','pairs.json'))):
        before=authority.stamp(p)
        if p.name not in ('report.json',):need(file_sha(p)==report['files'][p.name],'ADAPTATION_RETAINED_FILE')
        need(authority.stamp(p)==before,'ADAPTATION_RETAINED_CHANGED');state['stamps'][p]=before
    traces=torch.load(path/'train-predictions.pt',map_location='cpu',weights_only=True)
    weights=json.loads((path/'predictors.json').read_bytes())
    for arm in ARCHITECTURES:
        p=root/REFS['predictions'][0]/arm/'update-5.pt'
        need(weights[arm]['path']==str(p) and file_sha(p)==weights[arm]['pin'],'ADAPTATION_RETAINED_PREDICTOR')
    for row in traces:
        doc=state['items'][row['document']]
        need(row['document'] in TRAIN and doc['side']=='train' and row['predictor']==weights[row['architecture']]['pin']
             and row['input']==fingerprint(doc['frames'][:2]),'ADAPTATION_RETAINED_INPUT')
    state['prediction_reuse']=dict(report=RETAINED[1],readback=RETAINED_COLD[1],
        trace=report['trace_pin'],path=str(path/'train-predictions.pt'),predictor_calls=0)
    return traces,weights


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

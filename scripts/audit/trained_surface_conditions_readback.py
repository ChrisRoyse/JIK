"""Direct-forward cold replay and independent condition/source denominator recount."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import surface_condition_inputs as inputs
from audit.answer_surface_stream_readback import logits, baseline_atom
from audit.trained_answer_render import ROOT,REFS
from claims.process_conditions import write_committed
from model.native import NativeInput
from model.native_population_training import fingerprint
from model.stream_surface import BOS,EOS
from text.constellation_corpus import file_sha,need


def slots(row,specs):
    return {s.name:NativeInput(s.identity,(row['row'],),row['native'][s.name].clone().unsqueeze(0),
        torch.zeros((1,len(row['native'][s.name])),dtype=torch.bool),torch.tensor([False])) for s in specs}


def outcome(teacher,free,raw):
    target=torch.tensor([*raw,EOS]);loss=-torch.log_softmax(teacher.double(),1)[torch.arange(len(target)),target]
    values=bytes(v['value'] for v in free['events'] if v['kind']=='byte');last=free['events'][-1]
    need(free['decisions']==sum('token' in v for v in free['events']),'CONDITION_COLD_DECISIONS')
    completed=last['kind']=='atom_end'
    return dict(positions=len(target),correct=int((teacher.argmax(1)==target).sum()),loss_sum=float(loss.sum()),
        mean_loss=float(loss.mean()),source_bytes=len(raw),generated_bytes=len(values),decisions=free['decisions'],
        last=last['kind'],code=last.get('code'),censored=free['censored'],completed=completed,
        reference_fits_observer=len(raw)+1<=96,exact=completed and values==raw,generated_sha256=hashlib.sha256(values).hexdigest())


def same(actual,expected):
    need(set(actual)==set(expected),'CONDITION_COLD_KEYS')
    for key,v in actual.items():
        need(math.isclose(v,expected[key],rel_tol=1e-14,abs_tol=1e-12) if type(v) is float else v==expected[key],
             'CONDITION_COLD_SCORE')


def aggregate(rows):
    groups={}
    for row in rows:
        labels=[('all','all'),('side',row['side']),('train_present',str(row['train_present'])),('document',str(row['document']))]
        for kind,label in labels:
            key=f'{kind}:{label}';g=groups.setdefault(key,dict(atoms=0,modes={},true_shuffled_wins=0,ties=0,losses=0,
                                                            paired_gain_sum=0.,shuffled_free_equal=0))
            g['atoms']+=1;a=row['scores']['true'];b=row['scores']['shuffled'];delta=b['mean_loss']-a['mean_loss']
            g['true_shuffled_wins']+=delta>0;g['losses']+=delta<0;g['ties']+=delta==0;g['paired_gain_sum']+=delta
            g['shuffled_free_equal']+=a['generated_sha256']==b['generated_sha256']
            for mode,v in row['scores'].items():
                acc=g['modes'].setdefault(mode,dict(positions=0,correct=0,loss_sum=0.,atom_mean_loss_sum=0.,
                    completed=0,censored=0,refused=0,exact=0,reference_fits_observer=0,unique_surfaces=set()))
                for n in ('positions','correct','loss_sum','completed','censored','exact','reference_fits_observer'):acc[n]+=v[n]
                acc['atom_mean_loss_sum']+=v['mean_loss'];acc['refused']+=v['last']=='refuse'
                acc['unique_surfaces'].add(v['generated_sha256'])
    for g in groups.values():
        for v in g['modes'].values():
            v['unique_surfaces']=len(v['unique_surfaces']);v['mean_nll_per_position']=v['loss_sum']/v['positions']
    return groups


def predicted_pairs(root,state,rows):
    source=root/'trained-answer-render-cpu-v2-20261024';pin='99bcd5932cf874266c95bf3c6279590ba78193f828585ef364d63ab7fb8d385a'
    need(file_sha(source/'report.json')==pin,'CONDITION_PREDICTED_PIN');r=json.loads((source/'report.json').read_bytes())
    need(r['decoder']==state['decoder'].receipt,'CONDITION_PREDICTED_DECODER')
    p=source/'corpus_trained.pt';need(file_sha(p)==r['files'][p.name],'CONDITION_PREDICTED_FILE')
    results=torch.load(p,map_location='cpu',weights_only=True)
    name,cpin=REFS['cohort'];p=root/name/'report.json';need(file_sha(p)==cpin,'CONDITION_PREDICTED_COHORT')
    q=json.loads(p.read_bytes());p=root/name/'inputs.pt';need(file_sha(p)==q['files'][p.name],'CONDITION_PREDICTED_INPUTS')
    predictions=torch.load(p,map_location='cpu',weights_only=True)
    index={v['occurrence']:i for i,v in enumerate(state['rows'])};pairs=[]
    for i,v in enumerate(predictions):
        need(v['occurrence'] in index,'CONDITION_PREDICTED_OCCURRENCE');j=index[v['occurrence']];target=state['rows'][j]
        inputs.check_reference(v,target)
        logits=torch.stack([e['logits'] for e in results['teacher'][i]])
        pred=outcome(logits,results['free'][i],bytes(target['raw'].tolist()))
        pairs.append(dict(architecture=v['architecture'],proposal=v['proposal'],position=v['position'],occurrence=v['occurrence'],
                          train_present=v['train_present'],predicted=pred,controls=rows[j]['scores']))
    need(len(pairs)==16 and len({x['occurrence'] for x in pairs})==2,'CONDITION_PREDICTED_DENOMINATOR')
    return dict(source=pin,cohort=cpin,atoms=16,distinct_target_atoms=2,rows=pairs)


def run(a):
    a.output.mkdir(exist_ok=False);start=tick=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.source/'report.json')==a.pin,'CONDITION_COLD_PIN')
    r=json.loads((a.source/'report.json').read_bytes());base=Path(__file__).resolve().parents[1]
    for n,h in r['code'].items():need(file_sha(base/n)==h,'CONDITION_COLD_CODE')
    for n,h in r['files'].items():need(file_sha(a.source/n)==h,'CONDITION_COLD_FILE')
    state=inputs.load(a.root);model=state['decoder'].model
    need(r['binding']['decoder']==state['decoder'].receipt and r['binding']['ablation_state']==state['ablation_pin']
         and r['binding']['native_population']==state['input_pin'],'CONDITION_COLD_BINDING')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();rows=[None]*215;positions=free_positions=0
    for ordinal in (reversed(range(215)) if a.reverse else range(215)):
        row=state['rows'][ordinal];raw=bytes(row['raw'].tolist());tokens=[BOS,*raw]
        saved=torch.load(a.source/f'atoms/{ordinal:04d}.pt',map_location='cpu',weights_only=True)
        need(set(saved)=={'true','shuffled','no_cross'},'CONDITION_COLD_MODES');scores={}
        for mode in (reversed(r['modes']) if a.reverse else r['modes']):
            current=state['ablated'] if mode=='no_cross' else model
            native=slots(state['rows'][state['donors'][ordinal]] if mode=='shuffled' else row,model.specs)
            matrix=saved[mode]['teacher'];need(matrix.shape==(len(raw)+1,258),'CONDITION_COLD_TAIL')
            for position in (reversed(range(len(raw)+1)) if a.reverse else range(len(raw)+1)):
                left=max(0,16*(position//16)-16);prefix=tokens[left:position+1]
                value=logits(current,native,prefix,left)
                need(torch.equal(value,matrix[position]),'CONDITION_COLD_LOGITS');positions+=1
            generated=baseline_atom(current,native,96)
            need(fingerprint(generated)==fingerprint(saved[mode]['free']),'CONDITION_COLD_FREE');free_positions+=generated['decisions']
            scores[mode]=outcome(matrix,generated,raw);same(scores[mode],r['summary'][ordinal]['scores'][mode])
        metadata={k:row[k] for k in ('document','side','train_present','occurrence','uid')}
        for k,v in metadata.items():need(v==r['summary'][ordinal][k],'CONDITION_COLD_METADATA')
        rows[ordinal]=dict(metadata,ordinal=ordinal,donor=state['donors'][ordinal],scores=scores)
        if ordinal%10==0:print(json.dumps(dict(atom=ordinal,positions=positions,seconds=time.perf_counter()-start)),flush=True)
        del saved,matrix
    phases['independent_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    need(positions==3*42738,'CONDITION_COLD_DENOMINATOR')
    paired=predicted_pairs(a.root,state,rows);groups=aggregate(rows);state['fresh']();state['decoder'].close()
    need(not torch.cuda.is_initialized(),'CONDITION_COLD_GPU');phases['joins_and_recount']=time.perf_counter()-tick
    seconds=time.perf_counter()-start;need(sum(phases.values())/seconds>=.9,'CONDITION_COLD_PHASE_ATTRIBUTION')
    result=dict(complete=True,source=dict(path=str(a.source),pin=a.pin),code=r['code'],reverse=a.reverse,
        acceptance='author-checked',number_status='measured',all_direct_logits_exact=True,all_free_paths_exact=True,
        source_and_weights_unchanged=True,teacher_positions=positions,free_positions=free_positions,groups=groups,prediction_join=paired,
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,semantic_validity_established=False,
        independent_generalization_established=False,promoted_to_serving=False)
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds,all=groups['all:all'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reverse',action='store_true')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

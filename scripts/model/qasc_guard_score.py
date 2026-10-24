"""Fixed per-slot geometry and independent provider-target diagnostics; always unasserted."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_generation_data import load
from model.qasc_context import LANES,POLICIES,check
from audit.qasc_region_readback import read as read_region,read_json
from audit.qasc_surface_readback import read as read_surface,records
from audit.qasc_guard_population_readback import read as read_population
from audit.window_readback import readback as read_native
from model.guard_score import POLICY
from model.train_qasc_surfaces import lines
from model.train_qasc_regions import files
from text.evaluation_multiscale import file_sha
from text.connected_training_census import norm
from senses.trajectory_population import publish
from claims.qasc_grounding import assessment,source_records,SOURCE_SHA256

def unit(x):
    norms=np.linalg.norm(x,axis=1,keepdims=True);valid=np.isfinite(x).all(1)&(norms[:,0]>1e-12)
    return np.divide(x,norms,out=np.zeros_like(x),where=valid[:,None]),valid

def gather(output_records,native,target_native,measured):
    values=[];available=[];counts=[];instrument={}
    for name in LANES:
        rows=np.zeros((len(output_records),768),np.float64);windows=np.zeros(len(rows),np.int64)
        ids=[read_json(root/name/'report.json')['lens_id'] for root in (native,target_native,measured)]
        check(len(set(ids))==1,'QASC_GUARD_INSTRUMENT');instrument[name]=ids[0]
        for route,root in [('public',native),('target',target_native),('fresh',measured)]:
            with np.load(root/name/'slots.npz',allow_pickle=False) as z:
                ptr=z['offsets'];v=z['vectors']
                for i,r in enumerate(output_records):
                    if r['native']['route']!=route:continue
                    row=r['native']['row']+(1 if route=='fresh' else 0);check(0<=row<len(ptr)-1,'QASC_GUARD_ADDRESS')
                    n=int(ptr[row+1]-ptr[row]);windows[i]=n
                    if n==1:rows[i]=v[ptr[row]]
        check(np.all(windows>0),'QASC_GUARD_COVERAGE');values.append(rows);available.append(windows==1);counts.append(windows)
    return values,available,counts,instrument

def summarize(traces):
    result={}
    for policy in POLICIES:
        for arm in ('true','gaussian','predicted'):
            for split in ('calibration','evaluation'):
                for route in ('true','predicted'):
                    rr=[x for x in traces if (x['policy'],x['arm'],x['partition'],x['route'])==(policy,arm,split,route)]
                    result['/'.join((policy,arm,split,route))]=dict(rows=len(rr),complete=sum(x['status']=='complete_unasserted' for x in rr),
                        geometric_pass=sum(x['geometric_pass'] for x in rr),exact_provider_target=sum(x['anchor']['exact_provider_target'] for x in rr),
                        joint_geometry_exact_target=sum(x['geometric_pass'] and x['anchor']['exact_provider_target'] for x in rr),
                        geometric_nonexact=sum(x['geometric_pass'] and not x['anchor']['exact_provider_target'] for x in rr),
                        novel_exact_joint=sum(x['geometric_pass'] and x['anchor']['exact_provider_target'] and x['anchor']['not_in_training_targets'] for x in rr),
                        novel_noncopy_exact_joint=sum(x['geometric_pass'] and x['anchor']['exact_provider_target'] and x['anchor']['not_in_training_targets'] and not x['anchor']['copy_of_current_input'] for x in rr),
                        slot_pass={n:sum(x['slots'][n]['passed'] for x in rr) for n in LANES},asserted=0)
    return result

def score(ctx,output_records,candidates,raw,available,window_counts,region,source):
    traces=[];control_rows=[];thresholds={};calibration={}
    for policy in POLICIES:
        rr=ctx['policies'][policy];subset=[r for r in candidates if r['policy']==policy]
        with np.load(region/policy/'codebook.npz',allow_pickle=False) as book,np.load(region/policy/'seed-25/region/outputs.npz',allow_pickle=False) as pred:
            measured=[];valid=[];truth=[];predicted=[]
            for k,name in enumerate(LANES):
                x,ok=unit(raw[k]-book[name+'_mean'].astype(np.float64));measured.append(x);valid.append(ok&available[k])
                y,ok=unit(ctx['truth'][k].astype(np.float64)-book[name+'_mean'].astype(np.float64));check(ok.all(),'QASC_GUARD_TRUTH')
                truth.append(y[ctx['target_index']]);v,ok=unit(pred[f'slots_{k}'].astype(np.float64));check(ok.all(),'QASC_GUARD_PREDICTED');predicted.append(v)
        authentic=[c for c in subset if c['arm']=='true' and c['route']=='true' and c['partition']=='calibration'
                   and c['output'] is not None and all(v[c['output']] for v in valid)]
        calibration[policy]=dict(rows=[c['row'] for c in authentic],groups=len({rr[c['row']]['group'] for c in authentic}),minimum_complete=50,available=len(authentic)>=50)
        taus={name:float(np.quantile([np.dot(measured[k][c['output']],truth[k][c['row']]) for c in authentic],.1,method='linear')) if len(authentic)>=50 else None for k,name in enumerate(LANES)}
        thresholds[policy]=taus;train_text={norm(ctx['texts'][ctx['target_index'][i]]) for i,r in enumerate(rr) if r['partition']=='train'}
        for c in subset:
            row=c['row'];ix=c['output'];text=None if ix is None else output_records[ix]['text'];slots={};wanted=truth if c['route']=='true' else predicted
            for k,name in enumerate(LANES):
                good=ix is not None and valid[k][ix];cos=float(np.clip(np.dot(measured[k][ix],wanted[k][row]),-1,1)) if good else None
                status='measured' if good else c['status'] if ix is None else 'MNLM_GUARD_MULTI_WINDOW' if window_counts[k][ix]!=1 else 'MNLM_GUARD_INVALID_CENTERED_SLOT'
                slots[name]=dict(cosine=cos,threshold=taus[name],passed=cos is not None and taus[name] is not None and cos>=taus[name],status=status)
            target=source[row]['combinedfact'];choices=source[row]['choices'];answer=choices['text'][choices['label'].index(source[row]['answerKey'])]
            word_match=bool(text is not None and re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)',text,re.I))
            anchor=dict(exact_provider_target=text==target,not_in_training_targets=text is not None and norm(text) not in train_text,
                        copy_of_current_input=text is not None and text in [source[row][f] for f in ('question','fact1','fact2')],
                        provider_answer_string_present=word_match,answer_presence_is_correctness=False,
                        target_numbers_equal=text is not None and Counter(re.findall(r'\d+(?:\.\d+)?',text))==Counter(re.findall(r'\d+(?:\.\d+)?',target)),
                        nonexact_semantic_adjudication='unavailable' if text!=target else 'not_applicable_exact_provider_bytes')
            failing=[n for n,v in slots.items() if not v['passed']]
            grounding=assessment(text,target,row)
            reasons=['MNLM_PANEL_NOT_ADMITTED','MNLM_GUARD_IDENTITY_UNCERTIFIED','MNLM_GROUNDING_UNADJUDICATED' if text is not None else 'MNLM_NO_COMPLETE_CLAIM']
            if failing:reasons.append('MNLM_GUARD_SLOT_BELOW_OR_UNAVAILABLE')
            if len(authentic)<50:reasons.append('MNLM_GUARD_CALIBRATION_INSUFFICIENT')
            traces.append({**c,'slots':slots,'geometric_pass':not failing,'anchor':anchor,'grounding':grounding,
                'asserted':False,'verdict':'refuse','reasons':reasons,'failing_slots':failing})
        donor=np.load(region/policy/'donors.npy',allow_pickle=False)
        for c in [x for x in subset if x['arm']=='true' and x['route']=='true']:
            row=c['row']
            for kind,j in [('cached-source-identity',row),('different-group-target',int(donor[row]))]:
                cos={name:float(np.clip(np.dot(truth[k][j],truth[k][row]),-1,1)) for k,name in enumerate(LANES)}
                control_rows.append(dict(policy=policy,partition=c['partition'],row=row,donor_row=j,kind=kind,cosines=cos,
                    geometric_pass=all(taus[n] is not None and v>=taus[n] for n,v in cos.items()),semantic_negative_adjudicated=False,asserted=False))
    return traces,control_rows,thresholds,calibration

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','region-model','surface-model','guard-population','measured','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'QASC_GUARD_SCORE_EXISTS');started=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);region=read_region(a.region_model,ctx);surface=read_surface(a.surface_model,ctx,a.region_model)
    population=read_population(a.guard_population,a.population,a.native,a.targets,a.region_model,a.surface_model);native=read_native(a.measured,a.guard_population)
    fsv=read_json(a.proof/'guard-native-fsv-linux-v1/fsv.json');replay=read_json(a.proof/'guard-native-replay-linux-v1/report.json')
    check(fsv['passed']==fsv['total']==20 and fsv['cold_readback']['report_sha256']==replay['publication_sha256']==native['report_sha256']
          and all(x['all_replayed_vectors_bitwise_equal'] for x in replay['lanes'].values()) and replay['source_and_instruments_unchanged'],'QASC_GUARD_NATIVE_PROOF')
    output_records=records(a.guard_population/'outputs.jsonl');candidates=records(a.guard_population/'candidates.jsonl')
    check(file_sha(a.raw/'qasc-hf-training-v1/train.parquet')==SOURCE_SHA256,'QASC_GROUNDING_SOURCE')
    source=pq.read_table(a.raw/'qasc-hf-training-v1/train.parquet').to_pylist();a.output.mkdir(parents=True)
    phases['source_model_surface_and_native_cold_read']=time.perf_counter()-started;tick=time.perf_counter()
    raw,available,counts,instruments=gather(output_records,a.native,a.target_native,a.measured)
    traces,controls,thresholds,calibration=score(ctx,output_records,candidates,raw,available,counts,a.region_model,source)
    phases['native_gather_calibration_per_slot_and_anchor_diagnostics']=time.perf_counter()-tick;tick=time.perf_counter()
    lines(a.output/'traces.jsonl',traces);lines(a.output/'controls.jsonl',controls);publish(a.output/'thresholds.json',thresholds)
    lines(a.output/'source-grounding.jsonl',source_records(source))
    publish(a.output/'calibration.json',calibration);publish(a.output/'metrics.json',summarize(traces));publish(a.output/'instruments.json',instruments)
    metadata=files(a.output);phases['publication_and_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(schema='qasc-guard-scores-v1',complete=True,errors=[],files=metadata,counts=dict(traces=len(traces),controls=len(controls)),
            region_sha256=region['report_sha256'],surface_sha256=surface['report_sha256'],guard_population_sha256=population['report_sha256'],measured_sha256=native['report_sha256'],
            calibration_policy=POLICY,scientific_admission=False,served=False,semantic_far_certified=False,free_text_composition_proven=False,production_assertions=0,
            source_agreement_is_world_validity=False,grounded_promotions=0,epistemic_distinction_learned=False,
            phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall,metrics=summarize(traces))),flush=True)

if __name__=='__main__':main()

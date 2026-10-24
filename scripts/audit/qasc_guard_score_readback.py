"""Recompute guard thresholds, per-slot verdicts and source anchors from persisted readings."""
from collections import Counter
import math
import re
import numpy as np
from audit.qasc_region_readback import check,read_json,time_check,same
from audit.qasc_surface_readback import records
from model.qasc_context import LANES,POLICIES
from model.guard_score import POLICY
from text.connected_training_census import norm
from text.evaluation_multiscale import file_sha
import hashlib

def read(root,ctx,guard_population,measured,region,surface,source,raw,window_counts):
    r=read_json(root/'report.json');check(r['schema']=='qasc-guard-scores-v1' and r['complete'] and r['errors']==[],'GUARD_STATE');time_check(r)
    check(r['calibration_policy']==POLICY and all(r[k] is False for k in ('scientific_admission','served','semantic_far_certified','free_text_composition_proven')) and r['production_assertions']==0,'GUARD_CLAIMS')
    check(r['source_agreement_is_world_validity'] is False and r['grounded_promotions']==0 and r['epistemic_distinction_learned'] is False,'GUARD_GROUNDING_CLAIMS')
    check(r['region_sha256']==file_sha(region/'report.json') and r['surface_sha256']==file_sha(surface/'report.json')
          and r['guard_population_sha256']==file_sha(guard_population/'report.json') and r['measured_sha256']==file_sha(measured/'report.json'),'GUARD_PARENTS')
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(r['files'])|{'report.json'},'GUARD_FILES')
    for name,pin in r['files'].items():
        p=(root/name).resolve();check(p.is_relative_to(root.resolve()) and p.stat().st_size==pin['bytes'] and file_sha(p)==pin['sha256'],'GUARD_FILE')
    candidates=records(guard_population/'candidates.jsonl');outputs=records(guard_population/'outputs.jsonl');traces=records(root/'traces.jsonl');controls=records(root/'controls.jsonl')
    check(len(traces)==len(candidates)==3072 and len(controls)==1024 and r['counts']==dict(traces=3072,controls=1024),'GUARD_COVERAGE')
    refs=records(root/'source-grounding.jsonl');check(len(refs)==len(source)==8134,'GUARD_GROUNDING_COVERAGE')
    for i,ref in enumerate(refs):
        expected=dict(source='allenai/qasc',revision='a34ba204eb9a33b919c10cc08f4f1c8dae5ec070',artifact='qasc-hf-training-v1/train.parquet',
            artifact_sha256='b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1',row=i,
            fields={name:hashlib.sha256(source[i][name].encode('utf-8')).hexdigest() for name in ('fact1','fact2','combinedfact')},
            statement_kind='provider_assertion',world_validity='unadjudicated',derivation_status='provider_supplied_unchecked',event_time=None,event_time_status='not_provided')
        check(ref==expected,'GUARD_GROUNDING_SOURCE')
    tau_saved=read_json(root/'thresholds.json');cal_saved=read_json(root/'calibration.json');expected_control=[]
    for policy in POLICIES:
        rr=ctx['policies'][policy];subset=[x for x in candidates if x['policy']==policy]
        with np.load(region/policy/'codebook.npz',allow_pickle=False) as b,np.load(region/policy/'seed-25/region/outputs.npz',allow_pickle=False) as p:
            measured_vectors=[];valid=[];truth=[];prediction=[]
            for k,name in enumerate(LANES):
                vector=raw[k]-b[name+'_mean'].astype(float);n=np.sqrt(np.sum(vector*vector,axis=1));ok=np.isfinite(vector).all(1)&(n>1e-12)&(window_counts[k]==1)
                measured_vectors.append(np.divide(vector,n[:,None],out=np.zeros_like(vector),where=(n>1e-12)[:,None]));valid.append(ok)
                v=ctx['truth'][k].astype(float)-b[name+'_mean'].astype(float);v/=np.linalg.norm(v,axis=1)[:,None];truth.append(v[ctx['target_index']])
                v=p[f'slots_{k}'].astype(float);v/=np.linalg.norm(v,axis=1)[:,None];prediction.append(v)
        calibration=[c for c in subset if c['partition']=='calibration' and c['arm']=='true' and c['route']=='true'
                     and c['output'] is not None and all(ok[c['output']] for ok in valid)]
        expected=dict(rows=[c['row'] for c in calibration],groups=len({rr[c['row']]['group'] for c in calibration}),minimum_complete=50,available=len(calibration)>=50)
        check(cal_saved[policy]==expected,'GUARD_CALIBRATION_SCOPE');threshold={}
        for k,name in enumerate(LANES):
            if len(calibration)<50:threshold[name]=None
            else:
                values=sorted(float(np.sum(measured_vectors[k][c['output']]*truth[k][c['row']])) for c in calibration)
                pos=.1*(len(values)-1);low=math.floor(pos);high=math.ceil(pos);threshold[name]=values[low]+(values[high]-values[low])*(pos-low)
        check(same(threshold,tau_saved[policy],1e-12),'GUARD_THRESHOLDS')
        train_text={norm(source[i]['combinedfact']) for i,row in enumerate(rr) if row['partition']=='train'}
        for c in subset:
            t=traces[c['index']];check(all(t[k]==v for k,v in c.items()),'GUARD_TRACE_SOURCE')
            row=c['row'];ix=c['output'];text=None if ix is None else outputs[ix]['text'];wanted=truth if c['route']=='true' else prediction;slots={}
            for k,name in enumerate(LANES):
                good=ix is not None and valid[k][ix];cos=float(np.clip(np.sum(measured_vectors[k][ix]*wanted[k][row]),-1,1)) if good else None
                status='measured' if good else c['status'] if ix is None else 'MNLM_GUARD_MULTI_WINDOW' if window_counts[k][ix]!=1 else 'MNLM_GUARD_INVALID_CENTERED_SLOT'
                slots[name]=dict(cosine=cos,threshold=threshold[name],passed=cos is not None and threshold[name] is not None and cos>=threshold[name],status=status)
            check(same(slots,t['slots'],1e-12),'GUARD_SLOTS')
            fail=[name for name,v in slots.items() if not v['passed']];reasons=['MNLM_PANEL_NOT_ADMITTED','MNLM_GUARD_IDENTITY_UNCERTIFIED','MNLM_GROUNDING_UNADJUDICATED' if text is not None else 'MNLM_NO_COMPLETE_CLAIM']
            if fail:reasons.append('MNLM_GUARD_SLOT_BELOW_OR_UNAVAILABLE')
            if len(calibration)<50:reasons.append('MNLM_GUARD_CALIBRATION_INSUFFICIENT')
            check(t['failing_slots']==fail and t['geometric_pass']==(not fail) and t['verdict']=='refuse' and t['asserted'] is False and t['reasons']==reasons,'GUARD_VERDICT')
            target=source[row]['combinedfact'];opts=source[row]['choices'];answer=opts['text'][opts['label'].index(source[row]['answerKey'])]
            anchor=dict(exact_provider_target=text==target,not_in_training_targets=text is not None and norm(text) not in train_text,
                copy_of_current_input=text is not None and text in [source[row][f] for f in ('question','fact1','fact2')],
                provider_answer_string_present=bool(text is not None and re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)',text,re.I)),answer_presence_is_correctness=False,
                target_numbers_equal=text is not None and Counter(re.findall(r'\d+(?:\.\d+)?',text))==Counter(re.findall(r'\d+(?:\.\d+)?',target)),
                nonexact_semantic_adjudication='unavailable' if text!=target else 'not_applicable_exact_provider_bytes')
            check(anchor==t['anchor'],'GUARD_ANCHOR')
            expected=dict(claim_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest() if text is not None else None,source_record=row,
                source_agreement='exact_provider_assertion' if text==target else 'unassessed' if text is not None else 'no_complete_claim',
                source_agreement_is_world_validity=False,grounding_status='unadjudicated' if text is not None else 'not_applicable_no_claim',
                memory_layer='episodic',promotion_eligible=False,promotion_blockers=['BASE_FACTS_UNADJUDICATED','DERIVATION_UNCHECKED'] if text is not None else ['NO_COMPLETE_CLAIM'],
                truth_value='unknown',independent_grounding_receipts=[],checked_derivation=None,
                generated_claim_link='exact_provider_bytes' if text==target else 'semantic_link_unchecked' if text is not None else 'absent')
            check(t['grounding']==expected,'GUARD_GROUNDING_PROMOTION')
        donors=np.load(region/policy/'donors.npy',allow_pickle=False)
        for c in [x for x in subset if x['arm']=='true' and x['route']=='true']:
            row=c['row']
            for kind,j in [('cached-source-identity',row),('different-group-target',int(donors[row]))]:
                cos={name:float(np.clip(np.sum(truth[k][j]*truth[k][row]),-1,1)) for k,name in enumerate(LANES)}
                expected_control.append(dict(policy=policy,partition=c['partition'],row=row,donor_row=j,kind=kind,cosines=cos,
                    geometric_pass=all(threshold[n] is not None and v>=threshold[n] for n,v in cos.items()),semantic_negative_adjudicated=False,asserted=False))
    check(same(expected_control,controls,1e-12),'GUARD_CONTROLS')
    metrics={}
    for policy in POLICIES:
        for arm in ('true','gaussian','predicted'):
            for split in ('calibration','evaluation'):
                for route in ('true','predicted'):
                    tt=[t for t in traces if (t['policy'],t['arm'],t['partition'],t['route'])==(policy,arm,split,route)]
                    joint=[t for t in tt if t['geometric_pass'] and t['anchor']['exact_provider_target']]
                    metrics['/'.join((policy,arm,split,route))]=dict(rows=len(tt),complete=sum(t['status']=='complete_unasserted' for t in tt),
                        geometric_pass=sum(t['geometric_pass'] for t in tt),exact_provider_target=sum(t['anchor']['exact_provider_target'] for t in tt),
                        joint_geometry_exact_target=len(joint),geometric_nonexact=sum(t['geometric_pass'] and not t['anchor']['exact_provider_target'] for t in tt),
                        novel_exact_joint=sum(t['anchor']['not_in_training_targets'] for t in joint),novel_noncopy_exact_joint=sum(t['anchor']['not_in_training_targets'] and not t['anchor']['copy_of_current_input'] for t in joint),
                        slot_pass={name:sum(t['slots'][name]['passed'] for t in tt) for name in LANES},asserted=0)
    check(metrics==read_json(root/'metrics.json'),'GUARD_METRICS')
    check(read_json(root/'instruments.json')=={name:read_json(measured/name/'report.json')['lens_id'] for name in LANES},'GUARD_INSTRUMENTS')
    return dict(report_sha256=file_sha(root/'report.json'),all_3072_slot_and_anchor_traces_reconstructed=True,all_1024_controls_reconstructed=True,
                calibration_only_thresholds_reconstructed=True,metrics=metrics,scientific_admission=False,served=False,acceptance='author-checked')

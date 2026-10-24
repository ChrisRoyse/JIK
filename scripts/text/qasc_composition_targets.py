"""Private source-bound composition supervision with exact reuse of native readings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_context import POPULATION,NATIVE,POLICIES,LANES,rows
from audit.connected_population_readback import readback as population_read
from audit.window_readback import readback as native_read
from text.connected_training_census import sha,norm
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from senses.contract import canonical

def check(ok,code):
    if not ok:raise ValueError('MNLM_COMPOSITION_TARGET_'+code)

def jsonl(path,values):
    with path.open('xb') as f:
        for v in values:f.write(canonical(v)+b'\n')

def project(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    check(file_sha(a.population/'report.json')==POPULATION and file_sha(a.native/'report.json')==NATIVE,'PARENT_PIN')
    source=population_read(a.population,a.raw/'multirc-training-v1',a.raw/'qasc-hf-training-v1',a.raw/'multirc-v1');native=native_read(a.native,a.population)
    check(tuple(sorted(native['lenses']))==LANES,'ROSTER');public=pq.read_table(a.population/'connected/chunks/atoms.parquet').to_pylist()
    ref={r['id']:r for r in rows(a.population/'references.jsonl')};cases=[r for r in rows(a.population/'cases.jsonl') if r['family']=='qasc']
    private={r['case']:r for r in rows(a.population/'private/targets.jsonl') if r['family']=='qasc'}
    check(len(cases)==len(private)==8134,'CASES');public_lookup={r['entity']:r['row'] for r in public}
    phases['pinned_population_and_native_readback']=time.perf_counter()-start;tick=time.perf_counter()
    unique=[];lookup={};fresh=[];mapping=[]
    for c in cases:
        text=private[c['id']]['combined_statement'];key=sha(text);check(bool(text.strip()) and key==private[c['id']]['combined_sha256'],'TARGET_TEXT')
        if key not in lookup:
            if key in public_lookup:
                address=dict(route='public',family='connected',row=public_lookup[key]);check(public[address['row']]['text']==text,'CACHE_COLLISION')
            else:
                address=dict(route='fresh',family='targets',row=len(fresh));fresh.append(dict(row=len(fresh),chunk=0,entity=key,text=text))
            lookup[key]=len(unique);unique.append(dict(index=len(unique),text=text,sha256=key,utf8_bytes=len(text.encode()),native=address))
        target=unique[lookup[key]];check(target['text']==text,'TARGET_COLLISION')
        question=ref[c['question']]['unit'];facts=[ref[r]['unit'] for r in c['facts']]
        mapping.append(dict(case=c['id'],source_row=c['source_row'],source_field='combinedfact',target=target['index'],question_unit=question,fact_units=facts,
                            equals_current_fact=any(public[u]['text']==text for u in facts),equals_current_question=public[question]['text']==text,surface_codec_available=target['utf8_bytes']<=512))
    candidates=[r for r in ref.values() if r['family']=='multirc' and r['role']=='paragraph' and r['unit'] is not None and 1800<=len(public[r['unit']]['text'])<=3000]
    check(bool(candidates),'REAL_PROBE');pr=candidates[0];probe=dict(public_unit=pr['unit'],reference=pr['id'],text_sha256=pr['text_sha256'],role='probe-only',model_input=False,target=False)
    assignments={(r['policy'],r['case']):r for r in rows(a.population/'assignments.jsonl') if r['policy'] in POLICIES};policies={}
    for policy in POLICIES:
        rr=[dict(case=c['case'],**{k:v for k,v in assignments[policy,c['case']].items() if k not in ('policy','case')}) for c in mapping]
        train=[i for i,r in enumerate(rr) if r['partition']=='train'];inputs={norm(public[u]['text']) for i in train for u in [mapping[i]['question_unit'],*mapping[i]['fact_units']]}
        target_texts={norm(unique[mapping[i]['target']]['text']) for i in train}
        for i,r in enumerate(rr):
            text=norm(unique[mapping[i]['target']]['text']);r['target_in_training_generation_inputs']=text in inputs;r['target_in_training_targets']=text in target_texts
        check(not any(r['target_in_training_targets'] for r in rr if r['partition']!='train'),'TARGET_SPLIT_LEAKAGE');policies[policy]=rr
    phases['target_identity_cache_and_exposure_maps']=time.perf_counter()-tick;tick=time.perf_counter()
    jsonl(a.output/'targets.jsonl',unique);jsonl(a.output/'cases.jsonl',mapping);publish(a.output/'policies.json',policies);publish(a.output/'probe.json',probe)
    families={}
    for family,table in [('probe',[dict(row=0,chunk=0,entity=probe['text_sha256'],text=public[probe['public_unit']]['text'])]),('targets',fresh)]:
        root=a.output/family;(root/'chunks').mkdir(parents=True)
        check(bool(table),'NO_NEW_TARGETS');pq.write_table(pa.Table.from_pylist(table),root/'chunks/atoms.parquet',compression='zstd')
        publish(root/'summary.json',dict(chunks=len(table),chunks_sha256=file_sha(root/'chunks/atoms.parquet')));families[family]=dict(ready=True,summary_sha256=file_sha(root/'summary.json'))
    counts=dict(cases=len(mapping),unique_targets=len(unique),fresh_targets=len(fresh),cached_targets=len(unique)-len(fresh),measurement_inputs=len(fresh)+1,
                target_characters=sum(len(r['text']) for r in unique),fresh_characters=sum(len(r['text']) for r in fresh),probe_characters=len(public[probe['public_unit']]['text']),
                max_target_utf8_bytes=max(r['utf8_bytes'] for r in unique),over_capacity_cases=sum(not r['surface_codec_available'] for r in mapping),
                equals_current_fact_cases=sum(r['equals_current_fact'] for r in mapping),equals_current_question_cases=sum(r['equals_current_question'] for r in mapping),
                exposures={p:{s:dict(cases=sum(r['partition']==s for r in rr),targets_seen_as_training_inputs=sum(r['partition']==s and r['target_in_training_generation_inputs'] for r in rr)) for s in ('train','calibration','evaluation')} for p,rr in policies.items()})
    phases['source_bound_target_publication']=time.perf_counter()-tick;tick=time.perf_counter()
    check(file_sha(a.population/'report.json')==POPULATION and file_sha(a.native/'report.json')==NATIVE,'PARENT_DRIFT')
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='qasc-composition-targets-v1',complete=True,errors=[],calibration_only=True,families=families,files=files,counts=counts,
                population_sha256=POPULATION,public_native_sha256=NATIVE,source_parquet_sha256=file_sha(a.raw/'qasc-hf-training-v1/train.parquet'),lanes=list(LANES),
                intended_role='private supervision; never predictor input',probe_is_target=False,scientific_admission=False,served=False,sealed_evaluation=False,
                source_supplied_facts=True,free_text_composition_proven=False,code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','raw','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:r=project(a)
    except Exception as e:
        if a.output.exists():publish(a.output/'failure.json',dict(complete=False,error=str(e),type=type(e).__name__))
        raise
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),counts=r['counts'],wall_seconds=r['wall_seconds'])))

if __name__=='__main__':main()

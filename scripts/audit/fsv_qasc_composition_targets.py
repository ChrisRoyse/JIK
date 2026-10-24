"""Corrupt-copy proof of source/target, cache and private/probe boundaries."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.qasc_composition_target_readback import read,check
from model.qasc_context import rows
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

def main():
    p=argparse.ArgumentParser()
    for n in ('population','native','raw','targets','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    cold=read(a.targets,a.population,a.native,a.raw);publish(a.output/'cold.json',cold)
    originals={str(p):file_sha(p) for p in [a.targets/'report.json',*[p for p in a.targets.rglob('*') if p.is_file() and p.name!='report.json'],a.population/'report.json',a.native/'report.json',a.raw/'qasc-hf-training-v1/train.parquet']}
    shadow=a.output/'shadow';shutil.copytree(a.targets,shadow);phases['original_cold_and_shadow']=time.perf_counter()-start;tick=time.perf_counter();checks=[]
    def probe(name,file,mutate,code):
        path=shadow/file;old=path.read_bytes();header=(shadow/'report.json').read_bytes();before=file_sha(path)
        value=json.loads(old) if file.endswith('.json') else rows(path);mutate(value)
        check(path.resolve().is_relative_to(shadow.resolve()),'FSV_SHADOW')
        path.write_text(json.dumps(value,sort_keys=True) if file.endswith('.json') else ''.join(json.dumps(v,sort_keys=True)+'\n' for v in value),encoding='utf-8');action=file_sha(path)
        if file!='report.json':
            r=json.loads(header);r['files'][file]=dict(bytes=path.stat().st_size,sha256=file_sha(path));(shadow/'report.json').write_text(json.dumps(r,sort_keys=True),encoding='utf-8')
        try:read(shadow,a.population,a.native,a.raw)
        except ValueError as exc:check(code in str(exc),'FSV_WRONG_REFUSAL');error=str(exc)
        else:raise ValueError('MNLM_TARGET_READBACK_FSV_ACCEPTED')
        path.write_bytes(old);(shadow/'report.json').write_bytes(header);check(file_sha(path)==before,'FSV_RESTORE')
        checks.append(dict(name=name,boundary=file,before_sha256=before,action_sha256=action,after_sha256=file_sha(path),error=error,restored=True))
    probe('omitted target case','cases.jsonl',lambda v:v.pop(),'CASE_COVERAGE')
    probe('target inserted into predictor metadata','cases.jsonl',lambda v:v[0].update(target_text='not an allowed input field'),'CASE_SCHEMA')
    probe('wrong public question address','cases.jsonl',lambda v:v[0].update(question_unit=(v[0]['question_unit']+1)%63285),'INPUT_ADDRESS')
    probe('changed private target','targets.jsonl',lambda v:v[0].update(text=v[0]['text']+' changed'),'RAW_TARGET')
    def wrong_address(v):
        t=next(t for t in v if t['native']['route']=='fresh');t['native']['family']='probe';t['native']['row']=0
    probe('probe promoted to target','targets.jsonl',wrong_address,'FRESH_ADDRESS')
    cached=[t for t in rows(a.targets/'targets.jsonl') if t['native']['route']=='public']
    if cached:
        def wrong_cache(v):
            t=next(t for t in v if t['native']['route']=='public');t['native']['row']=(t['native']['row']+1)%63285
        probe('wrong cached native address','targets.jsonl',wrong_cache,'CACHE_ADDRESS')
    probe('probe claimed as model input','probe.json',lambda v:v.update(model_input=True),'PROBE_SCOPE')
    probe('changed original source pin','report.json',lambda v:v.update(source_parquet_sha256='0'*64),'RAW_PIN')
    probe('false public serving claim','report.json',lambda v:v.update(served=True),'CLAIMS')
    probe('inflated target count','report.json',lambda v:v['counts'].update(cases=8135),'COUNTS')
    phases['negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,a.population,a.native,a.raw)==cold,'FSV_FINAL_RESTORE');after={n:file_sha(Path(n)) for n in originals};check(originals==after,'FSV_ORIGINAL_DRIFT')
    publish(a.output/'before.json',originals);publish(a.output/'after.json',after);phases['restored_and_immutable_originals']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(passed=True,checks=checks,targets_sha256=cold['report_sha256'],scientific_admission=False,served=False,acceptance='author-checked',
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(passed=True,controls=len(checks),sha256=file_sha(a.output/'report.json'),seconds=wall)))

if __name__=='__main__':main()

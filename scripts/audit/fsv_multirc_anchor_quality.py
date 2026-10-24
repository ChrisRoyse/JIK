"""Raw-source, contradictory-label and exclusion-boundary FSV for quality overlay."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.multirc_anchor_quality import audit_raw,check
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def main():
    p=argparse.ArgumentParser()
    for n in ('population','raw','quality','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};source=a.raw/'multirc-training-v1/train.json'
    r=json.loads((a.quality/'report.json').read_bytes());check(r['schema']=='multirc-anchor-quality-v1' and r['scientific_admission'] is False and r['model_input'] is False,'FSV_SCOPE')
    for n,pin in r['before'].items():check(file_sha(Path(n))==pin,'FSV_PARENT')
    for n,pin in r['files'].items():check(file_sha(a.quality/n)==pin,'FSV_FILE')
    names=('conflicts','excluded-cases','contrasts');data=[json.loads((a.quality/(n+'.json')).read_bytes()) for n in names]
    baseline=audit_raw(a.population,source,*data);check(baseline==r['raw_readback'],'FSV_BASELINE')
    original={str(p):file_sha(p) for p in [a.quality/'report.json',*[a.quality/(n+'.json') for n in names],source]}
    phases['pinned_original_readback']=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/'shadow';shadow.mkdir();checks=[]
    for n in names:shutil.copyfile(a.quality/(n+'.json'),shadow/(n+'.json'))
    shutil.copyfile(source,shadow/'train.json')
    def probe(name,file,change,error):
        p=shadow/file;before=p.read_bytes();v=json.loads(before);change(v);p.write_text(json.dumps(v,sort_keys=True),encoding='utf-8')
        action=file_sha(p)
        try:audit_raw(a.population,shadow/'train.json',*[json.loads((shadow/(n+'.json')).read_bytes()) for n in names])
        except ValueError as exc:check(error in str(exc),'FSV_ERROR');message=str(exc)
        else:raise ValueError('MNLM_MULTIRC_QUALITY_FSV_ACCEPTED')
        p.write_bytes(before);check(p.read_bytes()==before,'FSV_RESTORE')
        checks.append(dict(name=name,boundary=file,before_sha256=file_sha(p),action_sha256=action,after_sha256=file_sha(p),error=message,restored=True))
    first=data[0][0]['members'][0]
    def raw_change(v):v['data'][first['row']]['paragraph']['questions'][first['source_question']]['question']+=' altered'
    probe('altered original question','train.json',raw_change,'RAW_CONFLICTS')
    def label_change(v):v[0]['members'][0]['label']=not v[0]['members'][0]['label']
    probe('altered conflict anchor','conflicts.json',label_change,'RAW_CONFLICTS')
    probe('omitted conflicted case','excluded-cases.json',lambda v:v.pop(),'RAW_EXCLUSIONS')
    probe('omitted genuine question contrast','contrasts.json',lambda v:v.pop(),'RAW_CONTRASTS')
    phases['owned_shadow_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    restored=audit_raw(a.population,shadow/'train.json',*[json.loads((shadow/(n+'.json')).read_bytes()) for n in names]);check(restored==baseline,'FSV_FINAL')
    after={n:file_sha(Path(n)) for n in original};check(original==after,'FSV_ORIGINALS')
    phases['restored_readback_and_immutable_originals']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(passed=True,checks=checks,quality_sha256=file_sha(a.quality/'report.json'),raw_readback=baseline,before=original,after=after,
        acceptance='author-checked',scientific_admission=False,served=False,wall_seconds=wall,phases_seconds=phases,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(passed=True,controls=len(checks),sha256=file_sha(a.output/'report.json'))))

if __name__=='__main__':main()

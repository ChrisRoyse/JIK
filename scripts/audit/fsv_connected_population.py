"""Real population readback and bounded source/view/split adversarial controls."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.connected_population_readback import readback,bound_result,read_data,verify_views,digest,check,rows
from audit.connected_census_readback import sources
from audit.connected_split_readback import read_splits
from senses.trajectory_population import publish


def main():
    p=argparse.ArgumentParser()
    for k in ('population','multirc','qasc','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};tick=time.perf_counter();checks=[]
    originals={k:{v.name:digest(v) for v in getattr(a,k).iterdir() if v.is_file()} for k in ('multirc','qasc','parent')}
    original_files={p.relative_to(a.population).as_posix():digest(p) for p in a.population.rglob('*') if p.is_file()}
    baseline=readback(a.population,a.multirc,a.qasc,a.parent);publish(a.output/'cold-readback.json',baseline)
    multi,qasc=sources(a.multirc,a.qasc,a.parent);shadow=a.output/'shadow';shadow.mkdir();result=shadow/'population'
    shutil.copytree(a.population,result)
    for k in ('multirc','qasc','parent'):shutil.copytree(getattr(a,k),shadow/k)
    phases['original_cold_read_and_owned_shadow']=time.perf_counter()-tick;tick=time.perf_counter()
    def mutate(name,path,transform,action,error,rebind=False):
        check(path.resolve().is_relative_to(shadow.resolve()),'SHADOW_BOUNDARY');before=path.read_bytes();pin=digest(path);report=result/'report.json';saved=report.read_bytes()
        modified=transform(before);check(before!=modified,'MUTATION_NOOP');path.write_bytes(modified);changed=digest(path)
        if rebind:
            r=json.loads(saved);r['files'][path.relative_to(result).as_posix()]=dict(bytes=path.stat().st_size,sha256=changed);report.write_text(json.dumps(r),encoding='utf-8')
        observed=None
        try:action()
        except ValueError as exc:observed=str(exc)
        finally:
            path.write_bytes(before)
            if rebind:report.write_bytes(saved)
        check(observed==error,'CONTROL_'+name);check(digest(path)==pin,'RESTORE_'+name)
        checks.append(dict(name=name,before_sha256=pin,action_sha256=changed,error=observed,after_sha256=digest(path),restored=True))
    def mutate_json(fn,jsonl=False):
        def transform(b):
            x=[json.loads(line) for line in b.splitlines()] if jsonl else json.loads(b);fn(x)
            return ('\n'.join(json.dumps(r) for r in x)+'\n').encode() if jsonl else json.dumps(x).encode()
        return transform
    def view_check():bound_result(result);return verify_views(read_data(result),multi,qasc)
    def split_check():bound_result(result);return read_splits(result,read_data(result))
    mutate('raw_source_tamper',shadow/'qasc/train.parquet',lambda b:b[:-1],lambda:sources(shadow/'multirc',shadow/'qasc',shadow/'parent'),'MNLM_CONNECTED_READ_SOURCE_BYTES')
    mutate('false_source_binding',result/'report.json',mutate_json(lambda x:x['source_pins'].update(qasc='0'*64)),lambda:bound_result(result),'MNLM_CONNECTED_POP_READ_SOURCE_PINS')
    phases['source_boundary']=time.perf_counter()-tick;tick=time.perf_counter()
    mut=[('label_in_observation','references.jsonl',lambda x:x[0].update(isAnswer=True),'PUBLIC_REF_KEYS'),
         ('wrong_surface_unit','references.jsonl',lambda x:x[0].update(unit=1),'SOURCE_TEXT'),
         ('hierarchy_edge_lost','documents.jsonl',lambda x:x[0]['edges'].pop(),'HIERARCHY'),
         ('private_target_changed','private/targets.jsonl',lambda x:x[0]['labels'].__setitem__(0,not x[0]['labels'][0]),'PRIVATE_TARGETS'),
         ('empty_surface_claimed','references.jsonl',lambda x:next(r for r in x if r['unit'] is None).update(unit=0),'EMPTY_MEASURED')]
    for name,file,fn,error in mut:mutate(name,result/file,mutate_json(fn,True),view_check,'MNLM_CONNECTED_POP_READ_'+error,True)
    phases['observation_target_hierarchy_boundary']=time.perf_counter()-tick;tick=time.perf_counter()
    mutate('owner_binding_changed',result/'cohorts/qasc-fact/owners.json',mutate_json(lambda x:x[0]['owners'].__setitem__(0,'not-a-source-case')),split_check,'MNLM_CONNECTED_POP_READ_CATALOG_OWNERS',True)
    mutate('shared_fact_link_removed',result/'partitions.json',mutate_json(lambda x:x['qasc-fact-disjoint']['links'].pop()),split_check,'MNLM_CONNECTED_POP_READ_PARTITIONS',True)
    mutate('evaluation_moved_to_training',result/'assignments.jsonl',mutate_json(lambda x:next(r for r in x if r['partition']=='evaluation').update(partition='train'),True),split_check,'MNLM_CONNECTED_POP_READ_ASSIGNMENTS',True)
    mutate('false_scientific_admission',result/'report.json',mutate_json(lambda x:x.update(scientific_admission=True)),lambda:bound_result(result),'MNLM_CONNECTED_POP_READ_FALSE_CLAIM_scientific_admission')
    phases['leakage_split_claim_boundary']=time.perf_counter()-tick;tick=time.perf_counter()
    restored=readback(result,shadow/'multirc',shadow/'qasc',shadow/'parent');check({k:v for k,v in restored.items() if k!='wall_seconds'}=={k:v for k,v in baseline.items() if k!='wall_seconds'},'RESTORED_COLD_READ')
    after={k:{v.name:digest(v) for v in getattr(a,k).iterdir() if v.is_file()} for k in originals};check(after==originals,'ORIGINAL_SOURCE_DRIFT')
    check({p.relative_to(a.population).as_posix():digest(p) for p in a.population.rglob('*') if p.is_file()}==original_files,'ORIGINAL_POPULATION_DRIFT')
    phases['restored_cold_read_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=True,checks=checks,before_sources=originals,after_sources=after,population_files=original_files,baseline=baseline,
        code_sha256={n:digest(Path(__file__).parent/n) for n in ('fsv_connected_population.py','connected_population_readback.py','connected_split_readback.py')},
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,measurements='author-checked',scientific_admission=False)
    publish(a.output/'report.json',result);print(json.dumps(dict(passed=True,checks=len(checks),seconds=wall,report_sha256=digest(a.output/'report.json'))))


if __name__=='__main__':main()

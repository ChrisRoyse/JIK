"""Physical corrupt-copy proof of rendered source, native address and probe boundaries."""
import argparse,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.qasc_guard_population_readback import read
from audit.qasc_surface_readback import records
from audit.qasc_region_readback import check,read_json
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def main():
    p=argparse.ArgumentParser()
    for n in ('population','native','targets','region-model','surface-model','guard-population','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    args=(a.population,a.native,a.targets,a.region_model,a.surface_model)
    cold=read(a.guard_population,*args);publish(a.output/'cold.json',cold)
    before={str(p):file_sha(p) for p in a.guard_population.rglob('*') if p.is_file()}
    before.update({str(x/'report.json'):file_sha(x/'report.json') for x in args})
    shadow=a.output/'shadow';shutil.copytree(a.guard_population,shadow);phases['cold_and_shadow']=time.perf_counter()-start;tick=time.perf_counter();controls=[]
    def control(name,rel,change,expected):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()}
        old=file_sha(path)
        try:
            value=records(path) if rel.endswith('.jsonl') else read_json(path);change(value)
            path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in value) if rel.endswith('.jsonl') else json.dumps(value,sort_keys=True),encoding='utf-8')
            action=file_sha(path);check(old!=action,'GUARD_POP_FSV_NOOP')
            if rel!='report.json':
                r=read_json(shadow/'report.json');r['files'][rel]=dict(bytes=path.stat().st_size,sha256=action)
                (shadow/'report.json').write_text(json.dumps(r,sort_keys=True),encoding='utf-8')
            error=None
            try:read(shadow,*args)
            except ValueError as e:error=str(e)
            check(error is not None and expected in error,'GUARD_POP_FSV_'+name)
        finally:
            for f,raw in saved.items():f.write_bytes(raw)
            check(all(f.read_bytes()==raw for f,raw in saved.items()),'GUARD_POP_FSV_RESTORE')
        controls.append(dict(name=name,boundary=rel,before_sha256=old,action_sha256=action,after_sha256=file_sha(path),observed=error,refused=True))
    control('altered output bytes','outputs.jsonl',lambda v:v[0].update(text=v[0]['text']+'x'),'OUTPUT_TEXT')
    control('wrong native row','outputs.jsonl',lambda v:v[0]['native'].update(row=(v[0]['native']['row']+1)%2991),'FRESH_TEXT')
    control('dropped failed attempt','candidates.jsonl',lambda v:v.pop(),'CANDIDATES')
    control('asserted candidate','candidates.jsonl',lambda v:v[0].update(asserted=True),'SOURCE_FIELDS')
    control('probe used as target','probe.json',lambda v:v.update(target=True),'PROBE_SCOPE')
    control('false admission','report.json',lambda v:v.update(scientific_admission=True),'CLAIMS')
    phases['negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,*args)==cold,'GUARD_POP_FSV_FINAL');after={n:file_sha(Path(n)) for n in before};check(before==after,'GUARD_POP_FSV_ORIGINALS')
    publish(a.output/'before.json',before);publish(a.output/'after.json',after);phases['restored_readback']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked',scientific_admission=False,served=False))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),controls=len(controls),cold=cold)),flush=True)

if __name__=='__main__':main()

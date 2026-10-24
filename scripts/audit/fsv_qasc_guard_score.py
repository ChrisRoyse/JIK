"""Cold-read guard geometry, source provenance and prohibited memory promotion."""
import argparse,json,shutil,sys,time
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_generation_data import load
from model.qasc_guard_score import gather
from audit.qasc_region_readback import read as read_region,read_json,check
from audit.qasc_surface_readback import read as read_surface,records
from audit.qasc_guard_population_readback import read as read_population
from audit.window_readback import readback as read_native
from audit.qasc_guard_score_readback import read
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def main():
    p=argparse.ArgumentParser()
    for n in ('population','native','targets','target-native','raw','proof','region-model','surface-model','guard-population','measured','score','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);read_region(a.region_model,ctx);read_surface(a.surface_model,ctx,a.region_model)
    read_population(a.guard_population,a.population,a.native,a.targets,a.region_model,a.surface_model);read_native(a.measured,a.guard_population)
    source=pq.read_table(a.raw/'qasc-hf-training-v1/train.parquet').to_pylist();raw,_,counts,_=gather(records(a.guard_population/'outputs.jsonl'),a.native,a.target_native,a.measured)
    args=(ctx,a.guard_population,a.measured,a.region_model,a.surface_model,source,raw,counts)
    cold=read(a.score,*args);publish(a.output/'cold.json',cold)
    before=dict(ctx['before'])
    for root in (a.region_model,a.surface_model,a.guard_population,a.measured,a.score):
        before.update({str(p):file_sha(p) for p in root.rglob('*') if p.is_file()})
    phases['all_source_and_output_cold_read']=time.perf_counter()-started;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.score,shadow);controls=[]
    phases['owned_shadow_copy']=time.perf_counter()-tick;tick=time.perf_counter()
    def control(name,rel,change,expected):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()};old=file_sha(path)
        try:
            value=records(path) if rel.endswith('.jsonl') else read_json(path);change(value)
            path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in value) if rel.endswith('.jsonl') else json.dumps(value,sort_keys=True),encoding='utf-8')
            action=file_sha(path);check(action!=old,'GUARD_SCORE_FSV_NOOP')
            if rel!='report.json':
                r=read_json(shadow/'report.json');r['files'][rel]=dict(bytes=path.stat().st_size,sha256=action)
                (shadow/'report.json').write_text(json.dumps(r,sort_keys=True),encoding='utf-8')
            error=None
            try:read(shadow,*args)
            except ValueError as e:error=str(e)
            check(error is not None and expected in error,'GUARD_SCORE_FSV_'+name)
        finally:
            for f,blob in saved.items():f.write_bytes(blob)
            check(all(f.read_bytes()==blob for f,blob in saved.items()),'GUARD_SCORE_FSV_RESTORE')
        controls.append(dict(name=name,boundary=rel,before_sha256=old,action_sha256=action,after_sha256=file_sha(path),observed=error,refused=True))
    control('changed threshold','thresholds.json',lambda v:v['qasc-composition'].update({'all-mpnet-base-v2':-1.0}),'GUARD_THRESHOLDS')
    control('wrong calibration row','calibration.json',lambda v:v['qasc-composition']['rows'].__setitem__(0,-1),'GUARD_CALIBRATION_SCOPE')
    control('fabricated slot','traces.jsonl',lambda v:v[0]['slots']['all-mpnet-base-v2'].update(cosine=-2.0),'GUARD_SLOTS')
    control('false assertion','traces.jsonl',lambda v:v[0].update(asserted=True),'GUARD_TRACE_SOURCE')
    control('wrong source provenance','source-grounding.jsonl',lambda v:v[0].update(row=1),'GUARD_GROUNDING_SOURCE')
    control('source validity invented','source-grounding.jsonl',lambda v:v[0].update(world_validity='grounded'),'GUARD_GROUNDING_SOURCE')
    control('episode promoted without proof','traces.jsonl',lambda v:v[0]['grounding'].update(memory_layer='grounded',promotion_eligible=True),'GUARD_GROUNDING_PROMOTION')
    control('unknown equated with false','traces.jsonl',lambda v:v[0]['grounding'].update(truth_value='false'),'GUARD_GROUNDING_PROMOTION')
    control('different group called false','controls.jsonl',lambda v:v[1].update(semantic_negative_adjudicated=True),'GUARD_CONTROLS')
    control('false learned epistemics','report.json',lambda v:v.update(epistemic_distinction_learned=True),'GUARD_GROUNDING_CLAIMS')
    control('missing trace','traces.jsonl',lambda v:v.pop(),'GUARD_COVERAGE')
    phases['eleven_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,*args)==cold,'GUARD_SCORE_FSV_FINAL');after={n:file_sha(Path(n)) for n in before};check(before==after,'GUARD_SCORE_FSV_ORIGINALS')
    publish(a.output/'before.json',before);publish(a.output/'after.json',after);phases['restored_cold_read_and_source_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked',scientific_admission=False,served=False))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),controls=len(controls),wall_seconds=wall,metrics=cold['metrics'])),flush=True)

if __name__=='__main__':main()

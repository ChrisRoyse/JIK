"""Real persisted predictor readback and restored, internally rehashed negative controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_generation_data import load
from audit.qasc_region_readback import read,check,read_json
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def write(path,value):path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'FSV_EXISTS');started=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);a.output.mkdir(parents=True)
    before={**ctx['before'],str(a.model/'report.json'):file_sha(a.model/'report.json')};publish(a.output/'before.json',before)
    cold=read(a.model,ctx);publish(a.output/'cold.json',cold);phases['source_and_result_cold_read']=time.perf_counter()-started;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.model,shadow);phases['owned_shadow_copy']=time.perf_counter()-tick;tick=time.perf_counter();controls=[]
    def control(name,rel,change,code):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()}
        child=None
        if len(Path(rel).parts)==4:
            child=path.parent/'report.json';saved[child]=child.read_bytes()
        try:
            if rel.endswith('.npz'):
                with np.load(path,allow_pickle=False) as z:value={n:z[n].copy() for n in z.files}
                change(value);np.savez(path,**value)
            elif rel.endswith('.npy'):
                value=np.load(path,allow_pickle=False);change(value);np.save(path,value)
            else:value=read_json(path);change(value);write(path,value)
            if rel!='report.json':
                report=read_json(shadow/'report.json');report['files'][rel]=dict(bytes=path.stat().st_size,sha256=file_sha(path))
                if child:
                    sub=read_json(child);sub['files'][path.name]=report['files'][rel];write(child,sub)
                    key=child.relative_to(shadow).as_posix();report['files'][key]=dict(bytes=child.stat().st_size,sha256=file_sha(child))
                    report['results'][child.parent.relative_to(shadow).as_posix()]=file_sha(child)
                write(shadow/'report.json',report)
            observed=None
            try:read(shadow,ctx)
            except ValueError as e:observed=str(e)
            check(observed is not None and code in observed,'CONTROL_'+name)
            controls.append(dict(name=name,expected=code,observed=observed,refused=True))
        finally:
            for p,body in saved.items():p.write_bytes(body)
            check(all(p.read_bytes()==body for p,body in saved.items()),'RESTORE')
    base='qasc-composition/seed-25/region/'
    control('private target/input map','gather.npz',lambda z:z['original'].__setitem__((0,0),z['original'][0,0]+1),'INPUT_TARGET_MAP')
    control('training-only centering','qasc-composition/codebook.npz',lambda z:z['all-mpnet-base-v2_mean'].__setitem__(0,z['all-mpnet-base-v2_mean'][0]+.1),'BOOK_MEAN_TRAIN_ONLY')
    heldout=next(i for i,r in enumerate(ctx['policies']['qasc-composition']) if r['partition']=='evaluation')
    control('held-out training row','qasc-composition/seed-25/training-rows.npy',lambda z:z.__setitem__((0,0),heldout),'TRAIN_ORDER')
    control('objective target replaced',base+'objective.npz',lambda z:z['targets'].__setitem__(0,(z['targets'][0]+1)%7883),'OBJECTIVE_TARGETS')
    control('zero gradients',base+'objective.json',lambda z:z.update(gradients={k:0 for k in z['gradients']}),'GRADIENTS')
    control('nonfinite prediction',base+'outputs.npz',lambda z:z['slots_0'].__setitem__((0,0),np.nan),'OUTPUT_VALUES')
    control('inflated metric',base+'metrics.json',lambda z:z['primary']['evaluation']['centered_cosine_by_sense'].__setitem__(0,1.5),'METRICS')
    control('instrument substituted','lanes.json',lambda z:z['all-mpnet-base-v2'].update(lens_id='0'*64),'SOURCE_SCOPE')
    control('false serving','report.json',lambda z:z.update(served=True),'CLAIMS')
    control('teacher-forced claim','report.json',lambda z:z.update(teacher_forced_regions=True),'CLAIMS')
    control('missing fit','report.json',lambda z:z['results'].pop('qasc-composition/seed-25/direct'),'FIT_COVERAGE')
    phases['persisted_corruption_refusals_and_restorations']=time.perf_counter()-tick;tick=time.perf_counter()
    restored=read(shadow,ctx);check(restored==cold,'RESTORED_READBACK')
    after={p:file_sha(Path(p)) for p in before};check(before==after,'ORIGINAL_DRIFT');publish(a.output/'after.json',after)
    phases['restored_full_read_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(acceptance='author-checked',source_report_sha256=cold['report_sha256'],cold_sha256=file_sha(a.output/'cold.json'),
            controls=controls,originals_unchanged=True,shadow_restored=True,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
            scientific_admission=False,served=False,code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(controls=len(controls),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':main()

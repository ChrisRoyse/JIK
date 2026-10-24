"""Source, conditioning and strict-surface corruption controls on restored owned copies."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_generation_data import load
from audit.qasc_region_readback import read as read_region,read_json,check
from audit.qasc_surface_readback import read,records
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def write(p,value):p.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','region-model','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'SURFACE_FSV_EXISTS');started=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);read_region(a.region_model,ctx)
    a.output.mkdir(parents=True);cold=read(a.model,ctx,a.region_model);publish(a.output/'cold.json',cold)
    before={**ctx['before'],str(a.region_model/'report.json'):file_sha(a.region_model/'report.json'),str(a.model/'report.json'):file_sha(a.model/'report.json')}
    publish(a.output/'before.json',before);phases['input_and_output_cold_read']=time.perf_counter()-started;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.model,shadow);phases['owned_shadow_copy']=time.perf_counter()-tick;tick=time.perf_counter();controls=[]
    def control(name,rel,change,expected):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()}
        child=path.parent/'report.json' if len(Path(rel).parts)==3 else None
        if child:saved[child]=child.read_bytes()
        try:
            if rel.endswith('.npz'):
                with np.load(path,allow_pickle=False) as z:value={n:z[n].copy() for n in z.files}
                change(value);np.savez(path,**value)
            elif rel.endswith('.npy'):
                value=np.load(path,allow_pickle=False);change(value);np.save(path,value)
            elif rel.endswith('.jsonl'):
                value=records(path);change(value);path.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in value),encoding='utf-8')
            else:value=read_json(path);change(value);write(path,value)
            if rel!='report.json':
                report=read_json(shadow/'report.json');report['files'][rel]=dict(bytes=path.stat().st_size,sha256=file_sha(path))
                if child:
                    sub=read_json(child);sub['files'][path.name]=report['files'][rel];write(child,sub);key=child.relative_to(shadow).as_posix()
                    report['files'][key]=dict(bytes=child.stat().st_size,sha256=file_sha(child));report['results'][child.parent.relative_to(shadow).as_posix()]=file_sha(child)
                write(shadow/'report.json',report)
            observed=None
            try:read(shadow,ctx,a.region_model)
            except ValueError as e:observed=str(e)
            check(observed is not None and expected in observed,'SURFACE_FSV_'+name);controls.append(dict(name=name,expected=expected,observed=observed,refused=True))
        finally:
            for f,raw in saved.items():f.write_bytes(raw)
            check(all(f.read_bytes()==raw for f,raw in saved.items()),'SURFACE_RESTORE')
    base='qasc-composition/true/'
    control('target byte altered','complete-target-bytes.npz',lambda z:z['ids'].__setitem__((0,1),(z['ids'][0,1]+1)%256),'SURFACE_COMPLETE_BYTES')
    heldout=next(i for i,r in enumerate(ctx['policies']['qasc-composition']) if r['partition']=='evaluation')
    control('evaluation row trains','qasc-composition/training-rows.npy',lambda z:z.__setitem__((0,0),heldout),'SURFACE_TRAIN_SCOPE')
    control('evaluation selection moved','qasc-composition/selected.npz',lambda z:z['evaluation'].__setitem__(0,(z['evaluation'][0]+1)%8134),'SURFACE_SELECTION')
    control('predictor seed replaced','qasc-composition/conditioning.json',lambda z:z.update(actual_predictor_seed=26),'SURFACE_CONDITION_SOURCE')
    control('actual prediction hash changed','qasc-composition/conditioning.json',lambda z:z['hashes']['predicted'].__setitem__(0,'0'*64),'SURFACE_ACTUAL_PREDICTIONS')
    control('objective loss fabricated',base+'objective.json',lambda z:z.update(loss=z['loss']+1),'SURFACE_OBJECTIVE')
    control('likelihood inflated',base+'metrics.json',lambda z:z['evaluation']['true'].update(bits_per_byte=0),'SURFACE_METRICS')
    control('missing render hidden',base+'renders.jsonl',lambda z:z.pop(),'SURFACE_RENDER_COVERAGE')
    def break_eos(z):
        i=next(i for i,r in enumerate(z) if r['status']=='complete_unasserted');z[i]['tokens']=[v for v in z[i]['tokens'] if v!=257]
    control('missing EOS claimed complete',base+'renders.jsonl',break_eos,'SURFACE_RENDER_CODEC')
    control('output asserted',base+'renders.jsonl',lambda z:z[0].update(asserted=True),'SURFACE_RENDER_CODEC')
    control('false serving','report.json',lambda z:z.update(served=True),'SURFACE_CLAIMS')
    phases['eleven_persisted_refusal_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,ctx,a.region_model)==cold,'SURFACE_RESTORED_COLD');after={p:file_sha(Path(p)) for p in before};check(before==after,'SURFACE_SOURCE_DRIFT')
    publish(a.output/'after.json',after);phases['restored_readback_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
            cold_sha256=file_sha(a.output/'cold.json'),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
            scientific_admission=False,served=False,acceptance='author-checked',code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(controls=len(controls),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':main()

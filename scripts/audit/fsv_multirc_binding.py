"""Retained corrupt-copy controls across source, training, and asserted meaning."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multirc_context import load,check
from audit.multirc_result_readback import read
from audit.multirc_binding_readback import read_json
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def write(p,v):p.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'FSV_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);a.output.mkdir(parents=True)
    before={**ctx['before'],str(a.model/'report.json'):file_sha(a.model/'report.json')};publish(a.output/'before.json',before)
    cold=read(a.model,ctx,a.native,a.raw);publish(a.output/'cold.json',cold);phases['complete_source_result_cold_read']=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.model,shadow);phases['owned_shadow_copy']=time.perf_counter()-tick;tick=time.perf_counter();controls=[]
    def control(name,rel,change,expected):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()};child=None
        if rel!='report.json' and path.parent!=shadow and (path.parent/'report.json').exists():
            child=path.parent/'report.json';saved[child]=child.read_bytes()
        try:
            if rel.endswith('.npz'):
                with np.load(path,allow_pickle=False) as z:v={n:z[n].copy() for n in z.files}
                change(v);np.savez(path,**v)
            elif rel.endswith('.npy'):v=np.load(path,allow_pickle=False);change(v);np.save(path,v)
            elif rel.endswith('.pt'):v=torch.load(path,map_location='cpu',weights_only=True);change(v);torch.save(v,path)
            else:v=read_json(path);change(v);write(path,v)
            if rel!='report.json':
                report=read_json(shadow/'report.json');report['files'][rel]=dict(bytes=path.stat().st_size,sha256=file_sha(path))
                if child:
                    sub=read_json(child);sub['files'][path.name]=report['files'][rel];write(child,sub);key=child.relative_to(shadow).as_posix()
                    report['files'][key]=dict(bytes=child.stat().st_size,sha256=file_sha(child))
                    if child.parent.name!='preparation':report['results'][child.parent.relative_to(shadow).as_posix()]=file_sha(child)
                write(shadow/'report.json',report)
            observed=None
            try:read(shadow,ctx,a.native,a.raw)
            except ValueError as exc:observed=str(exc)
            check(observed is not None and expected in observed,'FSV_CONTROL_'+name)
            controls.append(dict(name=name,expected=expected,observed=observed,refused=True))
        finally:
            for p,b in saved.items():p.write_bytes(b)
            check(all(p.read_bytes()==b for p,b in saved.items()),'FSV_RESTORE')
    base='seed-25/joint/'
    control('private raw label changed','preparation/labels.npy',lambda v:v.__setitem__(0,1-v[0]),'RAW_LABELS')
    control('native question source changed','preparation/all-mpnet-base-v2.npz',lambda v:v['question'].__setitem__(0,v['question'][0]+1),'MAP_QUESTION')
    control('held-out mean leakage','preparation/all-mpnet-base-v2.npz',lambda v:v['mean'].__setitem__(0,v['mean'][0]+.1),'MEAN_TRAIN_ONLY')
    heldout=next(i for i,r in enumerate(ctx['rows']) if r['partition']=='evaluation' and r['available'])
    control('evaluation row trained','seed-25/training-rows.npy',lambda v:v.__setitem__((0,0),heldout),'TRAIN_ORDER')
    control('initial objective label changed',base+'objective.npz',lambda v:v['labels'].__setitem__(0,1-v['labels'][0]),'OBJECTIVE_SCOPE')
    control('all gradients fabricated zero',base+'objective.json',lambda v:v.update(gradients={k:0 for k in v['gradients']}),'GRADIENTS')
    control('nonfinite checkpoint',base+'model.pt',lambda v:next(iter(v.values())).reshape(-1).__setitem__(0,float('nan')),'FINAL_WEIGHTS')
    control('inflated answer metric',base+'metrics.json',lambda v:v['evaluation']['clean'].update(accuracy=1.5),'METRICS')
    control('invented swap truth',base+'interventions.json',lambda v:v['question-swap'].update(semantic_labels_for_swaps='false'),'INTERVENTION_METRICS')
    control('world truth from source label','report.json',lambda v:v.update(world_truth_assessed=True),'CLAIMS')
    control('learned understanding invented','report.json',lambda v:v.update(epistemic_understanding_proven=True),'CLAIMS')
    control('failed fit omitted','report.json',lambda v:v['results'].pop('seed-25/no-question'),'FIT_COVERAGE')
    phases['persisted_negative_controls_and_restoration']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,ctx,a.native,a.raw)==cold,'RESTORED_READBACK');after={p:file_sha(Path(p)) for p in before};check(before==after,'ORIGINAL_DRIFT');publish(a.output/'after.json',after)
    phases['restored_complete_read_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(acceptance='author-checked',source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,scientific_admission=False,served=False,code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(controls=len(controls),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':main()

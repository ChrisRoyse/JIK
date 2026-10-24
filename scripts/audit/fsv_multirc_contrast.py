"""Restored physical corrupt-copy checks for source pairs, objectives and claims."""
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
from audit.multirc_contrast_readback import read
from audit.multirc_binding_readback import read_json
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def write(p,v):p.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','baseline','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'CONTRAST_FSV_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);a.output.mkdir(parents=True)
    before={**ctx['before'],str(a.model/'report.json'):file_sha(a.model/'report.json'),str(a.baseline/'report.json'):file_sha(a.baseline/'report.json')}
    publish(a.output/'before.json',before);cold=read(a.model,ctx,a.native,a.raw,a.baseline);publish(a.output/'cold.json',cold)
    phases['source_pairs_and_full_result_readback']=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.model,shadow);phases['owned_shadow_copy']=time.perf_counter()-tick;tick=time.perf_counter();controls=[]
    def control(name,rel,change,expected):
        path=shadow/rel;saved={path:path.read_bytes(),shadow/'report.json':(shadow/'report.json').read_bytes()};child=None
        if rel!='report.json' and path.parent!=shadow and (path.parent/'report.json').exists():child=path.parent/'report.json';saved[child]=child.read_bytes()
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
            try:read(shadow,ctx,a.native,a.raw,a.baseline)
            except ValueError as exc:observed=str(exc)
            check(observed is not None and expected in observed,'CONTRAST_FSV_CONTROL_'+name)
            controls.append(dict(name=name,expected=expected,observed=observed,refused=True))
        finally:
            for p,b in saved.items():p.write_bytes(b)
            check(all(p.read_bytes()==b for p,b in saved.items()),'CONTRAST_FSV_RESTORE')
    base='seed-25/joint-rank/'
    control('raw source label','preparation/labels.npy',lambda v:v.__setitem__(0,1-v[0]),'RAW_LABELS')
    control('native source address','preparation/all-mpnet-base-v2.npz',lambda v:v['question'].__setitem__(0,v['question'][0]+1),'MAP_QUESTION')
    control('mean contaminated','preparation/all-mpnet-base-v2.npz',lambda v:v['mean'].__setitem__(0,v['mean'][0]+.1),'MEAN_TRAIN_ONLY')
    control('invented pair-group weight','training-groups.json',lambda v:v['within_question'][0].update(possible_pairs=v['within_question'][0]['possible_pairs']+1),'CONTRAST_SOURCE_GROUPS')
    heldout=next(i for i,r in enumerate(ctx['rows']) if r['partition']=='evaluation' and r['available'])
    control('held-out endpoint','seed-25/schedule.npz',lambda v:v['rows'].__setitem__((0,0),heldout),'CONTRAST_TRAIN_SCOPE')
    control('wrong pair source group','seed-25/schedule.npz',lambda v:v['groups'].__setitem__((0,0),(v['groups'][0,0]+1)%3309),'CONTRAST_PAIR_SOURCE_MEMBERSHIP')
    control('negative pair position','seed-25/schedule.npz',lambda v:v['pairs'].__setitem__((0,0,0),-1),'CONTRAST_PAIR_ADDRESS')
    control('wrong fit label',base+'training-labels.npy',lambda v:v.__setitem__(0,1-v[0]),'CONTRAST_TRAIN_LABELS')
    control('ranking disabled in objective',base+'objective.json',lambda v:v.update(ranked=False),'CONTRAST_OBJECTIVE_SCOPE')
    control('objective score fabricated',base+'objective.json',lambda v:v.update(loss=v['loss']+.25),'CONTRAST_OBJECTIVE_ARITHMETIC')
    control('nonfinite checkpoint',base+'model.pt',lambda v:next(iter(v.values())).reshape(-1).__setitem__(0,float('nan')),'CONTRAST_WEIGHTS')
    control('inflated answer metric',base+'metrics.json',lambda v:v['evaluation']['clean'].update(accuracy=1.5),'CONTRAST_METRICS')
    control('invented swap truth',base+'interventions.json',lambda v:v['question-swap'].update(semantic_labels_for_swaps='false'),'CONTRAST_INTERVENTION_METRICS')
    control('invented paired interval','pair-comparisons.json',lambda v:v['seed-25/joint-rank/joint-ce']['descriptive_envelope'][0].__setitem__(1,1.5),'CONTRAST_PAIR_INTERVALS')
    control('unauthorized episodic promotion','research-advancement.json',lambda v:v.update(episodic_promotion_authorized=True),'CONTRAST_ADVANCEMENT')
    control('world grounding fabricated','report.json',lambda v:v.update(world_truth_assessed=True),'CONTRAST_CLAIMS')
    control('fit omitted','report.json',lambda v:v['results'].pop('seed-25/no-context-rank'),'CONTRAST_FIT_COVERAGE')
    phases['persisted_negative_controls_and_restoration']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read(shadow,ctx,a.native,a.raw,a.baseline)==cold,'CONTRAST_RESTORED_READBACK')
    after={p:file_sha(Path(p)) for p in before};check(after==before,'CONTRAST_ORIGINAL_DRIFT');publish(a.output/'after.json',after)
    phases['restored_full_readback_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(acceptance='author-checked',source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,scientific_admission=False,served=False,code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(controls=len(controls),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':main()

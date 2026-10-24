"""Three-boundary corrupt-copy FSV on actual connected-data fitted results."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_context import load,check
from audit.qasc_combination_readback import read
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

def main():
    p=argparse.ArgumentParser()
    for n in ('population','native','raw','result','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.raw);cold=read(a.result,ctx);publish(a.output/'cold.json',cold['cold'])
    original={str(a.result/'report.json'):file_sha(a.result/'report.json'),**ctx['before']}
    phases['original_source_and_result_cold_read']=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/'shadow';shutil.copytree(a.result,shadow);base=(shadow/'report.json').read_bytes();report=json.loads(base);checks=[]
    def rewrite(path,value):
        check(path.resolve().is_relative_to(shadow.resolve()),'FSV_SHADOW_PATH');path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')
    def probe(name,filename,mutate,error,reseal=True):
        path=shadow/filename;before=path.read_bytes();old_report=(shadow/'report.json').read_bytes()
        record=dict(name=name,boundary=filename,before_sha256=file_sha(path),expected_error=error)
        mutate(path);record['action_sha256']=file_sha(path)
        if reseal and filename!='report.json':
            r=json.loads(old_report);r['files'][filename]=dict(bytes=path.stat().st_size,sha256=file_sha(path));rewrite(shadow/'report.json',r)
        try:read(shadow,ctx)
        except ValueError as exc:record['error']=str(exc);check(error in str(exc),'FSV_WRONG_REFUSAL')
        else:raise ValueError('MNLM_QASC_FSV_ACCEPTED_'+name)
        path.write_bytes(before);(shadow/'report.json').write_bytes(old_report)
        record['after_sha256']=file_sha(path);record['restored']=path.read_bytes()==before and (shadow/'report.json').read_bytes()==old_report;check(record['restored'],'FSV_RESTORE');checks.append(record)
    def changed_npz(path,key,index,value):
        with np.load(path) as z:data={n:z[n].copy() for n in z.files}
        data[key][index]=value;np.savez(path,**data)
    probe('native gather points to wrong public unit','gather.npz',lambda p:changed_npz(p,'maps',(0,0),0 if ctx['maps'][0,0]!=0 else 1),'COLD_GATHER')
    def wrong_labels(path):
        values=np.load(path);values[0]=(values[0]+1)%8;np.save(path,values)
    probe('altered private training target','labels.npy',wrong_labels,'COLD_LABELS')
    path='qasc-composition/seed-25/training-rows.npy'
    def wrong_training(p):
        values=np.load(p);values[0,0]=next(i for i,r in enumerate(ctx['policies']['qasc-composition']) if r['partition']=='evaluation');np.save(p,values)
    probe('evaluation row used for training',path,wrong_training,'COLD_TRAINING_SPLIT')
    prefix='qasc-composition/seed-25/both-facts/'
    probe('wrong objective target',prefix+'objective.npz',lambda p:changed_npz(p,'labels',0,99),'COLD_OBJECTIVE_TARGET')
    def wrong_loss(p):
        value=json.loads(p.read_bytes());value['loss']+=1;rewrite(p,value)
    probe('fabricated objective loss',prefix+'objective.json',wrong_loss,'COLD_OBJECTIVE_LOSS')
    def wrong_scores(p):
        with np.load(p) as z:values={n:z[n].copy() for n in z.files}
        values['probabilities'][0]=np.roll(values['probabilities'][0],1);np.savez(p,**values)
    probe('changed probabilities with resealed file',prefix+'outputs.npz',wrong_scores,'COLD_METRICS')
    def wrong_metric(p):
        value=json.loads(p.read_bytes());value['evaluation']['accuracy']+=.01;rewrite(p,value)
    probe('inflated reported accuracy',prefix+'metrics.json',wrong_metric,'COLD_METRICS')
    def false_claim(p):
        value=json.loads(p.read_bytes());value['scientific_admission']=True;rewrite(p,value)
    probe('candidate promoted to admitted','report.json',false_claim,'COLD_CLAIMS',False)
    partial=a.output/'wrong-native';partial.mkdir();bad=json.loads((a.native/'report.json').read_bytes());bad['complete']=False;publish(partial/'report.json',bad)
    try:load(a.population,partial,a.raw)
    except ValueError as exc:check('NATIVE_PIN' in str(exc),'FSV_NATIVE_REFUSAL');checks.append(dict(name='changed native publication pin',error=str(exc),before_sha256=file_sha(a.native/'report.json'),action_sha256=file_sha(partial/'report.json'),after_sha256=file_sha(a.native/'report.json'),restored=True))
    else:raise ValueError('MNLM_QASC_FSV_NATIVE_ACCEPTED')
    phases['owned_shadow_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    restored=read(shadow,ctx)['cold'];check(restored==cold['cold'],'FSV_FINAL_RESTORE')
    after={p:file_sha(Path(p)) for p in original};check(after==original,'FSV_ORIGINAL_DRIFT')
    publish(a.output/'before.json',original);publish(a.output/'after.json',after)
    phases['restored_cold_and_immutable_originals']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(passed=True,controls=checks,result_sha256=cold['cold']['report_sha256'],acceptance='author-checked',scientific_admission=False,served=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(passed=True,controls=len(checks),wall_seconds=wall,sha256=file_sha(a.output/'report.json'))))

if __name__=='__main__':main()

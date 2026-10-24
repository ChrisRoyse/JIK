"""Read the sealed forecasts and physical process outcomes without model execution."""
import argparse
import copy
from datetime import datetime,timedelta
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims import attempt_forecast
from claims.attempt_observation import AttemptObservations
from claims.process_conditions import write_committed
from model import process_anchor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    started=time.perf_counter()
    root=a.root
    check(not a.output.exists(),'ATTEMPT_TRANSFER_READBACK_EXISTS')
    forecast_root=root/'forecast'
    pin=(forecast_root/'COMMITTED').read_text()
    seal,payload=attempt_forecast.read(forecast_root,pin)
    observation_pin=(root/'execution/COMMITTED').read_text()
    observed=AttemptObservations(root/'execution',observation_pin,forecast_root,pin)
    feedback=json.loads((root/'feedback.json').read_bytes())
    before={str(p):sha(p.read_bytes()) for folder in (forecast_root,root/'execution')
            for p in folder.rglob('*') if p.is_file()}
    authenticated=time.perf_counter()
    results={}
    for key,entry in observed.entries.items():
        actual=observed.read(key)
        condition=entry['condition']
        saved=payload['predictions'][condition]
        banks=[v['termination']['bank'] for v in saved['models'].values()]
        check(banks[0] == banks[1] and len(banks[0]['rows']) == 19,'ATTEMPT_READER_BANK')
        scores={n:process_anchor.score(v['termination'],actual,key) for n,v in saved['models'].items()}
        check(scores == feedback['scores'][condition]['models'],'ATTEMPT_READER_SCORES')
        truth=int(actual['result']['returncode'] != 0)
        prior=[r['outcome'] for r in banks[0]['rows'] if r['observation'] is not None]
        bases=dict(always_success=[1.,0.,0.],always_failure=[0.,1.,0.],binary_uniform=[.5,.5,0.],
                   observed_frequency=[prior.count(i)/len(prior) for i in range(3)])
        check(saved['baselines'] == bases,'ATTEMPT_READER_BASELINES')
        independently_scored={}
        for model_name,prediction in saved['models'].items():
            prediction=prediction['termination']
            rows=dict(prediction['instruments'],common=prediction['common'])
            lanes={}
            for name,row in rows.items():
                probabilities=[sum(p for r,p in zip(banks[0]['rows'],row['region_probabilities'])
                                   if r['outcome']==i) for i in range(3)]
                check(all(abs(x-y)<1e-12 for x,y in zip(probabilities,row['probabilities'])),
                      'ATTEMPT_READER_PROBABILITIES')
                score=sum((v-float(i==truth))**2 for i,v in enumerate(row['probabilities']))
                expected=scores[model_name]['common'] if name=='common' else scores[model_name]['instruments'][name]
                check(score==expected['brier'],'ATTEMPT_READER_BRIER')
                lanes[name]=dict(probabilities=row['probabilities'],brier=score)
            independently_scored[model_name]=lanes
        baseline_scores={n:dict(probabilities=p,brier=sum((v-float(i==truth))**2 for i,v in enumerate(p)))
                         for n,p in bases.items()}
        check(baseline_scores == feedback['scores'][condition]['baselines'],'ATTEMPT_READER_BASELINE_SCORES')
        gap=attempt_forecast.match(seal,payload,condition,actual)
        check(gap==feedback['scores'][condition]['seconds_before_execution'],'ATTEMPT_READER_CLOCK')
        results[condition]=dict(observation=key,returncode=actual['result']['returncode'],
            models=independently_scored,baselines=baseline_scores,seconds_before_execution=gap)
    reconstructed=time.perf_counter()
    controls=[]
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'ATTEMPT_READER_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('ATTEMPT_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    with tempfile.TemporaryDirectory(prefix='attempt-forecast-control-',dir=root.parent) as tmp:
        clone=Path(tmp)/'forecast'
        shutil.copytree(forecast_root,clone)
        p=clone/'forecast.json'
        p.write_bytes(p.read_bytes()+b' ')
        refuse('changed_forecast','PROPOSAL_FORECAST_PIN',lambda:attempt_forecast.read(clone,pin))
    key=sorted(observed.entries)[0]
    actual=copy.deepcopy(observed.read(key))
    actual['started']=(datetime.fromisoformat(seal['committed_at'])-timedelta(seconds=1)).isoformat()
    refuse('late_forecast','PROPOSAL_FORECAST_LATE',
           lambda:attempt_forecast.match(seal,payload,observed.entries[key]['condition'],actual))
    with tempfile.TemporaryDirectory(prefix='attempt-observation-control-',dir=root.parent) as tmp:
        clone=Path(tmp)/'execution'
        shutil.copytree(root/'execution',clone)
        shutil.copyfile(root/'forecast-restart.json',Path(tmp)/'forecast-restart.json')
        name=sorted(observed.manifest['files'])[0]
        p=clone/name
        p.write_bytes(p.read_bytes()+b' ')
        refuse('changed_observation','PROPOSAL_OBSERVATION_BYTES',
               lambda:AttemptObservations(clone,observation_pin,forecast_root,pin))
    after={str(p):sha(p.read_bytes()) for folder in (forecast_root,root/'execution')
           for p in folder.rglob('*') if p.is_file()}
    check(before==after,'ATTEMPT_READER_MUTATED')
    finished=time.perf_counter()
    value=dict(schema='attempt-transfer-readback-v1',forecast=pin,observation=observation_pin,
        feedback=sha((root/'feedback.json').read_bytes()),results=results,controls=controls,
        immutable_before=before,immutable_after=after,
        phases=dict(authentication=authenticated-started,reconstruction=reconstructed-authenticated,
                    controls=finished-reconstructed),wall_seconds=finished-started,
        acceptance='author-checked',number_status='measured',scientific_admission=False,serving=False,
        task_correctness_established=False,useful_retained_learning_established=False)
    result_pin=write_committed(a.output,value)
    print(json.dumps(dict(pin=result_pin,results=results,controls=controls,wall_seconds=value['wall_seconds'])))


if __name__=='__main__':
    main()

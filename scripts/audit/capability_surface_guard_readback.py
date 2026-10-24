"""Cold byte/native/score readback of surface guard receipts, without calling the guard."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.capability_population import need,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from senses.trajectory_population import read_pinned


def read(path):return json.loads(Path(path).read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();need(not a.output.exists(),'SURFACE_COLD_EXISTS')
    report=read_pinned(a.source/'report.json',a.pin)
    plan=read_pinned(Path(report['plan']['path'])/'report.json',report['plan']['pin'])
    mapping=read_pinned(Path(report['plan']['path'])/'mapping.json',plan['mapping_sha256'])
    loaded={}
    for key,bank in report['native_banks'].items():
        root=Path(bank['native']);files=report['bank_files'][key]
        for name,pin in files.items():need(sha((root/name).read_bytes())==pin,'SURFACE_COLD_NATIVE_HASH')
        loaded[key]={}
        for name in read(root/'report.json')['lenses']:
            with np.load(root/name/'slots.npz',allow_pickle=False) as z:
                loaded[key][name]=(z['vectors'].copy(),z['offsets'].copy())
    results=[]
    for index,item in enumerate(report['cases']):
        folder=a.source/str(index);record=mapping['records'][index]
        need(read(folder/'report.json')==item and item['record']==record,'SURFACE_COLD_CASE')
        need(sha((folder/'measured.npz').read_bytes())==item['measured_sha256']
             and sha((folder/'receipt.json').read_bytes())==item['receipt_sha256'],'SURFACE_COLD_FILES')
        receipt=read(folder/'receipt.json');source=record['source'];root=Path(source['prediction'])
        original=read_pinned(root/'report.json',source['report_pin'])['cases'][source['case']]
        case=root/str(source['case']);raw=b''.join(mapping['units'][k].encode() for k in record['fragments'])
        need(sha(raw)==receipt['source_sha256']==record['source_sha256']
             and len(raw)==receipt['source_bytes']==record['source_bytes'],'SURFACE_COLD_SOURCE')
        if record['purpose']=='source_replay_control':
            need(read(case/'source.json')['text'].encode()==raw and sha(raw)==original['surface']['sha256'],
                 'SURFACE_COLD_SOURCE_CONTROL')
        for name in ('prediction','target'):
            value=original['files'][name]
            need(sha((case/value['name']).read_bytes())==value['sha256'],'SURFACE_COLD_MODEL_BINDING')
        with np.load(case/'prediction.npz',allow_pickle=False) as z:predicted={n:z[n][0] for n in z.files}
        with np.load(folder/'measured.npz',allow_pickle=False) as z:measured={n:z[n] for n in z.files}
        guard=item['guard'];need(guard['policy']==digest(report['policy']) and guard['receipt']==digest(receipt),'SURFACE_COLD_GUARD_JOIN')
        errors={}
        for name,values in measured.items():
            parts=[]
            for join in receipt['joins']:
                vectors,ptr=loaded[join['bank']][name];row=join['row'];parts.append(vectors[ptr[row]:ptr[row+1]])
            reconstructed=np.concatenate(parts)
            need(np.array_equal(values,reconstructed) and sha(values.astype('<f4').tobytes())==receipt['slots'][name]['sha256'],
                 'SURFACE_COLD_NATIVE_JOIN')
            x,y=values.astype(np.float64),predicted[name].astype(np.float64)
            x/=np.linalg.norm(x,axis=1,keepdims=True);y/=np.linalg.norm(y,axis=1,keepdims=True)
            # Proof-case-only matrix; never a corpus matrix or a serving operation.
            similarity=x@y.T;left=similarity.max(1);right=similarity.max(0);slot=guard['slots'][name]
            error=max(float(np.max(np.abs(left-slot['candidate_window_maxima']))),
                      float(np.max(np.abs(right-slot['predicted_window_maxima']))))
            score=float(min(left.mean(),right.mean()))
            need(error<=1e-12 and abs(score-slot['score'])<=1e-12
                 and slot['passes']==(score>=report['policy']['threshold']),'SURFACE_COLD_SCORE')
            errors[name]=error
        need(guard['geometric_conjunction']==all(s['passes'] for s in guard['slots'].values())
             and guard['accepted'] is False and guard['serving'] is False
             and guard['reasons']==['PANEL_NOT_ADMITTED','SURFACE_GUARD_NOT_CALIBRATED','ANCHOR_CHECK_REQUIRED'],
             'SURFACE_COLD_OUTER_GATE')
        results.append(dict(case=record['case'],purpose=record['purpose'],family=record['family'],
            bytes=len(raw),native_byte_exact=True,score_errors=errors,accepted=False))
    need(len(report['controls'])==6 and report['terminal_refusals']==mapping['refusals'],'SURFACE_COLD_CONTROLS')
    result=dict(schema='capability-surface-guard-readback-v1',source=str(a.source),source_pin=a.pin,cases=results,
        generated_cases=sum(r['purpose']=='generated' for r in results),terminal_refusals=report['terminal_refusals'],
        seconds=time.perf_counter()-start,number_status='measured',acceptance='author-checked',
        learned_capability=False,scientific_admission=False,serving=False,producer_sha256=sha(Path(__file__).read_bytes()))
    print(json.dumps(dict(pin=write_committed(a.output,result),cases=len(results),generated=result['generated_cases'])),flush=True)


if __name__=='__main__':main()

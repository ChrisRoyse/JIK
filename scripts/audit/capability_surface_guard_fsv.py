"""Bind retained native measurements to exact surface bytes and exercise per-slot refusals."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.capability_prediction_readback import arrays
from claims.capability_population import need,sha
from claims.process_conditions import write_committed
from model.capability_surface_guard import POLICY,evaluate
from model.proposal_input import load
from senses.trajectory_population import read_pinned


def read(path):return json.loads(Path(path).read_bytes())


def refused(code,action):
    try:action()
    except ValueError as exc:
        need(code in str(exc),'SURFACE_GUARD_FSV_WRONG_ERROR')
        return dict(expected=code,error=str(exc))
    raise ValueError('SURFACE_GUARD_FSV_NEGATIVE_ACCEPTED:'+code)


def independent_scores(a,b,reported):
    """Separate scalar-dot readback; only this proof case's windows are visited."""
    left=[];right=[]
    a,b=a.astype(np.float64),b.astype(np.float64)
    def cosine(x,y):return np.dot(x,y)/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(y,y)))
    for x in a:left.append(max(cosine(x,y) for y in b))
    for y in b:right.append(max(cosine(x,y) for x in a))
    error=max(np.max(np.abs(np.array(left)-reported['candidate_window_maxima'])),
              np.max(np.abs(np.array(right)-reported['predicted_window_maxima'])))
    need(error<=1e-12 and abs(min(np.mean(left),np.mean(right))-reported['score'])<=1e-12,
         'SURFACE_GUARD_FSV_INDEPENDENT_SCORE')
    return float(error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-pin',required=True)
    p.add_argument('--native',type=Path);p.add_argument('--native-pin')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=tick=time.perf_counter()
    plan=read_pinned(a.plan/'report.json',a.plan_pin)
    mapping=read_pinned(a.plan/'mapping.json',plan['mapping_sha256']);banks=dict(mapping['banks'])
    if plan['fresh']:
        need(a.native is not None and a.native_pin is not None,'SURFACE_GUARD_FRESH_REQUIRED')
        banks['fresh']=dict(native=str(a.native),native_pin=a.native_pin,source_pin=a.plan_pin)
    need(not a.output.exists(),'SURFACE_GUARD_FSV_EXISTS');a.output.mkdir()
    loaded={};traces={};specs=None;bank_files={}
    for key,item in banks.items():
        root=Path(item['native']);native,current,files=load(root,item['native_pin'],item['source_pin'])
        need(specs is None or current==specs,'SURFACE_GUARD_FSV_IDENTITY_CHANGE')
        loaded[key]=native;specs=current;bank_files[key]=files;traces[key]={}
        wanted={v['row'] for v in mapping['mapping'].values() if v['bank']==key}
        for name in current:
            selected={}
            with gzip.open(root/name/'traces.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line)
                    if row['input'] in wanted:selected[row['input']]=row
            need(set(selected)==wanted,'SURFACE_GUARD_FSV_TRACE_MISSING')
            traces[key][name]=selected
    phases=dict(native_authentication=time.perf_counter()-tick);tick=time.perf_counter()
    identities={n:s.identity for n,s in specs.items()};results=[];controls=[]
    for index,record in enumerate(mapping['records']):
        source=record['source'];root=Path(source['prediction']);case=root/str(source['case'])
        prediction=read_pinned(root/'report.json',source['report_pin'])['cases'][source['case']]
        meta=read(case/'view.json')
        need({s['name']:s['identity'] for s in meta['specs']}==identities,'SURFACE_GUARD_FSV_PREDICTOR_IDENTITY')
        for name in ('prediction','target'):
            info=prediction['files'][name]
            need(sha((case/info['name']).read_bytes())==info['sha256'],'SURFACE_GUARD_FSV_PREDICTOR_BYTES')
        predicted={n:x[0] for n,x in arrays(case/'prediction.npz').items()}
        measured={n:[] for n in specs};parts=[];joins=[]
        for pin in record['fragments']:
            piece=mapping['units'][pin].encode('utf-8');need(sha(piece)==pin,'SURFACE_GUARD_FSV_FRAGMENT')
            parts.append(piece);where=mapping['mapping'][pin];bank,row=where['bank'],where['row']
            for name in specs:
                values,offsets=loaded[bank][name];lo,hi=map(int,offsets[row:row+2]);trace=traces[bank][name][row]
                need(trace['trace']['surface_sha256']==pin and trace['vector_span']==[lo,hi]
                     and hi>lo and len(trace['trace']['windows'])==hi-lo,'SURFACE_GUARD_FSV_WINDOW_JOIN')
                measured[name].append(values[lo:hi])
            joins.append(dict(fragment=pin,bank=bank,row=row))
        raw=b''.join(parts)
        need(sha(raw)==record['source_sha256'] and len(raw)==record['source_bytes'],'SURFACE_GUARD_FSV_CONSERVATION')
        measured={n:np.concatenate(v) for n,v in measured.items()}
        receipt=dict(source_sha256=sha(raw),source_bytes=len(raw),instruments=identities,joins=joins,
            slots={n:dict(shape=list(v.shape),sha256=sha(v.astype('<f4').tobytes())) for n,v in measured.items()})
        if record['purpose']=='source_replay_control':
            reference=arrays(case/'target.npz')
            need(all(np.array_equal(v,reference[n][0]) for n,v in measured.items()),'SURFACE_GUARD_FSV_REFERENCE_NATIVE')
            need(read(case/'source.json')['text'].encode('utf-8')==raw,'SURFACE_GUARD_FSV_REFERENCE_SOURCE')
        else:
            candidate=Path(plan['runtime']['path'])/str(record['case'])/'candidate.bin'
            need(candidate.read_bytes()==raw,'SURFACE_GUARD_FSV_GENERATED_SOURCE')
        result=evaluate(raw,measured,predicted,identities,receipt,policy=POLICY)
        source_control=(evaluate(raw,measured,measured,identities,receipt,policy=POLICY)
                        if record['purpose']=='source_replay_control' else None)
        errors={n:independent_scores(measured[n],predicted[n],result['slots'][n]) for n in specs}
        need(not result['accepted'] and result['reasons']==[
            'PANEL_NOT_ADMITTED','SURFACE_GUARD_NOT_CALIBRATED','ANCHOR_CHECK_REQUIRED'],'SURFACE_GUARD_FSV_OUTER_GATE')
        if source_control:need(source_control['geometric_conjunction'] and not source_control['accepted'],'SURFACE_GUARD_FSV_CONTROL')
        if not controls:
            def check(raw_=raw,measured_=measured,ids_=identities,receipt_=receipt,policy_=POLICY):
                return evaluate(raw_,measured_,predicted,ids_,receipt_,policy=policy_)
            missing=dict(measured);missing.pop(next(iter(specs)))
            controls.append(refused('MISSING_SLOT',lambda:check(measured_=missing)))
            controls.append(refused('GUARD_SOURCE',lambda:check(raw_=raw+b' ')))
            controls.append(refused('GUARD_UTF8',lambda:check(raw_=b'\xff')))
            wrong=dict(identities);wrong[next(iter(specs))]='0'*64
            controls.append(refused('GUARD_INSTRUMENT',lambda:check(ids_=wrong)))
            damaged={n:v.copy() for n,v in measured.items()};damaged[next(iter(specs))][0,0]+=1
            controls.append(refused('NATIVE_BYTES',lambda:check(measured_=damaged)))
            controls.append(refused('GUARD_POLICY',lambda:check(policy_=dict(POLICY,threshold=.69))))
        folder=a.output/str(index);folder.mkdir()
        np.savez(folder/'measured.npz',**measured);write_committed(folder/'receipt.json',receipt)
        output=dict(record=record,guard=result,source_control=source_control,independent_score_errors=errors,
            measured_sha256=sha((folder/'measured.npz').read_bytes()),receipt_sha256=sha((folder/'receipt.json').read_bytes()),
            external_truth='unestablished_by_native_remeasurement',semantic_false_accept_rate=None)
        write_committed(folder/'report.json',output);results.append(output)
    phases['guard_and_readback']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='capability-surface-guard-fsv-v1',plan=dict(path=str(a.plan),pin=a.plan_pin),
        cases=results,controls=controls,terminal_refusals=mapping['refusals'],native_banks=banks,bank_files=bank_files,
        policy=POLICY,phases_seconds=phases,seconds=wall,phase_coverage=sum(phases.values())/wall,
        generated_guard_cases=sum(r['record']['purpose']=='generated' for r in results),
        number_status='measured',acceptance='author-checked',scientific_admission=False,serving=False,
        learned_capability=False,producer_sha256=sha(Path(__file__).read_bytes()))
    need(report['phase_coverage']>=.9,'SURFACE_GUARD_FSV_TIMING')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),cases=len(results),
        generated=report['generated_guard_cases'],refusals=len(mapping['refusals']))),flush=True)


if __name__=='__main__':main()

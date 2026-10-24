"""Content-addressed measurement plan for completed predictions and source controls."""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need,sha
from claims.process_conditions import write_committed
from claims.repository_cohort_population import fragments
from senses.trajectory_population import read_pinned
from text.capability_context_plan import historical_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime','cache_config','output'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--runtime-pin',required=True);p.add_argument('--cache-pin',required=True)
    a=p.parse_args(); start=tick=time.perf_counter()
    runtime=read_pinned(a.runtime/'report.json',a.runtime_pin)
    cache,banks=historical_cache(read_pinned(a.cache_config,a.cache_pin))
    phases=dict(source_and_cache=time.perf_counter()-tick);tick=time.perf_counter()
    records=[];units={};refusals=[]
    for item in runtime['cases']:
        source=item['source']; root=Path(source['prediction']);case=root/str(source['case'])
        report=read_pinned(root/'report.json',source['report_pin']);record=report['cases'][source['case']]
        field=json.loads((case/'source.json').read_bytes());raw=field['text'].encode('utf-8')
        need(field['field']==record['field'] and sha(raw)==record['surface']['sha256'], 'SURFACE_PLAN_REFERENCE')
        candidates=[('source_replay_control',raw)]
        if item['complete']:
            raw=(a.runtime/str(item['case'])/'candidate.bin').read_bytes()
            need(sha(raw)==item['files']['candidate.bin']['sha256'] and len(raw)==item['bytes']
                 and item['status']=='COMPLETE' and item['eos'],'SURFACE_PLAN_COMPLETION')
            candidates.append(('generated',raw))
        else:refusals.append(dict(case=item['case'],family=item['family'],status=item['status']))
        for purpose,raw in candidates:
            need(raw and raw.decode('utf-8').encode('utf-8')==raw,'SURFACE_PLAN_UTF8')
            pieces=list(fragments(raw));need(b''.join(pieces)==raw,'SURFACE_PLAN_CONSERVATION')
            pins=[]
            for piece in pieces:
                key=sha(piece);text=piece.decode('utf-8')
                need(key not in units or units[key]==text,'SURFACE_PLAN_COLLISION')
                units[key]=text;pins.append(key)
            records.append(dict(case=item['case'],family=item['family'],purpose=purpose,source=source,
                source_sha256=sha(raw),source_bytes=len(raw),fragments=pins))
    fresh=sorted(k for k in units if k not in cache);positions={k:i for i,k in enumerate(fresh)}
    mapping={};used=set()
    for key,text in units.items():
        if key in cache:
            old=cache[key];need(old['text']==text,'SURFACE_PLAN_CACHE_COLLISION')
            mapping[key]=dict(bank=old['bank'],row=old['row']);used.add(old['bank'])
        else:mapping[key]=dict(bank='fresh',row=positions[key])
    phases['selection']=time.perf_counter()-tick;tick=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False);families={};probe=None
    if fresh:
        candidates=[(k,v) for k,v in cache.items() if 1800<=len(v['text'])<=3000]
        need(candidates,'SURFACE_PLAN_PROBE')
        key,value=max(candidates,key=lambda pair:(len(pair[1]['text']),pair[0]))
        rows=[dict(row=i,chunk=0,entity=k,text=units[k],start=0,end=len(units[k])) for i,k in enumerate(fresh)]
        probe=dict(row=len(rows),sha256=key,model_input=False)
        rows.append(dict(row=len(rows),chunk=0,entity=key,text=value['text'],start=0,end=len(value['text'])))
        folder=a.output/'surface';(folder/'chunks').mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows),folder/'chunks/atoms.parquet')
        pin=write_committed(folder/'summary.json',dict(chunks=len(rows),
            chunks_sha256=sha((folder/'chunks/atoms.parquet').read_bytes())))
        families['surface']=dict(ready=True,summary_sha256=pin)
    pin=write_committed(a.output/'mapping.json',dict(records=records,units=units,mapping=mapping,
        banks={k:banks[k] for k in sorted(used)},probe=probe,refusals=refusals))
    phases['publication']=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(schema='capability-surface-plan-v1',complete=True,errors=[],calibration_only=True,
        families=families,mapping_sha256=pin,runtime=dict(path=str(a.runtime),pin=a.runtime_pin),
        cache_config=dict(path=str(a.cache_config),pin=a.cache_pin),fresh=len(fresh),reused=len(units)-len(fresh),
        fresh_bytes=sum(len(units[k].encode()) for k in fresh),inputs=len(fresh)+bool(fresh),
        phases_seconds=phases,seconds=wall,phase_coverage=sum(phases.values())/wall,
        scientific_admission=False,serving=False,number_status='measured',producer_sha256=sha(Path(__file__).read_bytes()))
    need(result['phase_coverage']>=.9,'SURFACE_PLAN_TIMING')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),fresh=len(fresh),reused=len(units)-len(fresh))))


if __name__=='__main__':main()

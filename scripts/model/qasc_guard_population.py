"""Deduplicate strict rendered bytes and reuse exact native readings before new forwards."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_generation_data import load,TARGETS,TARGET_NATIVE
from model.qasc_context import LANES,POLICIES,NATIVE,POPULATION,rows,check
from audit.qasc_region_readback import read as read_region,read_json
from audit.qasc_surface_readback import read as read_surface,records
from model.train_qasc_surfaces import lines
from model.train_qasc_regions import files
from text.evaluation_multiscale import file_sha
from text.connected_training_census import sha
from senses.trajectory_population import publish

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','region-model','surface-model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'GUARD_POPULATION_EXISTS');started=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);region=read_region(a.region_model,ctx);surface=read_surface(a.surface_model,ctx,a.region_model)
    fsv=read_json(a.proof/'surface-fsv-linux-v1/report.json');replay=read_json(a.proof/'surface-replay-linux-v1/report.json')
    check(fsv['source_report_sha256']==replay['model_report_sha256']==surface['report_sha256'] and len(fsv['controls'])==11
          and all(x['refused'] for x in fsv['controls']) and fsv['originals_unchanged'] and fsv['shadow_restored']
          and replay['all_cases_and_bytes_replayed'] and replay['originals_unchanged'],'GUARD_SURFACE_PROOF')
    public=pq.read_table(a.population/'connected/chunks/atoms.parquet').to_pylist();private=rows(a.targets/'targets.jsonl')
    cache={r['text']:dict(route='public',row=r['row']) for r in public}
    for t in private:
        address=t['native'];cache.setdefault(t['text'],dict(route='public' if address['route']=='public' else 'target',row=address['row'] if address['route']=='public' else address['row']+1))
    phases['parent_cold_reads_and_cache_index']=time.perf_counter()-started;tick=time.perf_counter()
    outputs=[];lookup={};fresh=[];candidates=[]
    for policy in POLICIES:
        for arm in ('true','gaussian','predicted'):
            source=a.surface_model/policy/arm/'renders.jsonl'
            for j,r in enumerate(records(source)):
                ix=None
                if r['status']=='complete_unasserted':
                    text=r['text'];key=sha(text)
                    if key not in lookup:
                        if text in cache:address=cache[text]
                        else:
                            address=dict(route='fresh',row=len(fresh));fresh.append(dict(row=len(fresh),chunk=0,entity=key,text=text))
                        lookup[key]=len(outputs);outputs.append(dict(index=len(outputs),text=text,sha256=key,utf8_bytes=len(text.encode()),native=address))
                    ix=lookup[key];check(outputs[ix]['text']==text,'GUARD_HASH_COLLISION')
                candidates.append(dict(index=len(candidates),policy=policy,arm=arm,source_file=policy+'/'+arm+'/renders.jsonl',source_row=j,
                    case=r['case'],row=r['row'],partition=r['partition'],route=r['route'],status=r['status'],output=ix,asserted=False))
    check(len(candidates)==3072,'GUARD_CANDIDATE_COVERAGE')
    # Select the already proven overflowing source, not a character-length guess.
    probes=[read_json(a.native/n/'probe.json') for n in LANES];probe=probes[0]
    check(all((p['family'],p['row'],p['text_sha256'])==(probe['family'],probe['row'],probe['text_sha256']) for p in probes),'GUARD_SHARED_PROBE')
    check(probe['family']=='connected' and all(len(p['trace']['windows'])>1 for p in probes),'GUARD_PROBE_OVERFLOW')
    text=public[probe['row']]['text'];check(sha(text)==probe['text_sha256'],'GUARD_PROBE_SOURCE')
    phases['output_dedup_cache_and_source_probe']=time.perf_counter()-tick;tick=time.perf_counter();a.output.mkdir(parents=True)
    lines(a.output/'outputs.jsonl',outputs);lines(a.output/'candidates.jsonl',candidates)
    publish(a.output/'probe.json',dict(public_unit=probe['row'],text_sha256=probe['text_sha256'],role='probe-only',model_input=False,target=False))
    families={}
    for family,table in [('probe',[dict(row=0,chunk=0,entity=sha(text),text=text)]),('rendered',fresh)]:
        folder=a.output/family;(folder/'chunks').mkdir(parents=True)
        schema=pa.schema([('row',pa.int64()),('chunk',pa.int64()),('entity',pa.string()),('text',pa.string())])
        pq.write_table(pa.Table.from_pylist(table,schema=schema),folder/'chunks/atoms.parquet',compression='zstd')
        publish(folder/'summary.json',dict(chunks=len(table),chunks_sha256=file_sha(folder/'chunks/atoms.parquet')))
        families[family]=dict(ready=bool(table),summary_sha256=file_sha(folder/'summary.json'))
    counts=dict(attempts=len(candidates),statuses=dict(Counter(r['status'] for r in candidates)),unique_complete_outputs=len(outputs),
                fresh_outputs=len(fresh),cached_outputs=len(outputs)-len(fresh),cached_routes=dict(Counter(r['native']['route'] for r in outputs if r['native']['route']!='fresh')),
                fresh_characters=sum(len(r['text']) for r in fresh),measurement_inputs=len(fresh)+1,
                max_complete_utf8_bytes=max((r['utf8_bytes'] for r in outputs),default=0),probe_characters=len(text))
    phases['source_bound_publication']=time.perf_counter()-tick;tick=time.perf_counter();metadata=files(a.output)
    phases['final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(schema='qasc-rendered-native-population-v1',calibration_only=True,complete=True,errors=[],families=families,files=metadata,
            counts=counts,population_sha256=POPULATION,public_native_sha256=NATIVE,targets_sha256=TARGETS,target_native_sha256=TARGET_NATIVE,
            region_report_sha256=region['report_sha256'],surface_report_sha256=surface['report_sha256'],lanes=list(LANES),
            scientific_admission=False,served=False,guard_evaluated=False,free_text_composition_proven=False,
            phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),counts=counts,wall_seconds=wall)),flush=True)

if __name__=='__main__':main()

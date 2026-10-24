"""Pinned connected-evidence population for provisional frozen-sense research."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.connected_training_census import load_sources,PINS,check
from text.connected_training_capture import MULTIRC_PARENT
from text.evaluation_multiscale import file_sha
from text.connected_observations import build,leakage_catalog
from text.connected_leakage import overlap,splits,PAIR_VISIT_BOUND,SEED
from senses.contract import canonical
from senses.trajectory_population import publish

PAYLOAD_PINS={'census.json':'dd75415662ecf54dfb7fb176a52330b77244d973a268b069323254a5668cd428',
 'multirc-documents.json':'6d5a80e9deb46c9679682428c1c69fdc8bc056e5b5f075849191f98c6426d0d6',
 'qasc-cases.json':'93fffb0cc54e36a2d888ff98b0d5e433ba9af1a9a9f5503cd7d550f09ce32513'}


def write_rows(path,rows):
    with path.open('xb') as f:
        for r in rows:f.write(canonical(r)+b'\n')


def project(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};tick=time.perf_counter()
    for name,pin in PAYLOAD_PINS.items():check(file_sha(a.census/name)==pin,'CENSUS_PAYLOAD_PIN')
    multirc,qasc=load_sources(a.multirc,a.qasc,a.parent);docs=json.loads((a.census/'multirc-documents.json').read_bytes())
    phases['source_and_census_pins']=time.perf_counter()-tick;tick=time.perf_counter()
    data=build(multirc,qasc,docs);catalogs=leakage_catalog(data)
    phases['source_addressed_views_and_catalogs']=time.perf_counter()-tick;tick=time.perf_counter()
    public=a.output/'connected/chunks';public.mkdir(parents=True);private=a.output/'private';private.mkdir()
    pq.write_table(pa.Table.from_pylist(data['units']),public/'atoms.parquet',compression='zstd')
    for name,key in (('references','references'),('cases','cases'),('documents','documents')):write_rows(a.output/(name+'.jsonl'),data[key])
    write_rows(private/'targets.jsonl',data['targets'])
    publish(a.output/'connected/summary.json',dict(chunks=len(data['units']),chunks_sha256=file_sha(public/'atoms.parquet')))
    phases['observation_and_private_target_publication']=time.perf_counter()-tick;tick=time.perf_counter()
    overlaps={};overlap_stats={}
    for name,catalog in catalogs.items():
        root=a.output/'cohorts'/name;overlaps[name],overlap_stats[name]=overlap(catalog,root);publish(root/'owners.json',catalog)
    phases['bounded_complete_lexical_overlap']=time.perf_counter()-tick;tick=time.perf_counter()
    assignments,policies=splits(data,catalogs,overlaps);write_rows(a.output/'assignments.jsonl',assignments);publish(a.output/'partitions.json',policies)
    counts=dict(public_unique_texts=len(data['units']),public_unique_characters=sum(len(u['text']) for u in data['units']),
        references=len(data['references']),missing_surfaces=sum(r['unit'] is None for r in data['references']),
        roles=dict(Counter(r['family']+':'+r['role'] for r in data['references'])),cases=dict(Counter(c['family'] for c in data['cases'])),
        structural_edges=sum(len(d['edges']) for d in data['documents']),source_alias_methods=dict(Counter(d['source_alias']['method'] for d in data['documents'])),
        private_targets=len(data['targets']),quarantined_cases=sum(bool(t['quarantine_reasons']) for t in data['targets']),overlap=overlap_stats,
        partitions={k:{n:v for n,v in p.items() if n not in ('links','groups')} for k,p in policies.items()})
    publish(a.output/'counts.json',counts);phases['policy_assignment_and_counts']=time.perf_counter()-tick;tick=time.perf_counter()
    for name,pin in PAYLOAD_PINS.items():check(file_sha(a.census/name)==pin,'CENSUS_DRIFT')
    for root,pin in ((a.multirc,PINS['multirc']),(a.qasc,PINS['qasc']),(a.parent,MULTIRC_PARENT)):check(file_sha(root/'report.json')==pin,'SOURCE_REPORT_DRIFT')
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='connected-grounding-population-v1',complete=True,errors=[],files=files,source_pins={**PINS,'multirc_parent':MULTIRC_PARENT},census_payloads=PAYLOAD_PINS,
        families={'connected':dict(ready=True,summary_sha256=file_sha(a.output/'connected/summary.json'))},counts=counts,
        public_observation_columns=['row','chunk','entity','text'],private_target_path='private/targets.jsonl',target_identity_used_for_split_only=True,class_labels_used_for_split=False,
        split_seed=SEED,prefix_pair_visit_bound=PAIR_VISIT_BOUND,empty_observations_measured=False,calibration_only=True,sealed_evaluation=False,scientific_admission=False,served=False,
        world_entity_source_time_sealed=False,cross_source_overlap_sealed=False,encoder_exposure_certified=False,full_composition_novelty_certified=False,native_text_floor=5,
        research_scope='connected-text development only; MultiRC research-only; candidate instruments unadmitted',observations_removed=0,
        code_sha256={n:file_sha(Path(__file__).parent/n) for n in ('connected_population.py','connected_observations.py','connected_leakage.py')},
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser()
    for name in ('multirc','qasc','parent','census','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:r=project(a)
    except Exception as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():publish(a.output/'failure.json',dict(complete=False,error=str(exc),error_type=type(exc).__name__))
        raise
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),counts=r['counts'],wall_seconds=r['wall_seconds'],phase_coverage=r['phase_coverage'])))


if __name__=='__main__':main()

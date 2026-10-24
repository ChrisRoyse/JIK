"""Immutable quality overlay: contradictory option anchors and question contrasts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.connected_population_readback import readback
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

def check(ok,code):
    if not ok:raise ValueError('MNLM_MULTIRC_QUALITY_'+code)
def rows(path):return [json.loads(x) for x in path.read_bytes().splitlines()]
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def collect(population):
    units=pq.read_table(population/'connected/chunks/atoms.parquet')['text'].to_pylist()
    refs={r['id']:r for r in rows(population/'references.jsonl')};targets={t['case']:t for t in rows(population/'private/targets.jsonl')}
    assignments={r['case']:r for r in rows(population/'assignments.jsonl') if r['policy']=='multirc'};groups=defaultdict(list)
    for c in rows(population/'cases.jsonl'):
        if c['family']!='multirc' or not assignments[c['id']]['eligible']:continue
        for j,rid in enumerate(c['options']):
            r=refs[rid];check(r['unit'] is not None,'EMPTY_ELIGIBLE');text=units[r['unit']]
            groups[c['document'],norm(text)].append(dict(case=c['id'],question_sha256=refs[c['question']]['text_sha256'],option_text=text,option=j,
                label=targets[c['id']]['labels'][j],row=c['source_row'],source_question=c['source_question'],partition=assignments[c['id']]['partition'],group=assignments[c['id']]['group']))
    conflicts=[];excluded=defaultdict(set)
    for (doc,text),members in sorted(groups.items()):
        for question in sorted({m['question_sha256'] for m in members}):
            mm=[m for m in members if m['question_sha256']==question]
            if {m['label'] for m in mm}!={0,1}:continue
            exact=any(len({m['label'] for m in mm if m['option_text']==surface})>1 for surface in {m['option_text'] for m in mm})
            reason='exact-surface-label-conflict' if exact else 'normalized-surface-label-conflict'
            conflicts.append(dict(document=doc,option_normalized=text,question_sha256=question,reason=reason,members=mm))
            for m in mm:excluded[m['case']].add(reason)
    exclusion=[dict(case=case,reasons=sorted(reasons),partition=assignments[case]['partition'],group=assignments[case]['group']) for case,reasons in sorted(excluded.items())]
    contrasts=[]
    for (doc,text),members in sorted(groups.items()):
        mm=[m for m in members if m['case'] not in excluded]
        if {m['label'] for m in mm}!={0,1}:continue
        check(len({m['partition'] for m in mm})==len({m['group'] for m in mm})==1,'SPLIT_CROSSING')
        contrasts.append(dict(document=doc,option_normalized=text,members=mm,positive=sum(m['label']==1 for m in mm),negative=sum(m['label']==0 for m in mm)))
    return conflicts,exclusion,contrasts

def audit_raw(population,raw,conflicts,excluded,contrasts):
    """Rebuild eligibility/conflicts from original fields, not producer grouping rows."""
    data=json.loads(raw.read_bytes());cases=[c for c in rows(population/'cases.jsonl') if c['family']=='multirc']
    assignments={r['case']:r for r in rows(population/'assignments.jsonl') if r['policy']=='multirc'}
    lookup={(c['source_row'],c['source_question']):c for c in cases};original=defaultdict(list);bad=set()
    for row,d in enumerate(data['data']):
        for qi,q in enumerate(d['paragraph']['questions']):
            c=lookup[row,qi]
            if not assignments[c['id']]['eligible']:continue
            for oi,a in enumerate(q['answers']):original[d['id'],norm(a['text'])].append((row,qi,oi,a['isAnswer'],q['question'],a['text'],c['id']))
    expected_conflicts={}
    for key,entries in original.items():
        for question in {x[4] for x in entries}:
            ms=[x for x in entries if x[4]==question]
            if {m[3] for m in ms}!={0,1}:continue
            exact=any({m[3] for m in ms if m[5]==text}=={0,1} for text in {m[5] for m in ms})
            reason='exact-surface-label-conflict' if exact else 'normalized-surface-label-conflict';bad.update(m[6] for m in ms)
            expected_conflicts[key+(digest(question),)]=(reason,sorted(m[:4] for m in ms))
    observed_conflicts={(g['document'],g['option_normalized'],g['question_sha256']):(g['reason'],sorted((m['row'],m['source_question'],m['option'],m['label']) for m in g['members'])) for g in conflicts}
    check(observed_conflicts==expected_conflicts,'RAW_CONFLICTS');check({r['case'] for r in excluded}==bad,'RAW_EXCLUSIONS')
    expected={key:sorted(m[:4] for m in entries if m[6] not in bad) for key,entries in original.items() if {m[3] for m in entries if m[6] not in bad}=={0,1}}
    observed={(g['document'],g['option_normalized']):sorted((m['row'],m['source_question'],m['option'],m['label']) for m in g['members']) for g in contrasts}
    check(observed==expected,'RAW_CONTRASTS');return dict(original_field_equal=True,conflicts=len(conflicts),excluded_cases=len(bad),contrast_groups=len(contrasts))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','raw','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    readback(a.population,a.raw/'multirc-training-v1',a.raw/'qasc-hf-training-v1',a.raw/'multirc-v1')
    source=a.raw/'multirc-training-v1/train.json'
    before={str(p):file_sha(p) for p in [a.population/n for n in ('report.json','cases.jsonl','references.jsonl','private/targets.jsonl','assignments.jsonl')]+[source]}
    phases['source_population_and_split_readback']=time.perf_counter()-start;tick=time.perf_counter()
    conflicts,excluded,contrasts=collect(a.population)
    for n,v in (('conflicts',conflicts),('excluded-cases',excluded),('contrasts',contrasts)):publish(a.output/(n+'.json'),v)
    phases['linear_grouping_and_overlay_publication']=time.perf_counter()-tick;tick=time.perf_counter()
    cold=audit_raw(a.population,source,*[json.loads((a.output/(n+'.json')).read_bytes()) for n in ('conflicts','excluded-cases','contrasts')])
    summary={}
    for split in ('train','calibration','evaluation'):
        gs=[g for g in contrasts if g['members'][0]['partition']==split]
        summary[split]=dict(contrast_groups=len(gs),opposite_label_pairs=sum(g['positive']*g['negative'] for g in gs),documents=len({g['document'] for g in gs}),
            split_components=len({g['members'][0]['group'] for g in gs}),questions=len({m['case'] for g in gs for m in g['members']}),excluded_cases=sum(r['partition']==split for r in excluded))
    after={n:file_sha(Path(n)) for n in before};check(before==after,'SOURCE_DRIFT')
    phases['independent_raw_readback_and_immutable_parents']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='multirc-anchor-quality-v1',before=before,after=after,summary=summary,raw_readback=cold,
        conflict_reasons=dict(Counter(g['reason'] for g in conflicts)),files={n+'.json':file_sha(a.output/(n+'.json')) for n in ('conflicts','excluded-cases','contrasts')},
        operational_rule='Exclude listed cases from new anchor training; keep original records and partitions; report exclusions beside full-cohort evaluation.',
        original_labels_changed=False,original_partitions_changed=False,diagnostic_metadata_only=True,model_input=False,scientific_admission=False,served=False,
        code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked'))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),summary=summary,raw_readback=cold)))

if __name__=='__main__':main()

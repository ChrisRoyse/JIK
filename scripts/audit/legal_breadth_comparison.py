"""Recount the full matched breadth/repetition experiment without rerunning models."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
import torch
from audit import legal_breadth_comparison_source as source
from audit import legal_breadth_comparison_metrics as metrics
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need


def check_summary(cell,s):
    for arm in ('breadth','repeated'):
        original=s.refs[arm]['evaluations']['final']['summary'][cell['side']+':'+cell['source']][cell['subset']]
        kind,name=cell['metric'].split('/',1)
        n=original.get('permuted_target_rows',0) if kind=='target_permutation_gap' else original['rows']
        need(n==cell['rows'],'BREADTH_COMPARE_SUMMARY_SUPPORT')
        if not n:continue
        actual=cell['means'][arm]
        if kind in ('accuracy','prefix_accuracy'):
            expected={k:v[int(name)] for k,v in original[kind].items() if v[int(name)] is not None}
        elif kind in ('conditional_ce','conditional_brier'):
            expected=dict(model=original[kind][int(name)])
        elif kind=='target_permutation_gap':expected=dict(model=original['correct_minus_permuted_log_score'][int(name)])
        elif kind=='count_accuracy':expected=dict(model=original['native'][name]['count_accuracy'])
        else:
            need(kind=='ordered_distance','BREADTH_COMPARE_METRIC_KIND')
            expected={k:original['native'][name]['ordered_'+k] for k in ('model','copy_current','train_mean')}
        need(actual.keys()==expected.keys() and all(abs(actual[k]-expected[k])<=2e-12 for k in actual),
             'BREADTH_COMPARE_SUMMARY_RECOUNT')


def rejected(action,code):
    try:action()
    except ValueError as exc:
        need(code in str(exc),'BREADTH_COMPARE_WRONG_CONTROL');return str(exc)
    raise ValueError('BREADTH_COMPARE_CONTROL_ACCEPTED:'+code)


def controls(s,pair,cell):
    a,b=pair;out={}
    for name,field,code in [('target','target','BREADTH_COMPARE_TARGET_ORDER'),
                            ('order','uids','BREADTH_COMPARE_TARGET_ORDER')]:
        bad=deepcopy(b);bad[field].reshape(-1)[0]+=1
        out[name]=rejected(lambda:source.validate_pair(a,bad,s.roster),code)
    bad=deepcopy(b);bad['masks']['all'][0]=False
    out['mask']=rejected(lambda:source.validate_pair(a,bad,s.roster),'BREADTH_COMPARE_MASKS')
    bad=deepcopy(b);bad['predictions']['copy_current'][0,0]+=1
    out['baseline']=rejected(lambda:source.validate_pair(a,bad,s.roster),'BREADTH_COMPARE_BASELINE')
    bad=deepcopy(b);bad['native'][s.roster[0]]['ordered_distances'][0,1]+=.1
    out['native_baseline']=rejected(lambda:source.validate_pair(a,bad,s.roster),'BREADTH_COMPARE_NATIVE_BASELINE')
    bad=deepcopy(b);bad['conditional_ce'][0,0]+=.1
    out['saved_score']=rejected(lambda:metrics.arrays(bad),'BREADTH_COMPARE_SCORE_RECOUNT')
    bad=deepcopy(s.selection);bad[1]['case_component']=bad[0]['case_component']
    out['duplicate_case']=rejected(lambda:s.validate_selection(bad),'BREADTH_COMPARE_CASES')
    out['missing_document']=rejected(lambda:s.validate_selection(s.selection[:-1]),'BREADTH_COMPARE_CASES')
    out['wrong_pin']=rejected(lambda:source.raw(s.root/source.REFS['breadth'][0]/'report.json','0'*64),
                              'BREADTH_COMPARE_BYTES')
    bad=deepcopy(cell);bad['means']['breadth']['model']+=.1
    out['changed_summary']=rejected(lambda:check_summary(bad,s),'BREADTH_COMPARE_SUMMARY_RECOUNT')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    s=source.Source(a.root);code=source.code();records=[];first=None
    phases['authority']=time.perf_counter()-tick;tick=time.perf_counter()
    with (a.output/'documents.jsonl').open('x') as f:
        for selected in s.selection:
            pair=s.pair(selected)
            if first is None:first=pair
            rows=metrics.document(*pair,selected)
            for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
            records.extend(rows)
    phases['read_authenticate_recount']=time.perf_counter()-tick;tick=time.perf_counter()
    draws=a.output/'draws';draws.mkdir();cells=metrics.aggregate(records,draws)
    for cell in cells:check_summary(cell,s)
    write_committed(a.output/'cells.json',cells)
    phases['paired_cases_intervals']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(s,first,next(c for c in cells if c['status']=='measured'))
    write_committed(a.output/'controls.json',negative)
    write_committed(a.output/'sources.json',dict(files=s.files,refs=source.REFS,schedule=s.refs['schedule']['summary']))
    s.fresh();need(source.code()==code and not torch.cuda.is_initialized(),'BREADTH_COMPARE_STATE_CHANGED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'BREADTH_COMPARE_PHASES')
    report=dict(complete=True,acceptance='author-checked',number_status='measured',code=code,files=files,
        refs=source.REFS,documents=len(s.selection),records=len(records),cells=len(cells),controls=negative,
        seconds=seconds,phases=phases,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_bytes=sum(x['bytes'] for x in s.files.values()),bootstrap_draws=metrics.DRAWS,
        interval_scope='descriptive paired case resampling; one training seed; no multiplicity-adjusted admission',
        evaluation_scope='existing sources; case-disjoint from fit; exposed after evaluation; not external sealed generalization',
        shared_representation_pretraining=True,subset_presence_against='full TRAIN, shared by both arms',
        target_permutation_is_input_shuffle=False,native_metric='ordered distance; lower is better',
        native_forward_replayed_here=False,repeated_training_cold_replay='pending; aggregate readback is a different claim',
        scientific_admission=False,model_promotion=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,seconds=seconds,documents=len(s.selection),cells=len(cells))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

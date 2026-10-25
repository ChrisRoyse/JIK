"""Independent original-journal/metadata recount of source exposure."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


def check(ok, code):
    if not ok: raise ValueError('MNLM_EXPOSURE_READBACK_'+code)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path, pin):
    raw = Path(path).read_bytes()
    check(hashlib.sha256(raw).hexdigest() == pin, 'PIN')
    return json.loads(raw)


def same(actual, expected):
    if type(expected) is float:
        check(type(actual) is float and math.isfinite(actual)
              and math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12), 'NUMBER')
    elif isinstance(expected, dict):
        check(type(actual) is dict and set(actual) == set(expected), 'KEYS')
        for key in expected: same(actual[key],expected[key])
    elif isinstance(expected, list):
        check(type(actual) is list and len(actual) == len(expected), 'LENGTH')
        for x,y in zip(actual,expected,strict=True): same(x,y)
    else:
        check(type(actual) is type(expected) and actual == expected, 'VALUE')


def match(row, expected):
    # Compact rows retain original floats exactly; only aggregate arithmetic has a tolerance.
    check(type(row) is dict and row == expected, 'JOINED_ROW')


def refuses(fn):
    try: fn()
    except ValueError as error:
        check(str(error).startswith('MNLM_EXPOSURE_READBACK_'), 'NEGATIVE_CODE')
        return str(error)
    raise AssertionError('MNLM_EXPOSURE_READBACK_NEGATIVE_ACCEPTED')


def run(a):
    began = time.perf_counter(); cpu = time.process_time()
    report = load(a.root/'report.json',a.pin)
    check(report['complete'] and report['schema']=='placement-source-exposure-v1'
          and report['optimizer_updates']==0 and report['gpu_used'] is False, 'SCOPE')
    inputs = report['inputs']
    boundary_path = next(Path(p) for p in inputs if p.endswith('/seal/boundary.json'))
    corpus_path = next(Path(p) for p in inputs if p.endswith('/legal-source-corpus-v1-20261022/report.json'))
    boundary = load(boundary_path,inputs[str(boundary_path)])
    corpus = load(corpus_path,inputs[str(corpus_path)])
    plan = boundary['binding']['source']['plan']['train_documents']; selected = set(plan)
    check(len(plan)==len(selected)==91540 and boundary['branches']['shared']['step']==len(plan), 'PLAN')
    metadata = corpus_path.parent/'documents.jsonl'; journal = boundary_path.parent/'shared/steps.jsonl'
    check(set(inputs)=={str(boundary_path),str(corpus_path),str(metadata),str(journal)}, 'INPUTS')
    check(inputs[str(metadata)]==corpus['files']['documents.jsonl']['sha256']
          ==boundary['binding']['source']['metadata']['documents']
          and inputs[str(journal)]==boundary['files']['shared/steps.jsonl'], 'BINDINGS')
    mapping = {}; metadata_hash = hashlib.sha256(); metadata_count = 0
    with metadata.open('rb') as stream:
        for index,raw in enumerate(stream):
            metadata_hash.update(raw); value=json.loads(raw); metadata_count += 1
            if index in selected:
                mapping[index]=(value['source_path'],int(value['atom_end'])-int(value['atom_start'])-1)
    check(metadata_hash.hexdigest()==inputs[str(metadata)] and len(mapping)==len(plan), 'METADATA')
    counts=Counter(mapping[n][0] for n in plan); seen=Counter(); buckets={}; runs=[]; transitions=Counter()
    journal_hash=hashlib.sha256(); compact_hash=hashlib.sha256(); first=None
    with journal.open('rb') as original,(a.root/'rows.jsonl').open('rb') as compact:
        for index,document in enumerate(plan):
            raw=original.readline(); saved=compact.readline()
            check(raw.endswith(b'\n') and saved.endswith(b'\n'), 'SHORT')
            journal_hash.update(raw); compact_hash.update(saved)
            r=json.loads(raw); u=r['update']; row=json.loads(saved); source,targets=mapping[document]
            step=index+1; quarter=(seen[source]*4)//counts[source]
            check(r['cursor']==index and u['step']==step and u['document']==document
                  and u['targets']==targets and targets>0, 'SOURCE_ROW')
            expected=dict(step=step,document=document,source=source,targets=targets,
                          source_training_ordinal=seen[source],quarter=quarter,loss=u['loss'],journal_hash=r['hash'])
            match(row,expected)
            if first is None: first=deepcopy(expected)
            seen[source] += 1
            for key in ((source,None),(source,quarter)):
                b=buckets.setdefault(key,dict(updates=0,targets=0,first_step=step,last_step=step,
                                              loss=Fraction(0),weighted=Fraction(0)))
                b['updates']+=1; b['targets']+=targets; b['last_step']=step
                b['loss']+=Fraction.from_float(u['loss'])
                b['weighted']+=Fraction.from_float(targets*u['loss'])
            if not runs or runs[-1]['source'] != source:
                if runs: transitions[(runs[-1]['source'],source)]+=1
                runs.append(dict(source=source,first_step=step,last_step=step,updates=0,targets=0))
            b=runs[-1];b['last_step']=step;b['updates']+=1;b['targets']+=targets
        check(not original.read(1) and not compact.read(1), 'EXTRA')
    check(journal_hash.hexdigest()==inputs[str(journal)]
          and compact_hash.hexdigest()==report['output_files']['rows.jsonl'], 'STREAM_PINS')
    total=sum(buckets[(s,None)]['targets'] for s in counts);check(total==6455636, 'TARGET_TOTAL')
    def reduce(b):
        return dict(updates=b['updates'],targets=b['targets'],first_step=b['first_step'],last_step=b['last_step'],
                    document_mean_loss=float(b['loss'])/b['updates'],target_weighted_loss=float(b['weighted'])/b['targets'])
    sources=[]
    for name in sorted(counts):
        b=buckets[(name,None)];value=dict(source=name,**reduce(b))
        value.update(update_fraction=b['updates']/len(plan),target_fraction=b['targets']/total,
                     quarters=[dict(quarter=q,**reduce(buckets[(name,q)])) for q in range(4)])
        sources.append(value)
    expected=dict(updates=len(plan),targets=total,metadata_documents=metadata_count,sources=sources,runs=runs,
                  transitions=[dict(left=l,right=r,count=n) for (l,r),n in sorted(transitions.items())],
                  training_order='frozen_document_plan',loss_scope='online_before_each_update_not_heldout_accuracy',
                  causal_order_effect_established=False,forgetting_established=False,architecture_winner=False)
    summary=load(a.root/'summary.json',report['output_files']['summary.json']);same(summary,expected)
    controls={}
    for field,value in (('document',plan[1]),('source','unobserved-source'),('targets',first['targets']+1),
                        ('quarter',1),('loss',first['loss']+1),('journal_hash','0'*64),('step',2)):
        wrong=dict(first);wrong[field]=value
        controls[field]=refuses(lambda:match(wrong,first))
    controls['missing_row']=refuses(lambda:match(None,first))
    changed=deepcopy(summary);changed['sources'][0]['updates']+=1
    controls['summary']=refuses(lambda:same(changed,expected))
    controls['wrong_pin']=refuses(lambda:load(a.root/'report.json','0'*64))
    check(all(sha(p)==h for p,h in inputs.items()),'SOURCE_CHANGED')
    check(all(sha(p)==h for p,h in report['code'].items()),'PRODUCER_CODE')
    check(sha(a.root/'report.json')==a.pin,'PUBLICATION_CHANGED')
    value=dict(complete=True,cold_readback=True,producer=a.pin,inputs=inputs,
               rows_pin=compact_hash.hexdigest(),summary_pin=report['output_files']['summary.json'],
               updates=len(plan),targets=total,source_runs=len(runs),negative_controls=controls,
               original_journal_and_metadata_recounted=True,source_bytes_unchanged=True,
               aggregate_absolute_tolerance=1e-12,code={str(Path(__file__).resolve()):sha(__file__)},
               seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               acceptance='author-checked',number_status='measured',gpu_used=False)
    with a.output.open('xb') as stream:
        stream.write(json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

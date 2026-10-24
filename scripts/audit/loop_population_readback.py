"""Independent source-field and provenance readback for identity loop inputs."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

PROFILE='5f839acceb76bd4bd2db269af9651b0df9e072b4a001588068434391a42ea4f6'
RECORDS='2f2149ca502f2db2713e1262e1c4eef3fa581e8b405e9c61cedf84c12d217a13'


def check(ok, code):
    if not ok:
        raise ValueError('MNLM_LOOP_READBACK_'+code)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readback(root, source):
    report=json.loads((root/'report.json').read_bytes())
    check(report['source_pins']==dict(profile=PROFILE,records=RECORDS), 'SOURCE_PIN')
    check(sha(source/'profile.md')==PROFILE and sha(source/'records.jsonl')==RECORDS, 'SOURCE_BYTES')
    check(report['complete'] and not report['errors'] and report['units']==77 and report['identity_components']==10 and report['corrections']==22,
          'COUNTS')
    check(all(report[n] is False for n in ('scientific_admission','meaning_asserted','behavioural_compliance_established','served')), 'SCOPE')
    check(set(report['files'])=={'references.jsonl','identity/summary.json','identity/chunks/atoms.parquet'}, 'FILES')
    for name,pin in report['files'].items():
        check(sha(root/name)==pin['sha256'] and (root/name).stat().st_size==pin['bytes'], 'HASH')
    summary=json.loads((root/'identity/summary.json').read_bytes())
    check(report['families']=={'identity':dict(ready=True,summary_sha256=sha(root/'identity/summary.json'))}
          and summary['chunks']==77 and summary['chunks_sha256']==sha(root/'identity/chunks/atoms.parquet'), 'JOIN')
    source_rows={r['id']:r for r in map(json.loads,(source/'records.jsonl').read_bytes().splitlines())}
    refs=list(map(json.loads,(root/'references.jsonl').read_bytes().splitlines()))
    rows=pq.read_table(root/'identity/chunks/atoms.parquet').to_pylist()
    check(len(refs)==len(rows)==77, 'LENGTH')
    wanted={(r['id'],'text') for r in source_rows.values() if r['type']=='standard'}
    wanted|={(r['id'],f) for r in source_rows.values() if r['type']=='correction' for f in ('echo','response','not_chosen')}
    seen=set()
    for i,(ref,row) in enumerate(zip(refs,rows)):
        check(ref['unit']==row['row']==i and row['chunk']==0 and ref['partition']=='development_exposed'
              and ref['epistemic_scope']=='operator_profile_interpretation', 'ROW')
        check(hashlib.sha256(row['text'].encode()).hexdigest()==ref['text_sha256'], 'TEXT_HASH')
        if i==76:
            profile=(source/'profile.md').read_bytes().decode('utf-8')
            lo,hi=ref['source_char_span']
            check(ref['role']=='probe_only' and ref['record']=='profile' and ref['field'] is None
                  and ref['source_sha256']==PROFILE and row['entity']=='profile:overflow-probe'
                  and row['start']==lo and row['end']==hi and 1800<=hi-lo<=3000 and row['text']==profile[lo:hi], 'PROBE')
        else:
            pair=ref['record'],ref['field']
            check(pair in wanted and pair not in seen, 'ROSTER')
            seen.add(pair);r=source_rows[pair[0]]
            check(row['text']==r[pair[1]] and ref['refs']==r['refs'] and ref['source_tag']==r['tag']
                  and ref['exposes']==r.get('exposes',[]) and ref['source_sha256']==RECORDS
                  and row['entity']==':'.join(pair) and row['start']==0 and row['end']==len(row['text'])
                  and ref['role']==('standard' if pair[1]=='text' else pair[1]), 'SOURCE_FIELD')
    check(seen==wanted, 'COVERAGE')
    return dict(complete=True,units=len(rows),profile_sha256=PROFILE,records_sha256=RECORDS,
                report_sha256=sha(root/'report.json'),acceptance='author-checked')

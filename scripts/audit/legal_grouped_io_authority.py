"""Pinned physical authority for the grouped native document consumer."""
import json
from pathlib import Path

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

PROOFS = {
    'qualification': ('legal-grouped-document-qualification-v1-20261023/report.json',
                      '6e0bde299f7eeaa61b9b9a8b3875feb10c31a9971c2a079a6494ba8bd74f1905'),
    'cold': ('legal-grouped-document-cold-v1-20261023/report.json',
             '7a62c62742a8fc97ab92252e43b03b102e0df4b7fe830183f6eb3746aee80639'),
    'readback': ('phase77-grouped-qualification-recount.json',
                 '54f15aa618fb428a21cef4617734ec98f9b7654e9d85498afd26cb7211de03a0'),
    'cold_readback': ('phase77-grouped-cold-recount.json',
                      '1adb631e280f6f6d1dd28d7a5c7eaf37dc8fb677bdc913b0ccebaf6a9f433d5b')}


def validate(records):
    q,c,r,s=(records[n] for n in PROOFS)
    need(all(x['complete'] and not x['scientific_admission'] for x in (q,c,r,s)), 'GROUPED_IO_PROOF_COMPLETE')
    need(q['binding']==c['binding'] and c['previous_pin']==PROOFS['qualification'][1]
         and q['documents']==c['documents'] and q['controls']==c['controls'], 'GROUPED_IO_COLD_AUTHORITY')
    need(all(x['all_native_bytes_exact'] and x['all_consumer_inputs_exact'] and x['all_handles_closed'] for x in (q,c)),
         'GROUPED_IO_CONSUMER_AUTHORITY')
    need(r['input_pin']==PROOFS['qualification'][1] and s['input_pin']==PROOFS['cold'][1]
         and all(x['all_native_bytes_exact'] and x['all_consumer_fields_exact'] and x['no_unrequested_reads'] for x in (r,s))
         and r['counts']==s['counts'] and r['checked_native_slot_arrays']==s['checked_native_slot_arrays']==24756,
         'GROUPED_IO_INDEPENDENT_AUTHORITY')
    need(q['binding']['handle_limit']==64 and q['binding']['tile']==128 and q['binding']['output_bytes']==16<<20,
         'GROUPED_IO_PHYSICAL_POLICY')


def authenticate(root):
    paths={};records={}
    for name,(relative,pin) in PROOFS.items():
        path=Path(root)/relative;need(file_sha(path)==pin,'GROUPED_IO_PROOF_PIN')
        records[name]=json.loads(path.read_bytes());paths[path]=stamp(path)
        for relative,expected in records[name].get('files',{}).items():
            f=path.parent/relative;need(file_sha(f)==expected,'GROUPED_IO_PHYSICAL_PROOF');paths[f]=stamp(f)
    validate(records)
    for n,pin in records['qualification']['binding']['code'].items():
        need(file_sha(Path(__file__).resolve().parents[1]/n)==pin,'GROUPED_IO_SOURCE_CHANGED')
    def fresh():
        need(all(stamp(p)==s for p,s in paths.items()),'GROUPED_IO_PROOF_CHANGED')
    fresh()
    return dict(proofs={n:p for n,(_,p) in PROOFS.items()},
                source=records['qualification']['binding']['source'],
                cache=records['qualification']['binding']['cache']),fresh


def require_match(root, pin, candidate_pin, current_code):
    path=Path(root)/'report.json';need(file_sha(path)==pin,'GROUPED_MATCH_PIN')
    r=json.loads(path.read_bytes())
    need(r['complete'] and r['candidate_pin']==candidate_pin and r['device']=='cuda'
         and r['candidate_code']==current_code and r['all_updates_exact'] and r['all_checkpoint_tensors_exact']
         and r['all_evaluation_payloads_exact'] and r['architectures']==['shared','own_slot','whole_proposals'],
         'GROUPED_MATCH_FULL_AUTHORITY')
    need(r['updates']==15 and r['training_rows']==12312 and r['checkpoints']==18,'GROUPED_MATCH_SCOPE')
    for relative,expected in r['files'].items():need(file_sha(Path(root)/relative)==expected,'GROUPED_MATCH_PHYSICAL')
    return dict(path=str(root),pin=pin,baseline=r['baseline_pin'],candidate=candidate_pin)

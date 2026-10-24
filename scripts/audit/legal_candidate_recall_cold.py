"""Fresh-index readback of every persisted candidate-profile result."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
from audit import legal_candidate_recall_readback as source
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane,pinned
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    report=pinned(a.previous/'report.json',a.previous_pin)
    need(report['complete'] and report['plan']['code']==source.code(),'CANDIDATE_COLD_AUTHORITY')
    for name,pin in report['files'].items():need(file_sha(a.previous/name)==pin,'CANDIDATE_COLD_FILE_PIN')
    lane=CertifiedIndexLane(a.root/source.CERTIFICATE[0],source.CERTIFICATE[1])
    certificate=pinned(a.root/source.CERTIFICATE[0],source.CERTIFICATE[1]);ref=certificate['result']['source_receipt']
    receipt=pinned(Path(ref['path']),ref['pin']);exact={r['uid']:r['exact_hits'] for r in receipt['assessment']['rows']}
    need(list(exact)==report['plan']['query_uids'],'CANDIDATE_COLD_SCOPE')
    initial=lane.index.index.expansion_search;results={}
    try:
        for count,expansion in source.PROFILES:
            name=f'candidates{count}-expansion{expansion}'
            rows=pinned(a.previous/(name+'.json'),report['profiles'][name]['pin'])
            lane.index.index.expansion_search=expansion
            actual=[source.query(lane,uid,count) for uid in exact]
            need(actual==rows['rows'],'CANDIDATE_COLD_QUERY_REPLAY')
            groups=source.recount(rows['rows'],exact,report['plan']['groups'])
            need(groups==rows['groups']==report['profiles'][name]['groups'],'CANDIDATE_COLD_RECOUNT')
            results[name]=dict(queries=len(actual),groups=groups,all_candidate_sets_scores_work_exact=True)
    finally:lane.index.index.expansion_search=initial
    lane.fresh();seconds=time.perf_counter()-began
    r=dict(complete=True,previous_pin=a.previous_pin,producer_code=source.code(),code_sha256=file_sha(Path(__file__)),
           results=results,seconds=seconds,promoted=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),seconds=seconds)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--previous-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

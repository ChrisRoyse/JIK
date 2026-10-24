"""Stream legal native galleries into separate instrument indexes and measure retrieval recall."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from claims.process_conditions import write_committed
from kernel.native_window_index import POLICY,NativeWindowIndex,exact_queries,multiplex,publish
from text.constellation_corpus import digest,file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs


class BankLane:
    def __init__(self,bank,name):self.bank,self.name=bank,name

    def __call__(self,uids):
        batch=self.bank.read_many(self.name,uids,memory_bytes=64<<20)
        return [batch.values[batch.offsets[i]:batch.offsets[i+1]] for i in range(len(uids))]


class ProofLane:
    def __init__(self,root,pin,source,name,expected_lens):
        root=Path(root);need(file_sha(root/'report.json')==pin,'ASSOCIATION_PROOF_PIN')
        report=json.loads((root/'report.json').read_bytes());need(report['complete'],'ASSOCIATION_PROOF_INCOMPLETE')
        for filename,ref in report['files'].items():need(file_sha(root/filename)==ref['sha256'],'ASSOCIATION_PROOF_FILE')
        records=json.loads((root/'inputs.json').read_bytes());self.uids=np.array([x['index'] for x in records],dtype='<i8')
        need(np.all(self.uids[1:]>self.uids[:-1]) and report['source']['pin']==source.pin,'ASSOCIATION_PROOF_SOURCE')
        for row in records:
            need(digest(row['text'].encode())==row['text_sha256']==source.arrays['hashes'][row['index']].tobytes().hex(),
                 'ASSOCIATION_PROOF_REAL_INPUT')
        need(report['instruments'][name]['lens_id']==expected_lens,'ASSOCIATION_PROOF_LENS')
        need(file_sha(root/name/'report.json')==report['instruments'][name]['pin'],'ASSOCIATION_PROOF_LANE')
        lane=json.loads((root/name/'report.json').read_bytes())
        for filename,ref in lane['files'].items():need(file_sha(root/name/filename)==ref['sha256'],'ASSOCIATION_PROOF_NATIVE_FILE')
        with np.load(root/name/'reference.npz',allow_pickle=False) as saved:
            self.values,self.bounds=saved['vectors'],saved['bounds']
        self.values.flags.writeable=False;self.bounds.flags.writeable=False
        self.rows={int(uid):i for i,uid in enumerate(self.uids)}

    def __call__(self,uids):
        need(all(uid in self.rows for uid in uids),'ASSOCIATION_PROOF_UNKNOWN_UID')
        return [self.values[self.bounds[self.rows[uid]]:self.bounds[self.rows[uid]+1]] for uid in uids]


def assess(index,read,query_uids):
    queries=dict(zip(query_uids,read(query_uids)));start=time.perf_counter()
    approximate=[index.query(uid,values,read,lens_id=index.report['binding']['lens_id']) for uid,values in queries.items()]
    ann_seconds=time.perf_counter()-start;start=time.perf_counter();exact=exact_queries(index.uids,queries,read)
    exact_seconds=time.perf_counter()-start
    for row in approximate:
        row['exact_hits']=exact[row['uid']];expected={u for u,_ in row['exact_hits']}
        row['recovered']=len(expected&{u for u,_ in row['hits']});row['recall']=row['recovered']/len(expected)
    recall=sum(r['recovered'] for r in approximate)/(len(approximate)*POLICY['top_k'])
    return dict(rows=approximate,recall=recall,recall_gate=.95,qualified=recall>=.95,
                approximate_seconds=ann_seconds,exact_seconds=exact_seconds)


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    source=ReuseInputs(a.source,a.source_pin);continuations=Continuations(a.continuations,a.continuations_pin)
    need(continuations.report['source']['pin']==source.pin,'ASSOCIATION_CONTINUATION_SOURCE')
    bank=NativeBatchSession(a.bank,a.bank_pin,source);roster=bank.config['roster']
    need(0<a.prefix_inputs<=bank.config['inputs'],'ASSOCIATION_PREFIX')
    uids=np.arange(a.prefix_inputs,dtype='<i8');query=np.flatnonzero((continuations.presence[:len(uids)]&1)==0)[:32].tolist()
    need(len(query)==32,'ASSOCIATION_QUERY_SUPPORT')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter();lanes={};native_results={}
    for name in roster:
        lane_start=time.perf_counter();read=BankLane(bank,name)
        binding=dict(source_pin=source.pin,bank_pin=bank.pin,lens_id=bank.config['instruments'][name]['lens_id'],
                     instrument=name,scope='complete_declared_prefix_retrieval_diagnostic',prefix_inputs=len(uids))
        root=a.output/name;pin=publish(root/'prefix',uids,read,binding)
        index=NativeWindowIndex(root/'prefix',pin,binding);result=assess(index,read,query)
        prefix_pin=write_committed(root/'prefix-assessment.json',result);native_results[name]=result['rows']
        proof=ProofLane(a.proof,a.proof_pin,source,name,binding['lens_id'])
        proof_binding={**binding,'scope':'separate_real_canonical_multiwindow_gallery','proof_pin':a.proof_pin,
                       'prefix_inputs':None}
        proof_pin=publish(root/'multiwindow',proof.uids,proof,proof_binding)
        proof_index=NativeWindowIndex(root/'multiwindow',proof_pin,proof_binding)
        # Every prior input is evaluated, including every real multi-window reading.
        variable=assess(proof_index,proof,proof.uids.tolist())
        assessment_pin=write_committed(root/'multiwindow-assessment.json',variable)
        lanes[name]=dict(prefix=dict(path=str(root/'prefix'),pin=pin,assessment_pin=prefix_pin,recall=result['recall']),
                         multiwindow=dict(path=str(root/'multiwindow'),pin=proof_pin,assessment_pin=assessment_pin,
                                          recall=variable['recall']),seconds=time.perf_counter()-lane_start)
        print(json.dumps(dict(instrument=name,**lanes[name])),flush=True)
    phases['construct_and_assess_lanes']=time.perf_counter()-tick;tick=time.perf_counter()
    union=multiplex(native_results);union_pin=write_committed(a.output/'multiplex.json',union)
    source.fresh();continuations.fresh();bank.fresh()
    phases['multiplex_and_freshness']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'ASSOCIATION_PHASE_ATTRIBUTION')
    report=dict(schema='legal-native-associations-proof-v1',complete=True,lanes=lanes,roster=roster,
        source=dict(path=str(a.source),pin=source.pin),bank=dict(path=str(a.bank),pin=bank.pin),
        continuations=dict(path=str(a.continuations),pin=a.continuations_pin),
        proof=dict(path=str(a.proof),pin=a.proof_pin),query_uids=query,prefix_inputs=len(uids),multiplex_pin=union_pin,
        multiplex_edges=len(union['edges']),single_instrument_edges=sum(len(x['readings'])==1 for x in union['edges']),
        reader_stats=bank.stats,phases_seconds=phases,seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code_sha256=file_sha(Path(__file__)),index_code_sha256=file_sha(Path(__file__).with_name('native_window_index.py')),
        new_encoder_calls=0,predictor_updates=0,scientific_admission=False,full_legal_corpus_index=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=seconds,
        multiplex_edges=report['multiplex_edges'],single_instrument_edges=report['single_instrument_edges'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','bank','continuations','proof'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--prefix-inputs',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

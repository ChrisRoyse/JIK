"""Real complete-document consumer comparison under unchanged source/label gates."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_cached_target_readback as cached
from audit import legal_training_statistics as training
from audit.legal_coalesced_io_readback import traced_reads
from audit.legal_native_io_readback import controls, usage
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import legal_cached_documents as temporal, legal_grouped_documents as grouped
from model.legal_cached_targets import LegalCachedTargets
from model.legal_native_resident_stream import ResidentNativeFrames
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need

PRIOR=('legal-coalesced-native-io-qualification-v1-20261023',
       '9e546d69ec8c806f71cd13cfc5478656f234fb8551ab3fe70029b972306c295d')
CACHE=('legal-target-cache-full-full-v2-20261023','77319fe3eaac3ea9db2ede167519a418151133b5c4641ef19ce6c901d2d5fc72')
COLD=('legal-target-cache-full-cold-v2-20261023','d4498a44902db8e642ab368e1e1ead435e7e87caed44cae274245215b2a9a77f')


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(cached.code(),**{n:file_sha(root/n) for n in [
        'model/legal_native_resident_stream.py','model/legal_cached_documents.py','model/legal_grouped_documents.py',
        'audit/legal_coalesced_io_readback.py','audit/legal_native_io_readback.py',
        'audit/legal_grouped_document_readback.py','audit/legal_grouped_document_recount.py']})


def run(a):
    began=tick=time.perf_counter();cpu0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pins=code()
    for name,pin in pins.items():
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(path)==pin,'GROUPED_SOURCE_COPY')
    prior=training.pinned(a.root/PRIOR[0]/'report.json',PRIOR[1])
    for name,pin in prior['files'].items():need(file_sha(a.root/PRIOR[0]/name)==pin,'GROUPED_PRIOR_BYTES')
    data,continuation,unused,_,_,authority=cached.producer.prepare(a.root)
    unused.close()
    cache=LegalCachedTargets(a.root/CACHE[0],CACHE[1],a.root/COLD[0],COLD[1],data,authority)
    args=(data,a.root/training.DEFAULTS['audit'][0],training.DEFAULTS['audit'][1])
    readers={n:ResidentNativeFrames(*args) for n in ['temporal','grouped']}
    functions=dict(temporal=temporal.read_document,grouped=grouped.read_document)
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    for row in prior['plan']:
        item,_=grouped.read_document(data,readers['grouped'],cache,row['document'],side='train',fit=True)
        need(item['native_pin']==row['native'] and item['uids']==row['uids'],'GROUPED_PRIOR_NATIVE')
        original=torch.load(a.root/PRIOR[0]/f'document-{row["document"]}.pt',map_location='cpu',weights_only=True)
        need(torch.equal(item['codes'],original['codes'][1:])
             and all(torch.equal(item['counts'][n],original['counts'][n][1:]) for n in data.roster),
             'GROUPED_PRIOR_LABELS')
    phases['native_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    # Fresh reader handles for both arms; authenticated descriptors are shared.
    for reader in readers.values():reader.close()
    readers={n:ResidentNativeFrames(*args) for n in functions}
    measurements=[];documents=[]
    for repeat,order in enumerate([['temporal','grouped'],['grouped','temporal'],['temporal','grouped']]):
        for mode in order:
            reader=readers[mode]
            for row in prior['plan']:
                before=usage();start=time.perf_counter();c0=time.process_time()
                item,cost=functions[mode](data,reader,cache,row['document'],side='train',fit=True)
                seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-c0
                after=usage();need(item['native_pin']==row['native'],'GROUPED_TIMED_NATIVE')
                if repeat==0:
                    path=a.output/f'{mode}-document-{row["document"]}.pt';torch.save(item,path)
                    documents.append(dict(mode=mode,document=row['document'],state=fingerprint(item),path=path.name))
                need(fingerprint(item)==next(r['state'] for r in documents if r['mode']==mode and r['document']==row['document']),
                     'GROUPED_REPEATED_CONSUMER_STATE')
                measurements.append(dict(mode=mode,repeat=repeat,document=row['document'],seconds=seconds,
                                         cpu_seconds=cpu_seconds,cost=cost,before=before,after=after))
    for row in prior['plan']:
        states=[r['state'] for r in documents if r['document']==row['document']]
        need(len(states)==2 and states[0]==states[1],'GROUPED_CONSUMER_NOT_EXACT')
    stats={n:dict(reader=r.stats.copy(),handles=r.io_stats.copy()) for n,r in readers.items()}
    phases['matched_consumers']=time.perf_counter()-tick;tick=time.perf_counter()
    for row in prior['traces']:
        cursor=[0]
        with traced_reads(a.output/f'{row["label"]}-reads.jsonl',cursor):
            frames=grouped.read_frames(readers['grouped'],row['uids'])
        need(fingerprint(frames)==row['native'],'GROUPED_TRACE_NATIVE')
        torch.save(frames,a.output/f'{row["label"]}-frames.pt')
    phases['actual_range_trace']=time.perf_counter()-tick;tick=time.perf_counter()
    seed=prior['plan'][0]['uids'];r=readers['grouped'];before=dict(r.stats)
    negatives=dict(nontrain_fit=rejected(lambda:grouped.read_document(data,r,cache,prior['plan'][0]['document'],side='query',fit=True)),
                   boolean_uid=rejected(lambda:grouped.read_frames(r,[True])),
                   empty=rejected(lambda:grouped.read_frames(r,[])),
                   consumer_order=rejected(lambda:grouped.DocumentReader(r,seed).frames(seed[::-1])))
    need(before==r.stats,'GROUPED_REFUSAL_READ_VALUES')
    duplicate=grouped.read_frames(r,[seed[0],seed[0]])
    for n in data.roster:
        need(duplicate[0][n].data_ptr()!=duplicate[1][n].data_ptr(),'GROUPED_DUPLICATE_ALIAS')
        saved=duplicate[1][n].clone();duplicate[0][n][0,0]+=1
        need(torch.equal(duplicate[1][n],saved),'GROUPED_DUPLICATE_MUTATION')
    readers['temporal'].close();negatives.update(controls(r,seed,a.output))
    cache.close();continuation.fresh();data.fresh();data.source.fresh()
    binding=dict(code=pins,prior=PRIOR[1],source=data.binding,cache=cache.binding,handle_limit=64,
                 tile=128,output_bytes=16<<20,read_order='source_uid',consumer_order='original_document')
    if a.previous:
        old=training.pinned(a.previous/'report.json',a.previous_pin)
        need(old['complete'] and old['binding']==binding and old['documents']==documents
             and old['controls']==negatives,'GROUPED_COLD_BINDING')
        for row in prior['traces']:
            for suffix in ['frames.pt','reads.jsonl']:
                name=f'{row["label"]}-{suffix}'
                need(file_sha(a.previous/name)==old['files'][name]==file_sha(a.output/name),'GROUPED_COLD_PHYSICAL')
    need(all(not r.maps and r.closed for r in readers.values()) and not torch.cuda.is_initialized(),'GROUPED_RESOURCE_TEARDOWN')
    phases['controls_and_cold']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'GROUPED_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,documents=documents,measurements=measurements,stats=stats,
        controls=negatives,files=files,previous_pin=a.previous_pin,phases_seconds=phases,seconds=seconds,
        cpu_seconds=time.process_time()-cpu0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        all_native_bytes_exact=True,all_consumer_inputs_exact=True,all_handles_closed=True,
        scientific_admission=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

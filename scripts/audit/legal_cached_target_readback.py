"""Source-bound readback of the full frozen target cache through its training consumer."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_target_cache as producer,legal_long_document_training as long
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_cached_targets import LegalCachedTargets
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(producer.code(),**long.code(),**{n:file_sha(root/n) for n in
        ['model/legal_cached_targets.py','audit/legal_cached_target_readback.py']})


def long_authority(a,data):
    q=producer.qualification.training.pinned(a.long/'report.json',a.long_pin)
    c=producer.qualification.training.pinned(a.long_cold/'report.json',a.long_cold_pin)
    need(q['complete'] and c['complete'] and q['mode']=='qualify' and c['mode']=='cold'
         and c['previous_pin']==a.long_pin and q['binding']==c['binding']
         and q['binding']['code']==long.code() and q['binding']['source']==data.binding
         and all(q[k]==c[k] for k in ['updates','targets','ledger_head']), 'CACHE_READER_LONG_AUTHORITY')
    for directory,report in [(a.long,q),(a.long_cold,c)]:
        for name,pin in report['files'].items():need(file_sha(directory/name)==pin,'CACHE_READER_LONG_PHYSICAL_PIN')
    return q


def read_proof(a,consumer,data,reader,labels,selected,long_report):
    codes=[];counts={n:[] for n in consumer.roster}
    for lo in range(0,len(selected),128):
        ids=selected[lo:lo+128].tolist();c,n=consumer.take(ids)
        tc,tn=consumer.take(tuple(ids))
        need(torch.equal(tc,c) and all(torch.equal(tn[k],n[k]) for k in n),'CACHE_READER_ADDRESS_CONTAINER')
        frames=reader.frames(ids);direct,actual_counts=labels.encode(frames,reference=True)
        need(np.array_equal(c.numpy(),direct) and all(np.array_equal(n[name].numpy(),actual_counts[:,i])
             for i,name in enumerate(consumer.roster)),'CACHE_READER_NATIVE_LABEL')
        codes.append(c)
        for name in consumer.roster:counts[name].append(n[name])
    plan=producer.qualification.training.pinned(a.long/'selection.json',long_report['binding']['selection'])
    documents=[]
    for row in plan['documents']:
        f=a.long/f"labels-{row['document']}.pt"
        need(file_sha(f)==long_report['files'][f.name],'CACHE_READER_DOCUMENT_LABEL_PIN')
        saved=torch.load(f,map_location='cpu',weights_only=True);uids=saved['uids']
        expected=data.mapping[row['start']:row['stop']].tolist()
        need(uids==expected and len(uids)==row['stop']-row['start'],'CACHE_READER_DOCUMENT_SOURCE')
        for lo in range(0,len(uids),128):
            c,n=consumer.take(uids[lo:lo+128]);hi=lo+len(c)
            need(torch.equal(c,saved['codes'][lo:hi])
                 and all(torch.equal(n[name],saved['counts'][name][lo:hi]) for name in consumer.roster),
                 'CACHE_READER_CANONICAL_DOCUMENT_CODES')
        joined=consumer.document(row['document'],side='train',fit=True,expected_bounds=(row['start'],row['stop']))
        need(joined['target_uids']==uids[1:] and torch.equal(joined['codes'],saved['codes'][1:])
             and all(torch.equal(joined['counts'][n],saved['counts'][n][1:]) for n in consumer.roster),
             'CACHE_READER_SUCCESSOR_JOIN')
        documents.append(dict(document=row['document'],atoms=len(uids),targets=len(uids)-1,
                              source_start=row['start'],source_stop=row['stop'],native_pin=saved['native_pin'],
                              target_state=fingerprint((joined['codes'],joined['counts'])),all_labels_exact=True))
    return dict(qualification_uids=selected.tolist(),codes=torch.cat(codes),
                counts={n:torch.cat(v) for n,v in counts.items()},documents=documents,
                full_train_baseline_rows=consumer.baselines['rows'],baseline=consumer.baselines,
                all_source_labels_exact=True,all_document_order_labels_equal_canonical=True)


def controls(a,consumer,data,binding):
    result={}
    for key in ['source','target_checkpoint','code']:
        changed=dict(binding);changed[key]={'changed':True}
        result['changed_'+key]=rejected(lambda:LegalCachedTargets(a.cache,a.cache_pin,a.cache_cold,a.cache_cold_pin,data,changed))
    result['changed_manifest']=rejected(lambda:LegalCachedTargets(a.cache,'0'*64,a.cache_cold,a.cache_cold_pin,data,binding))
    result['changed_cold']=rejected(lambda:LegalCachedTargets(a.cache,a.cache_pin,a.cache_cold,'0'*64,data,binding))
    result['out_of_range_uid']=rejected(lambda:consumer.take([consumer.rows]))
    result['missing_uid']=rejected(lambda:consumer.take([None]))
    document=int(np.flatnonzero(data.doc_sides==0)[0]);start,stop=map(int,data.documents[document])
    result['changed_document_boundary']=rejected(lambda:consumer.document(document,side='train',fit=True,expected_bounds=(start,stop-1)))
    held=int(np.flatnonzero(data.doc_sides==2)[0])
    result['held_side_fit']=rejected(lambda:consumer.document(held,side='query',fit=True))
    result['implicit_training_authority']=rejected(consumer.training_authority)
    # A real byte-for-byte copy exercises mapping invalidation without touching authority.
    copied=a.output/'negative'/'changed-export';shutil.copytree(a.cache,copied)
    witness=LegalCachedTargets(copied,a.cache_pin,a.cache_cold,a.cache_cold_pin,data,binding)
    changed=copied/'export'/'codes.npy'
    with changed.open('r+b') as f:f.seek(-1,2);f.write(b'\xff');f.flush();os.fsync(f.fileno())
    result['changed_consumed_array']=rejected(witness.fresh)
    for value in witness.maps:value._mmap.close()
    return result


def run(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(),'CACHE_READER_GPU_CONTEXT')
    for name,pin in code().items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(dest)==pin,'CACHE_READER_SOURCE_COPY')
    data,c,reader,labels,selected,authority=producer.prepare(a.root)
    long_report=long_authority(a,data)
    consumer=LegalCachedTargets(a.cache,a.cache_pin,a.cache_cold,a.cache_cold_pin,data,authority)
    binding=dict(code=code(),cache=consumer.binding,long=a.long_pin,long_cold=a.long_cold_pin,
                 selected=selected.tolist(),device='cpu',threads=1,source=data.binding)
    previous=producer.qualification.training.pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:need(previous['complete'] and previous['binding']==binding,'CACHE_READER_COLD_BINDING')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    payload=read_proof(a,consumer,data,reader,labels,selected,long_report)
    torch.save(payload,a.output/'proof.pt')
    need(fingerprint(torch.load(a.output/'proof.pt',map_location='cpu',weights_only=True))==fingerprint(payload),
         'CACHE_READER_PROOF_READBACK')
    if previous:
        need(file_sha(a.previous/'proof.pt')==previous['files']['proof.pt'],'CACHE_READER_COLD_PROOF_PIN')
        need(fingerprint(torch.load(a.previous/'proof.pt',map_location='cpu',weights_only=True))==fingerprint(payload),
             'CACHE_READER_COLD_PROOF')
    phases['native_and_document_reconciliation']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(a,consumer,data,authority)
    if previous:need(negative==previous['controls'],'CACHE_READER_COLD_CONTROLS')
    consumer.close();reader.close();c.fresh()
    need(not torch.cuda.is_initialized() and fingerprint(labels.model.state_dict())==authority['model_state'],
         'CACHE_READER_MODEL_OR_RUNTIME_CHANGED')
    phases['controls_final_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'CACHE_READER_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,mode='cold' if previous else 'qualify',previous_pin=a.previous_pin,
                documents=payload['documents'],qualification_rows=len(selected),proof_state=fingerprint(payload),
                baseline_rows=consumer.baselines['rows'],controls=negative,calls=consumer.calls,returned_rows=consumer.returned_rows,
                files=files,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                optimizer_updates=0,native_encoder_calls=0,skill_claim=False,implicit_training_authority=False,
                acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'));p.add_argument('--output',type=Path,required=True)
    for name in ['cache','cache-cold','long','long-cold']:
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise

"""Native scoring of presealed whole-answer TRAIN controls, with full denominator."""
import argparse
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_answer_public_pack import authenticate
from audit.legal_answer_cohort_scores import native_exact
from audit.legal_answer_exact_readback import read
from claims.process_conditions import write_committed
from model import legal_answer_exact_scores as scoring
from model.legal_answer_exact_control import DTYPE,bounds,prefix,answer_pin
from model.native_population_training import fingerprint
from text.constellation_corpus import canonical,digest,file_sha,need

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29');BASE=Path(__file__).resolve().parents[1]
ORIGINAL=('legal-answer-exact-control-v1-20261024','d3a037732205c9aa5e1bbe70b6f393cb7d8ae8fc3d6759d06d0db5ffa9ca3573')
COLD=('legal-answer-exact-control-cold-v1-20261024','8258ed25a9a7fca4b07c3e6bb1cd727881d076863726a8c1bb7264fee8415d72')
READBACK=('legal-answer-exact-readback-v1-20261024','4012f029a70cf877c0f936e1ec9c216677a0510a39ab0481b3073ed77a7ef9be')
PROOF=(7,1818,2923,19876)
NEW=('model/legal_answer_exact_scores.py','audit/legal_answer_exact_native.py',
     'audit/legal_answer_exact_native_readback.py')


def selection(root,source):
    r=read(root/ORIGINAL[0],ORIGINAL[1]);c=read(root/COLD[0],COLD[1])
    p=root/READBACK[0]/'report.json';need(file_sha(p)==READBACK[1],'EXACT_NATIVE_READBACK_PIN')
    b=json.loads(p.read_bytes())
    need(r['files']==c['files'] and r['source']==c['source']==b['source']==source.binding
         and b['producer']==ORIGINAL[1] and b['cold']==COLD[1] and b['all_selections_exact']
         and b['train_index_exact'],'EXACT_NATIVE_AUTHORITY')
    code={n:file_sha(BASE/n) for n in NEW}
    for n,pin in r['code'].items():
        need(file_sha(BASE/n)==pin,'EXACT_NATIVE_SOURCE_CODE');code[n]=pin
    for n in ('model/complete_answer_scores.py','model/complete_answer_summary.py',
        'audit/legal_answer_public_pack.py','audit/legal_answer_cohort_scores.py',
        'model/legal_native_resident_stream.py','model/legal_native_training_stream.py',
        'audit/complete_answer_comparison.py'):
        code[n]=file_sha(BASE/n)
    rows=np.load(root/ORIGINAL[0]/'selection.npy',allow_pickle=False)
    need(rows.dtype==DTYPE and np.array_equal(rows['document'],np.flatnonzero(source.sides!=0)),
         'EXACT_NATIVE_COVERAGE')
    for j,row in enumerate(rows):
        d=int(row['document']);key=prefix(source,d);donor=int(row['donor'])
        need(int(row['ordinal'])==j and tuple(row['uids'][:len(key)])==key
             and int(row['observed'])==len(key),'EXACT_NATIVE_PUBLIC')
        if donor>=0:
            _,a,b=bounds(source,donor)
            need(source.sides[donor]==0 and source.components[donor]!=source.components[d]
                 and prefix(source,donor)==key and (a,b)==(int(row['donor_answer_start']),int(row['donor_stop']))
                 and answer_pin(source.mapping[a:b])==row['answer'].tobytes().hex(),'EXACT_NATIVE_DONOR')
    return rows,code


def frames(source,data,reader,ids):
    result=[]
    for at in range(0,len(ids),128):
        part=ids[at:at+128];values=reader.frames(part)
        need(all([len(f[n]) for n in source.roster]==source.counts[u].tolist()
                 for u,f in zip(part,values,strict=True)),'EXACT_NATIVE_WINDOW_COUNT')
        native_exact(data,reader,part,values);result.extend(values)
    return tuple(result)


def retrieved_stream(path,donor,available):
    """Real retrieved frames, explicitly not model-predicted events."""
    path.mkdir()
    with (path/'events.bin').open('xb') as stream:
        for j,f in enumerate(donor):
            buffer=io.BytesIO();torch.save(dict(kind='retrieved_constellation',index=j,slots=f),buffer)
            raw=buffer.getvalue();stream.write(struct.pack('>Q',len(raw)));stream.write(raw)
        if available:
            buffer=io.BytesIO();torch.save(dict(kind='end',source='TRAIN_document_END'),buffer)
            raw=buffer.getvalue();stream.write(struct.pack('>Q',len(raw)));stream.write(raw)
        stream.flush();os.fsync(stream.fileno())
    return file_sha(path/'events.bin')


def main(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir()
    source,public_rows,data,continuation,reader=authenticate(a.root)
    rows,code=selection(a.root,source);roster=source.roster
    if a.mode=='full':
        need(a.proof is not None and a.proof_pin,'EXACT_NATIVE_PROOF_REQUIRED')
        need(file_sha(a.proof/'report.json')==a.proof_pin,'EXACT_NATIVE_PROOF_PIN')
        proof=json.loads((a.proof/'report.json').read_bytes())
        need(proof['complete'] and proof['mode']=='proof' and proof['all_native_exact']
             and proof['all_scores_exact'] and proof['documents']==list(PROOF)
             and proof['code']==code,'EXACT_NATIVE_PROOF_AUTHORITY')
        chosen=rows
    else:chosen=rows[np.isin(rows['document'],PROOF)]
    documents=chosen['document'].tolist();summary=scoring.Summary(roster)
    binding=dict(source=source.binding,selection=ORIGINAL[1],cold=COLD[1],readback=READBACK[1],
        code=code,policy=scoring.POLICY,mode=a.mode,documents=documents)
    binding_pin=write_committed(a.output/'binding.json',binding)
    phases=dict(authentication=time.perf_counter()-tick,native_read=0.,score=0.,persist=0.,proof_native=0.)
    native_calls=0;native_bytes=0;maximum=0;proofs={}
    with (a.output/'cases.jsonl').open('xb') as stream:
        for row in chosen:
            tick=time.perf_counter();d=int(row['document']);_,at,hi=bounds(source,d);expected=hi-at
            donor_id=int(row['donor']);available=donor_id>=0;status=int(row['status'])
            donor_ids=[] if not available else source.mapping[int(row['donor_answer_start']):int(row['donor_stop'])].tolist()
            overlap=min(len(donor_ids),expected);reference_ids=source.mapping[at:at+overlap].tolist()
            payload=int(source.counts[donor_ids+reference_ids].sum())*768*4
            free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            need(payload+(len(donor_ids)+len(reference_ids))*4096<=free//4,'EXACT_NATIVE_MEMORY')
            donor=frames(source,data,reader,donor_ids);reference=frames(source,data,reader,reference_ids)
            native_calls+=int(bool(donor_ids))+int(bool(reference_ids));native_bytes+=payload;maximum=max(maximum,payload)
            phases['native_read']+=time.perf_counter()-tick;tick=time.perf_counter()
            modal=scoring.metrics(donor,reference,expected=expected,roster=roster,status=status)
            case=dict(identity=dict(document=d,component=int(source.components[d]),
                side=('train','development','query')[int(source.sides[d])],collection=source.metadata[d]['source_path']),
                selection_status=status,donor=donor_id,selection_row=digest(row.tobytes()),
                native=dict(donor=fingerprint((donor_ids,donor)),reference=fingerprint((reference_ids,reference)),
                    donor_atoms=len(donor),compared_atoms=len(reference),bytes=payload),
                arms=scoring.arms(modal,status,roster))
            summary.add(case);phases['score']+=time.perf_counter()-tick;tick=time.perf_counter()
            if a.mode=='proof':
                full_ids=source.mapping[at:hi].tolist();full=frames(source,data,reader,full_ids)
                path=a.output/('document-'+str(d));path.mkdir();torch.save(dict(donor_ids=donor_ids,donor=donor,
                    reference_ids=full_ids,reference=full),path/'native.pt')
                trace=retrieved_stream(path/'retrieved',donor,available)
                proofs[str(d)]=dict(native=file_sha(path/'native.pt'),trace=trace)
            phases['proof_native']+=time.perf_counter()-tick;tick=time.perf_counter()
            stream.write(canonical(case)+b'\n');phases['persist']+=time.perf_counter()-tick
            if int(row['ordinal'])%4096==0:print(json.dumps(dict(stage='score',ordinal=int(row['ordinal']))),flush=True)
        tick=time.perf_counter();stream.flush();os.fsync(stream.fileno());phases['persist']+=time.perf_counter()-tick
    tick=time.perf_counter();result=summary.finish();reader_stats=dict(reader.stats);io_stats=dict(reader.io_stats)
    reader.close();source.close();public_rows._mmap.close();continuation.fresh()
    need(not torch.cuda.is_initialized() and all(file_sha(BASE/n)==p for n,p in code.items()),'EXACT_NATIVE_FINAL')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'EXACT_NATIVE_TIMING')
    report=dict(complete=True,acceptance='author-checked',mode=a.mode,code=code,source=binding['source'],
        binding=binding_pin,cases_pin=file_sha(a.output/'cases.jsonl'),documents=documents,roster=list(roster),
        summary=result,proofs=proofs,native_calls=native_calls,native_bytes=native_bytes,maximum_case_bytes=maximum,
        reader=reader_stats,io=io_stats,phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,encoder_calls=0,optimizer_updates=0,gpu_used=False,semantic_validity_established=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=pin,cases=len(documents),native_bytes=native_bytes,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--mode',choices=('proof','full'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--proof',type=Path);p.add_argument('--proof-pin');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:main(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

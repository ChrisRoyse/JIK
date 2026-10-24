"""Full TRAIN RQ initialization with real-window qualification and cold replay."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_training_statistics as training, legal_native_training_probe as probe
from audit.legal_residual_codebook_readback import CONFIG, POLICY as RQ_POLICY
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_native_training_stream import CertifiedNativeFrames
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_packed import PackedNativeRQ
from model import streaming_rq_initialization as stream
from senses.contract import digest
from text.constellation_corpus import file_sha, need

STATS=('legal-training-statistics-full-v1-20261023','b19b2b43607e82f8bca634bd7b42c53b9b0a6ce9d3be8f0fdd6d2fafaac6195f')
STATS_COLD=('legal-training-statistics-full-cold-v1-20261023','c8e6c000b8910e744d454dedfdc7492e45808694082f968a1182cf6ed5ec5741')
POLICY=dict(seed=RQ_POLICY['seed'],sizes=RQ_POLICY['sizes'],width=CONFIG['width'],native_batch=128,
            initialization_tile=stream.TILE,physical_workspace=stream.MEMORY,device='cpu',threads=1,
            full_cold_native_tiles=389,model_updates=0,order='ascending_TRAIN_present_unique_input')


def code():
    root=Path(__file__).resolve().parents[1]
    names=['audit/legal_streaming_rq_readback.py','model/streaming_rq_initialization.py',
           'model/native_rq_packed.py','model/native_rq_conditioning.py','model/native_residual_codebook.py',
           'model/native_prefix_transformer.py','model/loop_training_checkpoint.py']
    return dict(probe.pins(),**{n:file_sha(root/n) for n in names})


def coverage(values,uids):
    need(values.shape==(len(uids),CONFIG['width']) and values.dtype==np.float32,'STREAM_RQ_INPUT_COVERAGE')


@torch.no_grad()
def scalar_reference(latent,sizes):
    residual=latent.clone();books=[];codes=[];chosen=[]
    for size in sizes:
        selected=[0];distance=(residual-residual[0]).square().sum(-1)
        for _ in range(1,size):
            distance[selected]=-torch.inf;j=int(distance.argmax());need(bool(distance[j]>0),'STREAM_RQ_REFERENCE_COLLAPSE')
            selected.append(j);distance=torch.minimum(distance,(residual-residual[j]).square().sum(-1))
        book=residual[selected];ids=torch.cat([(residual[lo:lo+128,None]-book[None]).square().sum(-1).argmin(-1)
                                             for lo in range(0,len(residual),128)])
        books.append(book);codes.append(ids);chosen.append(selected);residual=residual-book[ids]
    return dict(books=books,codes=torch.stack(codes,-1),residual=residual,choices=chosen)


def proof_controls(path,pin,ids,out):
    values=np.load(path,mmap_mode='r',allow_pickle=False);real=np.array(values[:64],copy=True)
    negative=out/'controls';negative.mkdir()
    broken=real.copy();broken[0,0]=np.nan;np.save(negative/'nonfinite.npy',broken,allow_pickle=False)
    duplicate=np.repeat(real[:1],len(real),axis=0);np.save(negative/'duplicate.npy',duplicate,allow_pickle=False)
    result=dict(changed_pin=rejected(lambda:stream.open_latents(path,'0'*64,POLICY['sizes'],stream.TILE)),
        missing_coverage=rejected(lambda:coverage(values[:-1],ids)),
        insufficient_support=rejected(lambda:stream.memory_check(63,32,POLICY['sizes'],stream.TILE)),
        physical_memory=rejected(lambda:stream.memory_check(371,32,POLICY['sizes'],1<<30)),
        nonfinite=rejected(lambda:stream.initialize(negative/'nonfinite.npy',file_sha(negative/'nonfinite.npy'),
                                                   negative/'nonfinite',POLICY['sizes'])),
        collapse=rejected(lambda:stream.initialize(negative/'duplicate.npy',file_sha(negative/'duplicate.npy'),
                                                  negative/'duplicate',POLICY['sizes'])))
    values._mmap.close();return result


@torch.no_grad()
def encode(model,reader,uids,path):
    values=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(len(uids),CONFIG['width']))
    cursor=0;native_windows={n:0 for n in reader.roster};phases=dict(reading=0.,forward=0.,write=0.)
    for lo in range(0,len(uids),128):
        ids=[int(v) for v in uids[lo:lo+128]];tick=time.perf_counter();frames=reader.frames(ids)
        phases['reading']+=time.perf_counter()-tick;tick=time.perf_counter();latent=model.latent(frames)
        need(latent.shape==(len(ids),CONFIG['width']) and bool(torch.isfinite(latent).all()),'STREAM_RQ_ENCODER_OUTPUT')
        phases['forward']+=time.perf_counter()-tick;tick=time.perf_counter()
        values[lo:lo+len(ids)]=latent.numpy();cursor+=len(ids)
        for frame in frames:
            for n in reader.roster:native_windows[n]+=len(frame[n])
        phases['write']+=time.perf_counter()-tick
        if lo==0 or cursor==len(uids) or cursor//16384!=lo//16384:
            values.flush();print(json.dumps(dict(stage='native_to_latent',rows=cursor,total=len(uids))),flush=True)
    need(cursor==len(uids),'STREAM_RQ_ENCODER_COVERAGE');values.flush();values._mmap.close()
    return dict(rows=cursor,windows=native_windows,phases_seconds=phases)


@torch.no_grad()
def replay(model,reader,uids,path,pin):
    values=stream.open_latents(path,pin,POLICY['sizes'],stream.TILE);coverage(values,uids)
    count=(len(uids)+127)//128
    selected=sorted(sorted(range(count),key=lambda i:hashlib.sha256(f'{POLICY["seed"]}:{i}'.encode()).digest())
                    [:POLICY['full_cold_native_tiles']])
    rows=0
    for number in selected:
        lo=number*128;ids=[int(v) for v in uids[lo:lo+128]];actual=model.latent(reader.frames(ids))
        need(np.array_equal(actual.numpy(),values[lo:lo+len(ids)]),'STREAM_RQ_NATIVE_REPLAY')
        rows+=len(ids)
    values._mmap.close();return dict(rows=rows,tiles=selected,exact=True,canonical_batch=128)


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    records,bank,data,c,selected,held,_,_=training.authenticate(a.root)
    statsroot=a.root/STATS[0];sr=training.pinned(statsroot/'report.json',STATS[1])
    sc=training.pinned(a.root/STATS_COLD[0]/'report.json',STATS_COLD[1])
    need(sr['complete'] and sr['scope']=='full' and sc['complete'] and sc['previous_pin']==STATS[1]
         and sr['binding']==sc['binding'] and sr['binding']['source']==data.binding,'STREAM_RQ_STATISTICS_BINDING')
    for name in ['statistics.pt','uids.npy']:need(file_sha(statsroot/name)==sr['files'][name],'STREAM_RQ_STATISTICS_PIN')
    stats=torch.load(statsroot/'statistics.pt',map_location='cpu',weights_only=True)
    binding=dict(code=code(),policy=POLICY,source=data.binding,statistics=STATS[1],statistics_cold=STATS_COLD[1])
    scope=a.mode;previous=None
    if a.mode in ['full','cold']:
        previous=training.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['binding']==binding,'STREAM_RQ_PREVIOUS_BINDING')
        if a.mode=='full':
            cold=training.pinned(a.cold_proof/'report.json',a.cold_pin)
            need(previous['scope']=='qualify' and cold['complete'] and cold['mode']=='cold'
                 and cold['previous_pin']==a.previous_pin and cold['binding']==binding,'STREAM_RQ_QUALIFICATION')
        else:scope=previous['scope'];need(scope in ['qualify','full'],'STREAM_RQ_COLD_SCOPE')
    uids=training.save_uids(a.output/'uids.npy',c.presence,selected if scope=='qualify' else None)
    if scope=='full':need(file_sha(a.output/'uids.npy')==sr['files']['uids.npy'],'STREAM_RQ_ALL_TRAIN')
    dimensions={n:len(stats['centers'][n]) for n in data.roster}
    model=probe.construct(PackedNativeRQ,stats,dimensions);before=digest(tensor_fingerprint(model.state_dict()))
    reader=CertifiedNativeFrames(data,a.root/training.DEFAULTS['audit'][0],training.DEFAULTS['audit'][1])
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();native_replay=None;encoded=None
    if a.mode=='cold' and scope=='full':
        path=Path(previous['latent']['path']);pin=previous['latent']['pin']
        native_replay=replay(model,reader,uids,path,pin)
    else:
        path=a.output/'latents.npy';encoded=encode(model,reader,uids,path);pin=file_sha(path)
        if scope=='full':need(encoded['windows']=={n:sr['lanes'][n]['windows'] for n in data.roster},'STREAM_RQ_WINDOW_COVERAGE')
    reader.finish();need(before==digest(tensor_fingerprint(model.state_dict())),'STREAM_RQ_ENCODER_CHANGED')
    phases['native_encoding_or_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    initialized=stream.initialize(path,pin,a.output/'initialization',POLICY['sizes'],
        progress=lambda row:print(json.dumps(dict(stage='book_initialized',**row)),flush=True))
    phases['streaming_initialization']=time.perf_counter()-tick;tick=time.perf_counter();reference=None
    if scope=='qualify':
        latent=torch.from_numpy(np.load(path,allow_pickle=False));ref=scalar_reference(latent,POLICY['sizes'])
        model.initialize_latents(latent)
        need(ref['choices']==initialized['choices'] and all(torch.equal(x,y) and torch.equal(x,z)
            for x,y,z in zip(ref['books'],initialized['books'],model.books)),'STREAM_RQ_REFERENCE_BOOKS')
        for name in ['codes','residual']:
            need(np.array_equal(ref[name].numpy(),np.load(a.output/'initialization'/(name+'.npy'),allow_pickle=False)),
                 'STREAM_RQ_REFERENCE_'+name.upper())
        alternative=stream.initialize(path,pin,a.output/'alternative-tile',POLICY['sizes'],tile=97)
        need(alternative['choices']==initialized['choices'] and all(torch.equal(x,y)
             for x,y in zip(alternative['books'],initialized['books'])),'STREAM_RQ_TILE_CHOICES')
        reference=dict(original_books_exact=True,choices_codes_residual_exact=True,alternative_tile=97)
    else:
        with torch.no_grad():
            for book,value in zip(model.books,initialized['books']):book.copy_(value)
            model.initialized.fill_(True)
    model.check_statistics();model_state=model.state_dict()
    torch.save(model_state,a.output/'initialized.pt')
    need(tensor_fingerprint(torch.load(a.output/'initialized.pt',weights_only=True,map_location='cpu'))==tensor_fingerprint(model_state),
         'STREAM_RQ_PERSISTED_STATE')
    controls=proof_controls(path,pin,uids,a.output)
    if a.mode=='cold':
        need(pin==previous['latent']['pin'] and file_sha(a.output/'uids.npy')==previous['files']['uids.npy'],
             'STREAM_RQ_COLD_LATENTS')
        for name in ['initialization/codes.npy','initialization/residual.npy','initialized.pt']:
            need(file_sha(a.previous/name)==previous['files'][name],'STREAM_RQ_COLD_PREVIOUS_PIN')
            need(file_sha(a.output/name)==previous['files'][name],'STREAM_RQ_COLD_OUTPUT')
        need(initialized['choices']==previous['choices'],'STREAM_RQ_COLD_CHOICES')
    data.fresh();c.fresh();reader.close();uids._mmap.close()
    phases['reference_controls_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'STREAM_RQ_TIMING_COVERAGE')
    result=dict(complete=True,mode=a.mode,scope=scope,binding=binding,previous_pin=a.previous_pin,files=files,
        latent=dict(path=str(path),pin=pin,rows=initialized['rows']),choices=initialized['choices'],
        initialization_phases=initialized['phases'],working_bytes=initialized['working_bytes'],reference=reference,
        native_encoding=encoded,native_replay=native_replay,reader_stats=reader.stats,controls=controls,
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_updates=0,
        scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--mode',choices=['qualify','full','cold'],required=True);p.add_argument('--output',type=Path,required=True)
    for name in ['previous','cold-proof']:p.add_argument('--'+name,type=Path)
    for name in ['previous-pin','cold-pin']:p.add_argument('--'+name)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise

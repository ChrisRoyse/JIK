"""Exact real inversion inputs, timed reader comparison and physical mutation controls."""
import argparse
from copy import deepcopy
import cProfile
import json
import os
from pathlib import Path
import pstats
import resource
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_input_profile as profile
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.legal_native_tile_stream import TileNativeFrames
from model.legal_surface_inputs import LegalSurfaceInputs
from model.native_population_training import fingerprint
from model.surface_device_runtime import batches,packed
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_native_store import SHARD_INPUTS

PROFILE='legal-input-profile-v2-20261024'
PROFILE_PIN='6cc0ca0b22909b3feda66e2e2e9c804c85d3c72a5f10b73642fdc220b5085818'


def fresh_reader(state,root):
    name,pin=profile.source.statistics.DEFAULTS['audit']
    return TileNativeFrames(state['data'],root/name,pin)


def physical_controls(state,root,output,uid):
    key=(state['data'].roster[0],uid//SHARD_INPUTS*SHARD_INPUTS)
    original=state['data'].native.descriptor(*key);results={}
    for action in ('vector_after_read','metadata_after_read','replace_after_read','nonfinite_copy'):
        directory=output/action;directory.mkdir();d=dict(original,root=directory,stamps=dict(original['stamps']))
        for filename in ('vectors.npy','report.json'):
            old=original['root']/filename;new=directory/filename
            shutil.copyfile(old,new);need(file_sha(new)==file_sha(old),'TILE_PROOF_PHYSICAL_COPY')
            del d['stamps'][old];d['stamps'][new]=stamp(new)
        state['data'].native.descriptors[key]=d;reader=fresh_reader(state,root);view=reader._view;fired=[]
        class Probe:
            def __init__(self,value):self.value=value
            @property
            def fd(self):return self.value.fd
            def __getitem__(self,selected):
                value=self.value[selected]
                if not fired:
                    fired.append(action)
                    if action=='nonfinite_copy':
                        value=value.copy();value[0,0]=np.nan
                    elif action=='replace_after_read':
                        path=directory/'vectors.npy';path.rename(directory/'retained-vector.npy')
                        shutil.copyfile(directory/'retained-vector.npy',path)
                    else:
                        path=directory/('vectors.npy' if action=='vector_after_read' else 'report.json')
                        with path.open('ab') as f:f.write(b'\n');f.flush();os.fsync(f.fileno())
                return value
        def wrapped(name,start,descriptor):
            value=view(name,start,descriptor)
            return Probe(value) if (name,start)==key else value
        reader._view=wrapped
        try:
            expected={'vector_after_read':'TILE_NATIVE_FD_CHANGED','metadata_after_read':'NATIVE_BATCH_SHARD_CHANGED',
                      # Renaming the open inode changes ctime; the FD guard rejects first.
                      'replace_after_read':'TILE_NATIVE_FD_CHANGED','nonfinite_copy':'TRAIN_STREAM_NONFINITE'}[action]
            message=rejected(lambda:reader.frames([uid]),expected)
            need(fired==[action] and reader.closed and not reader.maps and reader.active is None,'TILE_PROOF_REFUSED_RESOURCES')
            results[action]=dict(code=message,closed=True,handles=0)
        finally:
            state['data'].native.descriptors[key]=original
            reader.close()
    return results


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={};state=reader=None
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        prior=profile.source.authority.pinned(a.root/PROFILE,PROFILE_PIN)
        state=profile.source.sources(a.root);base=Path(__file__).resolve().parents[1]
        need(all(file_sha(base/n)==h for n,h in prior['code'].items()),'TILE_PROOF_REFERENCE_CODE')
        code=dict(prior['code'],**{n:file_sha(base/n) for n in ('model/legal_native_tile_stream.py','audit/legal_native_tile_proof.py')})
        reader=fresh_reader(state,a.root)
        candidate=LegalSurfaceInputs(state['data'],state['surfaces'],reader,state['config'],bank_pin=state['data'].native.pin)
        adapters=dict(reference=state['adapter'],candidate=candidate);cases={}
        for start in prior['starts']:
            name=f'tile-{start}.pt';need(file_sha(a.root/PROFILE/name)==prior['files'][name],'TILE_PROOF_REFERENCE_FILE')
            saved=torch.load(a.root/PROFILE/name,map_location='cpu',weights_only=True)
            need(fingerprint(saved)==prior['payload_pins'][str(start)],'TILE_PROOF_REFERENCE_PAYLOAD')
            cases[str(start)]=dict(ids=[e['occurrence'] for e in saved],pin=fingerprint(saved))
        phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter()
        for name,case in cases.items():
            examples=adapters['reference'].fetch(case['ids'],side='train',fit=True)
            need(fingerprint(profile.payload(examples))==case['pin'],'TILE_PROOF_COLD_SOURCE')
        # Select by native-window count only; no outcome or similarity is read.
        possibilities=[];first=state['data'].roster[0]
        for (name,start),d in state['data'].native.descriptors.items():
            if name==first:
                lengths=np.diff(d['offsets']);index=int(lengths.argmax())
                if int(lengths[index])>1:possibilities.append((int(lengths[index]),start+index))
        extra=None
        for count,uid in sorted(possibilities,reverse=True):
            indices=state['data'].source.occurrences(uid);eligible=indices[state['data'].sides[indices]==0]
            if len(eligible):extra=int(eligible[0]);break
        if extra is not None:
            examples=adapters['reference'].fetch([extra],side='train',fit=True)
            value=profile.payload(examples);persist(a.output/'multi-window-source.pt',value)
            cases['multi_window']=dict(ids=[extra],pin=fingerprint(value))
        phases['cold_source_and_multi_window_selection']=time.perf_counter()-tick;tick=time.perf_counter()
        rows=[];packing=[]
        for repeat,order in enumerate((('reference','candidate'),('candidate','reference'),('reference','candidate'))):
            for mode in order:
                for name,case in cases.items():
                    cpu=time.process_time();wall=time.perf_counter()
                    examples=adapters[mode].fetch(case['ids'],side='train',fit=True)
                    row=dict(repeat=repeat,mode=mode,case=name,seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
                    value=profile.payload(examples);need(fingerprint(value)==case['pin'],'TILE_PROOF_EXACT_INPUT')
                    rows.append(row)
                    if repeat==0 and mode=='candidate':
                        persist(a.output/f'candidate-{name}.pt',value)
                        need(fingerprint(torch.load(a.output/f'candidate-{name}.pt',map_location='cpu',weights_only=True))==case['pin'],
                             'TILE_PROOF_OUTPUT_READBACK')
                print(json.dumps(dict(repeat=repeat,mode=mode,seconds=sum(r['seconds'] for r in rows if r['repeat']==repeat and r['mode']==mode))),flush=True)
        phases['matched_reads_and_persistence']=time.perf_counter()-tick;tick=time.perf_counter()
        # Consume actual adapter outputs through the unchanged decoder batching path.
        for name,case in cases.items():
            pins={}
            for mode in adapters:
                examples=adapters[mode].fetch(case['ids'],side='train',fit=True)
                pins[mode]=[fingerprint(packed(b)) for b in batches(examples,state['config'],state['specs'],'gathered',128)]
            need(pins['reference']==pins['candidate'],'TILE_PROOF_DECODER_INPUTS')
            packing.append(dict(case=name,batches=len(pins['reference']),pins=pins['reference']))
        persist(a.output/'decoder-inputs.pt',packing);phases['decoder_input_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        profiles={}
        for mode in adapters:
            prof=cProfile.Profile();prof.enable()
            for case in cases.values():adapters[mode].fetch(case['ids'],side='train',fit=True)
            prof.disable();prof.dump_stats(str(a.output/f'{mode}.prof'))
            profiles[mode]=[dict(path=path,line=line,name=name,calls=calls,self_seconds=own,cumulative_seconds=cum)
                for (path,line,name),(_,calls,own,cum,_) in pstats.Stats(prof).stats.items()]
        write_committed(a.output/'profiles.json',profiles);phases['profile']=time.perf_counter()-tick;tick=time.perf_counter()
        original=cases[str(prior['starts'][0])]['ids'];chosen=original[:3]
        for ids in (chosen[::-1],[chosen[0],chosen[0],chosen[1]]):
            left=profile.payload(adapters['reference'].fetch(ids,side='train',fit=True))
            right=profile.payload(candidate.fetch(ids,side='train',fit=True))
            need(fingerprint(left)==fingerprint(right),'TILE_PROOF_REORDER_DUPLICATE')
        controls=physical_controls(state,a.root,a.output,int(state['data'].mapping[chosen[0]]))
        before=(state['surfaces'].stats['calls'],reader.stats['calls'])
        held=next(e.occurrence for e in state['examples'] if e.side=='query')
        controls['held_out']=rejected(lambda:candidate.fetch([held],side='train',fit=True),'LEGAL_LEARNING_CROSS_SPLIT')
        controls['too_many']=rejected(lambda:candidate.fetch(original+[original[0]],side='train',fit=True),'LEGAL_SURFACE_MEMORY_TILE')
        need(before==(state['surfaces'].stats['calls'],reader.stats['calls']),'TILE_PROOF_REFUSAL_ACCESS')
        reader.close();controls['closed']=rejected(lambda:reader.frames([int(state['data'].mapping[chosen[0]])]),'RESIDENT_NATIVE_CLOSED')
        profile.source.close(state);need(not reader.maps and not torch.cuda.is_initialized(),'TILE_PROOF_CLOSURE')
        phases['controls_and_closure']=time.perf_counter()-tick;tick=time.perf_counter()
        if a.previous:
            old=profile.source.authority.pinned(a.previous,a.previous_pin)
            need(old['code']==code and old['cases']==cases and old['controls']==controls,'TILE_PROOF_COLD_BINDING')
            for name in [f'candidate-{n}.pt' for n in cases]+['decoder-inputs.pt']:
                need(file_sha(a.previous/name)==old['files'][name],'TILE_PROOF_COLD_FILE')
                need(fingerprint(torch.load(a.previous/name,map_location='cpu',weights_only=True))
                     ==fingerprint(torch.load(a.output/name,map_location='cpu',weights_only=True)),'TILE_PROOF_COLD_EXACT')
        need(all(file_sha(base/n)==h for n,h in code.items()),'TILE_PROOF_CODE_CHANGED')
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['cold_and_final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TILE_PROOF_PHASES')
        report=dict(complete=True,code=code,files=files,profile_pin=PROFILE_PIN,previous_pin=a.previous_pin,cases=cases,
            rows=rows,controls=controls,decoder_input_batches=sum(r['batches'] for r in packing),multi_window_observed=extra is not None,
            phases=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            handle_limit=64,payload_limit=16<<20,all_inputs_exact=True,gpu_used=False,training_updates=0,promoted=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if reader is not None and not reader.closed:reader.close()
        if state is not None:
            if not state['reader'].closed:state['reader'].close()
            if not state['surfaces'].closed:state['surfaces'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

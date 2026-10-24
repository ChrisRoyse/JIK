"""Physical legal source -> native decoder inputs -> complete-byte gradients."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_placement_comparison_training as comparison
from audit import legal_training_statistics as statistics
from audit.capability_surface_fsv import check_schedule
from claims.process_conditions import write_committed
from model.contracts import ModelConfig
from model.legal_native_resident_stream import ResidentNativeFrames
from model.legal_surface_inputs import LegalSurfaceInputs
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder, schedule, whole_surface_loss
from text.constellation_corpus import file_sha, need
from text.legal_surface_bytes import SurfaceBytes

SURFACE=('legal-surface-bytes-full-v1-20261023','9557b97cdb190e9ca2b004b21d1e23ecd932c984e96aa14ea6af9fa5345375f7')
READBACK=('legal-surface-bytes-full-readback-v1-20261023','9cdb7d53885528444d011b5ae61f571774fc415ae77bffe8bfdb2795450573f3')


def code():
    root=Path(__file__).resolve().parents[1]
    names=set(statistics.code_pins()) | {
        'model/legal_surface_inputs.py','text/legal_surface_bytes.py','model/legal_learning_source.py',
        'model/legal_native_training_stream.py','model/legal_native_resident_stream.py',
        'text/legal_native_store.py','text/legal_native_batch.py','model/native.py','model/contracts.py',
        'model/network.py','model/attention.py','model/stream_surface.py','senses/contract.py',
        'audit/capability_surface_fsv.py','audit/legal_surface_decoder_proof.py',
        'audit/legal_surface_decoder_readback.py','audit/legal_placement_comparison_training.py'}
    return {n:file_sha(root/n) for n in sorted(names)}


def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError) as error:return str(error)
    raise ValueError('LEGAL_SURFACE_NEGATIVE_ACCEPTED')


def run(a):
    began=tick=time.perf_counter(); process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pins=code(); prior=comparison.pinned(a.root/READBACK[0],READBACK[1])
    need(prior['complete'] and prior['all_bytes_exact'] and prior['all_offsets_exact']
         and prior['input_pin']==SURFACE[1], 'LEGAL_SURFACE_FULL_READBACK')
    source,bundle=comparison.source_bundle(a.root)
    records,bank,data,c,_,_,_,_=statistics.authenticate(a.root)
    config=ModelConfig();surfaces=SurfaceBytes(a.root/SURFACE[0],SURFACE[1],data.source)
    audit=statistics.DEFAULTS['audit'];reader=ResidentNativeFrames(data,a.root/audit[0],audit[1])
    adapter=LegalSurfaceInputs(data,surfaces,reader,config,bank_pin=bank.pin)
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    controls={};before=(deepcopy(surfaces.stats),deepcopy(reader.stats))
    held={s:next(i for i in bundle['items'] if i['side']==s)['start'] for s in ['development','query']}
    for side,index in held.items():
        controls['fit_'+side]=rejected(lambda s=side,i=index:adapter.fetch([i],side=s,fit=True))
    controls['wrong_side']=rejected(lambda:adapter.fetch([held['query']],side='train',fit=True))
    need(before==(surfaces.stats,reader.stats),'LEGAL_SURFACE_INELIGIBLE_IO')
    controls['wrong_bank']=rejected(lambda:LegalSurfaceInputs(data,surfaces,reader,config,bank_pin='0'*64))
    controls['admission']=rejected(adapter.training_authority)
    saved=[];train=[];count=slots=windows=surface_bytes=0;exposure={}
    for item in bundle['items']:
        indices=list(range(item['start'],item['stop']))
        examples=adapter.fetch(indices,side=item['side'],fit=item['side']=='train')
        need(len(examples)==len(item['frames']),'LEGAL_SURFACE_DOCUMENT_COVERAGE')
        values=[]
        for k,example in enumerate(examples):
            raw=example.raw;plan=list(schedule(raw));check_schedule(raw,plan)
            frame={n:x.values[0].clone() for n,x in example.native.items()}
            need(all(torch.equal(v,item['frames'][k][n]) for n,v in frame.items()),'LEGAL_SURFACE_REFERENCE_NATIVE')
            value=dict(occurrence=example.occurrence,uid=example.uid,raw=torch.tensor(list(raw),dtype=torch.uint8),
                native=frame,row=next(iter(example.native.values())).rows[0],schedule=plan,
                train_present=bool(c.presence[example.uid]&1))
            values.append(value);count+=1;slots+=len(frame);windows+=sum(len(v) for v in frame.values())
            surface_bytes+=len(raw);key=item['side']+('/seen_train' if value['train_present'] else '/not_in_train')
            exposure[key]=exposure.get(key,0)+1
            if item['side']=='train':train.append(example)
        saved.append(dict(document=item['document'],start=item['start'],stop=item['stop'],side=item['side'],observations=values))
    torch.save(saved,a.output/'observations.pt')
    need(fingerprint(torch.load(a.output/'observations.pt',weights_only=True))==fingerprint(saved),'LEGAL_SURFACE_SAVED_INPUTS')
    phases['physical_join_and_persistence']=time.perf_counter()-tick;tick=time.perf_counter()
    chosen={train[0].occurrence:train[0],max(train,key=lambda e:len(e.raw)).occurrence:max(train,key=lambda e:len(e.raw))}
    for predicate in [lambda e:not e.raw.decode('utf-8').isascii(),lambda e:any(x.values.shape[1]>1 for x in e.native.values())]:
        example=next(e for e in train if predicate(e));chosen[example.occurrence]=example
    torch.manual_seed(20261023);model=StreamSurfaceDecoder(config,adapter.specs);model.train()
    initial={n:v.detach().clone() for n,v in model.state_dict().items()};torch.save(initial,a.output/'initial.pt')
    gradients=[]
    for index,example in enumerate(chosen.values()):
        model.zero_grad(set_to_none=True);loss=whole_surface_loss(model,example.raw,example.native,backward=True)
        values={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        need(all(bool(v.isfinite().all()) for v in values.values()),'LEGAL_SURFACE_GRADIENT_FINITE')
        norms={}
        for spec in adapter.specs:
            for prefix in ['projectors.'+spec.name+'.','blocks.0.branches.'+spec.name+'.']:
                norms[prefix]=sum(float(v.square().sum()) for n,v in values.items() if n.startswith(prefix))
        need(all(v>0 for v in norms.values()) and bool(values['tokens.weight'].abs().any())
             and bool(values['output.weight'].abs().any()),'LEGAL_SURFACE_GRADIENT_DISCONNECTED')
        file=f'gradients-{index}.pt';torch.save(values,a.output/file)
        gradients.append(dict(occurrence=example.occurrence,uid=example.uid,loss=loss,file=file,
            gradient_state=fingerprint(values),branch_norms=norms))
    need(fingerprint(initial)==fingerprint(model.state_dict()),'LEGAL_SURFACE_UNREQUESTED_UPDATE')
    phases['decoder_forward_backward']=time.perf_counter()-tick;tick=time.perf_counter()
    example=train[0];frame={n:x.values[0] for n,x in example.native.items()}
    controls['wrong_uid']=rejected(lambda:adapter.wrap(example.occurrence,example.uid+1,example.raw,frame,side='train'))
    controls['changed_surface']=rejected(lambda:adapter.wrap(example.occurrence,example.uid,b'x'+example.raw,frame,side='train'))
    controls['missing_slot']=rejected(lambda:adapter.wrap(example.occurrence,example.uid,example.raw,dict(list(frame.items())[1:]),side='train'))
    spec=adapter.specs[0];wrong=replace(example.native[spec.name],identity='0'*64)
    controls['wrong_instrument']=rejected(lambda:wrong.check(spec,config))
    for mode in ['future_prefix','missing_tail','changed_target']:
        bad=list(schedule(example.raw))
        if mode=='future_prefix':bad[0]['prefix'][0]=example.raw[0]
        elif mode=='missing_tail':bad.pop()
        else:bad[0]['labels'][0]=(example.raw[0]+1)%256
        controls[mode]=rejected(lambda b=bad:check_schedule(example.raw,b))
    adapter.fresh();reader.close();surfaces.close()
    need(reader.closed and not reader.maps and surfaces.closed and surfaces.fd is None,
         'LEGAL_SURFACE_HANDLES_NOT_CLOSED')
    need(not torch.cuda.is_initialized(),'LEGAL_SURFACE_GPU_INITIALIZED')
    need(code()==pins,'LEGAL_SURFACE_CODE_CHANGED')
    for name,pin in pins.items():
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(path)==pin,'LEGAL_SURFACE_SOURCE_COPY')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_close_and_seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'LEGAL_SURFACE_PHASE_COVERAGE')
    result=dict(complete=True,code=pins,files=files,binding=adapter.binding,source=source,
        surface_readback=READBACK[1],config=asdict(config),instruments=[asdict(s) for s in adapter.specs],
        observations=count,slots=slots,native_windows=windows,surface_bytes=surface_bytes,exposure=exposure,
        gradients=gradients,negative_controls=controls,ineligible_io=False,handles_closed=True,
        optimizer_steps=0,gpu_initialized=False,native_encoder_calls=0,parameters=sum(p.numel() for p in model.parameters()),
        source_claims_are_world_truth=False,decoder_condition='true_same_atom_native_readings',
        predictor_generation_tested=False,learned_capability=False,scientific_admission=False,
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        number_status='measured',acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,observations=count,windows=windows,bytes=surface_bytes,
        gradients=len(gradients),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

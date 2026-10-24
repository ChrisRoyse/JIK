"""Qualify, fit and cold-read native residual reconstruction on the growth cohort."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_rq_growth as growth
from audit.legal_rq_growth_readback import check_chain, recount
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_residual_codebook import NativeResidualCodebook
from model.native_rq_data import NativeRQData, training_components
from model.native_rq_residual import ResidualConditionedNativeRQ, RECIPES
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

POLICY = dict(growth.POLICY, recipes=list(RECIPES), population='large', qualification_updates=32,
              baseline_first32_exact=False, initial_baseline_exact=True,
              output='immutable_TRAIN_baseline_plus_sqrt_variance_times_zero_initialized_residual')


def pins():
    base=Path(__file__).resolve().parents[1]
    return dict(growth.pins(), **{n:file_sha(base/n) for n in
        ['model/native_rq_residual.py','audit/legal_rq_residual.py','audit/legal_rq_growth_readback.py']})


def make(dimensions, recipe, stats):
    torch.manual_seed(POLICY['seed'])
    model=ResidualConditionedNativeRQ(dimensions,growth.CONFIG,POLICY['sizes'],recipe,stats)
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'],
                                foreach=False,fused=False)
    return model,optimizer


def snapshot(path,model,optimizer,number,binding):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=number,binding=binding),path)


def load_snapshot(path,pin,binding,number):
    need(file_sha(path)==pin,'RQ_RESIDUAL_CHECKPOINT_PIN')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    need(saved['binding']==binding and saved['step']==number,'RQ_RESIDUAL_CHECKPOINT_BINDING')
    return saved


@torch.no_grad()
def probe(model,reader,uids,stats,*,initial=False):
    frames=tuple(f for _,part in reader.tiles(uids) for f in part)
    native_before=tensor_fingerprint(frames);out=model(frames)
    _,values,_=model.quantize(out['latent'])
    raw=NativeResidualCodebook.reconstruct(model,values,[{n:len(f[n]) for n in reader.roster} for f in frames])
    maximum=0.
    for frame,decoded,residual in zip(frames,out['decoded'],raw):
        for i,name in enumerate(reader.roster):
            w=len(frame[name]);base=(stats['centers'][name].float().expand(w,-1) if model.output_baseline=='mean_residual'
                                   else stats['positions'][name].float()[:w])
            if initial:need(torch.equal(base,decoded[name]),'RQ_RESIDUAL_INITIAL_BASELINE')
            expected=base.numpy().astype(np.float64)+np.sqrt(float(stats['variance'][name].float()))*residual[name].numpy().astype(np.float64)
            maximum=max(maximum,float(np.max(np.abs(expected-decoded[name].numpy()))))
    need(maximum<=POLICY['arithmetic_error'],'RQ_RESIDUAL_AFFINE_REFERENCE')
    need(native_before==tensor_fingerprint(frames),'RQ_RESIDUAL_NATIVE_CHANGED');model.check_statistics()
    payload=dict(codes=out['codes'],latent=out['latent'],decoded=out['decoded'])
    return dict(inputs=len(uids),initial_baseline_exact=initial,affine_max_error=maximum,
                output_pin=digest(tensor_fingerprint(payload)),native_pin=digest(native_before)),payload


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prior=growth.pinned(a.growth/'report.json',a.growth_pin)
    need(prior['complete'] and prior['mode']=='fit' and prior['code']==growth.pins(),'RQ_RESIDUAL_GROWTH')
    baseline=growth.pinned(a.baseline/'report.json',a.baseline_pin)
    selection=growth.pinned(a.growth/'selection.json',prior['selection_pin'])
    q=Path(prior['qualification']['path']);old=growth.pinned(q/'report.json',prior['qualification']['pin'])
    stat_path=q/'large-statistics.pt';need(file_sha(stat_path)==old['statistic_pins']['large'],'RQ_RESIDUAL_STATISTICS')
    stats=torch.load(stat_path,map_location='cpu',weights_only=True)
    source=ReuseInputs(baseline['source']['path'],baseline['source']['pin'])
    continuation=Continuations(Path(baseline['continuations']['path']),baseline['continuations']['pin'])
    split,cold=continuation.report['split'],continuation.report['cold'];bank=baseline['bank']
    data=LegalLearningSource(source,split['path'],split['pin'],cold['path'],cold['pin'],bank['path'],bank['pin'])
    need(data.binding==prior['source_binding']==selection['source_binding'],'RQ_RESIDUAL_SOURCE')
    reader=NativeRQData(data,Path(baseline['proof']['path']),baseline['proof']['pin'])
    uids=selection['populations']['large'];probe_uids=selection['populations']['small']
    need(stats['uids']==uids,'RQ_RESIDUAL_STATISTICS_SCOPE');_,cases=training_components(data,uids,continuation.presence)
    common=dict(source=data.binding,selection_pin=prior['selection_pin'],statistic_pin=file_sha(stat_path),code=pins(),policy=POLICY)
    previous=None
    if a.mode!='qualify':
        need(a.previous is not None and a.previous_pin is not None,'RQ_RESIDUAL_PREVIOUS_REQUIRED')
        previous=growth.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['common']==common and previous['growth_pin']==a.growth_pin,
             'RQ_RESIDUAL_PREVIOUS_BINDING')
    if a.mode=='fit':
        need(previous['mode']=='qualify' and a.cold_proof is not None,'RQ_RESIDUAL_QUALIFICATION_REQUIRED')
        cold_report=growth.pinned(a.cold_proof/'report.json',a.cold_pin)
        need(cold_report['complete'] and cold_report['mode']=='cold' and cold_report['previous_pin']==a.previous_pin
             and cold_report['common']==common,'RQ_RESIDUAL_COLD_QUALIFICATION')
    if a.mode=='cold':need(previous['mode'] in ['qualify','fit'],'RQ_RESIDUAL_COLD_PREVIOUS')
    phases=dict(source_and_bindings=time.perf_counter()-tick);tick=time.perf_counter();results={}
    for recipe in RECIPES:
        root=a.output/recipe;root.mkdir();began=time.perf_counter();binding=dict(common,recipe=recipe)
        model,optimizer=make(prior['dimensions'],recipe,stats)
        if a.mode=='qualify':
            latent_pin=growth.initialize(model,reader,uids)
            initial,payload=probe(model,reader,probe_uids,stats,initial=True)
            snapshot(root/'initial.pt',model,optimizer,0,binding);records=[]
            for i in range(POLICY['qualification_updates']):
                if i==POLICY['qualification_updates']-1:snapshot(root/'resume.pt',model,optimizer,i,binding)
                records.append(growth.step(model,optimizer,reader,uids,cases,i))
            snapshot(root/'final.pt',model,optimizer,POLICY['qualification_updates'],binding)
            after,payload=probe(model,reader,probe_uids,stats);torch.save(payload,root/'probe.pt')
            loss=growth.loss_reference(model,reader,uids,cases)
            controls={}
            if recipe=='position_residual':
                controls['unseen_position']=rejected(lambda:model.baseline(0,len(model.position_0)+1))
            with torch.no_grad():model.position_0[0,0]+=1
            controls['statistics_mutation']=rejected(model.check_statistics)
            results[recipe]=dict(initial_latent_pin=latent_pin,initial=initial,after=after,records=records,loss_reference=loss,
                controls=controls,seconds=time.perf_counter()-began,
                files={n:file_sha(root/n) for n in ['initial.pt','resume.pt','final.pt','probe.pt']})
        elif a.mode=='fit':
            reference=previous['results'][recipe];init_path=a.previous/recipe/'initial.pt'
            saved=load_snapshot(init_path,reference['files']['initial.pt'],binding,0)
            need(not saved['optimizer']['state'],'RQ_RESIDUAL_NONFRESH_OPTIMIZER')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            initial,_=probe(model,reader,probe_uids,stats,initial=True)
            need(initial==reference['initial'],'RQ_RESIDUAL_INITIAL_REPLAY')
            with (root/'updates.jsonl').open('x') as stream:
                for i in range(POLICY['updates']):
                    if i==POLICY['updates']-1:snapshot(root/'resume.pt',model,optimizer,i,binding)
                    row=growth.step(model,optimizer,reader,uids,cases,i);stream.write(json.dumps(row,sort_keys=True)+'\n')
                    if (i+1)%128==0:
                        stream.flush();print(json.dumps(dict(recipe=recipe,step=i+1,loss=row['loss'],seconds=time.perf_counter()-began)),flush=True)
            snapshot(root/'final.pt',model,optimizer,POLICY['updates'],binding)
            measured={'train':growth.evaluate(model,reader,uids,cases,stats,root/'train')}
            for name,side in [('development',1),('query',2)]:
                ids=selection['held'][name]
                measured[name]=growth.evaluate(model,reader,ids,growth.components_for(data,ids,side),stats,root/name)
            need(measured['train']['native_sha256']==stats['native_sha256'],'RQ_RESIDUAL_TRAIN_BYTES')
            results[recipe]=dict(initial=initial,initial_checkpoint=dict(path=str(init_path),pin=reference['files']['initial.pt']),
                last_update=row,evaluation={n:dict(pin=v['pin'],rows=v['rows'],distinct_codes=v['distinct_codes'],slices=v['slices']) for n,v in measured.items()},
                seconds=time.perf_counter()-began,files={n:file_sha(root/n) for n in ['resume.pt','final.pt','updates.jsonl']})
        else:
            reference=previous['results'][recipe];previous_root=a.previous/recipe
            count=POLICY['qualification_updates'] if previous['mode']=='qualify' else POLICY['updates']
            resume=load_snapshot(previous_root/'resume.pt',reference['files']['resume.pt'],binding,count-1)
            final=load_snapshot(previous_root/'final.pt',reference['files']['final.pt'],binding,count)
            if previous['mode']=='qualify':
                initial=load_snapshot(previous_root/'initial.pt',reference['files']['initial.pt'],binding,0)
                model.load_state_dict(initial['model']);checked,_=probe(model,reader,probe_uids,stats,initial=True)
                need(checked==reference['initial'],'RQ_RESIDUAL_COLD_INITIAL');expected=reference['records'][-1]
            else:
                need(file_sha(previous_root/'updates.jsonl')==reference['files']['updates.jsonl'],'RQ_RESIDUAL_LEDGER_PIN')
                records=[json.loads(line) for line in (previous_root/'updates.jsonl').read_text().splitlines()]
                check_chain(records,uids);expected=records[-1]
                need(expected==reference['last_update'],'RQ_RESIDUAL_LAST_UPDATE')
            model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
            actual=growth.step(model,optimizer,reader,uids,cases,count-1)
            need(actual==expected and tensor_fingerprint(model.state_dict())==tensor_fingerprint(final['model'])
                 and tensor_fingerprint(optimizer.state_dict())==tensor_fingerprint(final['optimizer']), 'RQ_RESIDUAL_COLD_STEP')
            measured={}
            if previous['mode']=='qualify':
                after,payload=probe(model,reader,probe_uids,stats)
                need(file_sha(previous_root/'probe.pt')==reference['files']['probe.pt'],'RQ_RESIDUAL_PROBE_PIN')
                saved_probe=torch.load(previous_root/'probe.pt',map_location='cpu',weights_only=True)
                need(after==reference['after'] and tensor_fingerprint(payload)==tensor_fingerprint(saved_probe),'RQ_RESIDUAL_COLD_PROBE')
                need(growth.loss_reference(model,reader,uids,cases)==reference['loss_reference'],'RQ_RESIDUAL_COLD_LOSS')
                wrong=dict(binding,recipe='wrong')
                controls=dict(wrong_checkpoint_binding=rejected(lambda:load_snapshot(previous_root/'final.pt',reference['files']['final.pt'],wrong,count)),
                              wrong_checkpoint_pin=rejected(lambda:load_snapshot(previous_root/'final.pt','0'*64,binding,count)))
            else:
                groups={'train':(uids,cases)}
                for name,side in [('development',1),('query',2)]:
                    ids=selection['held'][name];groups[name]=(ids,growth.components_for(data,ids,side))
                for name,(ids,components) in groups.items():
                    fresh=growth.evaluate(model,reader,ids,components,stats,root/name)
                    need(fresh['pin']==reference['evaluation'][name]['pin'],'RQ_RESIDUAL_COLD_OUTPUT')
                    arrays=[np.load(root/name/n,allow_pickle=False) for n in ['codes.npy','errors.npy','eligible.npy','windows.npy']]
                    measured[name]=dict(pin=fresh['pin'],**recount(fresh,*arrays,reader.roster))
                controls={}
            results[recipe]=dict(cold_step_exact=True,weight_pin=actual['after'],optimizer_pin=actual['optimizer'],
                controls=controls,evaluation=measured,seconds=time.perf_counter()-began)
        write_committed(root/'report.json',results[recipe])
        print(json.dumps(dict(recipe=recipe,mode=a.mode,complete=True,seconds=results[recipe]['seconds'])),flush=True)
    phases['qualification_fits_or_cold_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    source.fresh();data.fresh();continuation.fresh();phases['final_freshness']=time.perf_counter()-tick
    seconds=time.perf_counter()-start;need(sum(phases.values())/seconds>=.9,'RQ_RESIDUAL_PHASE_ATTRIBUTION')
    report=dict(schema='legal-rq-residual-v1',complete=True,mode=a.mode,common=common,growth_pin=a.growth_pin,
        previous_pin=a.previous_pin,results=results,phases_seconds=phases,seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        maximum_cached_native_bytes=reader.peak_cache_bytes,maximum_read_native_bytes=reader.maximum_read_bytes,
        model_promotion=False,scientific_admission=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['qualify','fit','cold'],required=True)
    for name in ['baseline','growth']:
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');p.add_argument('--cold-proof',type=Path);p.add_argument('--cold-pin')
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

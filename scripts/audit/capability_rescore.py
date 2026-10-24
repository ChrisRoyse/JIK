"""Frozen phase-12/14/18 target discrimination and byte conditioning traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.capability_conditioning import fingerprint,select_donors
from audit.capability_kernel_learning import parameter_hash
from audit.capability_region_diagnostic import load
from audit.capability_rescore_metrics import trace,array,discriminate,difference,summarize
from audit.fsv_identity_loop import configure
from claims.capability_population import need,sha
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease,run_cli
from model.capability_kernel_learning import pinned,predict
from model.capability_structure import FieldContext,predict_structured
from model.capability_structure_input import StructuredContexts


def install(model,reference):
    report=pinned(reference);path=Path(reference['path']).parent/'final.pt'
    need(sha(path.read_bytes())==report['final_checkpoint'],'RESCORE_CHECKPOINT')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    need(saved['source']==report['source'] and fingerprint(saved['optimizer'])==saved['optimizer_hash'],
         'RESCORE_CHECKPOINT_BINDING')
    structured=report['schema']=='capability-structure-learning-v1'
    if structured and 'context' not in model:model['context']=FieldContext(model['world']).to('cuda')
    if not structured and 'context' in model:del model['context']
    model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False).eval()
    need(parameter_hash(model)==report['final_parameters'],'RESCORE_PARAMETERS')
    model['world'].check_reference()
    return report,saved['optimizer_hash']


def forward(model,corpus,contexts,key,report):
    kind=corpus.tasks[key]['kind'];world=model['world']
    if 'context' in model:
        mode='boundaries' if report['arm']=='boundaries' else 'structured'
        return predict_structured(model,contexts.get(key,world.config,'cuda'),kind,mode=mode)
    return predict(world,model['request'],corpus.batch(key,'context',world.config,'cuda'),key=key,operation=kind)


def prior_rows(reference,report):
    name,pin=('queries.jsonl',report['queries_sha256']) if 'queries_sha256' in report else ('evaluation.jsonl',report['evaluation_sha256'])
    raw=(Path(reference['path']).parent/name).read_bytes();need(sha(raw)==pin,'RESCORE_PRIOR_ROWS')
    return {r['task']:r for r in map(json.loads,raw.splitlines()) if r.get('split','query')=='query'}


def source_raw(corpus,key):
    task=corpus.tasks[key];pop=corpus.sources[task['population']].population
    field=pop.fields[task['selections']['targets']['fields'][0]['id']]
    raw=field['text'].encode();need(sha(raw)==field['text_sha256'],'RESCORE_SOURCE')
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'RESCORE_EXISTS')
    a.output.mkdir();start=tick=time.perf_counter();phases={};source=json.loads(a.config.read_bytes())
    source['config_file_sha256']=sha(a.config.read_bytes())
    source['code']={n:sha((Path(__file__).parent/n).read_bytes()) for n in ('capability_rescore.py','capability_rescore_metrics.py')}
    try:
        gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-rescore');configure()
        model,corpus,parent,_=load(source['checkpoints']['phase12']);contexts=StructuredContexts(corpus)
        need(len(corpus.query)==170 and len(corpus.train)==319,'RESCORE_ROSTER')
        donors=select_donors(corpus);alternatives={k:[d for d in corpus.train if corpus.tasks[d]['kind']==corpus.tasks[k]['kind']
            and corpus.tasks[d]['split_group']!=corpus.tasks[k]['split_group']] for k in corpus.query}
        targets={k:corpus.target(k,model['world'].config,'cuda',fitting=k in corpus.train) for k in (*corpus.train,*corpus.query)}
        census=dict(train=len(corpus.train),query=len(corpus.query),pairs=sum(map(len,alternatives.values())),
            target_bytes=sum(len(raw) for raw,_ in targets.values()),query_bytes=sum(len(targets[k][0]) for k in corpus.query),
            native_bytes=sum(x.values.numel()*4 for _,slots in targets.values() for x in slots.values()))
        free,_=torch.cuda.mem_get_info();need(contexts.census()['native_bytes']+census['native_bytes']<free,'RESCORE_MEMORY')
        plan=dict(source=source,donors=donors,alternatives=alternatives,query=list(corpus.query),train=list(corpus.train),
            census=census,metric='per-lane-symmetric-MaxSim-source-discrimination',direction='shuffled-minus-correct-NLL',
            independent_groups={kind:len({corpus.tasks[k]['split_group'] for k in corpus.query if corpus.tasks[k]['kind']==kind})
                                for kind in sorted({corpus.tasks[k]['kind'] for k in corpus.query})})
        write_committed(a.output/'plan.json',plan);print(json.dumps(dict(stage='plan',census=census)),flush=True)
        phases['authenticate_materialize']=time.perf_counter()-tick;tick=time.perf_counter()
        oracle={};rows=[];files={};identities={};decoder_identity=None
        with torch.no_grad(),(a.output/'scores.jsonl').open('x') as out:
            for arm,reference in source['checkpoints'].items():
                report,moments=install(model,reference);before=fingerprint(model.state_dict());prior=prior_rows(reference,report)
                decoder_pin=fingerprint(model['decoder'].state_dict())
                if decoder_identity is None:decoder_identity=decoder_pin
                need(decoder_pin==decoder_identity,'RESCORE_DECODER_CHANGED')
                predictions={}
                for key in dict.fromkeys([*corpus.query,*donors.values()]):
                    output=forward(model,corpus,contexts,key,report)
                    predictions[key]=(output['predicted_constellation'].slots,int(next(iter(output['world_owner'].values()))[0]))
                root=a.output/arm;root.mkdir()
                for index,key in enumerate(corpus.query):
                    at=time.perf_counter();cpu=time.process_time();raw,target=targets[key];donor=donors[key]
                    predicted,region=predictions[key];wrong,_=predictions[donor]
                    correct_trace,correct_info=trace(model['decoder'],raw,predicted)
                    wrong_trace,wrong_info=trace(model['decoder'],raw,wrong)
                    old=prior[key];old_score=old['scores']['predicted'] if 'scores' in old else old['surface']
                    need(correct_info['canonical_loss']==old_score['loss'],'RESCORE_PRIOR_DRIFT')
                    if key not in oracle:
                        true_trace,true_info=trace(model['decoder'],raw,target)
                        other_trace,other_info=trace(model['decoder'],raw,targets[donor][1])
                        oracle[key]=(true_trace,other_trace,true_info,other_info)
                    true_trace,other_trace,true_info,other_info=oracle[key]
                    old_true=old['scores']['target'] if 'scores' in old else old['target_surface']
                    need(true_info['canonical_loss']==old_true['loss'],'RESCORE_TRUE_PRIOR_DRIFT')
                    native_path=root/(key+'.npz');native={}
                    for tag,slots in [('predicted',predicted),('wrong',wrong)]:
                        for name,slot in slots.items():native[tag+'__'+name]=array(slot).cpu().numpy()
                    np.savez(native_path,**native,correct=correct_trace,shuffled=wrong_trace,
                             oracle_correct=true_trace,oracle_shuffled=other_trace)
                    files[str(native_path.relative_to(a.output))]=sha(native_path.read_bytes())
                    candidates=alternatives[key];task=corpus.tasks[key]
                    row=dict(arm=arm,task=key,family=task['family'],operation=task['kind'],group=task['split_group'],
                        donor=donor,candidates=candidates,identical_surfaces=[targets[d][0]==raw for d in candidates],
                        source_sha256=sha(raw),source_bytes=len(raw),region=region,
                        lanes=discriminate(predicted,target,[targets[d][1] for d in candidates]),
                        traces=dict(correct=correct_info,shuffled=wrong_info,oracle_correct=true_info,oracle_shuffled=other_info),
                        deltas=dict(predicted=difference(correct_trace,wrong_trace),oracle=difference(true_trace,other_trace)),
                        seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                    out.write(json.dumps(row,sort_keys=True)+'\n');out.flush();rows.append(row)
                    if index%20==0:print(json.dumps(dict(arm=arm,index=index)),flush=True)
                need(fingerprint(model.state_dict())==before,'RESCORE_WEIGHT_DRIFT')
                identities[arm]=dict(model=before,optimizer=moments,decoder=decoder_pin)
                phases[arm]=time.perf_counter()-tick;tick=time.perf_counter()
            os.fsync(out.fileno())
        files['scores.jsonl']=sha((a.output/'scores.jsonl').read_bytes())
        files['plan.json']=sha((a.output/'plan.json').read_bytes())
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'RESCORE_JOURNAL')
        report=dict(schema='capability-rescore-v1',source=source,files=files,rows=len(rows),summary=summarize(rows),
            identities=identities,gpu_before=gpu,model_updates=0,scientific_admission=False,learned_capability=False,
            serving=False,generated_surface_evaluated=False,exposure='development_exposed_group_holdout',
            number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick;wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'RESCORE_PHASE_COVERAGE');report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':run_cli(main)

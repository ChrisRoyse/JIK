"""Real legal-native reciprocal optimizer qualification and same-runtime replay."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

# This is an explicit CPU qualification. PyTorch's optimizer graph-capture
# health check can initialize an available accelerator even for CPU tensors.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_reciprocal_region_readback as proof
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import native_document_training as checkpoints
from model import native_reciprocal_training as steps
from model.native_population_training import cpu, fingerprint
from model.native_reciprocal_region import NativeReciprocalRegion
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need

QUAL = ('legal-reciprocal-region-qualification-v1-20261023',
        '7113d7b74eab7a9e2e63261aa1cc5e11db8b9b00209d56345f98ff4b80052af4')
COLD = ('legal-reciprocal-region-cold-v1-20261023',
        'f3bb0311b8dd46c5376c5214838b459b1bd2445148498af56795505b08e7c7f7')
POLICY = dict(device='cpu', threads=1, seed=20261025, epochs=2, learning_rate=1e-4,
              weight_decay=.01, foreach=False, fused=False, masks_per_update=36,
              scope='exposed_reciprocal_optimizer_qualification', loss_recount_error=5e-7)


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(proof.code(), **{n:file_sha(root/n) for n in
        ['model/native_reciprocal_training.py', 'audit/legal_reciprocal_training.py',
         'model/native_document_training.py']})


def make(target, pin):
    model = NativeReciprocalRegion(target, target_pin=pin, seed=POLICY['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'],
        weight_decay=POLICY['weight_decay'], foreach=False, fused=False)
    return model, optimizer


def load_source(root):
    q, c = proof.pinned(root/QUAL[0], QUAL[1]), proof.pinned(root/COLD[0], COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin'] == QUAL[1]
         and q['binding'] == c['binding'] and q['results_pin'] == c['results_pin']
         and q['binding']['code'] == proof.code(), 'RECIPROCAL_TRAIN_PROOF_AUTHORITY')
    for directory, report in [(root/QUAL[0],q), (root/COLD[0],c)]:
        for name, pin in report['files'].items():
            need(file_sha(directory/name) == pin, 'RECIPROCAL_TRAIN_PROOF_FILE')
    selection = proof.prior.training.pinned(root/QUAL[0]/'selection.json', q['binding']['selection'])
    payload = torch.load(root/QUAL[0]/'inputs.pt', map_location='cpu', weights_only=True)
    metadata, frames, source_binding = proof.source(root)
    need(selection['atoms'] == metadata and q['binding']['source'] == source_binding
         and fingerprint(frames) == selection['native_pin'] == fingerprint(payload['frames'])
         and fingerprint((payload['codes'],payload['counts'])) == selection['labels'],
         'RECIPROCAL_TRAIN_SOURCE_RECONCILIATION')
    stats_root = root/proof.prior.initialization.STATS[0]
    stats_report = proof.pinned(stats_root, proof.prior.initialization.STATS[1])
    need(file_sha(stats_root/'statistics.pt') == stats_report['files']['statistics.pt'], 'RECIPROCAL_TRAIN_STATS')
    stats = torch.load(stats_root/'statistics.pt',map_location='cpu',weights_only=True)
    target, fit, ref = proof.prior.load_target(root,stats,{n:len(stats['centers'][n]) for n in frames[0]})
    need(ref == q['binding']['target'] and fit['binding']['source'] == source_binding['source'],
         'RECIPROCAL_TRAIN_FROZEN_TARGET')
    need(proof.donor_map(metadata,frames,target.roster) == selection['donors'], 'RECIPROCAL_TRAIN_DONOR_RECOUNT')
    indices = [i for i,r in enumerate(metadata) if r['side']=='train' and selection['donors'][i]>=0]
    donors = [selection['donors'][i] for i in indices]
    item = dict(rows=[metadata[i] for i in indices],donor_rows=[metadata[i] for i in donors],
                frames=tuple(frames[i] for i in indices),wrong=tuple(frames[i] for i in donors),
                codes=payload['codes'][indices],counts=payload['counts'][indices])
    item.update(native_pin=fingerprint((item['frames'],item['wrong'])),
                label_pin=fingerprint((item['codes'],item['counts'])),
                source_pin=fingerprint((item['rows'],item['donor_rows'])))
    steps.check_item(item,target.roster)
    need(len(indices)==21 and len(frames)==64, 'RECIPROCAL_TRAIN_QUALIFICATION_SCOPE')
    return target, ref, q, selection, payload, item


def metrics(forecast, region, count, codes, counts, lane, indices, majority, count_mode):
    target = codes[indices].numpy(); predicted = forecast['codes'][indices].numpy()
    equal = predicted == target
    return dict(rows=len(indices), code_accuracy=equal.mean(0).tolist(),
                prefix_accuracy=np.logical_and.accumulate(equal,axis=1).mean(0).tolist(),
                train_subset_mode_accuracy=(target==majority).mean(0).tolist(),
                hidden_count_accuracy=float((forecast['counts'][lane][indices].numpy()==counts[indices].numpy()).mean()),
                train_subset_count_mode_accuracy=float((counts[indices]==count_mode).double().mean()),
                conditional_code_ce=[float(r['ce'][indices].mean()) for r in region],
                conditional_code_brier=[float(r['brier'][indices].mean()) for r in region],
                hidden_count_ce=float(count['ce'][indices].mean()),
                hidden_count_brier=float(count['brier'][indices].mean()))


@torch.no_grad()
def evaluate(model, selection, payload, item, output):
    output.mkdir(); records = {}; values = {}
    frames, codes, counts = payload['frames'], payload['codes'], payload['counts']
    majority = np.array([torch.bincount(item['codes'][:,l],minlength=s).argmax().item()
                         for l,s in enumerate(model.sizes)])
    eligible = [i for i,d in enumerate(selection['donors']) if d>=0]
    for index,(hidden,visible) in enumerate(steps.masks(model.roster)):
        lane = model.roster.index(hidden); count_mode = int(torch.bincount(item['counts'][:,lane]).argmax())
        context = model.encode_visible(tuple({n:f[n] for n in visible} for f in frames),hidden=hidden,visible=visible)
        out = model.loss_withheld(context,codes,counts[:,lane],hidden=hidden)
        region = [proof.recount(score,codes[:,l]) for l,score in enumerate(out['logits'])]
        count = proof.recount(out['count_logits'],counts[:,lane]-1)
        error = max([abs(float(r['ce'].mean())-float(loss)) for r,loss in zip(region,out['region'])]
                    + [abs(float(count['ce'].mean())-float(out['count']))])
        need(error<=POLICY['loss_recount_error'],'RECIPROCAL_TRAIN_NUMPY_CE')
        forecast = cpu(model.forecast(context))
        wrong_frames = tuple({n:frames[selection['donors'][i]][n] for n in visible} for i in eligible)
        wrong_context = model.encode_visible(wrong_frames,hidden=hidden,visible=visible)
        wrong = model.loss_withheld(wrong_context,codes[eligible],counts[eligible,lane],hidden=hidden)
        wrong_region = [proof.recount(score,codes[eligible,l]) for l,score in enumerate(wrong['logits'])]
        wrong_count = proof.recount(wrong['count_logits'],counts[eligible,lane]-1)
        wrong_forecast = cpu(model.forecast(wrong_context)); sides = {}
        for side in ['train','development','query']:
            members = [i for i,r in enumerate(selection['atoms']) if r['side']==side]
            matched = [i for i in members if i in eligible]; positions=[eligible.index(i) for i in matched]
            sides[side] = dict(all=metrics(forecast,region,count,codes,counts[:,lane],hidden,members,majority,count_mode),
                paired=metrics(forecast,region,count,codes,counts[:,lane],hidden,matched,majority,count_mode),
                wrong_context=metrics(wrong_forecast,wrong_region,wrong_count,codes[eligible],counts[eligible,lane],
                                      hidden,positions,majority,count_mode),
                unavailable_for_matched=len(members)-len(matched),
                correct_minus_wrong_log_score=[float(w['ce'][positions].mean()-r['ce'][matched].mean())
                                               for r,w in zip(region,wrong_region)])
        native = dict(forecast=forecast,logits=cpu(out['logits']),count_logits=cpu(out['count_logits']),
                      wrong_forecast=wrong_forecast,wrong_logits=cpu(wrong['logits']),
                      wrong_count_logits=cpu(wrong['count_logits']),eligible=eligible)
        path=output/(str(index)+'.pt');torch.save(native,path)
        need(fingerprint(torch.load(path,map_location='cpu',weights_only=True))==fingerprint(native),
             'RECIPROCAL_TRAIN_EVALUATION_READBACK')
        key=hidden+'--'+','.join(visible); values[key]=fingerprint(native)
        records[key]=dict(hidden=hidden,visible=list(visible),sides=sides,numpy_loss_error=error)
    result=dict(records=records,payloads=values,baseline_scope='21_matched_TRAIN_atoms_only',
                teacher_forcing='earlier_semantic_code_levels_only',scientific_admission=False)
    pin=write_committed(output/'report.json',result)
    return dict(pin=pin,records=records)


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for name,pin in code().items():
        f=a.output/'source'/name;f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(f)==pin,'RECIPROCAL_TRAIN_SOURCE_COPY')
    target,ref,q,selection,payload,item=load_source(a.root)
    target_pin=fingerprint(target.state_dict());native_pin=fingerprint(payload)
    need(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'RECIPROCAL_TRAIN_CPU_ISOLATION')
    binding=dict(code=code(),policy=POLICY,proof=QUAL[1],proof_cold=COLD[1],source=q['binding']['source'],
                 selection=q['binding']['selection'],target=ref,torch_version=str(torch.__version__),
                 cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                 item_source=item['source_pin'],item_native=item['native_pin'],item_labels=item['label_pin'])
    previous=proof.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'RECIPROCAL_TRAIN_COLD_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'RECIPROCAL_TRAIN_COLD_FILE')
    write_committed(a.output/'selection.json',selection);torch.save(item,a.output/'training-inputs.pt')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();arms={}
    for arm in ['paired','wrong_pair']:
        root=a.output/arm;root.mkdir();model,opt=make(target,ref['pin'])
        need(fingerprint(model.state_dict())==q['initial_state'],'RECIPROCAL_TRAIN_INITIAL_STATE')
        arm_binding=dict(binding,arm=arm);chain=digest(arm_binding);rows=0
        evaluation=dict(initial=evaluate(model,selection,payload,item,root/'initial'))
        cps={'0':checkpoints.checkpoint(root/'checkpoint-0.pt',model,opt,arm_binding,0,0,chain)}
        prior=previous['arms'][arm] if previous else None
        if prior:
            checkpoints.load(a.previous/arm/'checkpoint-0.pt',prior['checkpoints']['0']['pin'],model,opt,arm_binding,0,0,POLICY)
        old=(a.previous/arm/'steps.jsonl').open() if prior else None
        with (root/'steps.jsonl').open('x') as ledger:
            for step in range(POLICY['epochs']):
                update=steps.update(model,opt,item,step,POLICY,arm);rows+=update['targets']
                record=dict(parent=chain,update=update);chain=digest(record);record['hash']=chain
                if old:need(record==json.loads(old.readline()),'RECIPROCAL_TRAIN_COLD_UPDATE')
                ledger.write(canonical(record).decode()+'\n');ledger.flush();os.fsync(ledger.fileno())
                cp=root/f'checkpoint-{step+1}.pt'
                cps[str(step+1)]=checkpoints.checkpoint(cp,model,opt,arm_binding,step+1,rows,chain)
                if prior:
                    expected=prior['checkpoints'][str(step+1)]
                    need(cps[str(step+1)]['state']==expected['state'],'RECIPROCAL_TRAIN_COLD_CHECKPOINT')
                    model,opt=make(target,ref['pin'])
                    saved=checkpoints.load(a.previous/arm/cp.name,expected['pin'],model,opt,arm_binding,step+1,rows,POLICY)
                    need(saved['ledger']==chain,'RECIPROCAL_TRAIN_COLD_LEDGER')
                print(json.dumps(dict(arm=arm,step=step+1,masked_targets=rows)),flush=True)
        if old:need(not old.read(),'RECIPROCAL_TRAIN_COLD_EXTRA_UPDATE');old.close()
        evaluation['final']=evaluate(model,selection,payload,item,root/'final')
        before=fingerprint((model.state_dict(),opt.state_dict()));cp=root/'checkpoint-2.pt'
        bad_side=deepcopy(item);bad_side['rows'][0]['side']='query'
        bad_donor=deepcopy(item);bad_donor['donor_rows'][0]['uid']=bad_donor['rows'][0]['uid']
        bad_native=deepcopy(item);bad_native['frames'][0][model.roster[0]][0,0]+=1
        negative=dict(held_fit=rejected(lambda:steps.update(model,opt,bad_side,2,POLICY,arm)),
            donor_identity=rejected(lambda:steps.update(model,opt,bad_donor,2,POLICY,arm)),
            native_changed=rejected(lambda:steps.update(model,opt,bad_native,2,POLICY,arm)),
            label_changed=rejected(lambda:steps.update(model,opt,dict(item,label_pin='0'*64),2,POLICY,arm)),
            wrong_checkpoint=rejected(lambda:checkpoints.load(cp,'0'*64,model,opt,arm_binding,2,rows,POLICY)),
            wrong_cursor=rejected(lambda:checkpoints.load(cp,cps['2']['pin'],model,opt,arm_binding,3,rows,POLICY)),
            target_visible=rejected(lambda:model.encode_visible(item['frames'],hidden=model.roster[0],visible=model.roster[1:])))
        need(before==fingerprint((model.state_dict(),opt.state_dict())),'RECIPROCAL_TRAIN_CONTROL_MUTATION')
        need(target_pin==fingerprint(target.state_dict()) and native_pin==fingerprint(payload),
             'RECIPROCAL_TRAIN_REFERENCE_MUTATION')
        value=dict(updates=2,masked_targets=rows,ledger_head=chain,checkpoints=cps,evaluations=evaluation,controls=negative)
        if prior:
            need(evaluation==prior['evaluations'] and chain==prior['ledger_head'] and negative==prior['controls'],
                 'RECIPROCAL_TRAIN_COLD_RESULTS')
        arms[arm]=value;phases[arm]=time.perf_counter()-tick;tick=time.perf_counter()
    need(not torch.cuda.is_initialized(),'RECIPROCAL_TRAIN_UNEXPECTED_CUDA')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'RECIPROCAL_TRAIN_PHASE_COVERAGE')
    report=dict(complete=True,binding=binding,previous_pin=a.previous_pin,files=files,arms=arms,
        seconds=seconds,cpu_seconds=time.process_time()-process,phases_seconds=phases,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_context_created=False,cuda_optimizer_qualified=False,scientific_admission=False,model_promotion=False,
        number_status='measured',acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,seconds=seconds,updates_per_arm=2,masked_targets_per_arm=1512)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    args=parser.parse_args();need(bool(args.previous)==bool(args.previous_pin),'RECIPROCAL_TRAIN_COLD_ARGUMENTS');run(args)

"""Recount a completed placement branch's saved outputs without another GPU forward."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_efficient_predictor_training as existing
from audit import legal_grouped_placement_training as producer
from audit import legal_grouped_placement_readback as reader
from audit.legal_grouped_io_authority import authenticate
from claims.process_conditions import write_committed
from model import legal_placement_comparison as measure
from model.legal_cached_documents import categorical_baselines
from model.legal_grouped_documents import read_document
from model.legal_native_resident_stream import ResidentNativeFrames
from model.native_population_training import fingerprint
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_native_catalogue import attach


def plain(x):
    if isinstance(x,torch.Tensor):return x.tolist()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def compact(row):
    keys=('document','side','source','component','offset','proposal','target','predicted','measured_codes',
          'target_count','predicted_count','ce','brier','conditional_energy','conditional_frozen_baseline',
          'baselines','donor','native_distance')
    result=plain({k:row[k] for k in keys})
    result['shuffled']=None if row['shuffled'] is None else plain(dict(
        energy=row['shuffled']['energy'],ce=row['shuffled']['categorical']['ce'][0]))
    return result


def summarize(rows,roster):
    result={}
    for key in sorted({(r['side'],r['source'],r['proposal']) for r in rows}):
        selected=[r for r in rows if (r['side'],r['source'],r['proposal'])==key]
        target=np.asarray([r['target'] for r in selected]);pred=np.asarray([r['predicted'] for r in selected])
        equal=pred==target;roundtrip=pred==np.asarray([r['measured_codes'] for r in selected])
        paired=[r for r in selected if r['shuffled'] is not None]
        mean=lambda name:np.asarray([r[name] for r in selected],dtype=np.float64).mean(0).tolist()
        result['|'.join(key)]=dict(rows=len(selected),documents=len({r['document'] for r in selected}),
            code_accuracy=equal.mean(0).tolist(),prefix_accuracy=equal.cumprod(1).mean(0).tolist(),
            roundtrip_prefix=roundtrip.cumprod(1).mean(0).tolist(),ce=mean('ce'),brier=mean('brier'),
            conditional_energy=float(np.mean([r['conditional_energy']['loss'] for r in selected])),
            conditional_frozen_baseline=float(np.mean([r['conditional_frozen_baseline']['loss'] for r in selected])),
            paired_rows=len(paired),missing_donor_rows=len(selected)-len(paired),
            paired_energy_advantage=float(np.mean([r['shuffled']['energy']['loss']-r['conditional_energy']['loss'] for r in paired])) if paired else None,
            paired_ce_advantage=np.mean([np.asarray(r['shuffled']['ce'])-r['ce'] for r in paired],axis=0).tolist() if paired else None,
            count_accuracy={n:sum(r['target_count'][n]==r['predicted_count'][n] for r in selected)/len(selected) for n in roster},
            baseline_accuracy={n:(np.asarray([r['baselines'][n] for r in selected])==target).mean(0).tolist() for n in selected[0]['baselines']},
            baseline_available={n:(np.asarray([r['baselines'][n] for r in selected])>=0).sum(0).tolist() for n in selected[0]['baselines']},
            native_distance={n:{arm:float(np.mean([r['native_distance'][n][arm] for r in selected]))
                                for arm in ('model','copy_current','train_mean')} for n in roster})
    return result


def close_enough(left,right):
    if isinstance(left,dict):return set(left)==set(right) and all(close_enough(left[k],right[k]) for k in left)
    if isinstance(left,list):return len(left)==len(right) and all(close_enough(a,b) for a,b in zip(left,right))
    if isinstance(left,float):return np.isfinite(left) and np.isfinite(right) and abs(left-right)<=1e-12
    return left==right


def checked(payload,item,labels,stats,donors):
    row=payload['record'];i=row['offset'];d=item['document']
    need(type(i) is int and 0<=i<len(item['codes']) and row['document']==d and row['proposal']=='assembled'
         and (row['side'],row['source'],row['component'])==(item['side'],item['source_collection'],item['component'])
         and row['native_pin']==item['native_pin'] and row['label_pin']==item['label_pin'],'BRANCH_ROW_SOURCE')
    need(row['donor']==donors[str(d)][i] and (row['shuffled'] is None)==(row['donor']==-1)
         and (payload['wrong'] is None)==(row['donor']==-1),'BRANCH_ROW_DONOR')
    need(set(row['baselines'])==set(item['baselines']) and all(torch.equal(v,item['baselines'][n][i])
         for n,v in row['baselines'].items()),'BRANCH_ROW_BASELINE')
    need(set(payload['variances'])==set(labels.roster) and all(torch.equal(v,stats['variance'][n])
         for n,v in payload['variances'].items()),'BRANCH_ROW_VARIANCE')
    reader.check_row(payload,item,labels)
    future=item['frames'][i+1];variance=payload['variances']
    need(measure.numpy_energy(payload['conditional'][0],future,variance)==row['conditional_energy'],'BRANCH_ENERGY_TERMS')
    base={n:x[None].expand(4,*x.shape) for n,x in payload['frozen_base'].items()}
    need(measure.numpy_energy(base,future,variance)==row['conditional_frozen_baseline'],'BRANCH_FROZEN_ENERGY')
    if payload['wrong'] is not None:
        need(measure.numpy_energy(payload['wrong'][0],future,variance)==row['shuffled']['energy'],'BRANCH_SHUFFLED_TERMS')
    for n in labels.roster:
        for arm,value in [('copy_current',item['frames'][i][n]),('train_mean',stats['centers'][n].float()[None])]:
            need(abs(measure.distance(future[n],value)-row['native_distance'][n][arm])<=1e-12,'BRANCH_NATIVE_BASELINE')
    return row


def coverage(rows,document,lo,stop):
    need(len(rows)==stop-lo and [(r['record']['document'],r['record']['offset'],r['record']['proposal']) for r in rows]
         ==[(document,i,'assembled') for i in range(lo,stop)],'BRANCH_TILE_COVERAGE')


def negatives(payload,item,labels,stats,donors):
    out={}
    def reject(name,value,code):
        try:checked(value,item,labels,stats,donors)
        except (ValueError,RuntimeError) as e:need(code in str(e),'BRANCH_WRONG_REFUSAL:'+str(e));out[name]=str(e);return
        raise ValueError('BRANCH_CORRUPTION_ACCEPTED:'+name)
    for name,key,value,code in [('source','source','wrong-source','BRANCH_ROW_SOURCE'),
                               ('native_pin','native_pin','0'*64,'BRANCH_ROW_SOURCE'),
                               ('donor','donor',-2,'BRANCH_ROW_DONOR')]:
        bad=deepcopy(payload);bad['record'][key]=value;reject(name,bad,code)
    bad=deepcopy(payload);bad['target'][labels.roster[0]][0,0]+=.25;reject('target',bad,'PLACEMENT_READBACK_NATIVE_SOURCE')
    bad=deepcopy(payload);bad['probabilities'][0][0]+=.25;reject('probability',bad,'PLACEMENT_READBACK_PROBABILITY')
    bad=deepcopy(payload);bad['record']['conditional_energy']['loss']+=.25;reject('energy',bad,'PLACEMENT_READBACK_ENERGY')
    bad=deepcopy(payload);bad['record']['conditional_energy']['lanes'][labels.roster[0]]+=.25;reject('energy_term',bad,'BRANCH_ENERGY_TERMS')
    bad=deepcopy(payload);bad['record']['native_distance'][labels.roster[0]]['model']+=.25;reject('native_distance',bad,'PLACEMENT_READBACK_NATIVE_DISTANCE')
    bad=deepcopy(payload);bad['record']['baselines']['copy_current'][0]=-2;reject('baseline',bad,'BRANCH_ROW_BASELINE')
    try:coverage([],item['document'],0,1)
    except ValueError as e:need('BRANCH_TILE_COVERAGE' in str(e),'BRANCH_WRONG_REFUSAL');out['missing_row']=str(e)
    else:raise ValueError('BRANCH_MISSING_ROW_ACCEPTED')
    return out


def tensor_file(path,files):
    identity=stamp(path);need(path.is_file() and not path.is_symlink() and path.stat().st_size<=128<<20,'BRANCH_PAYLOAD_LAYOUT')
    raw=path.read_bytes();need(stamp(path)==identity,'BRANCH_PAYLOAD_CHANGED')
    files[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),identity=list(identity))
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    files={};saved=tensor_file(a.checkpoint,files);b=saved['binding'];branch=a.checkpoint.parent
    need(files[str(a.checkpoint)]['sha256']==a.checkpoint_pin and b['architecture']=='shared'
         and saved['step']==91540,'BRANCH_CHECKPOINT')
    donor_path=branch.parent/'donors.json';need(file_sha(donor_path)==b['donors'],'BRANCH_DONOR_PIN')
    donor=json.loads(donor_path.read_bytes());donor['metadata']={int(k):v for k,v in donor['metadata'].items()}
    measure.validate_donors(donor['metadata'],donor['donors'])
    src,bundle=producer.source_bundle(a.root);need(b['code']==producer.code(src),'BRANCH_PRODUCER_CODE')
    base=Path(__file__).resolve().parents[1];code=dict(b['code'])
    for n in ('audit/placement_branch_diagnostic.py','audit/legal_grouped_placement_readback.py','text/legal_native_catalogue.py'):
        code[n]=file_sha(base/n)
    need(all(file_sha(base/n)==h for n,h in code.items()),'BRANCH_CODE_PIN')
    args=argparse.Namespace(root=a.root,scope='full',arm='breadth')
    for n,(path,pin) in producer.REFS.items():setattr(args,n,a.root/path);setattr(args,n+'_pin',pin)
    data,continuation,old_reader,labels,cache,meta=existing.breadth.prepare(args)
    catalogue=attach(dict(data=data,reader=old_reader),a.catalogue,a.catalogue_pin);old_reader.close()
    proof=existing.small.qualification.training.DEFAULTS['audit'];native=ResidentNativeFrames(data,a.root/proof[0],proof[1])
    try:
        io_authority,io_fresh=authenticate(a.root)
        schedule,receipt=existing.load_schedule(args,data,cache,meta);plan=existing.selection(args,data,meta,schedule)
        binding=dict(source=data.binding,cache=cache.binding,metadata=meta.binding,schedule=producer.REFS['schedule'][1],plan=plan,io=io_authority)
        need(binding==b['source'] and labels.state==bundle['target_fingerprint'],'BRANCH_SOURCE_BINDING')
        need(len(plan['evaluations'])==192 and saved['step']==len(plan['train_documents'])
             and saved['rows']==receipt['summary']['targets'],'BRANCH_SOURCE_SCOPE')
        phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
        records={stage:[] for stage in ('initial','final')};controls={};expected=set();total=0;metadata={}
        handles={stage:(a.output/(stage+'.jsonl')).open('x') for stage in records}
        try:
            for selected in plan['evaluations']:
                d=selected['document'];item,_=read_document(data,native,cache,d,side=selected['side'],fit=False)
                item.update(component=int(data.components[d]),source_collection=meta.rows[d]['source_path'])
                need((item['source_start'],item['source_stop'])==(selected['start'],selected['stop']),'BRANCH_SOURCE_BOUNDS')
                baselines,_=categorical_baselines(cache,continuation,item['uids'][:-1]);item['baselines']={n:torch.from_numpy(v.copy()) for n,v in baselines.items()}
                metadata.update(measure.metadata([item]));need(metadata[d]==donor['metadata'][d],'BRANCH_METADATA')
                for lo in range(0,len(item['codes']),8):
                    stop=min(lo+8,len(item['codes']));name=f'document-{d}-rows-{lo}-{stop}-assembled.pt';expected.add(name)
                    for stage in records:
                        payloads=tensor_file(branch/stage/name,files);coverage(payloads,d,lo,stop)
                        for payload in payloads:
                            row=checked(payload,item,labels,bundle['statistics'],donor['donors'])
                            if not controls:controls=negatives(payload,item,labels,bundle['statistics'],donor['donors'])
                            records[stage].append(row);handles[stage].write(canonical(compact(row)).decode()+'\n');total+=1
                if len(metadata)%16==0:print(json.dumps(dict(documents=len(metadata),rows=total)),flush=True)
        finally:
            for h in handles.values():h.flush();os.fsync(h.fileno());h.close()
        need(metadata==donor['metadata'] and total==49644,'BRANCH_FULL_COVERAGE')
        for stage in records:need(expected=={p.name for p in (branch/stage).iterdir()},'BRANCH_FILE_COVERAGE')
        phases['source_payload_score_recount']=time.perf_counter()-tick;tick=time.perf_counter()
        summaries={stage:measure.summary(rows,labels.roster) for stage,rows in records.items()}
        for stage,rows in records.items():need(close_enough(summaries[stage],summarize([compact(r) for r in rows],labels.roster)),'BRANCH_SUMMARY_RECOUNT')
        # Original initial evaluation is already bound by the immutable retirement publication.
        retirement=a.root/'legal-answer-retirement-v1-20261024/seal/boundary.json'
        need(file_sha(retirement)=='1092333ca134641e4261681d9f97bbf27f0b526ddda0a99be91093cf2cf6a5f4','BRANCH_INITIAL_AUTHORITY')
        inherited=json.loads(retirement.read_bytes())['branches']['shared']
        need(fingerprint(records['initial'])==inherited['initial_rows'] and summaries['initial']==inherited['initial'],'BRANCH_INITIAL_CHANGED')
        need(all(stamp(Path(p))==tuple(v['identity']) and file_sha(Path(p))==v['sha256']
                 for p,v in files.items()),'BRANCH_ARTIFACT_CHANGED')
        need(all(file_sha(base/n)==h for n,h in code.items()) and file_sha(donor_path)==b['donors'],'BRANCH_AUTHORITY_CHANGED')
        phases['summaries_and_immutability']=time.perf_counter()-tick;tick=time.perf_counter()
        cache.close();native.close();continuation.fresh();meta.fresh();labels.fresh();io_fresh()
        phases['close']=time.perf_counter()-tick
        out=dict(complete=True,schema='placement-branch-diagnostic-v1',checkpoint=a.checkpoint_pin,checkpoint_state=fingerprint(saved),binding=b,
                 code=code,catalogue=catalogue,roster=list(labels.roster),files=files,controls=controls,
                 scores={stage:dict(path=stage+'.jsonl',pin=file_sha(a.output/(stage+'.jsonl')),records=fingerprint(rows)) for stage,rows in records.items()},
                 summaries=summaries,rows=total,documents=len(metadata),phases=phases,wall=time.perf_counter()-began,
                 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_forward_replayed=False,
                 untouched_external_test=False,scientific_admission=False,model_promoted=False,acceptance='author-checked')
        need(not torch.cuda.is_initialized() and sum(phases.values())/out['wall']>=.9,'BRANCH_RUNTIME')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',out),wall=out['wall'],rows=total)),flush=True)
    finally:
        if not native.closed:native.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','checkpoint','catalogue','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('checkpoint-pin','catalogue-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise

"""Fixed shared checkpoints, identical complete legal histories, explicit CPU diagnosis."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_efficient_predictor_training as existing
from audit import legal_grouped_placement_training as original
from audit import region_probability_readback as probabilities
from audit.legal_grouped_io_authority import authenticate
from audit.placement_source_exposure import read,save,sha
from model import legal_placement_comparison as measure
from model.legal_cached_documents import categorical_baselines
from model.legal_grouped_documents import read_document
from model.legal_native_resident_stream import ResidentNativeFrames
from model.native_placement_efficient import stream_for
from model.native_population_training import cpu,fingerprint
from model.region_count_probabilities import Tables
from text.legal_native_catalogue import attach

STEPS=(14336,15360,91540)
SEAL=('answer-refresh-retirement-v1-20261025/seal','35dd82d4e93b6476810a23301b4e75ffd57c4b7b50bb02e88b9f85be7ff7301d')
CATALOGUE=('native-catalogue-publication-v2-20261025/catalogue.json','6e218e778cb98a24ffff25060249ee0b10a6ff83a0a6085d3ebd4ca33cbf92d8')


def need(ok,code):
    if not ok:raise ValueError('RETENTION_'+code)


def reject(fn):
    try:fn()
    except (ValueError,RuntimeError) as error:return str(error)
    raise AssertionError('RETENTION_NEGATIVE_ACCEPTED')


class Source:
    def __init__(self,root):
        self.root=root;self.inputs={str(root/SEAL[0]/'boundary.json'):SEAL[1]}
        self.boundary=read(root/SEAL[0]/'boundary.json',SEAL[1]);self.binding=self.boundary['binding']
        self.source,self.bundle=original.source_bundle(root)
        args=argparse.Namespace(root=root,scope='full',arm='breadth')
        for n,(path,pin) in original.REFS.items():setattr(args,n,root/path);setattr(args,n+'_pin',pin)
        self.data,self.continuation,old,self.labels,self.cache,self.meta=existing.breadth.prepare(args)
        self.catalogue=attach(dict(data=self.data,reader=old),root/CATALOGUE[0],CATALOGUE[1]);old.close()
        self.inputs[str(root/CATALOGUE[0])]=CATALOGUE[1]
        proof=existing.small.qualification.training.DEFAULTS['audit']
        self.native=ResidentNativeFrames(self.data,root/proof[0],proof[1])
        authority,self.io_fresh=authenticate(root)
        schedule,receipt=existing.load_schedule(args,self.data,self.cache,self.meta)
        self.plan=existing.selection(args,self.data,self.meta,schedule)
        binding=dict(source=self.data.binding,cache=self.cache.binding,metadata=self.meta.binding,
                     schedule=original.REFS['schedule'][1],plan=self.plan,io=authority)
        need(binding==self.binding['source'] and len(self.plan['evaluations'])==192,'SOURCE_BINDING')
        need(self.labels.state==self.bundle['target_fingerprint'],'TARGET_STATE')
        self.target=deepcopy(self.labels.model).eval().requires_grad_(False)
        self.target_state=fingerprint(self.target.state_dict())
        counts,sizes,gpu,files,_,inherited=probabilities.authenticate(root)
        self.inputs.update(files);self.counts=counts;self.tables=Tables(counts,sizes)
        self.gpu={(r['document'],r['offset']):r for r in gpu['final']}
        need(len(self.gpu)==24822,'GPU_SCOPE')
        self.models={};self.checkpoints={}
        for step in STEPS:
            path=root/SEAL[0]/'shared'/f'checkpoint-{step}.pt'
            ref=self.boundary['branches']['shared']['checkpoints'][str(step)]
            need(sha(path)==ref['pin'],'CHECKPOINT_PIN');self.inputs[str(path)]=ref['pin']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.check(saved,step,ref)
            model,opt=original.runtime.make(self.bundle,self.target,'shared','cpu')
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
            original.efficient.validate_optimizer(model,opt,self.binding['policy'],step)
            need(fingerprint((model.state_dict(),opt.state_dict()))==fingerprint((saved['model'],saved['optimizer'])),'RESTORE')
            model.eval().requires_grad_(False);self.models[step]=model;self.checkpoints[step]=ref
        self.model_states={s:fingerprint(m.state_dict()) for s,m in self.models.items()}
        names=set(self.binding['code'])|set(inherited)|{'audit/placement_checkpoint_retention.py',
            'audit/placement_checkpoint_retention_readback.py','audit/placement_source_exposure.py',
            'audit/region_probability_readback.py','model/region_count_probabilities.py','text/legal_native_catalogue.py'}
        base=Path(__file__).resolve().parents[1]
        self.code={n:sha(base/n) for n in sorted(names)}

    def check(self,saved,step,ref):
        need(fingerprint(saved)==ref['state'] and saved['step']==step,'CHECKPOINT_STATE')
        need(saved['binding']==dict(self.binding,architecture='shared') and saved['noise_cursor']==self.binding['policy']['noise_seed']+step,
             'CHECKPOINT_BINDING')

    def item(self,selected):
        d=selected['document'];item,_=read_document(self.data,self.native,self.cache,d,side=selected['side'],fit=False)
        item.update(component=int(self.data.components[d]),source_collection=self.meta.rows[d]['source_path'])
        need((item['source_start'],item['source_stop'])==(selected['start'],selected['stop']),'DOCUMENT_BOUNDS')
        baseline,_=categorical_baselines(self.cache,self.continuation,item['uids'][:-1])
        item['baselines']={k:torch.from_numpy(v.copy()) for k,v in baseline.items()}
        need(len(item['codes'])>0 and item['side'] in ('development','query'),'HELDOUT_SCOPE')
        return item

    def finish(self):
        self.cache.close();self.native.close();self.continuation.fresh();self.meta.fresh();self.labels.fresh();self.io_fresh()
        need(fingerprint(self.target.state_dict())==self.target_state and all(fingerprint(m.state_dict())==self.model_states[s]
             for s,m in self.models.items()),'MODEL_CHANGED')
        need(all(sha(p)==pin for p,pin in self.inputs.items()),'INPUT_CHANGED')
        base=Path(__file__).resolve().parents[1]
        need(all(sha(base/n)==pin for n,pin in self.code.items()),'CODE_CHANGED')
        need(not torch.cuda.is_initialized(),'UNEXPECTED_GPU')


@torch.no_grad()
def forward(model,item,*,reference=False):
    stream=model.predictor.stream() if reference else stream_for(model)
    state=None;logits=[[],[],[]];pred=[];n=len(item['codes'])
    for lo in range(0,len(item['frames']),8):
        h,state=stream.step(item['frames'][lo:lo+8],context='retention:'+str(item['document']),offset=lo,state=state,detach_history=True)
        stop=min(lo+8,n)
        if lo>=n:continue
        hidden=h[:stop-lo]
        autonomous=model.predictor.forecast(hidden)['codes']
        if reference:values,_=model.predictor.region(hidden,item['codes'][lo:stop])
        else:values=model.predictor.loss(hidden,item['codes'][lo:stop],{k:v[lo:stop] for k,v in item['counts'].items()})['logits']
        for j,x in enumerate(values):logits[j].append(x)
        pred.append(autonomous)
    need(state.next_atom==len(item['frames']),'HISTORY_COVERAGE')
    values=tuple(torch.cat(x) for x in logits);codes=torch.cat(pred)
    scores=measure.categorical(values,item['codes'])
    return cpu(dict(logits=values,predicted=codes,**scores))


def identity(item):
    return dict(document=item['document'],source=item['source_collection'],side=item['side'],component=item['component'],
                targets=len(item['codes']),native_pin=item['native_pin'],label_pin=item['label_pin'],
                source_start=item['source_start'],source_stop=item['source_stop'])


def check_payload(payload,item):
    need(payload['identity']==identity(item),'PAYLOAD_IDENTITY')
    need(torch.equal(payload['target'],item['codes']) and torch.equal(payload['current'],item['baselines']['copy_current']),'PAYLOAD_TARGET')
    need(set(payload['models'])=={str(s) for s in STEPS},'PAYLOAD_STEPS')
    for value in payload['models'].values():
        need(value['predicted'].shape==item['codes'].shape and all(x.shape==(len(item['codes']),n)
             for x,n in zip(value['logits'],(4,16,64))),'PAYLOAD_LAYOUT')
        recounted=measure.categorical(value['logits'],item['codes'])
        need(fingerprint(recounted)==fingerprint({k:value[k] for k in recounted}),'PAYLOAD_SCORES')


def document_score(payload,baselines):
    targets=payload['target'];n=len(targets)
    out=dict(**payload['identity'],models={},baselines={})
    for step,v in payload['models'].items():
        eq=(v['predicted']==targets).double();ce=v['ce'].double();brier=v['brier'].double()
        out['models'][step]=dict(ce=ce.mean(0).tolist(),brier=brier.mean(0).tolist(),joint_ce=float(ce.sum(1).mean()),
                                  code_accuracy=eq.mean(0).tolist(),full_code_accuracy=float(eq.prod(1).mean()))
    for name in baselines[0]:
        ce=[[r[name]['ce'][j] for r in baselines] for j in range(3)]
        br=[[r[name]['brier'][j] for r in baselines] for j in range(3)]
        out['baselines'][name]=dict(ce=[math.fsum(x)/n for x in ce],brier=[math.fsum(x)/n for x in br],
                                   joint_ce=math.fsum(math.fsum(r[name]['ce']) for r in baselines)/n)
    return out


def summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['side']+'|'+r['source']].append(r)
    out={}
    for key,docs in sorted(groups.items()):
        count=sum(r['targets'] for r in docs)
        need(len(docs)==len({r['component'] for r in docs})==32,'CELL_CASES')
        cell=dict(documents=len(docs),targets=count,models={},baselines={},pairs={})
        for family in ('models','baselines'):
            for name in docs[0][family]:
                cell[family][name]={metric:math.fsum(r['targets']*r[family][name][metric] for r in docs)/count
                                   for metric in ('joint_ce',)}
                for metric in ('ce','brier'):
                    cell[family][name][metric]=[math.fsum(r['targets']*r[family][name][metric][j] for r in docs)/count for j in range(3)]
                if family=='models':
                    cell[family][name]['code_accuracy']=[math.fsum(r['targets']*r[family][name]['code_accuracy'][j] for r in docs)/count for j in range(3)]
                    cell[family][name]['full_code_accuracy']=math.fsum(r['targets']*r[family][name]['full_code_accuracy'] for r in docs)/count
        for earlier,later in ((14336,15360),(15360,91540),(14336,91540)):
            gaps=[r['models'][str(earlier)]['joint_ce']-r['models'][str(later)]['joint_ce'] for r in docs]
            cell['pairs'][str(earlier)+'->'+str(later)]=dict(atom_weighted_advantage=math.fsum(r['targets']*g for r,g in zip(docs,gaps))/count,
                document_mean_advantage=math.fsum(gaps)/len(gaps),wins=sum(g>0 for g in gaps),ties=sum(g==0 for g in gaps),losses=sum(g<0 for g in gaps))
        out[key]=cell
    need(len(out)==6 and sum(r['targets'] for r in rows)==24822 and len(rows)==192,'TOTAL_SUPPORT')
    return out


def run(a):
    began=tick=time.perf_counter();cpu0=time.process_time();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=Source(a.root);phases=dict(authentication=time.perf_counter()-tick);files={};docs=[];controls={};native_bytes=0
    phases.update(native_read=0.,forwards=0.,scores_and_persistence=0.)
    runtime=dict(max_abs_ce=0.,max_abs_brier=0.,changed_autonomous_codes=0,rows=0)
    try:
        for selected in source.plan['evaluations']:
            tick=time.perf_counter();item=source.item(selected);phases['native_read']+=time.perf_counter()-tick
            native_bytes+=sum(x.numel()*x.element_size() for f in item['frames'] for x in f.values())
            tick=time.perf_counter();models={str(s):forward(m,item) for s,m in source.models.items()};phases['forwards']+=time.perf_counter()-tick
            tick=time.perf_counter();payload=cpu(dict(identity=identity(item),target=item['codes'],current=item['baselines']['copy_current'],models=models))
            check_payload(payload,item);baselines=[]
            for i,(target,current) in enumerate(zip(payload['target'].tolist(),payload['current'].tolist())):
                prior=source.gpu[item['document'],i]
                need(prior['source']==item['source_collection'] and prior['side']==item['side'] and prior['component']==item['component']
                     and prior['target']==target and prior['baselines']['copy_current']==current,'GPU_JOIN')
                v=models['91540'];runtime['max_abs_ce']=max(runtime['max_abs_ce'],max(abs(x-y) for x,y in zip(v['ce'][i].tolist(),prior['ce'])))
                runtime['max_abs_brier']=max(runtime['max_abs_brier'],max(abs(x-y) for x,y in zip(v['brier'][i].tolist(),prior['brier'])))
                runtime['changed_autonomous_codes']+=v['predicted'][i].tolist()!=prior['predicted'];runtime['rows']+=1
                baselines.append(source.tables.scores(target,current))
            docs.append(document_score(payload,baselines))
            path=a.output/f"document-{item['document']}.pt";receipt=measure.persist(path,payload);files[path.name]=receipt
            if not controls:
                for name,key,value in (('source','source','wrong'),('document','document',-1),('offset','source_start',-1)):
                    bad=deepcopy(payload);bad['identity'][key]=value;controls[name]=reject(lambda:check_payload(bad,item))
                bad=deepcopy(payload);bad['target'][0,0]=(bad['target'][0,0]+1)%4
                controls['target']=reject(lambda:check_payload(bad,item))
                bad=deepcopy(payload);bad['models']['91540']['probabilities'][0][0,0]+=.25
                controls['probability']=reject(lambda:check_payload(bad,item))
                bad=deepcopy(payload);bad['models']['91540']['ce'][0,0]+=.25
                controls['score']=reject(lambda:check_payload(bad,item))
                bad=deepcopy(payload);bad['models'].pop('14336')
                controls['missing_checkpoint']=reject(lambda:check_payload(bad,item))
                witness=torch.load(a.root/SEAL[0]/'shared/checkpoint-14336.pt',map_location='cpu',weights_only=True)
                controls['wrong_checkpoint']=reject(lambda:source.check(witness,15360,source.checkpoints[15360]))
            phases['scores_and_persistence']+=time.perf_counter()-tick
            if len(docs)%16==0:print(json.dumps(dict(documents=len(docs),targets=sum(d['targets'] for d in docs))),flush=True)
        tick=time.perf_counter();result=summary(docs);controls['missing_document']=reject(lambda:summary(docs[:-1]))
        save(a.output/'documents.json',docs);save(a.output/'summary.json',result)
        source.finish();phases['closing']=time.perf_counter()-tick
        base=Path(__file__).resolve().parents[1]
        for name,pin in source.code.items():
            p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((base/name).read_bytes());need(sha(p)==pin,'CODE_COPY')
        out=dict(schema='placement-checkpoint-retention-v1',complete=True,inputs=source.inputs,code=source.code,catalogue=source.catalogue,
                 checkpoints=source.checkpoints,files=files,documents_pin=sha(a.output/'documents.json'),summary_pin=sha(a.output/'summary.json'),
                 source_binding=source.binding['source'],native_bytes=native_bytes,rows=24822,documents=192,model_rows=74466,
                 final_cpu_vs_saved_cuda=runtime,controls=controls,device='cpu',gpu_used=False,optimizer_updates=0,
                 phases=phases,seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu0,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 scope='Exposed source-split conditional region log scores; not whole-answer or legal-QA performance.',
                 architecture_winner=False,interleaving_benefit_established=False,acceptance='author-checked')
        need(.9<=sum(phases.values())/out['seconds']<=1.000001,'PHASE_COVERAGE')
        save(a.output/'report.json',out);print(json.dumps(dict(pin=sha(a.output/'report.json'),seconds=out['seconds'],runtime=runtime)),flush=True)
    finally:
        if not source.native.closed:source.native.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():save(a.output/'failure.json',dict(error=repr(error)))
        raise

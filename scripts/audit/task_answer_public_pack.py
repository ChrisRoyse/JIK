"""Export and independently read task prompts without giving generation an answer horizon."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from claims.process_conditions import write_committed
from model import task_answer_public_pack as packing
from model.answer_training_schedule import AnswerSchedule
from model.canonical_answer_inputs import CanonicalAnswerInputs
from model.native_population_training import fingerprint
from model.native_request_input import describe, NativeRequest
from text.constellation_corpus import canonical, digest, file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
RECOVERY = ROOT/'phase115-complete-answer-recovery-v1-20261024'
PLAN = dict(path=str(ROOT/'answer-native-plan-v1-20261023'),
            pin='22993ed7704f1d31ca48df404c0af14acff1d466624b412a9c55be2132cf1098')
CONSUMER = dict(path=str(RECOVERY/'bind'),pin='a4ecbd4ea8db243fee604d08806c02faeb4abe82c96369c1deae40459812bb16')
COLD = dict(path=str(RECOVERY/'bind-cold'),pin='fe8d25764215ae7001be0994e853bdfcd6978f89623c99d9ec40325aa6285591')
BASE = Path(__file__).resolve().parents[1]
CODE = ('model/task_answer_public_pack.py','audit/task_answer_public_pack.py',
        'model/native_request_input.py','model/native_population_training.py')


def read(ref):
    path=Path(ref['path'])/'report.json';need(file_sha(path)==ref['pin'],'TASK_PACK_SOURCE_PIN')
    return json.loads(path.read_bytes())


def source():
    first,second=read(CONSUMER),read(COLD)
    need(first['complete'] and first['mode']=='bind' and first['experimental_items_ready']
         and first['plan']==PLAN and second['complete'] and second['source']==CONSUMER
         and second['all_native_inputs_checked'] and second['all_learner_items_checked']
         and first['summary']==second['summary'] and not first['scientific_admission'], 'TASK_PACK_SOURCE_SCOPE')
    schedule=AnswerSchedule(PLAN['path'],PLAN['pin']);refs=first['native']
    store=CanonicalAnswerInputs(schedule,refs['native']['path'],refs['native']['pin'],refs['cold']['path'],refs['cold']['pin'])
    need(store.binding==first['native_binding'],'TASK_PACK_NATIVE_BINDING')
    binding=dict(plan=PLAN,consumer=CONSUMER,consumer_cold=COLD,native=store.binding,
                 code={n:file_sha(BASE/n) for n in CODE})
    return schedule,store,binding


def public_row(schedule,key):
    row=schedule.rows[key];public=schedule.request(key)
    return dict(task=key,split=row['split'],group=row['group'],family=row['family'],
        population=schedule.plan.tasks[key]['population'],
        fields=[dict(field=f['field'],input=f['input']) for f in public['fields']],
        descriptor=describe(schedule,key))


def publish(output,schedule,store,binding):
    output.mkdir();rows=[public_row(schedule,key) for key in sorted(schedule.rows)]
    ids=sorted({f['input'] for r in rows for f in r['fields']})
    write_committed(output/'inputs.json',ids);windows={}
    for n in store.roster:
        counts=[int(store.offsets[n][store.index[i]+1]-store.offsets[n][store.index[i]]) for i in ids]
        offsets=np.r_[np.int64(0),np.cumsum(counts,dtype='<i8')]
        np.save(output/(n+'.offsets.npy'),offsets,allow_pickle=False)
        values=np.lib.format.open_memmap(output/(n+'.windows.npy'),mode='w+',dtype='<f4',
                                        shape=(int(offsets[-1]),store.dimensions[n]))
        try:
            for j,i in enumerate(ids):
                k=store.index[i];a,b=store.offsets[n][k:k+2]
                values[offsets[j]:offsets[j+1]]=store.arrays[n][a:b]
            values.flush()
        finally:values._mmap.close()
        windows[n]=int(offsets[-1])
    with (output/'cases.jsonl').open('xb') as f:
        for row in rows:
            frames=tuple(store.frame(v['input']) for v in row['fields'])
            row['state']=fingerprint(dict(frames=frames,descriptor=row['descriptor']))
            row['pin']=packing.seal(row);packing.validate_row(row)
            f.write(canonical(row)+b'\n')
        f.flush();os.fsync(f.fileno())
    for p in output.iterdir():
        with p.open('rb') as f:os.fsync(f.fileno())
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in output.iterdir()}
    summary=dict(tasks=len(rows),unique_inputs=len(ids),context_fields=sum(len(r['fields']) for r in rows),
        splits={s:sum(r['split']==s for r in rows) for s in ('train','query')},windows=windows,
        payload_bytes=sum(windows[n]*store.dimensions[n]*4 for n in store.roster))
    return write_committed(output/'report.json',dict(schema=packing.SCHEMA,complete=True,policy=packing.POLICY,
        binding=binding,roster=list(store.roster),dimensions=store.dimensions,files=files,summary=summary))


def direct(pack,schedule,store):
    """Read original population requests and NPZ rows separately from the exporter."""
    total=0;read_bytes=0
    need(pack.report['binding']['native']==store.binding,'TASK_PACK_DIRECT_BINDING')
    for name in pack.roster:
        with np.load(store.source/name/'reference.npz',allow_pickle=False) as z:
            vectors,bounds=z['vectors'],z['bounds']
        for j,identity in enumerate(pack.ids):
            k=store.index[identity];a,b=map(int,bounds[k:k+2]);c,d=map(int,pack.offsets[name][j:j+2])
            need(b-a==d-c and vectors[a:b].tobytes()==pack.vectors[name][c:d].tobytes(), 'TASK_PACK_DIRECT_NATIVE')
            total+=b-a;read_bytes+=vectors[a:b].nbytes
    chain='0'*64
    for row in pack.cases:
        key=row['task'];pop,task,split=schedule.targets.source.selected(key)
        need(row['population']==pop.pin and row['split']==split and row['group']==task['split_group']
             and row['family']==pop.family,'TASK_PACK_DIRECT_PARTITION')
        fields=[pop.fields[k] for k in task['context']]
        need(row['fields']==[dict(field=f['id'],input=digest(f['text'].encode())) for f in fields],
             'TASK_PACK_DIRECT_FIELDS')
        need(row['descriptor']==describe(schedule,key),'TASK_PACK_DIRECT_DESCRIPTOR')
        value=pack.load(key,row['pin']);original=store.prompt(key)
        need(fingerprint(value['context'])==fingerprint(original['context'])
             and value['request'].descriptor==original['request'].descriptor
             and torch.equal(value['request'].features,original['request'].features),'TASK_PACK_DIRECT_PROMPT')
        chain=digest(bytes.fromhex(chain)+canonical(dict(task=key,pin=row['pin'])))
    return dict(tasks=len(pack.cases),unique_inputs=len(pack.ids),windows=total,bytes=read_bytes,
                all_native_bytes_exact=True,all_public_requests_exact=True,chain=chain)


def rejected(fn):
    try:
        v=fn()
        if isinstance(v,packing.TaskPublicPack):v.close()
    except (ValueError,RuntimeError) as e:return str(e)
    raise ValueError('TASK_PACK_NEGATIVE_PASSED')


def controls(pack,output):
    output.mkdir();results={}
    results['wrong_report_pin']=rejected(lambda:packing.TaskPublicPack(pack.root,'0'*64))
    row=pack.cases[0]
    results['wrong_request_pin']=rejected(lambda:pack.load(row['task'],'0'*64))
    expected=deepcopy(pack.report['binding']);expected['native']['source']='0'*64
    results['wrong_source_binding']=rejected(lambda:packing.TaskPublicPack(pack.root,pack.pin,expected))
    for fault in ('target_field','reordered_fields','wrong_split','native_byte','offset','missing_lane'):
        dest=output/fault;dest.mkdir();r=deepcopy(pack.report)
        for name in r['files']:(dest/name).symlink_to(os.path.relpath(pack.root/name,dest))
        if fault in ('target_field','reordered_fields','wrong_split'):
            name='cases.jsonl';rows=deepcopy(pack.cases)
            if fault=='target_field':rows[0]['answer_length']=1
            elif fault=='wrong_split':rows[0]['split']='query' if rows[0]['split']=='train' else 'train'
            else:
                selected=next(x for x in rows if len(x['fields'])>1);selected['fields']=selected['fields'][::-1]
            (dest/name).unlink();(dest/name).write_bytes(b''.join(canonical(x)+b'\n' for x in rows))
        else:
            name=pack.roster[0]+('.offsets.npy' if fault=='offset' else '.windows.npy')
            (dest/name).unlink()
            if fault=='missing_lane':del r['files'][name]
            else:
                shutil.copyfile(pack.root/name,dest/name)
                a=np.load(dest/name,mmap_mode='r+',allow_pickle=False)
                if fault=='offset':a[0]=1
                else:a[0,0]=np.nextafter(a[0,0],np.float32(np.inf))
                a.flush();a._mmap.close()
        if fault!='missing_lane':r['files'][name]=dict(sha256=file_sha(dest/name),bytes=(dest/name).stat().st_size)
        pin=write_committed(dest/'report.json',r)
        def probe():
            bad=packing.TaskPublicPack(dest,pin)
            try:
                for row in bad.cases:bad.load(row['task'],row['pin'])
            finally:bad.close()
        results[fault]=rejected(probe)
    reasons=dict(wrong_report_pin='TASK_PUBLIC_REPORT_PIN',wrong_request_pin='TASK_PUBLIC_REQUEST_PIN',
        wrong_source_binding='TASK_PUBLIC_SOURCE_BINDING',target_field='TASK_PUBLIC_PRIVATE_FIELD',
        reordered_fields='TASK_PUBLIC_CASE_PIN',wrong_split='TASK_PUBLIC_CASE_PIN',native_byte='TASK_PUBLIC_NATIVE_STATE',
        offset='TASK_PUBLIC_LAYOUT',missing_lane='TASK_PUBLIC_FILE_SET')
    need(all(results[k]=='MNLM_TEXT_CORPUS_'+v for k,v in reasons.items()),'TASK_PACK_CONTROL_REASON')
    return results


def cold(a):
    # Install after imports, before opening the pack. Any evidence read outside
    # this public publication is a refusal, including its private source paths.
    reads=set();allowed=a.input.resolve();out=a.output.resolve()
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(ROOT):
                need(path.is_relative_to(allowed) or path.is_relative_to(out),'TASK_PACK_PRIVATE_SOURCE_OPEN')
                reads.add(str(path))
    sys.addaudithook(audit)
    pack=packing.TaskPublicPack(a.input,a.pin);chain='0'*64
    try:
        for row in pack.cases:
            value=pack.load(row['task'],row['pin'])
            need(set(value)=={'context','request'},'TASK_PACK_PUBLIC_INTERFACE')
            # Execute feature materialization without a predictor or an answer.
            need(value['request'].materialize('cpu').shape[0]==len(value['context'])+1,'TASK_PACK_REQUEST_FEATURES')
            chain=digest(bytes.fromhex(chain)+canonical(dict(task=row['task'],pin=row['pin'])))
        summary=pack.report['summary'];pack.close()
    finally:
        if not pack.closed:pack.close()
    write_committed(a.output/'report.json',dict(complete=True,source=dict(path=str(a.input),pin=a.pin),
        summary=summary,chain=chain,private_source_opens=0,read_paths=sorted(reads),
        model_loads=0,encoder_calls=0,training_updates=0,scientific_admission=False,acceptance='author-checked'))


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1)
    try:
        if a.mode=='cold':cold(a);return
        schedule,store,binding=source();phases=dict(source=time.perf_counter()-tick);tick=time.perf_counter()
        write_committed(a.output/'before.json',dict(binding=binding,stats=store.stats))
        pin=publish(a.output/'pack',schedule,store,binding);phases['publish']=time.perf_counter()-tick;tick=time.perf_counter()
        pack=packing.TaskPublicPack(a.output/'pack',pin,binding)
        try:
            checked=direct(pack,schedule,store);phases['readback']=time.perf_counter()-tick;tick=time.perf_counter()
            negatives=controls(pack,a.output/'controls');phases['controls']=time.perf_counter()-tick;tick=time.perf_counter()
            summary=pack.report['summary'];pack.close()
        finally:
            if not pack.closed:pack.close()
        store.fresh();schedule.plan.fresh();schedule.targets.source.fresh()
        need(binding['code']=={n:file_sha(BASE/n) for n in CODE},'TASK_PACK_CODE_CHANGED')
        write_committed(a.output/'direct.json',checked);write_committed(a.output/'controls.json',negatives)
        phases['final']=time.perf_counter()-tick;elapsed=time.perf_counter()-began
        need(.9<=sum(phases.values())/elapsed<=1.001,'TASK_PACK_PHASE_COVERAGE')
        result=dict(complete=True,public_pack=dict(path=str(a.output/'pack'),pin=pin),summary=summary,
            checked=checked,controls=negatives,code=binding['code'],phases=phases,seconds=elapsed,
            cpu_seconds=time.process_time()-cpu,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            native_encoder_calls=0,model_loads=0,training_updates=0,scientific_admission=False,
            transfer_evaluated=False,acceptance='author-checked')
        result_pin=write_committed(a.output/'report.json',result)
        print(json.dumps(dict(pin=result_pin,pack=pin,summary=summary,seconds=elapsed)),flush=True)
    except BaseException as e:
        write_committed(a.output/'failure.json',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('publish','cold'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--pin')
    run(p.parse_args())

"""Qualify source-fed complete-answer epochs and physical checkpoint continuation."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.legal_complete_answer_consumer import prepare
from claims.process_conditions import write_committed
from model.legal_answer_epoch import AnswerEpoch, persist
from model.legal_complete_answer_source import PLAN
from model.native_population_training import cpu, fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
PARENT = ('open-window-answer-cpu-v2-20261024','a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339')
RECEIPTS = ('shared-saved-v1-20261024/short-receipts.json','bbbdbd95a1ab0fefb892f4eefe5dbb25d357cc513356bf906c4c7877ced09cf2')
NEW = ('model/legal_answer_epoch.py','audit/legal_answer_epoch.py')
DOCUMENTS = (15639,27195,32791,105748,106288)
ARCHITECTURES = ('shared','own_slot','whole_proposals')


def read(path, pin):
    need(file_sha(path) == pin,'ANSWER_EPOCH_PROOF_PIN')
    return json.loads(path.read_bytes())


def numerical(value):
    return dict(records=value['record']['records'], **{n:value[n] for n in ('gradients','early','model','optimizer')})


def negatives(epoch, output, reference):
    before = fingerprint(epoch.snapshot()); result = {}
    def reject(name, fn, code):
        try: fn()
        except (ValueError,RuntimeError) as exc:
            need(code in str(exc),'ANSWER_EPOCH_WRONG_REFUSAL:'+str(exc)); result[name] = str(exc)
        else: raise ValueError('ANSWER_EPOCH_NEGATIVE_ACCEPTED:'+name)
        need(fingerprint(epoch.snapshot()) == before,'ANSWER_EPOCH_REFUSAL_MUTATED_STATE')
    path = Path(reference['path'])
    reject('pin',lambda:epoch.restore(path,dict(reference,pin='0'*64),cursor=0,chain=epoch.chain),
           'ANSWER_EPOCH_CHECKPOINT_PIN')
    reject('cursor',lambda:epoch.restore(path,reference,cursor=1,chain=epoch.chain),'ANSWER_EPOCH_CHECKPOINT_CURSOR')
    reject('chain',lambda:epoch.restore(path,reference,cursor=0,chain='0'*64),'ANSWER_EPOCH_CHECKPOINT_CURSOR')
    reject('next_document',lambda:epoch.update(DOCUMENTS[1]),'ANSWER_EPOCH_NEXT_DOCUMENT')
    held = next(i for i,r in enumerate(epoch.corpus.rows) if int(r['side']) != 0)
    reject('held_out',lambda:epoch.corpus.example(held,fitting=True),'LEGAL_ANSWER_NONTRAIN_FIT')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    for name in ('binding','moments','counts','noise'):
        bad = deepcopy(saved)
        if name == 'binding': bad['binding']['architecture'] = 'wrong'
        if name == 'moments': bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
        if name == 'counts': bad['counts']['decisions'] += 1
        if name == 'noise': bad['noise_cursor'] += 1
        wrong = output/('wrong-'+name+'.pt'); receipt = persist(wrong,bad)
        code = {'binding':'ANSWER_EPOCH_CHECKPOINT_BINDING','moments':'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT',
                'counts':'ANSWER_EPOCH_CHECKPOINT_CURSOR','noise':'ANSWER_EPOCH_CHECKPOINT_CURSOR'}[name]
        reject(name,lambda:epoch.restore(wrong,receipt,cursor=0,chain=epoch.chain),code)
    return result


def run(a):
    began = tick = time.perf_counter(); phases = {}; a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),'ANSWER_EPOCH_CPU_PROOF')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    parent = read(a.root/PARENT[0]/'report.json',PARENT[1]); refs = read(a.root/RECEIPTS[0],RECEIPTS[1])
    physical = {a.root/PARENT[0]/'report.json':PARENT[1],a.root/RECEIPTS[0]:RECEIPTS[1]}
    references = {}; code = dict(parent['code'])
    for arch in ARCHITECTURES:
        ref = refs[arch+'-baseline']; cold = refs[arch+'-cold']; r = read(Path(ref['path'])/'report.json',ref['pin'])
        c = read(Path(cold['path'])/'report.json',cold['pin'])
        need(r['complete'] and r['qualified'] and c['qualified'] and c['complete'] and not r['gpu_used']
             and r['architecture'] == c['architecture'] == arch and len(r['rows']) == 5
             and [x['numerical_pin'] for x in r['rows']] == [x['numerical_pin'] for x in c['rows']],
             'ANSWER_EPOCH_REFERENCE_QUALIFICATION')
        references[arch] = (Path(ref['path']),r)
        physical.update({Path(ref['path'])/'report.json':ref['pin'],Path(cold['path'])/'report.json':cold['pin']})
        physical.update({Path(ref['path'])/n:h for n,h in r['files'].items()})
        physical.update({Path(cold['path'])/n:h for n,h in c['files'].items()})
        code.update(r['code'])
        name = arch+'/update-5.pt'; physical[a.root/PARENT[0]/name] = parent['files'][name]
    for path,pin in physical.items(): need(file_sha(path) == pin,'ANSWER_EPOCH_REFERENCE_BYTES')
    corpus,labels,continuation,bundle,source_code = prepare(a.root)
    try:
        code.update(source_code); base = Path(__file__).resolve().parents[1]
        code.update({n:file_sha(base/n) for n in NEW})
        need(all(file_sha(base/n) == h for n,h in code.items()),'ANSWER_EPOCH_CODE_AUTHORITY')
        target,target_pin = extension(bundle); policy = dict(bundle['policy'],device='cpu')
        previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
        if previous: need(previous['complete'] and previous['code'] == code,'ANSWER_EPOCH_COLD_CODE')
        source = dict(binding=corpus.binding, code=code, documents=DOCUMENTS, policy=policy,
            references={a:refs[a+'-baseline'] for a in ARCHITECTURES}, parent=PARENT, target=target_pin)
        write_committed(a.output/'source.json',source)
        initial = torch.load(a.root/PARENT[0]/'shared/update-5.pt',map_location='cpu',weights_only=True)
        full = AnswerEpoch(corpus,bundle,target,target_pin,'shared',initial,5,policy,code)
        expected = read(a.root/PLAN[0]/'report.json',PLAN[1])['summary']['groups']['train']
        census = full.counts(len(full.documents))
        need(full.documents == tuple(i for i,s in enumerate(corpus.data.doc_sides) if int(s) == 0)
             and all(census[n] == expected[n] for n in ('documents','atoms','answers','decisions','ends'))
             and list(census['windows'].values()) == expected['windows'], 'ANSWER_EPOCH_FULL_SCHEDULE')
        schedule = dict(binding=full.binding,counts=census,first=full.documents[0],last=full.documents[-1],
                        complete_source_order=True,optimizer_updates=0)
        write_committed(a.output/'full-schedule.json',schedule)
        del full,initial
        phases['authentication'] = time.perf_counter()-tick; results = {}; controls = {}
        for arch in ARCHITECTURES:
            tick = time.perf_counter(); out = a.output/arch; out.mkdir()
            initial = torch.load(a.root/PARENT[0]/arch/'update-5.pt',map_location='cpu',weights_only=True)
            epoch = AnswerEpoch(corpus,bundle,target,target_pin,arch,initial,5,policy,code,documents=DOCUMENTS)
            initial_receipt = persist(out/'checkpoint-0.pt',epoch.snapshot())
            controls[arch] = negatives(epoch,out,initial_receipt)
            epoch.restore(out/'checkpoint-0.pt',initial_receipt,cursor=0,chain=epoch.chain)
            checkpoints = [initial_receipt]; updates = []; timings = dict(read=0.,update=0.,persist=0.)
            for i,number in enumerate(DOCUMENTS):
                value,entry = epoch.update(number)
                ref_root,ref = references[arch]; expected = torch.load(ref_root/f'update-{i+6}.pt',map_location='cpu',weights_only=True)
                need(fingerprint(numerical(value)) == fingerprint(numerical(expected)), 'ANSWER_EPOCH_REFERENCE_UPDATE')
                need(value['record']['branch_gradients'] == expected['record']['branch_gradients']
                     and value['record']['storage']['remaining_saved_storage_bytes'] == 0,
                     'ANSWER_EPOCH_BRANCH_OR_STORAGE')
                start = time.perf_counter(); payload = persist(out/f'update-{i+6}.pt',value)
                checkpoint = persist(out/f'checkpoint-{i+1}.pt',epoch.snapshot())
                epoch.restore(checkpoint['path'],checkpoint,cursor=i+1,chain=entry['chain'])
                timings['persist'] += time.perf_counter()-start
                timings['read'] += entry['read_seconds']; timings['update'] += entry['update_seconds']
                row = dict(document=number,step=i+6,numerical=fingerprint(numerical(value)),
                           chain=entry['chain'],checkpoint_state=checkpoint['state'],counts=entry['counts'])
                if previous:
                    old = previous['results'][arch]['updates'][i]
                    need(row == old,'ANSWER_EPOCH_COLD_UPDATE')
                    file = a.previous/arch/f'update-{i+6}.pt'
                    need(file_sha(file) == previous['files'][str(file.relative_to(a.previous))],'ANSWER_EPOCH_COLD_BYTES')
                    old_value = torch.load(file,map_location='cpu',weights_only=True)
                    need(fingerprint(numerical(old_value)) == row['numerical'],'ANSWER_EPOCH_COLD_PHYSICAL')
                updates.append(row); checkpoints.append(checkpoint)
                write_committed(out/f'step-{i+1}.json',entry)
                del value,expected,payload
            need(epoch.counts(epoch.cursor)['decisions'] == 66 and epoch.cursor == 5,'ANSWER_EPOCH_FINAL_COVERAGE')
            results[arch] = dict(binding=epoch.binding,updates=updates,checkpoints=checkpoints,
                counts=epoch.counts(epoch.cursor),chain=epoch.chain,parameters=sum(p.numel() for p in epoch.model.parameters()),
                timings=timings,all_updates_exact=True,all_checkpoint_restores_exact=True)
            phases[arch] = time.perf_counter()-tick
            print(json.dumps(dict(architecture=arch,updates=5,decisions=66,timings=timings)),flush=True)
            del epoch,initial; gc.collect()
        tick = time.perf_counter(); corpus.close(); corpus.cache.close(); corpus.reader.close()
        labels.fresh(); continuation.fresh()
        need(all(file_sha(base/n) == h for n,h in code.items()),'ANSWER_EPOCH_FINAL_CODE')
        for path,pin in physical.items(): need(file_sha(path) == pin,'ANSWER_EPOCH_FINAL_PROOF')
        if previous: need(controls == previous['controls'],'ANSWER_EPOCH_COLD_CONTROLS')
        phases['final_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001,'ANSWER_EPOCH_PHASES')
        files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        report = dict(complete=True,code=code,source=source,schedule=schedule,results=results,controls=controls,files=files,
            optimizer_updates=15,decisions=198,previous_pin=a.previous_pin,cold_reproduced=previous is not None,
            phases=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_used=False,frozen_encoder_calls=0,promoted=False,full_population_trained=False,
            capability_demonstrated=False,acceptance='author-checked')
        need(not torch.cuda.is_initialized(),'ANSWER_EPOCH_UNEXPECTED_GPU')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if not corpus.closed: corpus.close(); corpus.cache.close(); corpus.reader.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

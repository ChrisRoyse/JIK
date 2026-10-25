"""CPU proof of exact optimized continuation, retained-tail replay and evaluations."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as original
from claims.process_conditions import write_committed
from model.batched_placement_continuation import BatchedPlacementContinuation
from model import placement_resume_publication as publication
from model import placement_resume_tail as tail
from model.legal_placement_resume import recount
from model.native_population_training import fingerprint
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need

QUALIFICATION = ('placement-validation-cold-v4-20261025',
                 '8d7f827e6a186bfaa6c488b32f39a8af0d04a17effd3e99e6ef7bd6fb5fc143a')
BOUNDARY = ('placement-tail-proof-v1-20261024/boundary',
            '2b0b8aa5dd0f6e3e6116fb4c50f978e40e91e3bf62f7d69098e49afbfb55e78c')
REFERENCE = ('placement-driver-original-v1-20261024',
             'af334059da30b3a6cf1961b97c4c3b838fb00aeb8d2762a80f68c54a6202ce40')


def reject(call, code):
    try: call()
    except (ValueError, RuntimeError) as exc:
        need(code in str(exc), 'BATCHED_CONTINUATION_WRONG_REFUSAL:' + str(exc)); return str(exc)
    raise ValueError('BATCHED_CONTINUATION_ACCEPTED:' + code)


def controls(root, make, source, ref, prefix, initial, saved, census, item, value):
    root.mkdir(); engine = make(); cursor = saved['step']; chain = saved['chain']
    engine.restore(source, ref, prefix, initial, cursor=cursor, chain=chain)
    before = fingerprint(engine.snapshot()); out = {}
    def restore(p=source, r=ref, c=cursor):
        return engine.restore(p, r, prefix, initial, cursor=c, chain=chain)
    out['physical_pin'] = reject(lambda:restore(r=dict(ref, pin='0'*64)), 'CHECKPOINT_PIN')
    out['cursor'] = reject(lambda:restore(c=cursor+1), 'CHECKPOINT_BINDING')
    for label in ('noise', 'optimizer'):
        bad = deepcopy(saved)
        if label == 'noise': bad['noise_cursor'] += 1
        else:
            state = next(iter(bad['optimizer']['state'].values()))
            state['exp_avg'].reshape(-1)[0] += 1
        p = root/(label+'.pt'); r = original.measure.persist(p, bad)
        out[label] = reject(lambda:restore(p, r), 'CHECKPOINT_CURSOR' if label=='noise' else 'RESUME_STATE')
    bad = dict(item, document=census.documents[0])
    out['document'] = reject(lambda:engine.update(bad), 'NEXT_DOCUMENT')
    bad = dict(item, codes=item['codes'][:-1])
    out['census'] = reject(lambda:engine.update(bad), 'ITEM_CENSUS')
    need(fingerprint(engine.snapshot()) == before, 'BATCHED_CONTINUATION_REFUSAL_MUTATED_STATE')
    witness = prefix.parent/'retained-tail.jsonl'
    damaged = root/'truncated.jsonl'; damaged.write_bytes(witness.read_bytes()[:-1])
    out['truncated_tail'] = reject(lambda:tail.Replay(damaged,value,cursor), 'REPLAY_SOURCE')
    # Rehash a physically persisted but numerically wrong witness. The outer
    # chain remains valid; only exact execution can catch this false loss.
    rows = [json.loads(x) for x in witness.read_bytes().splitlines()]
    rows[0]['update']['loss'] += 1.0
    chain = saved['chain']; raw = []
    for row in rows:
        row['parent'] = chain; row['hash'] = digest({k:v for k,v in row.items() if k!='hash'})
        chain = row['hash']; raw.append(canonical(row)+b'\n')
    wrong = root/'wrong.jsonl'; wrong.write_bytes(b''.join(raw))
    with wrong.open('rb') as stream:
        counts = recount(prefix, saved, initial, census)
        wrong_value = tail.scan(stream, wrong.stat().st_size, counts, census, saved['binding'])
    journal = root/'steps.jsonl'; journal.write_bytes(prefix.read_bytes()); old = file_sha(journal)
    with journal.open('ab') as stream, tail.Replay(wrong,wrong_value,cursor) as replay:
        out['numerical_replay'] = reject(lambda:engine.commit(item,stream,replay), 'REPLAY_MISMATCH')
    need(file_sha(journal)==old and not (root/'report.json').exists(), 'BATCHED_CONTINUATION_REFUSAL_COMMIT')
    out['poisoned_state'] = reject(engine.snapshot, 'RESUME_POISONED')
    out['prefix_unchanged'] = True
    return out


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    q = original.pinned(a.root/QUALIFICATION[0],QUALIFICATION[1])
    need(q['complete'] and q['cold_readback_exact'] and q['all_updates_exact'] and q['midpoint_resume_exact']
         and q['binding']['device']=='cpu' and not q['gpu_qualified'], 'BATCHED_CONTINUATION_QUALIFICATION')
    base = Path(__file__).resolve().parents[1]
    manifest,census = publication.boundary(a.root/BOUNDARY[0],BOUNDARY[1],execution=publication.code())
    reference = original.pinned(a.root/REFERENCE[0],REFERENCE[1]); original.physical(a.root/REFERENCE[0],reference)
    source,bundle = original.source_bundle(a.root); code = dict(original.code(source),**publication.code())
    code.update(q['binding']['code'])
    for n in ('audit/batched_continuation_proof.py','model/batched_placement_continuation.py',
              'audit/captured_answer_readback.py'):
        code[n] = file_sha(base/n)
    need(all(file_sha(base/n)==h for n,h in code.items()), 'BATCHED_CONTINUATION_SOURCE')
    for n,h in code.items(): publication.copy_file(base/n,a.output/'source'/n,h)
    data = original.load_data(argparse.Namespace(root=a.root,scope='small'),bundle)
    target = data['labels'].model; metadata = original.measure.metadata(data['items'])
    donors = original.measure.donor_plan(metadata); original.measure.validate_donors(metadata,donors)
    donors_pin = write_committed(a.output/'donors.json',dict(metadata=metadata,donors=donors))
    binding = dict(code=original.code(source),policy=dict(bundle['policy'],device='cpu'),source=data['binding'],
                   target=bundle['target_ref'],scope='small',device='cpu',torch_version=str(torch.__version__),
                   donors=donors_pin,reader_match=manifest['binding']['reader_match'])
    need(binding==manifest['binding']==reference['binding'] and census.documents==tuple(data['train'])
         and len(census.documents)==5 and census.counts(5)['rows']==66, 'BATCHED_CONTINUATION_BINDING')
    previous = original.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['code']==code and previous['binding']==binding
             and previous['reference']==REFERENCE[1] and previous['boundary']==BOUNDARY[1]
             and previous['pid']!=os.getpid(), 'BATCHED_CONTINUATION_COLD_BINDING')
        original.physical(a.previous,previous)
    phases = dict(authentication=time.perf_counter()-tick); results={}; timings={}; cases={}; physical={}
    replayed = 0; evaluation_rows = 0; comparisons = dict(tensors=0,bytes=0)
    for name in original.runtime.original.ARCHITECTURES:
        tick=time.perf_counter(); out=a.output/name; out.mkdir(); branch=dict(binding,architecture=name)
        def make(): return BatchedPlacementContinuation(bundle,target,branch,census)
        inherited=manifest['branches'][name]; initial_root=a.root/BOUNDARY[0]/name
        initial=torch.load(initial_root/'checkpoint-0.pt',map_location='cpu',weights_only=True)
        if previous:
            src=a.previous/name/'checkpoint-3.pt'; ref=previous['results'][name]['checkpoints']['3']
            prefix_source=a.previous/name/'steps.jsonl'
        else:
            src=initial_root/'checkpoint-2.pt'; ref=inherited['checkpoints']['2']; prefix_source=initial_root/'steps.jsonl'
        need(file_sha(src)==ref['pin'],'BATCHED_CONTINUATION_RESTORE_PIN')
        saved=torch.load(src,map_location='cpu',weights_only=True)
        need(fingerprint(saved)==ref['state'],'BATCHED_CONTINUATION_RESTORE_STATE')
        prefix=out/'steps.jsonl'; counts=recount(prefix_source,saved,initial,census,copy_to=prefix,allow_suffix=bool(previous))
        full_journal=a.root/REFERENCE[0]/name/'steps.jsonl'
        witness=tail.seal(full_journal,out,counts,census,branch,full_journal.stat().st_size)
        tail.validate(out,witness,counts,census,branch)
        cases[name]=controls(out/'controls',make,src,ref,prefix,initial,saved,census,data['read'](census.documents[saved['step']]),witness)
        engine=make(); engine.restore(src,ref,prefix,initial,cursor=saved['step'],chain=saved['chain'])
        need(inherited['initial']==reference['results'][name]['initial']
             and inherited['initial_rows']==reference['results'][name]['initial_rows'], 'BATCHED_CONTINUATION_INITIAL')
        # Initial evaluations are reused by reference after physical authentication,
        # and checked tensor-by-tensor against the original experiment.
        paths=sorted(n for n in manifest['files'] if n.startswith(name+'/initial/'))
        for rel in paths:
            c=rd.exact(rd.load(a.root/REFERENCE[0],rel),rd.load(a.root/BOUNDARY[0],rel))
            for k in comparisons:comparisons[k]+=c[k]
        phases[name+'_restore']=time.perf_counter()-tick; tick=time.perf_counter(); checkpoints={}
        with prefix.open('ab') as stream, tail.Replay(out/'retained-tail.jsonl',witness,saved['step']) as replay:
            for cursor in range(saved['step'],len(census.documents)):
                record,gradients,noise=engine.commit(data['read'](census.documents[cursor]),stream,replay)
                need(record['update']==bundle['cpu_reference'][name]['updates'][cursor]['record'], 'BATCHED_CONTINUATION_PRIOR_UPDATE')
                path=out/f'checkpoint-{engine.step}.pt'; checkpoints[str(engine.step)]=original.measure.persist(path,engine.snapshot())
                expected=rd.load(a.root/REFERENCE[0],name+'/'+path.name)
                c=rd.exact(expected,rd.load(a.output,name+'/'+path.name))
                for k in comparisons:comparisons[k]+=c[k]
                del gradients,noise
            receipt=replay.finish(); replayed+=receipt['records']
        need(prefix.read_bytes()==full_journal.read_bytes(), 'BATCHED_CONTINUATION_LEDGER')
        timings[name]=time.perf_counter()-tick; phases[name+'_updates']=timings[name]; tick=time.perf_counter()
        rows,_=original.measure.evaluate(engine.model,target,data['labels'],data['items'],donors,bundle['statistics'],out/'final')
        final=original.measure.summary(rows,target.roster); row_pin=fingerprint(rows)
        need(final==reference['results'][name]['final'] and row_pin==reference['results'][name]['final_rows'],
             'BATCHED_CONTINUATION_FINAL_SCORES')
        final_paths=sorted(n for n in reference['files'] if n.startswith(name+'/final/'))
        need(final_paths==sorted(str(p.relative_to(a.output)) for p in (out/'final').iterdir()), 'BATCHED_CONTINUATION_EVALUATION_SCOPE')
        for rel in final_paths:
            old=rd.load(a.root/REFERENCE[0],rel); new=rd.load(a.output,rel); c=rd.exact(old,new)
            for k in comparisons:comparisons[k]+=c[k]
            evaluation_rows+=len(new)
        result=dict(chain=engine.chain,model=fingerprint(engine.model.state_dict()),optimizer=fingerprint(engine.optimizer.state_dict()),
                    rows=census.counts(engine.step)['rows'],step=engine.step,final=final,final_rows=row_pin,
                    initial_rows=inherited['initial_rows'])
        if previous:need(result==previous['results'][name]['final_state'],'BATCHED_CONTINUATION_COLD_RESULT')
        results[name]=dict(final_state=result,checkpoints=checkpoints,restore=dict(path=str(src),**ref),
                           replay=receipt,initial=dict(path=str(initial_root/'initial'),boundary=BOUNDARY[1],rows=inherited['initial_rows']))
        phases[name+'_evaluation']=time.perf_counter()-tick
        print(json.dumps(dict(architecture=name,replayed=receipt['records'],complete=True)),flush=True)
        del engine,rows,saved,initial; gc.collect()
    tick=time.perf_counter(); data['close']()
    need(fingerprint(target.state_dict())==bundle['target_fingerprint'] and not torch.cuda.is_initialized(), 'BATCHED_CONTINUATION_FROZEN_RUNTIME')
    need(all(file_sha(base/n)==h for n,h in code.items()), 'BATCHED_CONTINUATION_SOURCE_CHANGED')
    need(file_sha(a.root/BOUNDARY[0]/'boundary.json')==BOUNDARY[1] and file_sha(a.root/REFERENCE[0]/'report.json')==REFERENCE[1],
         'BATCHED_CONTINUATION_INPUT_CHANGED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['publication']=time.perf_counter()-tick; wall=time.perf_counter()-began
    need(.9<=sum(phases.values())/wall<=1.000001, 'BATCHED_CONTINUATION_PHASE_COVERAGE')
    report=dict(complete=True,kind='batched-placement-continuation-cpu-proof-v1',pid=os.getpid(),code=code,binding=binding,
                qualification=QUALIFICATION[1],reference=REFERENCE[1],boundary=BOUNDARY[1],previous_pin=a.previous_pin,
                results=results,files=files,controls=cases,comparison=comparisons,evaluation_rows=evaluation_rows,
                replayed=replayed,phases=phases,timings=timings,wall=wall,cpu=time.process_time()-cpu,
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,gpu_qualified=False,
                cold_readback_exact=bool(previous),all_evaluations_exact=True,all_retained_updates_exact=True,
                production_promoted=False,process_signals_sent=0,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,replayed=replayed,evaluation_rows=evaluation_rows,comparison=comparisons,wall=wall,cold=bool(previous))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
        raise

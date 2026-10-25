"""Full legal TRAIN schedules and a real, small CPU proof of order-only updates."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Explicit student-runtime proof; no encoder execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.placement_source_exposure import encoded, read, save, sha
from model.placement_source_order import Schedule, need

EXPOSURE = ('placement-source-exposure-v2-20261025', '60368e9934ff7c647a2addbcb3896b023a519e5c879b92486fb4642a15017461')
COLD = ('placement-source-exposure-cold-v2-20261025.json', '685837fbca1d2f78fb9e8e66d5c1c76da9b0b5bd593d0f44eaf0663422ba328f')
BOUNDARY = ('answer-refresh-retirement-v1-20261025/seal/boundary.json', '35dd82d4e93b6476810a23301b4e75ffd57c4b7b50bb02e88b9f85be7ff7301d')
CORPUS = ('legal-source-corpus-v1-20261022', 'ec47076aaaa3d313e823f83e935be59be066933ae3c9631d4cd963afb50bdf57')


def jsonl(path, rows):
    with path.open('xb') as stream:
        for row in rows: stream.write(encoded(row)+b'\n')
        stream.flush(); os.fsync(stream.fileno())
    return sha(path)


def reject(operation):
    try: operation()
    except (ValueError, RuntimeError, AssertionError) as error: return str(error)
    raise AssertionError('MNLM_PLACEMENT_ORDER_NEGATIVE_ACCEPTED')


def full_rows(root):
    inputs = {str(root/BOUNDARY[0]): BOUNDARY[1], str(root/EXPOSURE[0]/'report.json'): EXPOSURE[1],
              str(root/COLD[0]): COLD[1], str(root/CORPUS[0]/'report.json'): CORPUS[1]}
    seal, exposure, cold, corpus = [read(p, pin) for p, pin in inputs.items()]
    need(exposure['complete'] and cold['complete'], 'PRIOR_COMPLETE')
    metadata = root/CORPUS[0]/'documents.jsonl'
    inputs[str(metadata)] = corpus['files']['documents.jsonl']['sha256']
    compact = root/EXPOSURE[0]/'rows.jsonl'
    inputs[str(compact)] = exposure['output_files']['rows.jsonl']
    need(seal['binding']['source']['metadata']['documents'] == inputs[str(metadata)], 'METADATA_BINDING')
    documents = seal['binding']['source']['plan']['train_documents']; selected = set(documents)
    mapping = {}; metadata_hash = hashlib.sha256()
    with metadata.open('rb') as stream:
        for ordinal, raw in enumerate(stream):
            metadata_hash.update(raw)
            if ordinal in selected:
                r = json.loads(raw); mapping[ordinal] = r
    need(metadata_hash.hexdigest() == inputs[str(metadata)], 'METADATA_PIN')
    rows = []; compact_hash = hashlib.sha256()
    with compact.open('rb') as stream:
        for step, raw in enumerate(stream, 1):
            compact_hash.update(raw); e = json.loads(raw)
            need(step <= len(documents) and e['step'] == step and e['document'] == documents[step-1], 'EXPOSURE_JOIN')
            m = mapping[e['document']]
            need(e['source'] == m['source_path'] and e['targets'] == m['atom_end']-m['atom_start']-1, 'EXPOSURE_FIELDS')
            rows.append(dict(document=e['document'], source=e['source'], targets=e['targets'],
                             start=m['atom_start'], stop=m['atom_end'], original_step=step, side='train'))
    need(compact_hash.hexdigest() == inputs[str(compact)] and len(rows) == 91540
         and sum(r['targets'] for r in rows) == 6455636, 'EXPOSURE_PIN_OR_SCOPE')
    return rows, seal['binding']['policy']['noise_seed'], inputs


def describe(entries):
    previous = None; run = 0; longest = Counter(); counts = Counter(); targets = Counter(); transitions = 0
    first = {}; last = {}
    for r in entries:
        s = r['source']; counts[s] += 1; targets[s] += r['targets']; first.setdefault(s, r['step']); last[s] = r['step']
        if s == previous: run += 1
        else:
            transitions += previous is not None; previous = s; run = 1
        longest[s] = max(longest[s], run)
    return dict(documents=len(entries), targets=sum(targets.values()), transitions=transitions,
                sources={s:dict(documents=n, targets=targets[s], longest_run=longest[s], first=first[s], last=last[s])
                         for s,n in sorted(counts.items())})


def run(a):
    began = time.perf_counter(); cpu0 = time.process_time(); a.output.mkdir(exist_ok=False)
    rows, seed, inputs = full_rows(a.root); files = {}; summaries = {}
    for mode in ('original', 'interleaved'):
        entries = Schedule(rows, mode, seed).entries()
        files[mode+'.jsonl'] = jsonl(a.output/(mode+'.jsonl'), entries); summaries[mode] = describe(entries)
    phases = dict(schedule=time.perf_counter()-began)
    import torch
    from audit.legal_grouped_placement_training import source_bundle, small_metadata, SOURCE
    from audit.legal_placement_runtime_readback import make, persist
    from model.native_population_training import cpu, fingerprint
    from model.native_rq_packed import PackedNativeRQ
    from model.placement_source_order import update
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    tick = time.perf_counter(); source, bundle = source_bundle(a.root); small_metadata(a.root, bundle['items'])
    inputs[str(a.root/SOURCE[0]/'report.json')] = SOURCE[1]
    inputs[str(a.root/SOURCE[0]/'bundle.pt')] = source['bundle_sha256']
    target = PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'], strict=True); target.eval().requires_grad_(False)
    frozen = fingerprint(target.state_dict()); policy = dict(bundle['policy'], device='cpu'); policy_before = deepcopy(policy)
    items = [i for i in bundle['items'] if i['side'] == 'train']; lookup = {i['document']:i for i in items}
    small = [dict(document=i['document'],source=i['source_collection'],start=i['start'],stop=i['stop'],
                  targets=len(i['codes']),original_step=s,side='train') for s,i in enumerate(items,1)]
    need(len(small) == 5 and sum(r['targets'] for r in small) == 66, 'SMALL_SCOPE')
    phases['bundle'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = {}; bad = deepcopy(small); bad[1]['document'] = bad[0]['document']
    controls['duplicate_document'] = reject(lambda: Schedule(bad,'interleaved',policy['noise_seed']))
    bad2 = deepcopy(small); bad2[0]['targets'] += 1
    controls['changed_targets'] = reject(lambda: Schedule(bad2,'interleaved',policy['noise_seed']))
    bad3 = deepcopy(small); bad3[0]['side'] = 'query'
    controls['nontrain_schedule'] = reject(lambda: Schedule(bad3,'interleaved',policy['noise_seed']))
    controls['reversed_original_order'] = reject(lambda: Schedule(small[::-1],'interleaved',policy['noise_seed']))
    results = {}; initial_states = {}; pairs = {}
    for architecture in ('shared','own_slot','whole_proposals'):
        results[architecture] = {}; pairs[architecture] = {}
        for mode in ('original','interleaved'):
            plan = Schedule(small, mode, policy['noise_seed']); model, optimizer = make(bundle,target,architecture,'cpu')
            prefix = architecture+'/'+mode; directory = a.output/prefix; directory.mkdir(parents=True)
            def snapshot(step):
                return cpu(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                                schedule=plan.entries(),policy=policy,target=frozen))
            initial = snapshot(0); initial_states[architecture,mode] = fingerprint((initial['model'],initial['optimizer']))
            receipt = persist(directory/'checkpoint-0.pt',initial); files[prefix+'/checkpoint-0.pt'] = receipt['sha256']
            records = []; checkpoints = [dict(step=0, **receipt)]; doc_samples = {}
            if architecture == 'shared' and mode == 'original':
                first = lookup[plan.documents[0].document]; before = fingerprint((model.state_dict(),optimizer.state_dict()))
                controls['wrong_next_document'] = reject(lambda: update(model,optimizer,items[1],0,policy,target=target,schedule=plan))
                controls['wrong_seed'] = reject(lambda: update(model,optimizer,first,0,dict(policy,noise_seed=policy['noise_seed']+1),target=target,schedule=plan))
                controls['nontrain_item'] = reject(lambda: update(model,optimizer,dict(first,side='query'),0,policy,target=target,schedule=plan))
                controls['wrong_optimizer_cursor'] = reject(lambda: update(model,optimizer,lookup[plan.documents[1].document],1,policy,target=target,schedule=plan))
                need(fingerprint((model.state_dict(),optimizer.state_dict())) == before, 'CONTROLS_MUTATED_STATE')
            for step, row in enumerate(plan.documents):
                record, gradients, noise = update(model,optimizer,lookup[row.document],step,policy,target=target,schedule=plan)
                if mode == 'original':
                    need(record == bundle['cpu_reference'][architecture]['updates'][step]['record'], 'ORIGINAL_REFERENCE')
                doc_samples[str(row.document)] = {k:record[k] for k in ('noise_seed','noise','native','labels','targets')}
                records.append(record)
                state = snapshot(step+1); state.update(record=record,gradients=gradients,noise=noise)
                receipt = persist(directory/f'checkpoint-{step+1}.pt',state)
                files[prefix+f'/checkpoint-{step+1}.pt'] = receipt['sha256']; checkpoints.append(dict(step=step+1,**receipt))
            files[prefix+'/updates.jsonl'] = jsonl(directory/'updates.jsonl',records)
            pairs[architecture][mode] = doc_samples
            results[architecture][mode] = dict(order=[r.document for r in plan.documents],checkpoints=checkpoints,
                                              final_state=fingerprint((model.state_dict(),optimizer.state_dict())),samples=doc_samples)
        need(initial_states[architecture,'original'] == initial_states[architecture,'interleaved'], 'INITIAL_DIFFERENCE')
        need(pairs[architecture]['original'] == pairs[architecture]['interleaved'], 'PAIRED_INPUT_OR_NOISE')
        need(results[architecture]['original']['order'] != results[architecture]['interleaved']['order'], 'ORDER_NOT_CHANGED')
    phases['updates_and_persistence'] = time.perf_counter()-tick; tick = time.perf_counter()
    need(policy == policy_before and fingerprint(target.state_dict()) == frozen, 'FROZEN_CHANGED')
    need(all(sha(p) == pin for p,pin in inputs.items()), 'INPUT_CHANGED')
    root = Path(__file__).resolve().parents[1]
    names = set(source['code']) | {'model/placement_source_order.py','model/native_placement_batched_checks.py',
        'model/native_batched_validation.py','audit/placement_source_order_proof.py','audit/placement_source_order_readback.py',
        'audit/placement_source_exposure.py','audit/legal_grouped_placement_training.py','audit/legal_placement_runtime_readback.py'}
    code = {n:sha(root/n) for n in sorted(names)}
    for name,pin in code.items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((root/name).read_bytes())
        need(sha(dest) == pin, 'SOURCE_COPY')
    files['results.json'] = save(a.output/'results.json', dict(schedules=summaries, small_rows=small, results=results))
    phases['final_checks'] = time.perf_counter()-tick
    report = dict(schema='placement-source-order-proof-v1',complete=True,inputs=inputs,files=files,code=code,
                  policy=policy,seed=seed,negative_controls=controls,device='cpu',gpu_used=False,encoder_calls=0,
                  updates=30,successor_targets=396,full_schedule_documents=91540,full_schedule_targets=6455636,
                  phases=phases,seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu0,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  original_reference_exact=True,paired_document_inputs_and_noise_exact=True,
                  accuracy_gain_measured=False,gpu_qualified=False,full_training_run=False,acceptance='author-checked')
    save(a.output/'report.json',report)
    print(json.dumps(dict(report_pin=sha(a.output/'report.json'),schedules=summaries,seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    try: run(a)
    except BaseException as error:
        if a.output.exists(): save(a.output/'failure.json',dict(error=repr(error)))
        raise

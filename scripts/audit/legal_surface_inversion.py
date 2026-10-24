"""Source-complete legal inversion plan, qualification, and leased GPU training."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv or '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'plan':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_training_statistics as statistics
from audit import surface_device_authority as authority
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import surface_inversion_training as training
from model.legal_native_resident_stream import ResidentNativeFrames
from model.legal_surface_inputs import LegalSurfaceInputs
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_surface_bytes import SurfaceBytes

SURFACE = ('legal-surface-bytes-full-v1-20261023', '9557b97cdb190e9ca2b004b21d1e23ecd932c984e96aa14ea6af9fa5345375f7')
READBACK = ('legal-surface-bytes-full-readback-v1-20261023', '9cdb7d53885528444d011b5ae61f571774fc415ae77bffe8bfdb2795450573f3')
NEW = ['model/surface_inversion_training.py', 'audit/legal_surface_inversion.py',
       'audit/legal_surface_inversion_readback.py', 'audit/legal_surface_inversion_queue.py']


def sources(root):
    config, specs, examples, meta, initial, _, saved, pins, fresh = authority.load(root)
    records, bank, data, c, _, _, _, _ = statistics.authenticate(root)
    prior = authority.pinned(root/READBACK[0], READBACK[1])
    need(prior['all_bytes_exact'] and prior['all_offsets_exact'] and prior['input_pin'] == SURFACE[1],
         'INVERSION_SURFACE_READBACK')
    surfaces = SurfaceBytes(root/SURFACE[0], SURFACE[1], data.source)
    name, pin = statistics.DEFAULTS['audit']; reader = ResidentNativeFrames(data, root/name, pin)
    adapter = LegalSurfaceInputs(data, surfaces, reader, config, bank_pin=bank.pin)
    need(adapter.specs == specs, 'INVERSION_MACHINERY')
    base = Path(__file__).resolve().parents[1]
    for n in [*statistics.code_pins(), *NEW, 'model/legal_surface_inputs.py',
              'model/legal_native_training_stream.py', 'model/legal_native_resident_stream.py', 'text/legal_surface_bytes.py']:
        pins[n] = file_sha(base/n)
    return dict(config=config, specs=specs, examples=examples, metadata=meta, initial=initial,
        saved=saved, code=pins, fresh=fresh, data=data, continuation=c, surfaces=surfaces, reader=reader, adapter=adapter)


def select(data):
    result = {}
    for side, bit in [('development', 1), ('query', 2)]:
        candidates = sorted((digest(dict(seed='inversion-evaluation-v1', source=data.source.pin,
                                          component=int(data.components[d]), document=d)), d)
                            for d in np.flatnonzero(data.doc_sides == bit).tolist())
        docs = []; seen = set()
        for _, d in candidates:
            component = int(data.components[d])
            if component in seen: continue
            seen.add(component); docs.append(d)
            if len(docs) == 100: break
        need(len(docs) == 100, 'INVERSION_EVALUATION_COMPONENTS'); result[side] = docs
    return result


def plan(s, output):
    data = s['data']; offsets = s['surfaces'].offsets
    ids = np.flatnonzero(data.sides == 0).astype('<i8'); counts = dict(atoms=len(ids), bytes=0, prefix_rows=0)
    for lo in range(0, len(ids), 16384):
        u = data.mapping[ids[lo:lo+16384]]; lengths = offsets[u+1]-offsets[u]
        counts['bytes'] += int(lengths.sum())
        # Width 32 provides history; every row scores at most 16 new bytes.
        positions = lengths+1
        counts['prefix_rows'] += int(((positions+15)//16).sum())
    need(len(ids) == data.report['support']['train']['atoms'], 'INVERSION_TRAIN_CENSUS')
    selected = select(data); held = {}
    for side, docs in selected.items():
        n = b = overlap = 0
        for d in docs:
            a,z = map(int, data.documents[d]); u = data.mapping[a:z]
            n += z-a; b += int((offsets[u+1]-offsets[u]).sum())
            overlap += int(np.count_nonzero(s['continuation'].presence[u] & 1))
        held[side] = dict(documents=docs, atoms=n, bytes=b, train_present=overlap, components=len(docs))
    np.save(output/'train.npy', ids, allow_pickle=False)
    return dict(schema='legal-inversion-plan-v1', complete=True, code=s['code'], policy=training.POLICY,
        binding=s['adapter'].binding, surface_readback=READBACK[1], train=counts, evaluation=held,
        config=asdict(s['config']), instruments=[asdict(x) for x in s['specs']], initial=fingerprint(s['initial']),
        updates=(len(ids)+127)//128, files={'train.npy':file_sha(output/'train.npy')},
        scientific_admission=False, untouched_test=False, number_status='measured', acceptance='author-checked')


def donor_ids(s, ids, train):
    data = s['data']; result = []
    for i in ids:
        component = data.components[np.searchsorted(data.documents[:,1], i, side='right')]
        # Deterministic address arithmetic only; IDs never enter model features.
        start = int(digest(dict(seed='inversion-donor-v1', occurrence=i))[:16], 16) % len(train)
        for offset in range(len(train)):
            d = int(train[(start+offset)%len(train)])
            dc = data.components[np.searchsorted(data.documents[:,1], d, side='right')]
            if dc != component and data.mapping[d] != data.mapping[i]: result.append(d); break
        else: raise ValueError('INVERSION_NO_MATCHED_DONOR')
    return result


def evaluation(s, model, groups, train, output):
    output.mkdir(); results = []
    for side, documents in groups.items():
        for document in documents:
            a,z = map(int, s['data'].documents[document]); scores = []; metadata = []
            byte_matches = exact_atoms = 0
            for start in range(a,z,128):
                ids = list(range(start,min(start+128,z))); donors = donor_ids(s, ids, train)
                examples = s['adapter'].fetch(ids, side=side)
                other = s['adapter'].fetch(donors, side='train')
                values, _ = training.paired(model, examples, other)
                for e,v in zip(examples,values['correct']):
                    match = v['argmax'] == torch.tensor([*e.raw,257])
                    byte_matches += int(match.sum()); exact_atoms += int(bool(match.all()))
                scores.extend([dict(correct=x, shuffled=y) for x,y in zip(values['correct'], values['shuffled'])])
                metadata.extend(dict(occurrence=i, uid=e.uid, donor=d, bytes=len(e.raw),
                    train_present=bool(s['continuation'].presence[e.uid]&1)) for i,e,d in zip(ids,examples,donors))
            file = output/f'{side}-{document}.pt'; torch.save(dict(scores=scores, metadata=metadata), file)
            positions = sum(len(x['correct']['losses']) for x in scores)
            correct = sum(float(x['correct']['losses'].double().sum()) for x in scores)
            shuffled = sum(float(x['shuffled']['losses'].double().sum()) for x in scores)
            results.append(dict(side=side, document=document, component=int(s['data'].components[document]),
                atoms=z-a, positions=positions, correct_sum=correct, shuffled_sum=shuffled,
                byte_matches=byte_matches, exact_atoms=exact_atoms,
                shuffled_minus_correct=(shuffled-correct)/positions, path=str(file), pin=file_sha(file)))
    return results


def close(s):
    s['adapter'].fresh(); s['fresh'](); s['reader'].close(); s['surfaces'].close()
    need(not s['reader'].maps and s['surfaces'].fd is None, 'INVERSION_HANDLES_OPEN')


def run(a):
    began = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False); leased = False; s = None
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        s = sources(a.root)
        if a.mode == 'plan':
            r = plan(s, a.output); close(s); s = None
            r['seconds'] = time.perf_counter()-began
            pin = write_committed(a.output/'report.json', r); print(json.dumps(dict(pin=pin, train=r['train'], evaluation=r['evaluation'])), flush=True); return
        need(a.plan and a.plan_pin, 'INVERSION_PLAN_REQUIRED')
        p = authority.pinned(a.plan, a.plan_pin)
        need(p['schema'] == 'legal-inversion-plan-v1' and p['binding'] == s['adapter'].binding
             and p['code'] == s['code'] and p['policy'] == training.POLICY, 'INVERSION_PLAN_BINDING')
        if a.mode == 'full':
            need(a.qualification and a.qualification_pin, 'INVERSION_CUDA_QUALIFICATION_REQUIRED')
            q = authority.pinned(a.qualification, a.qualification_pin)
            need(q['plan'] == a.plan_pin and q['code'] == s['code'] and q['device'] == 'cuda:0'
                 and q['cold_optimizer_updates'] == 3 and q['all_update_replay_exact']
                 and q['restored_midpoint_continuation_exact'] and q['every_evaluation_byte_exact'],
                 'INVERSION_CUDA_QUALIFICATION_SCOPE')
        full_train = np.load(a.plan/'train.npy', mmap_mode='r', allow_pickle=False)
        need(not a.cpu_proof or a.mode == 'proof', 'INVERSION_FULL_GPU_REQUIRED')
        if a.cpu_proof: device = torch.device('cpu')
        else:
            import subprocess
            (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
            need(torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'INVERSION_CUDA_REQUIRED')
            take_lease('29', 'legal-surface-inversion'); leased = True; device = torch.device('cuda:0')
        if a.mode == 'proof':
            train = np.array([e.occurrence for e in s['examples'] if e.side == 'train'], dtype='<i8')
            groups = {side:sorted({m['document'] for m in s['metadata'] if m['side'] == side})
                      for side in ['development','query']}; tile = 24
        else:
            train = full_train; groups = {side:v['documents'] for side,v in p['evaluation'].items()}; tile = 128
        need(len(train)>0 and bool((s['data'].sides[train] == 0).all()), 'INVERSION_TRAIN_MEMBERSHIP')
        binding = dict(plan=a.plan_pin, scope=a.mode, train=fingerprint(torch.from_numpy(np.array(train))),
            tile=tile, device=str(device), torch=str(torch.__version__), cuda=torch.version.cuda,
            policy=training.POLICY, initial=fingerprint(s['initial']))
        write_committed(a.output/'binding.json', binding)
        model, optimizer = training.create(s['config'], s['specs'], s['initial'], device)
        cursor = step = 0; ledger = digest(binding); checkpoints = []; rows = []
        resume_ref = None
        if a.resume:
            ref = json.loads(a.resume.read_bytes())
            cursor, step, ledger = training.restore(ref, model, optimizer, binding, size=len(train))
            need(cursor < len(train), 'INVERSION_ALREADY_COMPLETE')
            resume_ref = ref
        initial = evaluation(s, model, groups, full_train, a.output/'initial-evaluation')
        phases = dict(authentication_initial=time.perf_counter()-began, read=0., update=0., persistence=0.)
        with (a.output/'steps.jsonl').open('x') as log:
            while cursor < len(train):
                tick = time.perf_counter(); ids = train[cursor:cursor+tile].tolist()
                examples = s['adapter'].fetch(ids, side='train', fit=True)
                phases['read'] += time.perf_counter()-tick; tick = time.perf_counter()
                row, profile = training.update(model, optimizer, examples, step)
                phases['update'] += time.perf_counter()-tick; tick = time.perf_counter()
                step += 1; cursor += len(ids); row.update(cursor=cursor, previous=ledger)
                ledger = digest(row); record = dict(row=row, pin=ledger, profile=profile)
                log.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n'); log.flush()
                if a.mode == 'proof': rows.append(record)
                if a.mode == 'proof' or step%training.POLICY['checkpoint_steps'] == 0 or cursor == len(train):
                    os.fsync(log.fileno())
                    ref = training.checkpoint(a.output/f'checkpoint-{step:06d}.pt', model, optimizer, binding, cursor, step, ledger)
                    write_committed(a.output/f'checkpoint-{step:06d}.json', ref); checkpoints.append(ref)
                phases['persistence'] += time.perf_counter()-tick
                if step%128 == 0: print(json.dumps(dict(step=step, cursor=cursor, total=len(train), phases=phases)), flush=True)
        tick = time.perf_counter(); final = evaluation(s, model, groups, full_train, a.output/'final-evaluation')
        phases['final_evaluation'] = time.perf_counter()-tick
        tick = time.perf_counter(); close(s); code = s['code']; s = None
        base = Path(__file__).resolve().parents[1]
        for n,h in code.items():
            path = a.output/'source'/n; path.parent.mkdir(parents=True, exist_ok=True)
            need(file_sha(base/n) == h, 'INVERSION_CODE_CHANGED'); path.write_bytes((base/n).read_bytes())
        files = {str(x.relative_to(a.output)):file_sha(x) for x in a.output.rglob('*') if x.is_file()}
        phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'INVERSION_PHASE_COVERAGE')
        r = dict(complete=True, plan=a.plan_pin, binding=binding, code=code, files=files,
            cursor=cursor, steps=step, ledger=ledger, checkpoints=checkpoints, initial=initial, final=final,
            resume=resume_ref, phases=phases, seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if leased else 0,
            full_training=a.mode=='full', cpu_proof=a.cpu_proof, native_encoder_calls=0,
            qualification=a.qualification_pin,
            scientific_admission=False, capability_demonstrated=False, free_generation_tested=False,
            number_status='measured', acceptance='author-checked')
        pin = write_committed(a.output/'report.json', r); print(json.dumps(dict(pin=pin, cursor=cursor, steps=step, seconds=seconds)), flush=True)
    finally:
        if s is not None: close(s)
        if leased: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=authority.ROOT); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['plan','proof','full'], required=True); p.add_argument('--cpu-proof', action='store_true')
    p.add_argument('--plan', type=Path); p.add_argument('--plan-pin'); p.add_argument('--resume', type=Path)
    p.add_argument('--qualification', type=Path); p.add_argument('--qualification-pin')
    a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(e)))
        raise

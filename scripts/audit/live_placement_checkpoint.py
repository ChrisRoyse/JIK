"""Read an existing live-job checkpoint and its fixed committed prefix, without stopping it."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as worker
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.native_rq_packed import PackedNativeRQ


def require(ok, code):
    if not ok: raise ValueError('LIVE_CHECKPOINT_'+code)


def record_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def recount(path, saved, initial_model, *, copy_to=None):
    step = saved['step']; binding = saved['binding']; plan = binding['source']['plan']['train_documents']
    require(type(step) is int and 0 < step < len(plan), 'CURSOR')
    require(binding['architecture'] == 'shared' and binding['scope'] == 'full' and binding['device'] == 'cuda', 'SCOPE')
    chain = record_hash(binding); rows = 0; expected_model = initial_model; prefix = hashlib.sha256(); consumed = 0
    output = copy_to.open('xb') if copy_to else None
    try:
        with path.open('rb') as stream:
            for cursor in range(step):
                raw = stream.readline(); require(raw.endswith(b'\n'), 'TORN_PREFIX'); prefix.update(raw); consumed += len(raw)
                if output: output.write(raw)
                row = json.loads(raw); payload = {k:v for k,v in row.items() if k != 'hash'}
                require(set(row) == {'parent','cursor','update','hash'} and row['parent'] == chain
                    and row['cursor'] == cursor and row['hash'] == record_hash(payload), 'CHAIN')
                update = row['update']
                require(update['document'] == plan[cursor] and update['step'] == cursor+1
                    and update['noise_seed'] == binding['policy']['noise_seed']+cursor
                    and update['before'] == expected_model and update['targets'] > 0, 'UPDATE')
                expected_model = update['after']; chain = row['hash']; rows += update['targets']
            if copy_to is None: require(not stream.read(1), 'EXTRA_PREFIX')
        require(saved['chain'] == chain and saved['rows'] == rows and saved['noise_cursor'] == binding['policy']['noise_seed']+step, 'CHECKPOINT_CURSOR')
        require(fingerprint(saved['model']) == expected_model and fingerprint(saved['optimizer']) == update['optimizer'], 'CHECKPOINT_STATE')
        return dict(step=step,rows=rows,chain=chain,prefix_bytes=consumed,prefix_sha256=prefix.hexdigest(),
            last_document=plan[step-1],next_document=plan[step],remaining_documents=len(plan)-step,
            model=expected_model,optimizer=update['optimizer'])
    finally:
        if output: output.flush(); os.fsync(output.fileno()); output.close()


def run(a):
    began = time.perf_counter(); torch.set_num_threads(1); a.output.mkdir(exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    source = a.job/'shared'/f'checkpoint-{a.step}.pt'; initial_path = a.job/'shared/checkpoint-0.pt'
    before = rd.sha(source); snapshot = a.output/'checkpoint.pt'
    with snapshot.open('xb') as f: f.write(source.read_bytes()); f.flush(); os.fsync(f.fileno())
    require(rd.sha(snapshot) == before, 'COPY')
    saved = torch.load(snapshot,map_location='cpu',weights_only=True)
    initial_pin = rd.sha(initial_path); initial = torch.load(initial_path,map_location='cpu',weights_only=True)
    rd.exact(initial['binding'],saved['binding']); require(initial['step'] == 0 and initial['rows'] == 0, 'INITIAL')
    require(saved['step'] == a.step, 'NAMED_CURSOR')
    code = saved['binding']['code']; base = Path(__file__).resolve().parents[1]
    for name,pin in code.items():
        require(rd.sha(base/name) == pin and rd.sha(a.job/'source'/name) == pin, 'CODE_BINDING')
    prefix = a.output/'steps-prefix.jsonl'
    original = recount(a.job/'shared/steps.jsonl',saved,fingerprint(initial['model']),copy_to=prefix)
    cold = torch.load(snapshot,map_location='cpu',weights_only=True)
    require(recount(prefix,cold,fingerprint(initial['model'])) == original, 'PREFIX_READBACK')
    _,bundle = worker.source_bundle(a.root)
    target = PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True); target.eval().requires_grad_(False)
    require(fingerprint(target.state_dict()) == bundle['target_fingerprint'], 'TARGET')
    model,optimizer = worker.runtime.make(bundle,target,'shared','cpu'); reference = dict(pin=before,state=fingerprint(cold))
    policy = saved['binding']['policy']; binding = saved['binding']; chain = saved['chain']; rows = saved['rows']
    def restore(step=a.step, expected_chain=chain):
        worker.restore(snapshot,reference,model,optimizer,binding,step,rows,expected_chain,policy)
    restore()
    matched = rd.exact(dict(model=dict(model.state_dict()),optimizer=optimizer.state_dict()),dict(model=saved['model'],optimizer=saved['optimizer']))
    controls = {}
    def reject(name, fn, code):
        try: fn()
        except (ValueError,RuntimeError) as exc:
            require(code in str(exc), 'WRONG_REFUSAL'); controls[name] = str(exc)
        else: raise ValueError('LIVE_CHECKPOINT_CONTROL_ACCEPTED:'+name)
    reject('cursor',lambda:restore(a.step+1),'PLACEMENT_COMPARISON_CHECKPOINT_CURSOR')
    reject('chain',lambda:restore(expected_chain='0'*64),'PLACEMENT_COMPARISON_CHECKPOINT_CURSOR')
    first = next(iter(optimizer.state)); state = optimizer.state.pop(first)
    try: reject('missing_moments',lambda:worker.efficient.validate_optimizer(model,optimizer,policy,a.step),'INCOMPLETE_STATE')
    finally: optimizer.state[first] = state
    moment = state['exp_avg'].clone(); state['exp_avg'].view(-1)[0] = float('nan')
    try: reject('nonfinite_moment',lambda:worker.efficient.validate_optimizer(model,optimizer,policy,a.step),'OPTIMIZER_STATE')
    finally: state['exp_avg'].copy_(moment)
    restore(); rd.exact(dict(model.state_dict()),saved['model']); rd.exact(optimizer.state_dict(),saved['optimizer'])
    require(rd.sha(source) == before and rd.sha(initial_path) == initial_pin and not torch.cuda.is_initialized(), 'SOURCE_OR_DEVICE_CHANGED')
    files = {p.name:rd.sha(p) for p in a.output.iterdir() if p.is_file()}
    report = dict(complete=True,job=str(a.job),checkpoint=str(source),checkpoint_pin=before,initial_checkpoint_pin=initial_pin,
        census=original,restored_cpu_state_exact=True,comparison=matched,controls=controls,source_code=code,files=files,
        gpu_used=False,optimizer_updates=0,live_processes_changed=False,resume_execution_qualified=False,
        next_requirement='A versioned resume worker and identical CUDA continuation are still required before any handoff.',
        seconds=time.perf_counter()-began,acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),census=original,comparison=matched)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('root','job','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--step',type=int,required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

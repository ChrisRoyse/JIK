"""Run/reopen native-world learning against the sealed grounded execution journal."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from model.native_world_learning import NativeData, NativeToolLoop, increment
from model import native_residual_checkpoint as checkpoint, rehearsal_retention
from model import process_anchor
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import SenseError


def authenticate(root,pin):
    check(sha((root/'seal.json').read_bytes()) == pin,'NATIVE_WORLD_SOURCE_SEAL')
    seal = json.loads((root/'seal.json').read_bytes())
    for name,entry in seal['files'].items():
        check(sha((root/name).read_bytes()) == entry['sha256'] and (root/name).stat().st_size == entry['bytes'],
              'NATIVE_WORLD_SOURCE_BYTES')
    for name,entry in seal.get('external_native',{}).items():
        check(sha(Path(name).read_bytes()) == entry['sha256'],'NATIVE_WORLD_SOURCE_NATIVE')
    return dict(files=len(seal['files']),external_native=len(seal.get('external_native',{})))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--continuation-check',action='store_true')
    a = p.parse_args()
    check(not a.continuation_check or a.cold,'NATIVE_WORLD_CONTINUATION_MODE')
    started = time.perf_counter()
    proof = authenticate(a.source,a.pin)
    upstream = json.loads((a.source/'source.json').read_bytes())
    evidence = Path(upstream['root'])
    upstream_proof = authenticate(evidence,upstream['seal'])
    summary = json.loads((evidence/'summary.json').read_bytes())
    c = summary['config']
    root = a.output
    if not a.cold:
        root.mkdir(exist_ok=False)
        shutil.copyfile(a.source/'experience.sqlite3',root/'experience.sqlite3')
        write(root/'source.json',dict(root=str(a.source),seal=a.pin,upstream=upstream))
    source_journal = sha((a.source/'experience.sqlite3').read_bytes())
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','native-world-cold' if a.cold else 'native-world')
    gpu_name = 'gpu-continuation.txt' if a.continuation_check else 'gpu-cold.txt' if a.cold else 'gpu.txt'
    with (root/gpu_name).open('x') as stream:
        stream.write(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source,Path(c['population']),Path(c['native']),
                        population_pin=c['population_pin'],native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']),inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'],'NATIVE_WORLD_PARENT')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
    saved = json.loads(((root if a.cold else a.source)/'snapshot.json').read_bytes())
    session = NativeToolLoop(root/'experience.sqlite3',source,inputs,parent_root/'initial',model_pin=parent['initial'],
        standard=parent['standard'],world=parent['world'],contexts={'train:1':c['providers'][0]['task']},
        device='cuda',condition_inputs=conditions,routing=routing,growth_root=Path(c['output'])/'growth',
        candidate_root=a.source/'candidates',native_root=root/'candidates',
        expected_head=saved['head'],expected_contract=saved['contract'])
    check(session.snapshot() == saved,'NATIVE_WORLD_PREFIX_REPLAY')
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    torch.cuda.reset_peak_memory_stats()
    controls = []
    train = [v['forecast'] for v in summary['measured']['scored'][:2]]
    evaluate = [summary['measured']['scored'][2]['forecast']]
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'NATIVE_WORLD_CONTROL_'+name+':'+str(error))
        else:
            raise AssertionError('NATIVE_WORLD_CONTROL_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before,'NATIVE_WORLD_CONTROL_STATE')
        controls.append(dict(name=name,code=code))
    if not a.cold:
        tick = time.perf_counter()
        refuse('heldout_training','TOOL_LEARNING_SPLIT',lambda:NativeData(session,train=evaluate,
            evaluate=train,continuation='rehearsal-2'))
        data = NativeData(session,train=train,evaluate=evaluate,continuation='rehearsal-2')
        model,opt,migration = checkpoint.start(data)
        negative_pin = checkpoint.save(root/'negative-checkpoint',model,opt,data,step=0,parent=session.decisions[-1]['pin'])
        path = root/'negative-checkpoint/weights.pt'
        raw = path.read_bytes()
        broken = bytearray(raw)
        broken[len(broken)//2] ^= 1
        path.write_bytes(broken)
        refuse('corrupt_weights','NATIVE_WORLD_CHECKPOINT_BYTES',lambda:checkpoint.load(
            root/'negative-checkpoint',data,expected=negative_pin))
        path.write_bytes(raw)
        check(checkpoint.load(root/'negative-checkpoint',data,expected=negative_pin)[2]['step'] == 0,
              'NATIVE_WORLD_NEGATIVE_RESTORE')
        retained = copy.deepcopy(data.binding['grounded_retention'])
        data.binding['grounded_retention']['ceilings'][train[0]]['termination']['common'] = 2.
        refuse('forged_ceiling','GROUNDED_RETENTION_BINDING',lambda:rehearsal_retention.validate(data))
        data.binding['grounded_retention'] = retained
        rehearsal_retention.validate(data)
        phases['negative_controls'] = time.perf_counter()-tick
        tick = time.perf_counter()
        result = increment(session,root,name='native-1',train=train,evaluate=evaluate,continuation='rehearsal-2')
        write(root/'increment.json',result)
        phases['increment'] = time.perf_counter()-tick
        refuse('failed_selection','TOOL_CANDIDATE_NOT_SELECTED',lambda:session.require_selection('native-1'))
        saved = session.snapshot()
        write(root/'snapshot.json',saved)
        print(json.dumps(dict(stage='increment',seconds=phases['increment'],parameters=result['parameter_count'],
            changed=len(result['changed']),result=result['decision']['result'])),flush=True)
    else:
        check(session.decisions[-1]['kind'] == 'native_world_candidate'
              and session.decisions[-1]['result']['selected'] is False,'NATIVE_WORLD_COLD_DECISION')
        result = json.loads((root/'increment.json').read_bytes())
        check(result['decision'] == session.decisions[-1],'NATIVE_WORLD_COLD_RESULT')
    continuation = None
    if a.continuation_check:
        tick = time.perf_counter()
        data = NativeData(session,train=train,evaluate=evaluate,continuation='native-1')
        model,opt,continuity = checkpoint.start(data)
        inherited = data.evaluate(model)
        check(inherited == result['after'],'NATIVE_WORLD_CONTINUATION_BEHAVIOR')
        before = digest(tensor_fingerprint(model.state_dict()))
        up = {s.name:model.query['_native'][s.name].up.weight.detach().clone() for s in model.specs}
        disabled = {}
        try:
            with torch.no_grad():
                for s in model.specs:
                    model.query['_native'][s.name].up.weight.zero_()
                for key in train:
                    e = data.examples[key]
                    disabled[key] = process_anchor.predict(data.forward(model,key),e['view'].execution_anchor_bank,e['allowed'])
        finally:
            with torch.no_grad():
                for name,value in up.items():
                    model.query['_native'][name].up.weight.copy_(value)
        check(digest(tensor_fingerprint(model.state_dict())) == before,'NATIVE_WORLD_ABLATION_RESTORE')
        check(session.snapshot() == saved,'NATIVE_WORLD_CONTINUATION_STATE')
        continuation = dict(exact=True,offsets=data.binding['native_world']['offsets'],continuity=continuity,
            diagnostic_only_disabled_native_head=disabled,parameters_restored=True,no_optimizer_updates=True)
        phases['continuation_and_branch_control'] = time.perf_counter()-tick
    tick = time.perf_counter()
    head = session.head
    check(session.active_checkpoint == saved['active_model'],'NATIVE_WORLD_ACTIVE_CHANGED')
    session.close()
    check(sha((a.source/'experience.sqlite3').read_bytes()) == source_journal,'NATIVE_WORLD_SOURCE_MUTATION')
    phases['final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True,acceptance='author-checked',source=proof,upstream=upstream_proof,
        head=head,source_journal=source_journal,journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        controls=controls,cold=a.cold,continuation=continuation,
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        scientific_admission=False,serving=False)
    write(root/('continuation.json' if a.continuation_check else 'cold.json' if a.cold else 'report.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k != 'continuation'}),flush=True)


if __name__ == '__main__':
    run_cli(main)

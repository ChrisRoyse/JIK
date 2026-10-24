"""Qualify certified tile reads through real decoder updates and cold continuation."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_surface_inversion as source
from audit import legal_surface_inversion_readback as independent
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import surface_inversion_training as training
from model.legal_native_tile_stream import TileNativeFrames
from model.legal_surface_inputs import LegalSurfaceInputs
from model.native_population_training import cpu, fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha, need

QUALIFICATION = 'legal-native-tile-proof-cold-v2-20261024'
QUALIFICATION_PIN = 'e4e96bef4caf86bd9e312dcffce48d919cdb86271bdfa7ea5d190cfd73acf197'
CASES = ('0', '3273472', 'multi_window')


def persist(path, value):
    with path.open('xb') as f:
        torch.save(cpu(value), f); f.flush(); os.fsync(f.fileno())
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(saved) == fingerprint(value), 'TILE_UPDATE_PERSISTED_VALUE')
    return saved


def payload(examples):
    # Same typed serialization as the independently persisted phase120 inputs.
    return tuple(dict(occurrence=e.occurrence, uid=e.uid, side=e.side,
        raw=torch.tensor(list(e.raw), dtype=torch.uint8),
        native={n:dict(identity=x.identity, rows=x.rows, values=x.values, padding=x.padding,
                       absent=x.absent, indices=x.indices) for n,x in e.native.items()}) for e in examples)


def refused(fn, code):
    try: fn()
    except (ValueError, RuntimeError) as exc:
        need(code in str(exc), 'TILE_UPDATE_WRONG_REFUSAL:'+str(exc)); return str(exc)
    raise ValueError('TILE_UPDATE_NEGATIVE_ACCEPTED:'+code)


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    state = reader = None; leased = False; phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        qualified = source.authority.pinned(a.root/QUALIFICATION, QUALIFICATION_PIN)
        need(qualified['all_inputs_exact'] and qualified['multi_window_observed'], 'TILE_UPDATE_INPUT_QUALIFICATION')
        state = source.sources(a.root); base = Path(__file__).resolve().parents[1]
        need(all(file_sha(base/n) == h for n,h in qualified['code'].items()), 'TILE_UPDATE_QUALIFIED_CODE')
        code = dict(qualified['code'], **{'audit/legal_tile_update_proof.py':file_sha(__file__)})
        name,pin = source.statistics.DEFAULTS['audit']; reader = TileNativeFrames(state['data'], a.root/name, pin)
        candidate = LegalSurfaceInputs(state['data'], state['surfaces'], reader, state['config'], bank_pin=state['data'].native.pin)
        adapters = dict(reference=state['adapter'], candidate=candidate); cases = {}; census = {}
        for name in CASES:
            file = 'candidate-'+name+'.pt'; path = a.root/QUALIFICATION/file
            need(file_sha(path) == qualified['files'][file], 'TILE_UPDATE_INPUT_FILE')
            saved = torch.load(path, map_location='cpu', weights_only=True); case = qualified['cases'][name]
            need(fingerprint(saved) == case['pin'] and [e['occurrence'] for e in saved] == case['ids'], 'TILE_UPDATE_INPUT_BINDING')
            cases[name] = case
            census[name] = dict(atoms=len(saved), bytes=sum(e['raw'].numel() for e in saved),
                windows=sum(v['values'].shape[1] for e in saved for v in e['native'].values()))
        need(sum(v['atoms'] for v in census.values()) == 257, 'TILE_UPDATE_PROOF_CENSUS')
        write_committed(a.output/'census.json', census)
        device = torch.device('cpu' if a.cpu_proof else 'cuda:0')
        if not a.cpu_proof:
            (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
            need(torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'TILE_UPDATE_CUDA_REQUIRED')
            take_lease('29', 'legal-tile-update-proof'); leased = True
        binding = dict(qualification=QUALIFICATION_PIN, code=code, cases=cases, device=str(device),
                       torch=str(torch.__version__), cuda=torch.version.cuda, policy=training.POLICY,
                       initial=fingerprint(state['initial']), source=state['adapter'].binding)
        write_committed(a.output/'binding.json', binding)
        prior = None
        if a.previous:
            prior = source.authority.pinned(a.previous, a.previous_pin)
            need(prior['binding'] == binding and prior['all_updates_exact'], 'TILE_UPDATE_COLD_BINDING')
        phases['authentication']=time.perf_counter()-tick; tick=time.perf_counter()
        rows=[]; refs={}; controls={}; numerical_pins={}; access={}; references={}
        def fetch(mode, name):
            examples=adapters[mode].fetch(cases[name]['ids'], side='train', fit=True)
            need(fingerprint(payload(examples)) == cases[name]['pin'], 'TILE_UPDATE_LIVE_INPUT')
            return examples
        def save_numeric(name, value):
            saved=persist(a.output/name, value); numerical_pins[name]=fingerprint(saved)
            if prior:
                need(file_sha(a.previous/name) == prior['files'][name], 'TILE_UPDATE_COLD_FILE')
                need(fingerprint(torch.load(a.previous/name, map_location='cpu', weights_only=True))
                     == numerical_pins[name], 'TILE_UPDATE_COLD_NUMERICAL')
            return saved
        for mode in adapters:
            model,opt=training.create(state['config'], state['specs'], state['initial'], device)
            ledger=digest(binding); cursor=0
            for step,name in enumerate(CASES):
                start=time.perf_counter(); examples=fetch(mode,name); read_seconds=time.perf_counter()-start
                start=time.perf_counter(); row,profile=training.update(model,opt,examples,step)
                update_seconds=time.perf_counter()-start
                value=dict(record=row, gradients=cpu({n:p.grad for n,p in model.named_parameters()}),
                           model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()))
                path=f'{mode}-update-{step+1}.pt'; saved=save_numeric(path,value)
                if mode=='reference':references[name]=saved
                else:need(fingerprint(saved)==fingerprint(references[name]), 'TILE_UPDATE_REFERENCE_EXACT')
                cursor+=len(examples); ledger=digest(dict(previous=ledger, record=row))
                if step==0:
                    refs[mode]=training.checkpoint(a.output/f'{mode}-midpoint.pt',model,opt,binding,cursor,step+1,ledger)
                rows.append(dict(mode=mode,case=name,read_seconds=read_seconds,update_seconds=update_seconds,
                    numerical_pin=fingerprint(saved),positions=row['positions'],profile=profile))
                print(json.dumps(dict(mode=mode,case=name,step=step+1,objective=row['objective'],exact=True)),flush=True)
            del model,opt
        phases['paired_updates_and_readback']=time.perf_counter()-tick; tick=time.perf_counter()
        model,opt=training.create(state['config'],state['specs'],state['initial'],device)
        origin=fingerprint((model.state_dict(),opt.state_dict())); ref=refs['candidate']
        controls['wrong_binding']=refused(lambda:training.restore(ref,model,opt,{},size=257),'INVERSION_CHECKPOINT_BINDING')
        controls['wrong_cursor']=refused(lambda:training.restore(dict(ref,cursor=ref['cursor']+1),model,opt,binding,size=257),'INVERSION_CHECKPOINT_CURSOR')
        controls['wrong_pin']=refused(lambda:training.restore(dict(ref,pin='0'*64),model,opt,binding,size=257),'INVERSION_CHECKPOINT_PIN')
        need(fingerprint((model.state_dict(),opt.state_dict()))==origin,'TILE_UPDATE_NEGATIVE_MUTATION')
        cursor,step,ledger=training.restore(ref,model,opt,binding,size=257)
        need(cursor==128 and step==1,'TILE_UPDATE_RESUME_POSITION')
        need(fingerprint(cpu(model.state_dict()))==references[CASES[0]]['record']['after']
             and fingerprint(cpu(opt.state_dict()))==references[CASES[0]]['record']['optimizer'],'TILE_UPDATE_RESUME_STATE')
        for index,name in enumerate(CASES[1:],start=1):
            examples=fetch('candidate',name); native=fingerprint(payload(examples)); expected=references[name]
            need(fingerprint(model.state_dict())==expected['record']['before'],'TILE_UPDATE_RESUME_BEFORE')
            model.train(); opt.zero_grad(set_to_none=True)
            scores,objective=independent.score(model,examples,backward=True)
            gradients=cpu({n:p.grad for n,p in model.named_parameters()})
            need(fingerprint(gradients)==expected['record']['gradient'] and fingerprint(scores)==expected['record']['scores']
                 and objective==expected['record']['objective'],'TILE_UPDATE_INDEPENDENT_GRADIENT')
            opt.step(); values,moments=training.check_state(model,opt,index+1)
            need(fingerprint(values)==expected['record']['after'] and fingerprint(moments)==expected['record']['optimizer'],
                 'TILE_UPDATE_RESTORED_CONTINUATION')
            need(fingerprint(payload(examples))==native,'TILE_UPDATE_REPLAY_NATIVE_CHANGED')
            save_numeric(f'restored-update-{index+1}.pt',dict(scores=scores,objective=objective,gradients=gradients,model=values,optimizer=moments))
        origin=fingerprint((model.state_dict(),opt.state_dict())); held=next(e for e in state['examples'] if e.side=='query')
        controls['held_out_update']=refused(lambda:training.update(model,opt,[held],3),'INVERSION_TRAIN_ONLY')
        need(fingerprint((model.state_dict(),opt.state_dict()))==origin,'TILE_UPDATE_HELD_OUT_MUTATION')
        phases['independent_continuation_and_controls']=time.perf_counter()-tick; tick=time.perf_counter()
        access=dict(reference=dict(state['reader'].stats),candidate=dict(reader.stats),handles=dict(reader.io_stats))
        reader.close(); source.close(state)
        need(not reader.maps and (not a.cpu_proof or not torch.cuda.is_initialized()),'TILE_UPDATE_RUNTIME_CLOSURE')
        need(all(file_sha(base/n)==h for n,h in code.items()),'TILE_UPDATE_CODE_CHANGED')
        if prior:need(prior['controls']==controls and prior['numerical_pins']==numerical_pins,'TILE_UPDATE_COLD_RESULT')
        files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
        phases['final_readback_and_closure']=time.perf_counter()-tick; seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TILE_UPDATE_PHASES')
        report=dict(complete=True,binding=binding,code=code,files=files,census=census,rows=rows,controls=controls,
            numerical_pins=numerical_pins,checkpoints=refs,previous_pin=a.previous_pin,phases=phases,seconds=seconds,
            access=access,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if leased else 0,gpu_used=leased,
            optimizer_updates=8,all_updates_exact=True,independent_restored_continuation_exact=True,
            promoted=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if reader is not None and not reader.closed:reader.close()
        if state is not None:
            if not state['reader'].closed:state['reader'].close()
            if not state['surfaces'].closed:state['surfaces'].close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=source.authority.ROOT); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-proof',action='store_true'); p.add_argument('--previous',type=Path); p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

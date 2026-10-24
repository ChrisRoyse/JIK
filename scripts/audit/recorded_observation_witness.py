"""Real multimodal scene carry and cold replay without action/outcome supervision."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),
              str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.timed_world_witness import fingerprint,bundle
from audit.observation_checkpoint_transition import transition
from claims.timed_speech_memory import SpeechMemory
from lenscommon import take_lease,release_lease
from model.contracts import ModelConfig
from model.episode_input import EpisodeWorldInputs
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_identity_world import NativeIdentityWorld,SensoryIdentityCore
from model.recorded_observation_session import RecordedObservationSession,FLAGS
from model.timed_world_input import TimedWorldInputs
from senses.contract import digest
from senses.perception_corpus import pinned,sha
from senses.timed_readings import Bank,Inputs,need
from senses.timed_speech import save


def main(a):
    started=time.perf_counter();config=json.loads(a.config.read_bytes())
    root=Path(config['root']);stage='cold' if a.cold else 'warm'
    if not a.cold:
        root.mkdir(exist_ok=False);save(root/'config.json',config)
    else:need(json.loads((root/'config.json').read_bytes())==config,'OBSERVATION_CONFIG')
    c=config['source'];inputs_root=Path(c['inputs'])
    source=Inputs(inputs_root,c['input_pin'],Path(c['source']))
    bank_root=Path(c['bank']);manifest=pinned(bank_root/'bank.json',c['bank_pin'])
    bank=Bank(bank_root,c['bank_pin'],source,{n:digest(v) for n,v in manifest['instruments'].items()})
    retained=pinned(inputs_root/'rollback-retained.json',c['retained_pin'])
    need(sha(inputs_root/'rollback.sqlite')==c['memory_pin'],'OBSERVATION_MEMORY_SOURCE')
    memory_path=root/'memory.sqlite'
    if not a.cold:shutil.copy2(inputs_root/'rollback.sqlite',memory_path)
    need(sha(memory_path)==c['memory_pin'],'OBSERVATION_MEMORY_COPY')
    memory=SpeechMemory(memory_path,source.evidence,create=False,retained=retained)
    cfg=ModelConfig();selected=TimedWorldInputs(bank,memory,cfg)
    inputs=EpisodeWorldInputs(selected)
    identity=LoopInputs(Path(c['identity_population']),Path(c['identity_native']),Path(c['identity_source']))
    previous_root=Path(c['previous']);previous=pinned(previous_root/'report.json',c['previous_pin'])
    need(previous['binding']==inputs.binding,'OBSERVATION_PRIOR_BINDING')
    census={m['id']:{n:dict(rows=x.count,readings=len(x.pieces),dims=x.dims)
                     for n,x in lanes.items()} for m,lanes in zip(inputs.moments,inputs.lanes,strict=True)}
    need(sum(v['rows']*v['dims']*4 for lanes in census.values() for v in lanes.values())==2960381952,
         'OBSERVATION_CENSUS')
    order=sorted(range(len(inputs.moments)),key=lambda i:Fraction(*inputs.moments[i]['event_time']['at']))
    need(len(order)==4 and len(inputs.specs)==16,'OBSERVATION_PROOF_ROSTER')
    if not a.cold:
        save(root/'census.json',census);save(root/'source-records.json',inputs.records)
    else:
        need(json.loads((root/'census.json').read_bytes())==census and
             json.loads((root/'source-records.json').read_bytes())==inputs.records,'OBSERVATION_SOURCE_REPLAY')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name',
              '--format=csv,noheader'],text=True).strip(),'OBSERVATION_GPU_BUSY')
    (root/('gpu-'+stage+'.txt')).open('x').write(gpu)
    take_lease('28','recorded-scene-observation-'+stage)
    authenticated=time.perf_counter();session=None
    try:
        torch.set_num_threads(32);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20261020)
        core=IdentityLoopPredictor(cfg,identity.specs,identity.batch(identity.standard_units,cfg,'cpu'),
                                   source_id=identity.source_id)
        model=NativeIdentityWorld(SensoryIdentityCore(core,inputs.specs,inputs.temporal_catalogue),
                selected.batch([0,2],'cpu'),world_source_id=inputs.binding).cuda().eval()
        if not a.cold:
            states,migration=transition(previous_root/'initial',root/'initial',model,
                                       expected=previous['initial'],binding=inputs.binding)
            baseline=previous['outputs'][0]
            expected=pinned(previous_root/'report.json',c['previous_pin'])['outputs'][0]
            need(sha(previous_root/'outputs'/baseline['file'])==expected['sha256'],'OBSERVATION_LEGACY_OUTPUT')
            with torch.no_grad():
                old=model(inputs.batch(baseline['index']),(states[baseline['context']],),
                    allowed=torch.ones(len(next(iter(model.reference_rows.values()))),dtype=torch.bool,device='cuda'),
                    phase=baseline['phase'],event_ids=[digest(dict(transition='legacy-parity'))],
                    evidence_origin=['source_observation'])
            actual=bundle(old)
            wanted=torch.load(previous_root/'outputs'/baseline['file'],weights_only=True)
            need(set(actual)==set(wanted) and all(torch.equal(v,wanted[n]) for n,v in actual.items()),
                 'OBSERVATION_LEGACY_PARITY')
            migration.update(legacy_output=expected,legacy_exact=True)
            save(root/'transition.json',migration)
        else:migration=json.loads((root/'transition.json').read_bytes())
        def open_session(*,create,retained=None):
            return RecordedObservationSession(root/'observations.sqlite',model,inputs,
                checkpoint=root/'initial',checkpoint_pin=migration['after'],
                contexts=('scene','isolated'),create=create,retained=retained)
        loaded=time.perf_counter()
        if a.cold:
            warm=json.loads((root/'warm.json').read_bytes())
            journal_before=sha(root/'observations.sqlite')
            session=open_session(create=False,retained=warm['after']['retained'])
            need(session.snapshot()==warm['after'],'OBSERVATION_COLD_EXACT')
            need(fingerprint({n:p.detach().cpu() for n,p in model.state_dict().items()})==warm['weights'],
                 'OBSERVATION_COLD_WEIGHTS')
            after=session.snapshot();session.close();session=None
            need(sha(root/'observations.sqlite')==journal_before==warm['journal'],'OBSERVATION_COLD_JOURNAL')
            result=dict(warm=sha(root/'warm.json'),after=after,journal=journal_before,exact=True)
        else:
            session=open_session(create=True);before=session.snapshot()
            weights=fingerprint({n:p.detach().cpu() for n,p in model.state_dict().items()})
            steps=[]
            for index in order:
                isolated=session.snapshot()['contexts']['isolated']
                out=session.observe('scene',index,expected_head=session.journal.head)
                need(session.snapshot()['contexts']['isolated']==isolated,'OBSERVATION_CONTEXT_LEAK')
                steps.append(session.snapshot()['contexts']['scene'])
                print(json.dumps(dict(observation=len(steps),index=index,phase=out['phase'])),flush=True)
            scene=session.snapshot()['contexts']['scene']
            session.observe('isolated',order[-1],expected_head=session.journal.head)
            after=session.snapshot()
            need(after['contexts']['scene']==scene,'OBSERVATION_CONTEXT_LEAK')
            need(after['contexts']['isolated']['world']!=scene['world'],'OBSERVATION_HISTORY_UNUSED')
            need(all(v['identity_sequence']==0 for v in after['contexts'].values()),'OBSERVATION_FAKE_ACTION')
            need(fingerprint({n:p.detach().cpu() for n,p in model.state_dict().items()})==weights,
                 'OBSERVATION_WEIGHTS_CHANGED')
            controls=[]
            def refuse(name,code,fn):
                head=session.journal.head;count=session.journal.count
                try:fn()
                except Exception as exc:
                    need(getattr(exc,'code',str(exc))==code,'OBSERVATION_CONTROL:'+str(exc))
                else:raise ValueError('OBSERVATION_CONTROL_ACCEPTED:'+name)
                need(session.journal.head==head and session.journal.count==count,'OBSERVATION_CONTROL_MUTATED')
                controls.append(dict(name=name,code=code))
            head=session.journal.head
            refuse('duplicate','MNLM_CODATA_OBSERVATION_DUPLICATE',lambda:session.observe('scene',order[0],expected_head=head))
            refuse('reversed_time','MNLM_CODATA_OBSERVATION_ORDER',lambda:session.observe('isolated',order[0],expected_head=head))
            refuse('unknown_context','MNLM_CODATA_OBSERVATION_CONTEXT',lambda:session.observe('missing',0,expected_head=head))
            refuse('missing_episode','MNLM_CODATA_OBSERVATION_INDEX',lambda:session.observe('scene',99,expected_head=head))
            refuse('stale_writer','MNLM_CODATA_OBSERVATION_WRITER_CONFLICT',lambda:session.observe('scene',0,expected_head='0'*64))
            refuse('promote_reading_to_fact','MNLM_CODATA_OBSERVATION_PURPOSE',lambda:session.observe('scene',0,
                   expected_head=head,purpose='assert_world_truth'))
            # A separate retained source-memory copy exercises withdrawal, never edits the source.
            memory.close();shutil.copy2(memory_path,root/'withdrawn.sqlite')
            shadow=SpeechMemory(root/'withdrawn.sqlite',source.evidence,create=False,retained=retained)
            shadow_inputs=EpisodeWorldInputs(TimedWorldInputs(bank,shadow,cfg))
            moment=inputs.moments[order[0]]
            audio=next(shadow.memory.episodes[k] for k in moment['evidence']
                       if shadow.memory.episodes[k]['content'].get('medium')=='audio')
            shadow.apply(dict(op='invalidate',episode=audio['id'],reason='observation withdrawal control'),
                         expected_head=shadow.retained()['head'])
            refuse('withdrawn_support','MNLM_TIMED_MODEL_STALE',lambda:shadow_inputs.batch(order[0]))
            save(root/'withdrawn.json',shadow.snapshot());shadow.close()
            session.close();session=None
            result=dict(before=before,after=after,steps=steps,order=order,weights=weights,controls=controls,
                journal=sha(root/'observations.sqlite'),checkpoint=migration['after'],
                transition=sha(root/'transition.json'),
                census=sha(root/'census.json'),records=sha(root/'source-records.json'),
                source_memory=sha(memory_path),source_bytes=2960381952,
                source_measurements_need_outcome_labels=False,history_changes_state=True,
                cross_context_unchanged=True,action_phase_unchanged=True)
        ended=time.perf_counter()
        result.update(schema='recorded-observation-witness-v1',stage=stage,flags=FLAGS,
            number_status='measured',acceptance='author-checked',
            input_binding=inputs.binding,source_config=config,driver=sha(Path(__file__)),
            session_code=sha(Path(__file__).resolve().parents[1]/'model/recorded_observation_session.py'),
            transition_code=sha(Path(__file__).with_name('observation_checkpoint_transition.py')),
            phases=dict(authentication=authenticated-started,model_setup=loaded-authenticated,
                        observation_and_controls=ended-loaded),wall_seconds=ended-started)
        need(sha(inputs_root/'rollback.sqlite')==c['memory_pin'],'OBSERVATION_SOURCE_CHANGED')
        save(root/(stage+'.json'),result)
        print(json.dumps(dict(stage=stage,report=sha(root/(stage+'.json')),wall_seconds=result['wall_seconds'])),flush=True)
    finally:
        if session is not None:session.close()
        memory.close();release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--cold',action='store_true')
    main(p.parse_args())

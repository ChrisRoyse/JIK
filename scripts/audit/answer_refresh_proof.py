"""Prepare and read back the complete successor wiring without signalling current owners."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import answer_refresh_contract as contract
from audit import protected_corpus_queue_contract as decoder
from audit.legal_answer_successor import Children
from audit.surface_device_queue import need,sha,write
from claims import process_handoff as process


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False)
    if a.mode=='prepare':
        cfg=contract.prepare();write(a.output/'authority.json',cfg)
        contract.authenticate(a.output/'authority.json',sha(a.output/'authority.json'))
        print(json.dumps(dict(pin=sha(a.output/'authority.json'))),flush=True);return
    cfg,old,d,fresh=contract.authenticate(a.authority,a.pin);before=contract.live(cfg)
    controls={}
    def reject(label,fn,expected):
        try:fn()
        except ValueError as e:need(expected in str(e),'ANSWER_REFRESH_PROOF_WRONG_REFUSAL');controls[label]=str(e)
        else:raise ValueError('ANSWER_REFRESH_PROOF_ACCEPTED:'+label)
    for label,change,error in (
        ('missing_decoder_stage',lambda c:c['decoder_stages'].pop(),'STAGES'),
        ('missing_answer_stage',lambda c:c['answer_stages'].pop(),'STAGES'),
        ('missing_waiter',lambda c:c['owners'].pop('queue'),'OWNER_SCOPE'),
        ('fewer_answer_documents',lambda c:c['counts']['shared'].update(documents=91539),'POPULATION'),
        ('fewer_decoder_atoms',lambda c:c['decoder_train'].update(atoms=6547175),'POPULATION'),
        ('shorter_answers',lambda c:c['execution'].update(length_limit=16),'EXECUTION')):
        bad=deepcopy(cfg);change(bad);reject(label,lambda:contract.scope(bad,old,d),error)
    # Actual launch entry points must refuse before any model import, lease acquisition or output creation.
    for script in ('answer_pipeline_refresh.py','mixed_placement_program.py'):
        dest=a.output/(Path(script).stem+'-refused');log=a.output/(Path(script).stem+'.log')
        command=[sys.executable,'-I',str(contract.REPO/'scripts/audit'/script),'--authority',str(a.authority),
                 '--pin',a.pin,'--handoff',str(a.output/'absent'),'--handoff-pin','0'*64,'--output',str(dest)]
        if script=='answer_pipeline_refresh.py':command+=['--proof',str(a.output/'absent-proof'),'--proof-pin','0'*64]
        with log.open('xb') as f:status=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        need(status!=0 and 'ANSWER_REFRESH_LIVE_PREDECESSOR' in log.read_text() and not dest.exists(),
             'ANSWER_REFRESH_PROOF_LIVE_LAUNCH');controls[script]='ANSWER_REFRESH_LIVE_PREDECESSOR'
    children=Children(a.output,fresh)
    ref=children.run('seal-proof','answer_refresh_seal_proof.py',[],a.output/'seal-proof')
    seal=contract.read(ref,'report.json')
    need(seal['complete'] and seal['source']==cfg['code']['audit/answer_refresh_seal.py']
         and seal['reader']==cfg['code']['audit/answer_refresh_seal_readback.py']
         and seal['predictor_forwards']==0 and not seal['gpu_used'],'ANSWER_REFRESH_PROOF_SEAL')
    # Build all six protected stage commands from the unchanged function and real receipt addresses.
    wiring={};outputs={}
    for label in decoder.STAGES:
        script,args=decoder.arguments(label,d,outputs);wiring[label]=dict(script=script,arguments=args)
        # Only dependency addresses for an unexecuted plan, never fabricated successful results.
        outputs[label]=dict(path=str(a.output/('planned-'+label)),pin='UNEXECUTED')
    need([wiring[n]['script'] for n in decoder.STAGES]==[
        'protected_corpus_program.py','protected_corpus_program.py','protected_corpus_readback.py',
        'protected_corpus_readback.py','protected_corpus_program.py','protected_corpus_readback.py'],
        'ANSWER_REFRESH_PROOF_DECODER_PROGRAM')
    # Read actual prior payloads through their existing checks; there is no synthetic completion report.
    reject('incomplete_answer_population',lambda:contract.answer_complete(a.output,{}),'COMPLETION_STAGES')
    after=contract.live(cfg);fresh()
    need([process.binding(r) for r in before]==[process.binding(r) for r in after]
         and all(r['state'] in ('R','S','D') for r in after),'ANSWER_REFRESH_PROOF_OWNER_CHANGED')
    need('torch' not in sys.modules,'ANSWER_REFRESH_PROOF_MODEL_IMPORT')
    value=dict(complete=True,authority=a.pin,code=cfg['code'],answer_stages=cfg['answer_stages'],decoder_stages=cfg['decoder_stages'],
        seal_branch_proof=seal,seal_ref=ref,controls=controls,wiring=wiring,before=before,after=after,
        live_launch_refused=True,signals_sent=0,gpu_used=False,retirement_performed=False,seconds=time.perf_counter()-began,
        files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()},acceptance='author-checked')
    if a.previous:
        previous=contract.read(dict(path=str(a.previous/'report.json'),pin=a.previous_pin))
        need(previous['authority']==a.pin and previous['code']==cfg['code'] and previous['controls']==controls,
             'ANSWER_REFRESH_PROOF_COLD_BINDING')
        for n in ('comparison','results','source_pins','cuts','source','reader','proof_source'):
            need(previous['seal_branch_proof'][n]==seal[n],'ANSWER_REFRESH_PROOF_COLD_SEAL')
        for rel,h in previous['files'].items():need(sha(a.previous/rel)==h,'ANSWER_REFRESH_PROOF_COLD_BYTES')
        value.update(previous=a.previous_pin,cold_readback_exact=True)
    write(a.output/'report.json',value);print(json.dumps(dict(pin=sha(a.output/'report.json'),seconds=value['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','prove'),required=True)
    for n in ('authority','output','previous'):p.add_argument('--'+n,type=Path)
    for n in ('pin','previous-pin'):p.add_argument('--'+n)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e)))
        raise

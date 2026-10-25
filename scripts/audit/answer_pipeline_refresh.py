"""One serial owner for the full answer comparison and protected decoder program."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import answer_refresh_contract as contract
from audit import protected_corpus_queue_contract as decoder
from audit.legal_answer_successor import Children,paired_args,queue_steps
from audit.surface_device_queue import identity,need,sha,write


def run(a):
    began=time.perf_counter();cfg,old,d,fresh=contract.authenticate(a.authority,a.pin)
    contract.terminal(cfg)
    from audit.answer_refresh_retirement import handoff
    handoff(cfg,a.pin,a.handoff,a.handoff_pin)
    proof=contract.read(dict(path=str(a.proof/'report.json'),pin=a.proof_pin))
    need(proof['complete'] and proof.get('cold_readback_exact') is True
         and proof['authority']==a.pin and proof['code']==cfg['code']
         and proof['live_launch_refused'] and proof['answer_stages']==cfg['answer_stages']
         and proof['decoder_stages']==cfg['decoder_stages'],'ANSWER_REFRESH_PROOF')
    with (contract.REPO/'.lease/legal-answer-successor').open('a+') as lock,decoder.lock() as decoder_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(exist_ok=False)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
              driver_pin=sha(__file__),handoff=a.handoff_pin,proof=a.proof_pin))
        children=Children(a.output,fresh,lock.fileno());outputs={}
        common=['--authority',contract.ANSWER['path'],'--pin',contract.ANSWER['pin'],'--owner-fd',str(lock.fileno())]
        for stage in contract.answer.STAGES:
            path=a.output/f'inputs-{stage}.json';write(path,outputs)
            extra=['--inputs',str(path),'--inputs-pin',sha(path)]
            if stage=='placement':
                ref=children.run(stage,'mixed_placement_program.py',[
                    '--authority',str(a.authority),'--pin',a.pin,'--handoff',str(a.handoff),'--handoff-pin',a.handoff_pin,
                    '--owner-fd',str(lock.fileno())],a.output/stage)
            elif stage in ('prior-readback','capacity','training-readback'):
                ref=children.run(stage,'legal_answer_successor_worker.py',['--stage',stage,*common,*extra],a.output/stage)
            elif stage in ('tile','epoch','source'):
                previous={'tile':'placement','epoch':'tile','source':'epoch'}[stage]
                ref=queue_steps(children,old,stage,contract.answer.owner(Path(outputs[previous]['path'])),a.output)
            elif stage=='training':
                value=contract.read(outputs['capacity'],'report.json')
                need(value['capacity_qualified'] and not value['full_population_trained'],'ANSWER_REFRESH_CAPACITY')
                ref=children.run(stage,'legal_answer_source_training.py',value['arguments'],a.output/stage)
            else:ref=children.run(stage,'legal_answer_successor_evaluation.py',['--cuda',*common,*extra],a.output/stage)
            outputs[stage]=ref;write(a.output/f'completed-{stage}.json',ref)
        answer_pins=contract.answer_complete(a.output,outputs)
        write(a.output/'answer-complete.json',dict(complete=True,outputs=outputs,stages=answer_pins,
              full_population_trained=True,full_population_evaluated=True,architecture_winner=False))
        decoder_outputs={}
        children=Children(a.output,fresh,decoder_lock.fileno())
        for label in decoder.STAGES:
            script,args=decoder.arguments(label,d,decoder_outputs)
            ref=children.run('decoder-'+label,script,args,a.output/('decoder-'+label))
            decoder.child_scope(label,contract.read(ref,'report.json'),d,decoder_outputs)
            decoder_outputs[label]=ref;write(a.output/f'completed-decoder-{label}.json',ref)
        fresh();write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),
            outputs=outputs,decoder_outputs=decoder_outputs,full_population_trained=True,full_population_evaluated=True,
            protected_decoder_complete=True,original_complete=False,scientific_admission=False,architecture_winner=False,
            capability_demonstrated=False,seconds=time.perf_counter()-began,acceptance='author-checked'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','handoff','proof','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','handoff-pin','proof-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e)))
        raise

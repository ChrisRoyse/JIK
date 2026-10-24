"""Replay one sealed corrective update and read its actual region choices."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
import torch
from audit.corrective_open import open_session
from audit.fsv_field_world import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from lens.lenscommon import run_cli, take_lease
from model.constellation_learning import trainable
from model.corrective_learning import CorrectiveData
from model.loop_training_checkpoint import fingerprint
from model import field_checkpoint, corrective_step, rehearsal_retention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started, root = time.perf_counter(), a.output
    root.mkdir(exist_ok=False)
    proof = authenticate(a.source,a.pin)
    source = json.loads((a.source/'source.json').read_bytes())
    saved = json.loads((Path(source['root'])/'snapshot-final.json').read_bytes())
    recorded = json.loads((a.source/'learning.json').read_bytes())
    shutil.copyfile(Path(source['root'])/'experience.sqlite3',root/'experience.sqlite3')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','corrective-region-boundary')
    (root/'gpu-before.txt').write_text(gpu)
    configure()
    session, echo_proof = open_session(Path(source['root']),source['seal'],root,saved)
    data = CorrectiveData(session,recorded['decision']['spec'])
    model, optimizer, initial = field_checkpoint.load(a.source/'candidates/corrective-1-start',data,
                                                      expected=recorded['start_pin'])
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    original_scores = rehearsal_retention.scores

    def capture(scores):
        result = {}
        with torch.no_grad():
            for key in data.plan['train']:
                output = data.forward(model,key)
                c = output['predicted_constellation']
                result[key] = dict(region=c.regions[0],scores=scores[key],
                    predictions=c.pack(model.specs,model.config),
                    region_probabilities=torch.softmax(output['constellation_regions'].double(),dim=-1)[0].tolist())
        return result

    tick = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prepared = rehearsal_retention.prepare(data,model)
    before = capture(prepared['scores'])
    loss, details = data.loss(model)
    expected = recorded['losses'][0]
    check(float(loss.detach()) == expected['loss'] and details == expected['details'],'REGION_BOUNDARY_LOSS')
    loss.backward()
    named = trainable(model)
    gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                          for _,p in named])
    torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
    old = {n:p.detach().clone() for n,p in named}
    optimizer.step()
    captures = []

    def intercepted(d,m):
        check(d is data and m is model,'REGION_BOUNDARY_CAPTURE_OWNER')
        result = original_scores(d,m)
        captures.append(capture(result))
        return result

    rehearsal_retention.scores = intercepted
    try:
        update = corrective_step.apply(data,model,old,prepared,float(loss.detach()),gradient)
    finally:
        rehearsal_retention.scores = original_scores
    check(digest(update) == digest(expected['update']),'REGION_BOUNDARY_EXACT_UPDATE')
    check(len(captures) == len(update['trials'])+1,'REGION_BOUNDARY_CAPTURE_COUNT')
    for trial, values in zip(update['trials'],captures):
        check(trial['scores'] == {k:v['scores'] for k,v in values.items()},'REGION_BOUNDARY_TRIAL_SCORES')
    check(captures[-1] == captures[-2],'REGION_BOUNDARY_ACCEPTED_READBACK')
    phases['first_update_and_region_capture'] = time.perf_counter()-tick
    tick = time.perf_counter()
    check(session.snapshot() == saved,'REGION_BOUNDARY_JOURNAL_STATE')
    session.close()
    check(sha((root/'experience.sqlite3').read_bytes()) == source['journal'] ==
          sha((Path(source['root'])/'experience.sqlite3').read_bytes()),'REGION_BOUNDARY_JOURNAL_BYTES')
    phases['final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True,acceptance='author-checked',source=dict(root=str(a.source),seal=a.pin),
        source_proof=proof,echo_proof=echo_proof,code=sha(Path(__file__).read_bytes()),
        before=before,trials=[dict(scale=t['scale'],retained=t['retained'],values=c)
                             for t,c in zip(update['trials'],captures)],after=captures[-1],
        original_journal_unchanged=True,exact_update_replay=True,start=initial['fingerprint'],
        after_fingerprint=fingerprint(model,optimizer),phases=phases,wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall,serving=False,scientific_admission=False)
    write(root/'boundary.json',report)
    print(json.dumps(dict(complete=True,root=str(root),sha256=sha((root/'boundary.json').read_bytes()),
        regions={k:dict(before=v['region'],trials=[c[k]['region'] for c in captures[:-1]],
                        after=captures[-1][k]['region']) for k,v in before.items()},
        wall_seconds=wall,phase_coverage=report['phase_coverage'])),flush=True)


if __name__ == '__main__':
    run_cli(main)

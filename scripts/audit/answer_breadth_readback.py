"""Cold transfer, task-gradient and autonomous-answer readback.

Reconstructs named transfers and gradients without calling the transfer installer
or task update. The existing autonomous observer independently replays outputs.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_breadth_source as source
from audit import answer_history_source as history
from audit.answer_history_readback import autonomous, same
from audit.answer_breadth_training import binding, restore
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.native_placement_checked import validate_optimizer
from text.constellation_corpus import file_sha, need


def transfer(state, architecture, arm, saved, ledger):
    model, opt = source.make(state, architecture); initial = model.state_dict()
    donor = state['checkpoints'][arm]['model']; result = {}; assigned = {}; prefix = 'forecaster.predictor.'
    need(ledger['before'] == fingerprint(initial), 'BREADTH_READBACK_BEFORE')
    for name, value in initial.items():
        suffix = name.removeprefix(prefix); src = None; row = None
        if name.startswith(prefix+'projectors.'): src = suffix
        elif name.startswith(prefix+'encoder.'):
            if architecture == 'shared': src = suffix
            else:
                tail = suffix.removeprefix('encoder.lanes.'); lane, _, parameter = tail.partition('.')
                need(lane in model.roster, 'BREADTH_READBACK_LANE'); src = 'encoder.'+parameter
                if parameter in ('kind', 'query'): row = model.roster.index(lane)
        elif name.startswith(prefix+'region.'): src = suffix
        elif architecture == 'whole_proposals' and name.startswith(prefix+'readouts.'):
            tail = suffix.removeprefix('readouts.'); lane, _, parameter = tail.partition('.')
            if parameter.startswith('region.'): src = parameter
        if src is None:
            result[name] = value
        else:
            expected = donor[src] if row is None else donor[src][row:row+1]
            need(expected.shape == value.shape and expected.dtype == value.dtype, 'BREADTH_READBACK_SHAPE')
            result[name] = expected.clone(); assigned[name] = dict(source=src, row=row)
    same(assigned, ledger['mapping'], 'BREADTH_READBACK_MAPPING')
    same({n:fingerprint(v) for n,v in initial.items() if n not in assigned}, ledger['preserved'], 'BREADTH_READBACK_PRESERVED')
    same(result, saved['model'], 'BREADTH_READBACK_TRANSFER_WEIGHTS')
    need(ledger['source_state'] == fingerprint(donor) and ledger['after'] == fingerprint(result), 'BREADTH_READBACK_LEDGER')
    same(opt.state_dict(), saved['optimizer'], 'BREADTH_READBACK_RESET_OPTIMIZER')
    model.load_state_dict(result, strict=True)
    for name in assigned:
        need(model.state_dict()[name].untyped_storage().data_ptr() != donor[assigned[name]['source']].untyped_storage().data_ptr(),
             'BREADTH_READBACK_ALIAS')
    return model, opt


def replay(state, model, opt, value, request, step, saved):
    before = fingerprint(model.state_dict()); opt.zero_grad(set_to_none=True)
    c, a = len(value['context']), len(value['answers']); stream = model.stream(request); cache = None
    noise = torch.randn(a, state['policy']['samples'], state['policy']['noise_width'],
                        generator=torch.Generator().manual_seed(state['policy']['noise_seed']+step))
    rows = saved['record']['records']; seen = 0
    for pos, frame in enumerate((*value['context'], *value['answers'])):
        with torch.set_grad_enabled(pos >= c-1):
            hidden, cache = stream.step((frame,), context='requested-answer:'+str(value['document']),
                                       offset=pos, state=cache, detach_history=True)
        if pos < c-1: continue
        index = pos-c+1; row = rows[index]; ended = index == a; seen += 1
        need(row['end'] == ended and row['position'] == pos
             and row['answer_index'] == (None if ended else index), 'BREADTH_READBACK_LABEL')
        stop = model.stop_loss(hidden, ended); same(stop, row['stop'], 'BREADTH_READBACK_STOP')
        loss = stop['loss']/(a+1)
        if not ended:
            continuation = model.forecaster.losses(hidden, value['codes'][index:index+1],
                {n:x[index:index+1] for n,x in value['counts'].items()}, (value['answers'][index],),
                noise[index:index+1], target=state['target'])
            need(float(continuation['loss'].detach()) == row['continuation'], 'BREADTH_READBACK_CONTINUATION')
            loss = loss+continuation['loss']/a
        else: need(row['continuation'] is None, 'BREADTH_READBACK_TERMINAL')
        need(float(loss.detach()) == row['weighted_loss'], 'BREADTH_READBACK_LOSS'); loss.backward()
    need(seen == len(rows) == a+1 == saved['record']['decisions'], 'BREADTH_READBACK_DENOMINATOR')
    same({n:p.grad for n,p in model.named_parameters()}, saved['gradients'], 'BREADTH_READBACK_GRADIENTS')
    opt.step(); validate_optimizer(model, opt, state['policy'], step+1)
    same(model.state_dict(), saved['model'], 'BREADTH_READBACK_PARAMETERS')
    same(opt.state_dict(), saved['optimizer'], 'BREADTH_READBACK_MOMENTS')
    need(before == saved['record']['before'] and fingerprint(model.state_dict()) == saved['record']['after']
         and fingerprint(opt.state_dict()) == saved['record']['optimizer'], 'BREADTH_READBACK_UPDATE_LEDGER')
    return seen


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.pin, 'BREADTH_READBACK_PIN')
    report = json.loads((a.input/'report.json').read_bytes()); need(report['complete'], 'BREADTH_READBACK_COMPLETE')
    for n,h in report['files'].items(): need(file_sha(a.input/n) == h, 'BREADTH_READBACK_BYTES')
    state = source.load(a.root); need(state['code'] == report['code'], 'BREADTH_READBACK_CODE')
    same(state['census'], json.loads((a.input/'census.json').read_bytes()), 'BREADTH_READBACK_CENSUS')
    phases['authentication'] = time.perf_counter()-tick; decisions = rollouts = 0; negatives = {}; results = {}
    for architecture in history.ARCHITECTURES:
        for arm in source.ARMS:
            tick = time.perf_counter(); key = architecture+'/'+arm; dest = a.input/key
            initial = torch.load(dest/'initial.pt', map_location='cpu', weights_only=True)
            same(initial['binding'], binding(state, architecture, arm), 'BREADTH_READBACK_BINDING')
            ledger = json.loads((dest/'transfer.json').read_bytes())
            model, opt = transfer(state, architecture, arm, initial, ledger)
            rollouts += autonomous(state, model, torch.load(dest/'before.pt', map_location='cpu', weights_only=True))
            for step, doc in enumerate(history.TRAIN):
                model.train(); saved = torch.load(dest/f'update-{step+1}.pt', map_location='cpu', weights_only=True)
                same(saved['binding'], binding(state, architecture, arm), 'BREADTH_READBACK_BINDING')
                value, request = history.item(state, doc)
                decisions += replay(state, model, opt, value, request, step, saved)
                model, opt = restore(state, architecture, arm, saved, step+1)
            rollouts += autonomous(state, model, torch.load(dest/'after.pt', map_location='cpu', weights_only=True))
            results[key] = dict(model=fingerprint(model.state_dict()), optimizer=fingerprint(opt.state_dict()))
            need(results[key]['model'] == report['arms'][key]['final'] and
                 results[key]['optimizer'] == report['arms'][key]['optimizer'], 'BREADTH_READBACK_FINAL')
            phases[key] = time.perf_counter()-tick
            print(json.dumps(dict(arm=key, complete=True)), flush=True)
    tick = time.perf_counter(); dest = a.input/'shared/learned_corpus'
    initial = torch.load(dest/'initial.pt', map_location='cpu', weights_only=True)
    ledger = json.loads((dest/'transfer.json').read_bytes()); bad = deepcopy(initial)
    next(iter(bad['model'].values())).add_(1)
    negatives['changed_transfer'] = rejected(lambda: transfer(state, 'shared', 'learned_corpus', bad, ledger),
                                             'BREADTH_READBACK_TRANSFER_WEIGHTS')
    saved = torch.load(dest/'update-1.pt', map_location='cpu', weights_only=True); bad = deepcopy(saved)
    bad['optimizer']['state'] = {}; bad['optimizer_pin'] = fingerprint(bad['optimizer'])
    negatives['missing_moments'] = rejected(lambda: restore(state, 'shared', 'learned_corpus', bad, 1),
                                           'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')
    model, opt = transfer(state, 'shared', 'learned_corpus', initial, ledger)
    bad = deepcopy(saved); bad['record']['records'][-1]['end'] = False
    value, request = history.item(state, history.TRAIN[0])
    negatives['changed_end'] = rejected(lambda: replay(state, model, opt, value, request, 0, bad), 'BREADTH_READBACK_LABEL')
    state['fresh'](); need(not torch.cuda.is_initialized(), 'BREADTH_READBACK_GPU')
    need(decisions == report['total_decisions'] == 66 and rollouts == report['total_rollouts'] == 640, 'BREADTH_READBACK_TOTAL')
    phases['controls_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'BREADTH_READBACK_PHASES')
    result = dict(complete=True, acceptance='author-checked', source=dict(path=str(a.input), pin=a.pin), code=state['code'],
        decisions=decisions, rollouts=rollouts, results=results, controls=negatives, seconds=seconds, phases=phases,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, capability_demonstrated=False, scientific_admission=False)
    pin = write_committed(a.output/'report.json', result); print(json.dumps(dict(pin=pin, seconds=seconds)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

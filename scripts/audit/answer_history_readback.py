"""Separate reconstruction of feedback, gradients, optimizer and free answers.

Does not call the new training update, feedback function or history selector.
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
from audit import answer_history_source as source
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.native_request_stream import bind
from model.native_stochastic_placement import energy_loss
from model.native_placement_checked import validate_optimizer
from model.native_requested_answer import update as original_update
from text.constellation_corpus import file_sha, need


def same(left, right, code):
    need(fingerprint(left) == fingerprint(right), code)


def replay(state, model, opt, value, request, step, enabled, saved):
    before = fingerprint(model.state_dict()); opt.zero_grad(set_to_none=True)
    c, a = len(value['context']), len(value['answers'])
    seed = state['policy']['noise_seed']+step
    energy_noise = torch.randn(a, state['policy']['samples'], state['policy']['noise_width'],
                              generator=torch.Generator().manual_seed(seed))
    rows = saved['record']['records']; count = 0; selected_total = 0
    for proposal in model.proposals:
        original, select, readout = model._stream(proposal)
        stream = bind(original, model.request_projection, request); cache = None
        feedback_rng = torch.Generator().manual_seed(seed+1000000)
        for pos, frame in enumerate(value['context']):
            with torch.set_grad_enabled(pos == c-1):
                hidden, cache = stream.step((select(frame),), offset=pos, state=cache,
                    context='history-train:'+str(value['document']), detach_history=True)
        input_source = 'observed_context'; input_pin = fingerprint(value['context'][-1])
        for index in range(a+1):
            row = rows[count]; count += 1
            need(row['proposal'] == proposal and row['end'] == (index == a), 'HISTORY_REPLAY_ORDER')
            need(row['input_source'] == input_source and row['input'] == input_pin, 'HISTORY_REPLAY_INPUT')
            head, h = readout(hidden); logits = model.stop_logits(h, proposal)
            same(logits, row['stop_logits'], 'HISTORY_REPLAY_STOP')
            stop = -torch.log_softmax(logits.double(), dim=-1)[0, int(index == a)]
            loss = stop/(a+1)/len(model.proposals)
            selected = enabled and (index+step) % 2 == 0 and index < a
            need(row['selected_prediction'] == selected and row['forced_training_continuation'] == (index < a)
                 and row['predicted_end'] == bool(logits.argmax(-1).item()), 'HISTORY_REPLAY_SELECTION')
            selected_total += int(selected); next_frame = None
            if index < a:
                with torch.no_grad():
                    prediction = head.forecast(h)
                    noise = torch.randn(1, 1, state['policy']['noise_width'], generator=feedback_rng)
                    samples = model.forecaster.placement[proposal](h, prediction['codes'], prediction['counts'],
                        noise, target=state['target'])[0]
                    frame = {n:samples[n][0].detach().clone() for n in model.roster}
                same(dict(codes=cpu(prediction['codes']), counts=cpu(prediction['counts']), noise=noise,
                          slots=cpu(frame), source='predicted', validity='unassessed'),
                     row['feedback'], 'HISTORY_REPLAY_FEEDBACK')
                next_frame = frame if selected else value['answers'][index]
                need(fingerprint(next_frame) == row['next_input'], 'HISTORY_REPLAY_NEXT_INPUT')
                target_codes = value['codes'][index:index+1]
                target_counts = {n:x[index:index+1] for n,x in value['counts'].items()}
                discrete = head.loss(h, target_codes, target_counts)
                sample = model.forecaster.placement[proposal](h, target_codes, target_counts,
                    energy_noise[index:index+1], target=state['target'])
                native = energy_loss(sample, (value['answers'][index],), model.forecaster.placement[proposal].variances())
                continuation = discrete['loss']+native['loss']
                need(float(continuation.detach()) == row['continuation'], 'HISTORY_REPLAY_CONTINUATION')
                loss = loss+continuation/a/len(model.proposals)
            else:
                need(row['feedback'] is None and row['next_input'] is None and row['continuation'] is None,
                     'HISTORY_REPLAY_TERMINAL')
            need(float(loss.detach()) == row['weighted_loss'], 'HISTORY_REPLAY_LOSS')
            loss.backward()
            if next_frame is not None:
                hidden, cache = stream.step((select(next_frame),), offset=c+index, state=cache,
                    context='history-train:'+str(value['document']), detach_history=True)
                input_source = 'predicted' if selected else 'observed_answer'; input_pin = fingerprint(next_frame)
        cache = hidden = stream = original = None
    need(count == len(rows) == saved['record']['decisions'] and selected_total == saved['record']['predicted_inputs'],
         'HISTORY_REPLAY_DENOMINATOR')
    same({n:p.grad for n,p in model.named_parameters()}, saved['gradients'], 'HISTORY_REPLAY_GRADIENT')
    expected_prefixes = ['forecaster.predictor.encoder.']
    for lane in model.roster:
        expected_prefixes += [root+lane+'.' for root in ('forecaster.predictor.projectors.', 'request_projection.lanes.')]
        for proposal in model.proposals:
            expected_prefixes.append('ending.'+proposal+'.'+lane+'.')
            expected_prefixes += ['forecaster.placement.'+proposal+'.'+kind+'.'+lane+'.'
                                  for kind in ('context', 'region', 'noise', 'position', 'residual')]
    need(set(expected_prefixes) == set(saved['record']['branch_gradients']), 'HISTORY_REPLAY_BRANCH_COVERAGE')
    for prefix in expected_prefixes:
        group = {n:p.grad for n,p in model.named_parameters() if n.startswith(prefix)}
        need(group and any(bool(g.abs().any()) for g in group.values()) and
             fingerprint(group) == saved['record']['branch_gradients'][prefix], 'HISTORY_REPLAY_BRANCH')
    opt.step(); validate_optimizer(model, opt, state['policy'], step+1)
    same(model.state_dict(), saved['model'], 'HISTORY_REPLAY_PARAMETERS')
    same(opt.state_dict(), saved['optimizer'], 'HISTORY_REPLAY_OPTIMIZER')
    need(before == saved['record']['before'] and fingerprint(model.state_dict()) == saved['record']['after']
         and fingerprint(opt.state_dict()) == saved['record']['optimizer']
         and fingerprint(saved['gradients']) == saved['record']['gradients'], 'HISTORY_REPLAY_LEDGER')
    return count


def autonomous(state, model, saved):
    total = 0; model.eval()
    for row in saved:
        d = state['items'][row['document']]; need(d['side'] != 'train', 'HISTORY_REPLAY_EVAL_SIDE')
        desc = legal(2)
        if row['mode'] == 'wrong_operation': desc['operation'] = 'respond_to_source_question'
        generator = model.rollout(tuple(d['frames'][:2]), request=NativeRequest(desc), target=state['target'],
            proposal=row['proposal'], context='count-answer-proof', seed=20261024+d['document'])
        events = []; refusal = None
        try:
            for _ in range(len(d['frames'])-1):
                event = next(generator); events.append(cpu(event))
                if event['kind'] == 'end': break
        except ValueError as exc: refusal = str(exc)
        finally: generator.close()
        same(events, row['events'], 'HISTORY_REPLAY_AUTONOMOUS')
        need(refusal == row['refusal'], 'HISTORY_REPLAY_REFUSAL'); total += 1
        measured = source.previous.measured(dict(events=events, refusal=refusal), d['frames'][2:], model.roster, d['frames'][1])
        same(measured, row['metrics'], 'HISTORY_REPLAY_METRICS')
    return total


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.pin, 'HISTORY_READBACK_PIN')
    report = json.loads((a.input/'report.json').read_bytes()); need(report['complete'], 'HISTORY_READBACK_COMPLETE')
    for n, h in report['files'].items(): need(file_sha(a.input/n) == h, 'HISTORY_READBACK_BYTES')
    state = source.load(a.root); need(state['code'] == report['code'], 'HISTORY_READBACK_CODE')
    same(state['census'], json.loads((a.input/'census.json').read_bytes()), 'HISTORY_READBACK_CENSUS')
    phases['authenticate'] = time.perf_counter()-tick; total = rollouts = 0; negative = {}; results = {}
    for architecture in source.ARCHITECTURES:
        tick = time.perf_counter()
        # Unchanged teacher learner is also physically replayed.
        model, opt = source.make(state, architecture); value, request = source.item(state, source.TRAIN[0])
        row, grad = original_update(model, opt, value, 5, state['policy'], request=request, target=state['target'])
        original = torch.load(a.input/(architecture+'-original.pt'), map_location='cpu', weights_only=True)
        same(dict(record=row, gradients=grad, model=model.state_dict(), optimizer=opt.state_dict()),
             original, 'HISTORY_READBACK_ORIGINAL')
        for enabled in (False, True):
            key = architecture+('/mixed' if enabled else '/teacher'); model, opt = source.make(state, architecture)
            need(fingerprint((model.state_dict(), opt.state_dict())) == report['arms'][key]['initial'], 'HISTORY_READBACK_INITIAL')
            for cursor, doc in enumerate(source.TRAIN):
                path = a.input/key/f'update-{6+cursor}.pt'
                saved = torch.load(path, map_location='cpu', weights_only=True); value, request = source.item(state, doc)
                same(saved['binding'], dict(source=source.PARENT[1], region=state['pin'], code=state['code'],
                    architecture=architecture, predicted=enabled, document=doc, step=6+cursor), 'HISTORY_READBACK_BINDING')
                total += replay(state, model, opt, value, request, 5+cursor, enabled, saved)
                if cursor == 0: model, opt = source.make(state, architecture, saved, step=6)
            evaluation = torch.load(a.input/key/'evaluation.pt', map_location='cpu', weights_only=True)
            rollouts += autonomous(state, model, evaluation)
            results[key] = dict(updates=2, rollouts=len(evaluation), final=fingerprint(model.state_dict()),
                                optimizer=fingerprint(opt.state_dict()))
            need(results[key]['final'] == report['arms'][key]['final'] and
                 results[key]['optimizer'] == report['arms'][key]['optimizer'], 'HISTORY_READBACK_FINAL')
        phases[architecture] = time.perf_counter()-tick
    tick = time.perf_counter()
    path = a.input/'shared/mixed/update-6.pt'; saved = torch.load(path, map_location='cpu', weights_only=True)
    for name in ('feedback', 'selection'):
        bad = deepcopy(saved); row = bad['record']['records'][0]
        if name == 'feedback': next(iter(row['feedback']['slots'].values())).add_(1)
        else: row['selected_prediction'] = not row['selected_prediction']
        model, opt = source.make(state, 'shared'); value, request = source.item(state, source.TRAIN[0])
        negative[name] = rejected(lambda: replay(state, model, opt, value, request, 5, True, bad),
            'HISTORY_REPLAY_FEEDBACK' if name == 'feedback' else 'HISTORY_REPLAY_SELECTION')
    bad = deepcopy(saved); next(iter(bad['model'].values())).add_(1)
    negative['checkpoint'] = rejected(lambda: same(bad['model'], saved['model'], 'HISTORY_READBACK_CHECKPOINT'),
                                     'HISTORY_READBACK_CHECKPOINT')
    state['fresh'](); need(not torch.cuda.is_initialized(), 'HISTORY_READBACK_GPU')
    need(total == report['total_decisions'] and rollouts == 320, 'HISTORY_READBACK_TOTAL')
    phases['controls_source_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'HISTORY_READBACK_PHASES')
    result = dict(complete=True, acceptance='author-checked', source=dict(path=str(a.input), pin=a.pin),
        code=state['code'], total_decisions=total, total_rollouts=rollouts, results=results, controls=negative,
        seconds=seconds, phases=phases, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        capability_demonstrated=False, scientific_admission=False)
    pin = write_committed(a.output/'report.json', result); print(json.dumps(dict(pin=pin, seconds=seconds)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

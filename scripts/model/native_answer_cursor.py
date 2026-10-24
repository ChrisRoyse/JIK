"""Checkpointable autonomous answer generation; no future targets or lengths.

The existing generator remains the arithmetic reference. A cursor checkpoints
one live native history, its last hidden state and the placement RNG. A saved
END is terminal, not an assertion of correctness or successful completion.
"""
from pathlib import Path

import torch

from model.native_answer_sequence import versions
from model.native_branch_predictor import BranchCache, IndependentNativeStream
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest
from model.native_request_stream import bind
from text.constellation_corpus import file_sha, need

RUNTIME = ('model/native_answer_cursor.py', 'model/native_requested_answer.py',
    'model/native_answer_sequence.py', 'model/native_prefix_stream.py',
    'model/native_request_stream.py', 'model/native_request_input.py',
    'model/native_branch_predictor.py', 'model/native_open_window_answer.py',
    'model/native_stochastic_placement.py', 'model/native_region_predictor.py')


class AnswerCursor:
    def __init__(self, model, *, target, request, proposal, context, prefix_pin):
        need(isinstance(request, NativeRequest) and proposal in model.proposals
             and type(context) is str and bool(context) and type(prefix_pin) is str
             and len(prefix_pin) == 64 and not model.training, 'ANSWER_CURSOR_BINDING')
        self.model, self.target, self.request = model, target, request
        self.proposal, self.context = proposal, context
        self.placement = model.forecaster.placement[proposal]
        self.placement.check_target(target)
        self.original, self.select, self.readout = model._stream(proposal)
        self.stream = bind(self.original, model.request_projection, request)
        self.device = next(model.parameters()).device
        self.rng = torch.Generator(device=self.device)
        base = Path(__file__).resolve().parents[1]
        self.binding = dict(model=fingerprint(model.state_dict()), target=fingerprint(target.state_dict()),
            target_contract=self.placement.target_pin, request=request.pin, prefix=prefix_pin,
            proposal=proposal, context=context, device=str(self.device), torch=str(torch.__version__),
            architecture=model.forecaster.architecture,
            arithmetic=dict(threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                            tf32=torch.backends.cuda.matmul.allow_tf32),
            placement=dict(output_bytes=self.placement.output_bytes, noise_width=self.placement.noise_width,
                           max_windows=self.placement.max_windows),
            runtime={n:file_sha(base/n) for n in RUNTIME})
        self.model_versions, self.target_versions = versions(model), versions(target)
        self.state = self.hidden = None
        self.offset = self.generated = 0
        self.ended = self.closed = False
        self.failure = None
        self.hidden_version = None

    def _check(self):
        need(not self.closed and self.failure is None, 'ANSWER_CURSOR_CLOSED_OR_FAILED')
        need(not self.model.training and versions(self.model) == self.model_versions,
             'ANSWER_CURSOR_WEIGHTS_CHANGED')
        need(versions(self.target) == self.target_versions, 'ANSWER_CURSOR_TARGET_CHANGED')
        self.placement.check_target(self.target)
        need(self.request.pin == self.binding['request'], 'ANSWER_CURSOR_REQUEST_CHANGED')
        if self.hidden is not None:
            need(self.hidden._version == self.hidden_version, 'ANSWER_CURSOR_HIDDEN_CHANGED')

    def _remember(self, frame):
        self.hidden, self.state = self.stream.step((self.select(frame),), context=self.context,
            offset=self.offset, state=self.state, detach_history=True)
        self.offset += 1
        self.hidden_version = self.hidden._version

    @classmethod
    def start(cls, model, prefix, *, target, request, proposal, context, seed):
        need(isinstance(request, NativeRequest) and isinstance(prefix, (tuple, list)) and len(prefix) == request.observed
             and len(prefix) > 0 and all(set(f) == set(model.roster) for f in prefix)
             and type(seed) is int and seed >= 0, 'ANSWER_CURSOR_PREFIX')
        cursor = cls(model, target=target, request=request, proposal=proposal,
                     context=context, prefix_pin=fingerprint(prefix))
        cursor.rng.manual_seed(seed)
        with torch.no_grad():
            for frame in prefix: cursor._remember(frame)
        return cursor

    def __iter__(self):
        return self

    def __next__(self):
        self._check()
        if self.ended: raise StopIteration
        need(self.hidden is not None and self.offset == self.request.observed+self.generated,
             'ANSWER_CURSOR_POSITION')
        try:
            with torch.no_grad():
                head, value = self.readout(self.hidden)
                logits = self.model.stop_logits(value, self.proposal).detach().cpu()
                if bool(logits.argmax(-1).item()):
                    self.ended = True
                    return dict(kind='end', proposal=self.proposal, generated=self.generated,
                                stop_logits=logits.clone(), validity='unassessed')
                forecast = head.forecast(value)
                noise = torch.randn(1, 1, self.placement.noise_width, generator=self.rng, device=self.device)
                sample = self.placement(value, forecast['codes'], forecast['counts'], noise, target=self.target)[0]
                frame = {n:sample[n][0].detach() for n in self.model.roster}
                self._remember(frame)
                event = dict(kind='constellation', proposal=self.proposal, index=self.generated,
                    slots={n:x.clone() for n,x in frame.items()}, codes=forecast['codes'].clone(),
                    counts={n:x.clone() for n,x in forecast['counts'].items()}, stop_logits=logits.clone(),
                    feedback_index=self.offset-1, source='predicted', validity='unassessed')
                self.generated += 1
                return event
        except Exception as exc:
            self.failure = type(exc).__name__+':'+str(exc)
            raise

    def snapshot(self):
        self._check()
        need(self.state is not None and self.hidden is not None, 'ANSWER_CURSOR_UNSTARTED')
        if isinstance(self.stream, IndependentNativeStream):
            caches = {n:s.export(c) for (n,s),c in zip(self.stream.streams.items(), self.state.states, strict=True)}
        else: caches = {'assembled':self.stream.export(self.state)}
        value = cpu(dict(schema='native-answer-cursor-v1', binding=self.binding, caches=caches,
            hidden=self.hidden, rng=self.rng.get_state(), generated=self.generated,
            offset=self.offset, ended=self.ended, validity='unassessed'))
        # cpu() owns copies; callers cannot mutate live history through a receipt.
        return dict(payload=value, pin=fingerprint(value))

    @classmethod
    def restore(cls, saved, *, expected_pin, model, target, request, proposal, context, prefix_pin):
        need(set(saved) == {'payload', 'pin'} and saved['pin'] == expected_pin
             and fingerprint(saved['payload']) == expected_pin, 'ANSWER_CURSOR_CHECKPOINT_PIN')
        v = saved['payload']
        need(set(v) == {'schema','binding','caches','hidden','rng','generated','offset','ended','validity'}
             and v['schema'] == 'native-answer-cursor-v1' and v['validity'] == 'unassessed', 'ANSWER_CURSOR_SCHEMA')
        cursor = cls(model, target=target, request=request, proposal=proposal, context=context, prefix_pin=prefix_pin)
        need(v['binding'] == cursor.binding, 'ANSWER_CURSOR_RESTORE_BINDING')
        need(type(v['generated']) is int and v['generated'] >= 0 and type(v['offset']) is int
             and v['offset'] == request.observed+v['generated'] and type(v['ended']) is bool,
             'ANSWER_CURSOR_RESTORE_POSITION')
        independent = isinstance(cursor.stream, IndependentNativeStream)
        expected = set(cursor.stream.streams) if independent else {'assembled'}
        need(set(v['caches']) == expected, 'ANSWER_CURSOR_RESTORE_ROSTER')
        for cache in v['caches'].values():
            need(cache['next_atom'] == v['offset'], 'ANSWER_CURSOR_RESTORE_CACHE_POSITION')
        if independent:
            states = tuple(s.restore(v['caches'][n], context=context) for n,s in cursor.stream.streams.items())
            cursor.state = BranchCache(id(cursor.stream), states)
        else: cursor.state = cursor.stream.restore(v['caches']['assembled'], context=context)
        lanes = 1 if model.forecaster.architecture == 'whole_proposals' else len(model.roster)
        h = v['hidden']
        need(isinstance(h, torch.Tensor) and h.shape == (1, lanes, model.forecaster.predictor.config['width'])
             and h.dtype == torch.float32 and bool(h.isfinite().all()) and not h.requires_grad,
             'ANSWER_CURSOR_RESTORE_HIDDEN')
        rng = v['rng']; current = cursor.rng.get_state()
        need(isinstance(rng, torch.Tensor) and rng.dtype == current.dtype and rng.shape == current.shape,
             'ANSWER_CURSOR_RESTORE_RNG')
        cursor.rng.set_state(rng.cpu().clone())
        cursor.hidden = h.to(cursor.device).clone(); cursor.hidden_version = cursor.hidden._version
        cursor.generated, cursor.offset, cursor.ended = v['generated'], v['offset'], v['ended']
        cursor._check()
        return cursor

    def close(self):
        self.state = self.hidden = self.stream = self.original = self.select = self.readout = None
        self.closed = True

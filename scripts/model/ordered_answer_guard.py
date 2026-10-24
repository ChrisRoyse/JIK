"""Consume rendered atoms in prediction order; semantic acceptance stays external.

One atom is buffered within a physical allowance. Completed bytes are released;
the caller owns durable receipt storage. No reference answer or length is input.
"""
import codecs
import hashlib

import torch

from model.answer_surface_input import predicted_atom
from model.capability_surface_guard import POLICY, evaluate
from model.native_population_training import fingerprint
from text.constellation_corpus import need


class OrderedAnswerGuard:
    def __init__(self, predictions, specs, config, *, region_pin, region_sizes, trace_pin,
                 measurements, origin, buffer_bytes=16 << 20):
        need(origin in ('generated', 'observed_surface_control', 'observed_geometry_control'), 'ANSWER_GUARD_ORIGIN')
        need(type(buffer_bytes) is int and buffer_bytes > 0, 'ANSWER_GUARD_BUFFER')
        self.source = iter(predictions); self.specs = specs; self.config = config
        self.binding = dict(region_pin=region_pin, region_sizes=region_sizes, trace_pin=trace_pin)
        self.measurements = measurements; self.origin = origin; self.buffer_bytes = buffer_bytes
        self.identities = {s.name:s.identity for s in specs}
        self.completed = 0; self.proposal = None; self.current = None; self.raw = None
        self.terminal = False; self.geometry = True; self.bytes = 0; self.peak_buffer = 0
        self.measurement_calls = 0; self.sequence_hash = hashlib.sha256()

    def next_prediction(self):
        try: return next(self.source)
        except StopIteration as exc: raise ValueError('ANSWER_GUARD_PREDICTION_EXHAUSTED') from exc

    def release(self):
        self.current = self.raw = self.decoder = self.byte_hash = None

    def observe(self, event):
        need(not self.terminal, 'ANSWER_GUARD_AFTER_TERMINAL')
        need(type(event) is dict and event.get('serving') is False and event.get('validity') == 'unassessed',
             'ANSWER_GUARD_UNASSESSED_INPUT')
        kind = event.get('kind'); proposal = event.get('proposal')
        need(type(proposal) is str and proposal and (self.proposal is None or proposal == self.proposal),
             'ANSWER_GUARD_PROPOSAL')
        self.proposal = proposal
        if kind == 'atom_start':
            need(self.current is None and type(event.get('index')) is int and event['index'] == self.completed,
                 'ANSWER_GUARD_ATOM_ORDER')
            prediction = self.next_prediction()
            atom = predicted_atom(prediction, self.specs, self.config, **self.binding, position=self.completed)
            need(prediction['proposal'] == proposal and tuple(event['regions']) == tuple(atom.regions),
                 'ANSWER_GUARD_ATOM_BINDING')
            self.current = dict(prediction=fingerprint(prediction),
                slots={n:v.values[0].detach().cpu().numpy().copy() for n,v in atom.slots.items()})
            self.raw = bytearray(); self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
            self.byte_hash = hashlib.sha256(); return None
        if kind in ('byte', 'atom_end'):
            need(self.current is not None and type(event.get('atom')) is int and event['atom'] == self.completed,
                 'ANSWER_GUARD_SURFACE_ORDER')
            if kind == 'byte':
                need(type(event.get('position')) is int and event['position'] == len(self.raw)
                     and type(event.get('value')) is int and 0 <= event['value'] < 256
                     and event.get('token') == event['value'], 'ANSWER_GUARD_BYTE_POSITION')
                need(len(self.raw) < self.buffer_bytes, 'ANSWER_GUARD_PHYSICAL_BUFFER_EXHAUSTED')
                piece = bytes([event['value']]); self.raw.extend(piece); self.byte_hash.update(piece)
                self.peak_buffer = max(self.peak_buffer, len(self.raw))
                try: self.decoder.decode(piece, final=False)
                except UnicodeDecodeError:
                    return self.refuse('ANSWER_GUARD_UTF8')
                need(event.get('utf8_prefix_valid') is True, 'ANSWER_GUARD_UTF8_FLAG'); return None
            need(self.raw and event.get('token') == 257 and event.get('position') == len(self.raw)
                 and event.get('bytes') == len(self.raw) and event.get('sha256') == self.byte_hash.hexdigest(),
                 'ANSWER_GUARD_EOS_BYTES')
            try: self.decoder.decode(b'', final=True)
            except UnicodeDecodeError: return self.refuse('ANSWER_GUARD_UTF8')
            raw = bytes(self.raw); self.measurement_calls += 1
            measured, receipt = self.measurements(raw)
            guard = evaluate(raw, measured, self.current['slots'], self.identities, receipt, policy=POLICY)
            self.geometry &= guard['geometric_conjunction']
            record = dict(kind='atom_guard', index=self.completed, proposal=proposal, origin=self.origin,
                prediction=self.current['prediction'], guard=guard, bytes=len(raw), accepted=False, serving=False)
            self.sequence_hash.update(bytes.fromhex(fingerprint(record)))
            self.bytes += len(raw); self.completed += 1; self.release(); return record
        if kind == 'answer_end':
            need(self.current is None, 'ANSWER_GUARD_MISSING_ATOM_EOS')
            end = self.next_prediction()
            need(set(end) == {'kind','proposal','generated','stop_logits','validity'} and end['kind'] == 'end'
                 and end['proposal'] == proposal and end['validity'] == 'unassessed'
                 and type(end['generated']) is int and end['generated'] == self.completed
                 and type(event.get('atoms')) is int and event['atoms'] == self.completed, 'ANSWER_GUARD_END_COUNT')
            logits = end['stop_logits']
            need(isinstance(logits,torch.Tensor) and logits.dtype == torch.float32 and logits.shape == (1,2)
                 and not logits.requires_grad and bool(logits.isfinite().all()) and int(logits.argmax(-1)) == 1,
                 'ANSWER_GUARD_END_DECISION')
            sentinel = object(); need(next(self.source,sentinel) is sentinel, 'ANSWER_GUARD_PREDICTION_AFTER_END')
            self.terminal = True
            return dict(kind='answer_guard', atoms=self.completed, bytes=self.bytes, proposal=proposal,
                origin=self.origin, sequence=self.sequence_hash.hexdigest(), structural_complete=True,
                geometric_conjunction=self.geometry and self.completed > 0, accepted=False, serving=False,
                reasons=['PANEL_NOT_ADMITTED','SURFACE_GUARD_NOT_CALIBRATED','ANCHOR_CHECK_REQUIRED']
                        + ([] if self.completed else ['EMPTY_ANSWER']), peak_buffer=self.peak_buffer,
                measurement_calls=self.measurement_calls, semantic_validity_established=False)
        if kind == 'refuse':
            need(type(event.get('code')) is str and bool(event['code']), 'ANSWER_GUARD_REFUSAL_CODE')
            return self.refuse(event['code'])
        raise ValueError('ANSWER_GUARD_EVENT_KIND')

    def refuse(self, code):
        result = dict(kind='refuse', code=code, atoms=self.completed, proposal=self.proposal, origin=self.origin,
            accepted=False, serving=False, structural_complete=False, peak_buffer=self.peak_buffer,
            measurement_calls=self.measurement_calls)
        self.terminal = True; self.release(); return result

    def finish(self):
        need(self.terminal, 'ANSWER_GUARD_INCOMPLETE_STREAM')

    def close(self):
        self.release()
        if hasattr(self.source,'close'): self.source.close()

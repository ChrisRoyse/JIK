"""Target-free byte rendering of ordered predicted constellations.

Emitted bytes are unassessed diagnostic output, never a semantic guard verdict.
Atom EOS and predictor answer END are different events. The caller owns durable
output storage; this module retains only a bounded prefix and one atom's memories.
"""
from collections import deque
import codecs
import hashlib

import torch

from model.answer_surface_input import predicted_atom
from model.native_answer_sequence import versions
from model.stream_surface import BOS, EOS, position_batch
from text.constellation_corpus import need


def native_versions(native):
    return tuple((n,x.identity,x.rows,tuple(None if t is None else
        (id(t),t._version,tuple(t.shape),str(t.dtype),str(t.device))
        for t in [x.values,x.padding,x.absent,x.indices])) for n,x in native.items())


class SurfaceStepper:
    def __init__(self, model, native, *, cached=True):
        need(not model.training and type(cached) is bool, 'SURFACE_STREAM_FROZEN')
        need(all(len(x.rows)==1 for x in native.values()), 'SURFACE_STREAM_ONE_ATOM')
        self.model, self.native, self.cached = model, native, cached
        self.closed = False; self.calls = 0; self.projection_calls = 0
        self.weight_versions, self.input_versions = versions(model), native_versions(native)
        self.config, self.specs = repr(model.config), tuple(model.specs)
        self.memories = None
        with torch.no_grad():
            if cached:
                self.memories,_ = model.memories(native); self.projection_calls += 1
            else: model.validate_inputs(native)
        self.memory_versions = self.projected_versions()
        self.fresh()

    def projected_versions(self):
        return None if self.memories is None else tuple((n,id(t),t._version)
            for n,t in self.memories.items())

    def fresh(self):
        need(not self.closed, 'SURFACE_STREAM_CLOSED')
        need(not self.model.training and versions(self.model)==self.weight_versions
             and repr(self.model.config)==self.config and tuple(self.model.specs)==self.specs,
             'SURFACE_STREAM_WEIGHTS_CHANGED')
        need(native_versions(self.native)==self.input_versions, 'SURFACE_STREAM_INPUT_CHANGED')
        need(self.projected_versions()==self.memory_versions, 'SURFACE_STREAM_MEMORY_CHANGED')

    def logits(self, prefix, offset):
        self.fresh(); model = self.model
        need(isinstance(prefix,(list,tuple)) and 0<len(prefix)<=model.config.max_prefix
             and all(type(v) is int and 0<=v<model.config.vocabulary for v in prefix)
             and type(offset) is int and 0<=offset<2**24, 'SURFACE_STREAM_PREFIX')
        device = model.tokens.weight.device
        x = torch.tensor([prefix],dtype=torch.int64,device=device)
        offsets = torch.tensor([offset],dtype=torch.int64,device=device)
        with torch.no_grad():
            if self.cached:
                h = model.tokens(x)+position_batch(offsets,len(prefix),model.config.width)
                for block in model.blocks:
                    h = block(h,self.memories,self.native,[s.name for s in model.specs])
                logits = model.output(h)
                need(bool(logits.isfinite().all()), 'SURFACE_STREAM_NONFINITE')
            else:
                logits = model(x,self.native,offsets=offsets)['logits']
                self.projection_calls += 1
        self.calls += 1; self.fresh()
        return logits[0,-1].detach().clone()

    def close(self):
        self.memories = None; self.native = None; self.model = None; self.closed = True


def render_atom(model, native, *, cached=True, stride=16, capture_logits=False):
    """Yield bytes and actual EOS/refusal, with no reference length or text input."""
    need(type(stride) is int and 0<stride<=model.config.max_prefix, 'SURFACE_STREAM_STRIDE')
    stepper = SurfaceStepper(model,native,cached=cached)
    prefix = deque([BOS],maxlen=model.config.max_prefix)
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    digest = hashlib.sha256(); position = 0
    try:
        while True:
            left = max(0,(position//stride)*stride-(model.config.max_prefix-stride))
            count = position+1-left
            need(0<count<=len(prefix), 'SURFACE_STREAM_PREFIX_HISTORY')
            ids = list(prefix)[-count:]; logits = stepper.logits(ids,left)
            token = int(logits.argmax().item())
            record = dict(position=position,token=token,offset=left,prefix=ids,
                          validity='unassessed',serving=False)
            if capture_logits: record['logits'] = logits.cpu()
            if token==EOS:
                try: decoder.decode(b'',final=True); valid = position>0
                except UnicodeDecodeError: valid = False
                yield dict(record,kind='atom_end' if valid else 'refuse',
                    code=None if valid else ('SURFACE_STREAM_EMPTY' if not position else 'SURFACE_STREAM_UTF8'),
                    bytes=position,sha256=digest.hexdigest())
                return
            if not 0<=token<256:
                yield dict(record,kind='refuse',code='SURFACE_STREAM_SYMBOL',bytes=position,sha256=digest.hexdigest())
                return
            raw = bytes([token]); digest.update(raw); prefix.append(token); position += 1
            try: decoder.decode(raw,final=False); valid = True
            except UnicodeDecodeError: valid = False
            yield dict(record,kind='byte',value=token,utf8_prefix_valid=valid)
            if not valid:
                yield dict(kind='refuse',code='SURFACE_STREAM_UTF8',bytes=position,
                           sha256=digest.hexdigest(),validity='unassessed',serving=False)
                return
    finally:
        stepper.close()


def render_answer(events, model, *, region_pin, region_sizes, trace_pin, cached=True):
    """Pull each predicted atom, render it, then require the predictor's own END.

No reference count is accepted. A caller stopping observation has a censored
stream, never a complete answer. Native/source guards remain a separate gate.
"""
    completed = 0; proposal = None; source = iter(events); surface = None
    try:
        for event in source:
            need(isinstance(event,dict) and type(event.get('proposal')) is str
                 and bool(event['proposal']) and event.get('kind') in ('end','constellation'),
                 'ANSWER_RENDER_EVENT')
            need(proposal is None or event['proposal']==proposal, 'ANSWER_RENDER_PROPOSAL')
            proposal = event['proposal']
            if event['kind']=='end':
                need(set(event)=={'kind','proposal','generated','stop_logits','validity'}
                     and type(event['generated']) is int and event['generated']==completed
                     and event['validity']=='unassessed' and isinstance(event['stop_logits'],torch.Tensor)
                     and event['stop_logits'].dtype==torch.float32 and not event['stop_logits'].requires_grad
                     and event['stop_logits'].shape==(1,2) and bool(event['stop_logits'].isfinite().all())
                     and int(event['stop_logits'].argmax(-1))==1, 'ANSWER_RENDER_END')
                sentinel = object()
                need(next(source,sentinel) is sentinel, 'ANSWER_RENDER_AFTER_END')
                yield dict(kind='answer_end',atoms=completed,proposal=proposal,validity='unassessed',serving=False)
                return
            atom = predicted_atom(event,model.specs,model.config,region_pin=region_pin,
                                  region_sizes=region_sizes,trace_pin=trace_pin,position=completed)
            yield dict(kind='atom_start',index=completed,proposal=proposal,regions=atom.regions,
                       validity='unassessed',serving=False)
            surface = render_atom(model,atom.slots,cached=cached)
            ended = False
            for item in surface:
                yield dict(item,atom=completed,proposal=proposal)
                if item['kind']=='refuse': return
                ended |= item['kind']=='atom_end'
            need(ended, 'ANSWER_RENDER_MISSING_EOS')
            completed += 1; surface = None
        raise ValueError('MNLM_ANSWER_RENDER_MISSING_END')
    finally:
        if surface is not None: surface.close()
        if hasattr(source,'close'): source.close()

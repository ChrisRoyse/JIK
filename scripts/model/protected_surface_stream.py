"""Two resident native caches and one target-free byte stream for a protected decoder."""
from collections import deque
import codecs
import hashlib

import torch
from model.answer_surface_stream import SurfaceStepper
from model.stream_surface import BOS,EOS
from text.constellation_corpus import need


class ProtectedSurfaceStepper:
    def __init__(self,model,native):
        model.fresh();need(not model.training,'PROTECTED_STREAM_FROZEN');self.model=model
        self.parent=SurfaceStepper(model.parent,native);self.correction=None;self.closed=False
        try:self.correction=SurfaceStepper(model.correction,native)
        except BaseException:self.parent.close();raise

    def logits(self,prefix,offset):
        need(not self.closed,'PROTECTED_STREAM_CLOSED');self.model.fresh()
        logits=self.parent.logits(prefix,offset)+self.correction.logits(prefix,offset)
        need(bool(logits.isfinite().all()),'PROTECTED_STREAM_NONFINITE');return logits

    def close(self):
        if not self.closed:
            self.parent.close();self.correction.close();self.model=None;self.closed=True


def render_atom(model,native,*,stride=16,capture_logits=False):
    need(type(stride) is int and 0<stride<=model.config.max_prefix,'SURFACE_STREAM_STRIDE')
    stepper=ProtectedSurfaceStepper(model,native);prefix=deque([BOS],maxlen=model.config.max_prefix)
    decoder=codecs.getincrementaldecoder('utf-8')('strict');digest=hashlib.sha256();position=0
    try:
        while True:
            left=max(0,(position//stride)*stride-(model.config.max_prefix-stride));count=position+1-left
            need(0<count<=len(prefix),'SURFACE_STREAM_PREFIX_HISTORY')
            ids=list(prefix)[-count:];logits=stepper.logits(ids,left);token=int(logits.argmax())
            record=dict(position=position,token=token,offset=left,prefix=ids,validity='unassessed',serving=False)
            if capture_logits:record['logits']=logits.cpu()
            if token==EOS:
                try:decoder.decode(b'',final=True);valid=position>0
                except UnicodeDecodeError:valid=False
                yield dict(record,kind='atom_end' if valid else 'refuse',code=None if valid else ('SURFACE_STREAM_EMPTY' if not position else 'SURFACE_STREAM_UTF8'),
                           bytes=position,sha256=digest.hexdigest());return
            if not 0<=token<256:
                yield dict(record,kind='refuse',code='SURFACE_STREAM_SYMBOL',bytes=position,sha256=digest.hexdigest());return
            digest.update(bytes([token]));prefix.append(token);position+=1
            try:decoder.decode(bytes([token]),final=False);valid=True
            except UnicodeDecodeError:valid=False
            yield dict(record,kind='byte',value=token,utf8_prefix_valid=valid)
            if not valid:
                yield dict(kind='refuse',code='SURFACE_STREAM_UTF8',bytes=position,sha256=digest.hexdigest(),validity='unassessed',serving=False);return
    finally:stepper.close()

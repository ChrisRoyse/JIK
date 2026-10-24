"""Streaming surface generation from predicted slots, with explicit terminal failures."""
import codecs
from collections import deque
import hashlib
import json
import os
from pathlib import Path

import torch

from claims.capability_population import need, sha
from model.constellation import Constellation
from model.stream_surface import BOS, EOS, position_batch


@torch.no_grad()
def render(decoder, predicted, destination):
    """No target bytes enter this API; memory projections are invariant over generation."""
    need(isinstance(predicted, Constellation) and predicted.origin=='predicted',
         'SURFACE_RUNTIME_PREDICTION_REQUIRED')
    predicted.validate(decoder.specs,decoder.config)
    need(not decoder.training,'SURFACE_RUNTIME_EVAL_REQUIRED')
    destination=Path(destination); need(not destination.exists(),'SURFACE_RUNTIME_EXISTS')
    destination.mkdir(); native=predicted.slots
    memories,_=decoder.memories(native)
    device=decoder.tokens.weight.device
    prefix=deque([BOS],maxlen=decoder.config.max_prefix)
    utf8=codecs.getincrementaldecoder('utf-8')('strict')
    steps=bytes_written=0; status=''; ended=False; offset=0
    with (destination/'candidate.bin').open('xb') as raw, (destination/'tokens.jsonl').open('x') as trace:
        while True:
            offset=max(0,steps+1-decoder.config.max_prefix)
            if offset>=2**24:
                status='SURFACE_POSITION_UNREPRESENTABLE'; break
            tokens=torch.tensor([list(prefix)],device=device,dtype=torch.int64)
            offsets=torch.tensor([offset],device=device,dtype=torch.int64)
            h=decoder.tokens(tokens)+position_batch(offsets,len(prefix),decoder.config.width)
            for block in decoder.blocks: h=block(h,memories,native,[s.name for s in decoder.specs])
            logits=decoder.output(h)[0,-1]
            need(bool(torch.isfinite(logits).all()),'SURFACE_RUNTIME_NONFINITE')
            selected=int(logits.argmax()); values=logits.cpu().numpy().astype('<f4').tobytes()
            trace.write(json.dumps(dict(position=steps,offset=offset,token=selected,logits_sha256=sha(values)),sort_keys=True)+'\n')
            steps+=1
            if selected==EOS:
                ended=True
                try: utf8.decode(b'',final=True)
                except UnicodeDecodeError: status='SURFACE_UTF8_INCOMPLETE'; break
                status='COMPLETE' if bytes_written else 'SURFACE_EMPTY'; break
            if not 0<=selected<256:
                status='SURFACE_INVALID_SYMBOL'; break
            piece=bytes([selected]); raw.write(piece); bytes_written+=1; prefix.append(selected)
            try: utf8.decode(piece,final=False)
            except UnicodeDecodeError: status='SURFACE_UTF8_INVALID'; break
        raw.flush(); os.fsync(raw.fileno()); trace.flush(); os.fsync(trace.fileno())
    def fingerprint(path):
        h=hashlib.sha256(); count=0
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk); count+=len(chunk)
        return dict(bytes=count,sha256=h.hexdigest())
    return dict(schema='predicted-surface-runtime-v1',status=status,complete=status=='COMPLETE',
        eos=ended,steps=steps,bytes=bytes_written,max_prefix=decoder.config.max_prefix,
        maximum_total_tokens=None,memory_projection_calls=1,
        files={n:fingerprint(destination/n) for n in ('candidate.bin','tokens.jsonl')},
        target_bytes_accessed=False,scientific_admission=False,serving=False)

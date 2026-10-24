"""Explicit request features inside typed causal attention; native arrays stay frozen."""
import torch
from torch import nn

from model.attention import Linear
from model.native_branch_predictor import IndependentNativeStream
from model.native_prefix_stream import NativePrefixStream
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest, WIDTH
from text.constellation_corpus import need


class RequestProjection(nn.Module):
    def __init__(self,roster,width):
        super().__init__();self.lanes=nn.ModuleDict({n:Linear(WIDTH,width,bias=False) for n in roster})
        for p in self.parameters():nn.init.zeros_(p)


class RequestStream(NativePrefixStream):
    def __init__(self,original,projection,request):
        super().__init__(original.encoder,original.projectors,cache_bytes=original.cache_bytes)
        need(isinstance(request,NativeRequest),'REQUEST_STREAM_INPUT')
        self.projection=projection;self.request=request.pin;self.observed=request.observed
        self.features=request.materialize(self.encoder.query.device);self.feature_version=self.features._version

    def weights(self):
        return (*super().weights(),*(p for n in self.names for p in self.projection.lanes[n].parameters()))

    def weight_pin(self):
        # Checkpoint identity includes the explicit request and its learned path.
        # The parent hashes only native encoder/projectors, which is insufficient here.
        need(self.features._version==self.feature_version,'REQUEST_STREAM_FEATURES_CHANGED')
        return fingerprint(dict(native=super().weight_pin(),projection=self.projection.state_dict(),
                                request=self.request,features=self.features))

    def check_cache(self,state,context,offset,detached):
        need(self.features._version==self.feature_version,'REQUEST_STREAM_FEATURES_CHANGED')
        super().check_cache(state,context,offset,detached)

    def pack(self,frames,offset):
        x,atoms,queries=super().pack(frames,offset);parts=[]
        for j,frame in enumerate(frames):
            feature=self.features[min(offset+j,self.observed)]
            for name in self.names:
                vector=self.projection.lanes[name](feature[None])
                parts.append(vector.expand(len(frame[name])+1,-1))
        return x+torch.cat(parts,0)[None],atoms,queries


def bind(stream,projection,request):
    if isinstance(stream,NativePrefixStream):return RequestStream(stream,projection,request)
    need(isinstance(stream,IndependentNativeStream),'REQUEST_STREAM_KIND')
    stream.streams={n:RequestStream(s,projection,request) for n,s in stream.streams.items()}
    return stream

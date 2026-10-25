"""Durable, reference-free rendering; native END and surface EOS remain distinct."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from claims.process_conditions import write_committed as write
from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import render_atom as plain_atom
from model.complete_answer_trace import replay
from model.protected_surface_stream import render_atom as corrected_atom
from model.task_surface_scoring import native_pin
from text.constellation_corpus import canonical,digest,file_sha,need

ARMS=('parent','corrected')


class Sink:
    def __init__(self,root,request):
        self.root=Path(root);self.root.mkdir(exist_ok=False)
        self.request=write(self.root/'request.json',request);self.chain=self.request
        self.raw=(self.root/'surface.bin').open('xb');self.logits=(self.root/'logits.bin').open('xb')
        self.index=(self.root/'events.jsonl').open('x');self.records=self.bytes=self.decisions=0
        self.completed=self.attempted=0;self.refusal=None

    def emit(self,record,logits=None):
        row=dict(record,ordinal=self.records,previous=self.chain)
        if logits is not None:
            need(logits.dtype==torch.float32 and logits.shape==(258,) and bool(logits.isfinite().all()),'FREE_SURFACE_LOGITS')
            blob=logits.detach().cpu().numpy().astype('<f4',copy=False).tobytes()
            row['logits']=dict(offset=self.logits.tell(),bytes=len(blob),sha256=digest(blob))
            need(self.logits.write(blob)==len(blob),'FREE_SURFACE_SHORT_WRITE');self.decisions+=1
        if record['kind']=='byte':
            need(self.raw.write(bytes([record['value']]))==1,'FREE_SURFACE_SHORT_WRITE');self.bytes+=1
        self.chain=digest(canonical(row));row['chain']=self.chain
        self.index.write(json.dumps(row,sort_keys=True)+'\n');self.records+=1

    def seal(self,*,native,events,generated,ended):
        for f in (self.raw,self.logits,self.index):f.flush();os.fsync(f.fileno());f.close()
        value=dict(complete=True,request=self.request,files={n:file_sha(self.root/n) for n in
            ('request.json','surface.bin','logits.bin','events.jsonl')},chain=self.chain,records=self.records,
            generated_bytes=self.bytes,byte_decisions=self.decisions,attempted_atoms=self.attempted,
            completed_atoms=self.completed,surface_refusal=self.refusal,native_status=native['status'],
            native_refusal=native['refusal'],native_events=events,native_atoms=generated,native_ended=ended,
            answer_ended=ended and self.refusal is None and self.completed==generated,
            reference_visible=False,semantic_validity_established=False,serving=False)
        pin=write(self.root/'report.json',value);return dict(path=self.root.name,pin=pin,**value)

    def close(self):
        for f in (self.raw,self.logits,self.index):
            if not f.closed:f.close()


def collect(path,pin,models,*,output,region_pin,region_sizes,expected_request,decoder_pins):
    """Each arm runs until its own EOS/refusal; no target text or length argument."""
    began=time.perf_counter();path=Path(path);output=Path(output);output.mkdir(exist_ok=False)
    need(set(models)==set(decoder_pins)==set(ARMS),'FREE_SURFACE_ARMS')
    need(file_sha(path/'request.json')==expected_request,'FREE_SURFACE_SOURCE_REQUEST')
    source=json.loads((path/'request.json').read_bytes());sinks={};events=generated=0;ended=False
    for name,model in models.items():
        need({s.name:json.loads(s.definition_json)['instrument_lens_id'] for s in model.specs}==source['authority']['lens_ids'],
             'FREE_SURFACE_INSTRUMENTS')
        need(not model.training and model.config.vocabulary==258,'FREE_SURFACE_MODEL')
        sinks[name]=Sink(output/name,dict(schema='free-answer-surface-v1',arm=name,trace=pin,trace_path=str(path),
            trace_request=expected_request,region_pin=region_pin,region_sizes=list(region_sizes),decoder=decoder_pins[name],
            stride=16,max_prefix=model.config.max_prefix,reference_visible=False))
    try:
        # Draining this iterator authenticates the complete native trace even if a surface arm has refused.
        for event in replay(path,pin):
            events+=1
            if event['kind']=='end':ended=True;continue
            for name,model in models.items():
                sink=sinks[name]
                if sink.refusal is not None:continue
                atom=predicted_atom(event,model.specs,model.config,region_pin=region_pin,
                    region_sizes=region_sizes,trace_pin=pin,position=generated)
                slots={n:v.to(next(model.parameters()).device) for n,v in atom.slots.items()}
                sink.attempted+=1
                sink.emit(dict(kind='atom_start',atom=generated,regions=atom.regions,native=native_pin(slots)))
                renderer=(plain_atom if name=='parent' else corrected_atom)(model,slots,capture_logits=True)
                terminal=False
                try:
                    for record in renderer:
                        record=dict(record);logits=record.pop('logits',None)
                        sink.emit(dict(record,atom=generated),logits)
                        if record['kind']=='atom_end':sink.completed+=1;terminal=True
                        elif record['kind']=='refuse':sink.refusal=record['code'];terminal=True
                finally:renderer.close()
                need(terminal,'FREE_SURFACE_MISSING_EOS')
                print(json.dumps(dict(atom=generated,arm=name,bytes=sink.bytes,completed=sink.completed,
                    refusal=sink.refusal)),flush=True)
            generated+=1
        native=json.loads((path/'report.json').read_bytes())
        need(events==native['events'] and generated==native['generated_atoms'] and ended==native['model_ended'],
             'FREE_SURFACE_SOURCE_TERMINAL')
        arms={name:sink.seal(native=native,events=events,generated=generated,ended=ended) for name,sink in sinks.items()}
        result=dict(complete=True,trace=pin,trace_request=expected_request,arms=arms,events=events,
            generated_atoms=generated,model_ended=ended,reference_visible=False,seconds=time.perf_counter()-began,
            free_generation_tested=True,capability_demonstrated=False)
        result_pin=write(output/'report.json',result);return result,result_pin
    finally:
        for sink in sinks.values():sink.close()


def rows(root,pin):
    """Streaming frame authentication; consumers must drain to authenticate the seal."""
    root=Path(root);need(file_sha(root/'report.json')==pin,'FREE_SURFACE_REPORT_PIN')
    r=json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['reference_visible'] is False,'FREE_SURFACE_REPORT_SCOPE')
    for name,h in r['files'].items():need(file_sha(root/name)==h,'FREE_SURFACE_FILE_PIN')
    chain=r['request'];count=decisions=0
    with (root/'events.jsonl').open() as index,(root/'logits.bin').open('rb') as stream:
        for line in index:
            row=json.loads(line);h=row.pop('chain')
            need(row['ordinal']==count and row['previous']==chain and digest(canonical(row))==h,'FREE_SURFACE_CHAIN')
            chain=h;logits=None
            if 'logits' in row:
                ref=row['logits'];need(ref['offset']==stream.tell() and ref['bytes']==258*4,'FREE_SURFACE_FRAME')
                blob=stream.read(ref['bytes']);need(len(blob)==ref['bytes'] and digest(blob)==ref['sha256'],'FREE_SURFACE_LOGIT_BYTES')
                logits=torch.frombuffer(bytearray(blob),dtype=torch.float32).clone();decisions+=1
            count+=1;yield row,logits
        need(stream.read(1)==b'' and count==r['records'] and decisions==r['byte_decisions'] and chain==r['chain'],
             'FREE_SURFACE_RECOUNT')
    need(file_sha(root/'report.json')==pin,'FREE_SURFACE_CHANGED')

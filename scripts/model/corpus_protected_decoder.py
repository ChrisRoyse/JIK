"""Restore a completed protected corpus segment with explicit proof/full scope."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from model.contracts import ModelConfig
from model.native_population_training import fingerprint
from model.protected_corpus_lineage import origin, runtime
from model.protected_corpus_training import POLICY, progress
from model.protected_surface import ProtectedSurfaceDecoder, versions
from model.protected_surface_state import read as read_state
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp


def scope(parent, plan, producer, reader, *, plan_pin, run_pin, full, device):
    """Check scope before loading any correction tensor; partial fits stay partial."""
    p,r,c=plan,producer,reader;b=r['binding']
    need(type(full) is bool and device in ('cpu','cuda:0'), 'CORPUS_DECODER_DEVICE')
    need(p['complete'] and p['schema']=='protected-corpus-plan-v1' and p['policy']==POLICY,
         'CORPUS_DECODER_PLAN')
    need(r['complete'] and c['complete'] and r['plan']==c['plan']==plan_pin and c['input_pin']==run_pin,
         'CORPUS_DECODER_READBACK_BINDING')
    need(type(r['full_training']) is bool and r['full_training']==c['full_training']==full
         and b['scope']==('full' if full else 'proof') and (not full or device=='cuda:0')
         and r['cpu_proof']==(device=='cpu'), 'CORPUS_DECODER_TRAINING_SCOPE')
    size=p['train']['atoms'] if full else 384;steps=p['updates'] if full else 3
    need(size>0 and steps==(size+POLICY['atoms']-1)//POLICY['atoms']
         and b['size']==r['cursor']==c['ledger']['source_occurrences']==size
         and r['steps']==c['ledger']['rows']==steps, 'CORPUS_DECODER_INCOMPLETE_TRAINING')
    need(c['mode']=='read' and c['all_final_document_evaluation_bytes_exact']
         and c['evaluation']['independently_replayed'] and c['source_census']==p['train']
         and c['producer_implementation']==r['implementation'], 'CORPUS_DECODER_READBACK')
    need(r['device']==c['device']==b['device']==device and r['runtime']==c['runtime']==b['runtime']
         and b['torch']==b['runtime']['torch'] and b['cuda']==b['runtime']['cuda'], 'CORPUS_DECODER_RUNTIME')
    keys=('full','plan','cold','final','model','checkpoint','optimizer')
    need(all(r['parent'][k]==p['parent'][k]==parent[k] for k in keys)
         and b['parent']==parent['model'] and b['parent_checkpoint']==parent['checkpoint']
         and b['parent_optimizer']==parent['optimizer'], 'CORPUS_DECODER_PARENT')
    need(b['plan']==plan_pin and b['policy']==POLICY and b['config']==p['config']
         and b['code']==digest(r['implementation']), 'CORPUS_DECODER_PROGRAM')
    ref=r['checkpoints'][-1];last=c['ledger']['checkpoints'][-1]
    need(ref['cursor']==ref['size']==size and ref['step']==steps and ref['ledger']==r['ledger']
         and all(last[k]==ref[k] for k in ('step','pin','state'))
         and ref['origin']==r['origin'], 'CORPUS_DECODER_FINAL_REFERENCE')
    return ref


class CorpusProtectedDecoder:
    """One authenticated load; caller owns the parent and the CUDA lease."""
    def __init__(self,parent,*,plan,plan_pin,run,run_pin,readback,readback_pin,full=True):
        self.model=None;self.closed=False;self.files={};self.consumed={}
        parent.fresh();device=str(next(parent.model.parameters()).device)
        def read(path,pin):
            path=Path(path);before=stamp(path)
            need(not path.is_symlink() and file_sha(path)==pin,'CORPUS_DECODER_PIN')
            value=json.loads(path.read_bytes());need(stamp(path)==before,'CORPUS_DECODER_CHANGED')
            self.files[path]=before;self.consumed[str(path)]=pin;return value
        root=Path(run);p=read(Path(plan)/'report.json',plan_pin);r=read(root/'report.json',run_pin)
        c=read(Path(readback)/'report.json',readback_pin)
        ref=scope(parent.receipt,p,r,c,plan_pin=plan_pin,run_pin=run_pin,full=full,device=device)
        need(runtime(device)==r['runtime'],'CORPUS_DECODER_CURRENT_RUNTIME')
        base=Path(__file__).resolve().parents[1]
        for n,h in r['implementation'].items():
            path=base/n;before=stamp(path);need(file_sha(path)==h,'CORPUS_DECODER_IMPLEMENTATION')
            self.files[path]=before;self.consumed[str(path)]=h
        lineage=origin(ref,r['binding']);need(lineage['resume']==r['resume'],'CORPUS_DECODER_ORIGIN')
        # Track every metadata link in an already authenticated ancestry, once per load.
        tip=ref
        while True:
            o=read(tip['origin']['path'],tip['origin']['pin'])
            if o['resume'] is None:break
            previous=o['resume'];tip=read(previous['path'],previous['pin'])
            need(tip==previous['reference'],'CORPUS_DECODER_ANCESTRY')
        state_root=Path(ref['root'])
        need(state_root.resolve().parent==root.resolve() and not state_root.is_symlink(),
             'CORPUS_DECODER_CHECKPOINT_PATH')
        m=read(state_root/'manifest.json',ref['pin']);required=sum(x['bytes'] for x in m['shards'])
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
        need(required*4<available//2,'CORPUS_DECODER_PHYSICAL_MEMORY')
        paths=[state_root/x['name'] for x in m['shards']];before={path:stamp(path) for path in paths}
        binding=progress(r['binding'],ref['cursor'],ref['step'],ref['ledger'],ref['size'])
        payload,manifest=read_state(state_root,ref['pin'],binding=binding,parent=parent.receipt['model'])
        need(manifest==m and m['state']==ref['state'] and m['step']==ref['step']
             and fingerprint(payload['correction'])==ref['after'] and fingerprint(payload['optimizer'])==ref['optimizer'],
             'CORPUS_DECODER_FINAL_STATE')
        need(all(stamp(path)==s for path,s in before.items()),'CORPUS_DECODER_CHANGED')
        self.files.update(before)
        self.consumed.update({str(state_root/x['name']):x['sha256'] for x in m['shards']})
        config=ModelConfig(**p['config']);config.validate()
        self.model=ProtectedSurfaceDecoder(parent.model,config,seed=171)
        self.model.correction.load_state_dict(payload['correction'],strict=True)
        self.model.eval().requires_grad_(False)
        need(sum(v.numel() for v in self.model.correction.parameters())==p['parameters']
             and fingerprint(self.model.correction.state_dict())==ref['after']
             and self.model.parent_pin==parent.receipt['model'],'CORPUS_DECODER_RESTORED')
        self.config=config;self.specs=self.model.specs
        self.receipt=dict(plan=plan_pin,producer=run_pin,readback=readback_pin,checkpoint=ref['pin'],
            state=ref['state'],correction=ref['after'],optimizer=ref['optimizer'],parent=parent.receipt,
            train_atoms=ref['cursor'],optimizer_updates=ref['step'],full_training=full,device=device,
            optimizer_allocated=False,serving=False,semantic_validity_established=False,
            full_gradient_replay_established=c['full_gradient_replay'],shards=len(m['shards']),checkpoint_bytes=required)
        del payload
        self.weight_versions=versions(self.model);self.definition=asdict(config),tuple(self.specs);self.fresh()

    def fresh(self):
        need(not self.closed and self.model is not None,'CORPUS_DECODER_CLOSED')
        need(all(stamp(p)==s for p,s in self.files.items()),'CORPUS_DECODER_CHANGED')
        self.model.fresh()
        need(not self.model.training and all(not p.requires_grad and p.grad is None for p in self.model.parameters())
             and versions(self.model)==self.weight_versions and (asdict(self.model.config),tuple(self.model.specs))==self.definition,
             'CORPUS_DECODER_WEIGHTS_CHANGED')

    def close(self):
        self.model=None;self.closed=True

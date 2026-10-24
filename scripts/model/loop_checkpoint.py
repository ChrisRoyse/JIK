"""Create-only identity-loop/model snapshots with independently pinned readers."""
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from model.contracts import Instrument, ModelConfig, HEX
from model.identity_loop import ContextState, IdentityLoopPredictor
from model.native import NativeInput
from production.publication import sync_dir
from senses.contract import canonical, require

SCHEMA = 'intrinsic-loop-checkpoint-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code_pins():
    root=Path(__file__).parent
    return {n:sha((root/n).read_bytes()) for n in ('identity_loop.py','loop_checkpoint.py','contracts.py','native.py','attention.py','stream_attention.py')}


def save(model, states, root, *, parent=None):
    root=Path(root)
    model.check_reference()
    require(parent is None or HEX.fullmatch(parent), 'MNLM_LOOP_CHECKPOINT_PARENT', 'pin parent version')
    require(bool(states) and len({s.context for s in states})==len(states), 'MNLM_LOOP_CHECKPOINT_CONTEXT', 'distinct named contexts required')
    for s in states:
        require(s.standard_id==model.standard_id and type(s.sequence) is int and s.sequence>=0
                and s.hidden.shape==(len(model.specs),model.config.width)
                and s.hidden.dtype==torch.float32 and bool(torch.isfinite(s.hidden).all())
                and ((s.sequence==0 and s.last_event is None) or (s.sequence>0 and isinstance(s.last_event,str) and HEX.fullmatch(s.last_event))),
                'MNLM_LOOP_CHECKPOINT_CONTEXT', 'invalid or foreign context state')
    weights=model.state_dict()
    require(all(bool(torch.isfinite(t).all()) for t in weights.values()), 'MNLM_LOOP_CHECKPOINT_NONFINITE', 'do not commit invalid model state')
    size=sum(t.numel()*t.element_size() for t in weights.values())+sum(s.hidden.numel()*4 for s in states)
    require(size<=model.config.checkpoint_bytes, 'MNLM_LOOP_CHECKPOINT_BUDGET', 'snapshot exceeds fixed budget')
    weight_buffer, state_buffer=io.BytesIO(),io.BytesIO()
    torch.save(weights,weight_buffer)
    np.save(state_buffer,torch.stack([s.hidden.detach().cpu() for s in states]).numpy(),allow_pickle=False)
    payloads={'weights.pt':weight_buffer.getvalue(),'contexts.npy':state_buffer.getvalue()}
    require(sum(map(len,payloads.values()))<=model.config.checkpoint_bytes, 'MNLM_LOOP_CHECKPOINT_BUDGET', 'serialization exceeds fixed budget')
    manifest=dict(schema=SCHEMA, config=asdict(model.config), instruments=[asdict(s) for s in model.specs],
                  source_id=model.source_id, standard_id=model.standard_id, reference_rows=model.reference_rows,
                  contexts=[dict(context=s.context, sequence=s.sequence, last_event=s.last_event) for s in states],
                  files={n:dict(sha256=sha(v),bytes=len(v)) for n,v in payloads.items()}, parent=parent,
                  torch_version=str(torch.__version__), code_sha256=code_pins(),
                  scientific_admission=False, meaning_asserted=False, behavioural_compliance_established=False,
                  optimizer_resume_supported=False)
    raw=canonical(manifest)
    require(len(raw)<=1<<20, 'MNLM_LOOP_CHECKPOINT_BUDGET', 'metadata exceeds fixed envelope')
    require(not root.exists(), 'MNLM_LOOP_CHECKPOINT_EXISTS', 'published versions are immutable')
    root.mkdir(parents=True)
    for name,payload in {**payloads,'manifest.json':raw,'COMMITTED':sha(raw).encode()}.items():
        with (root/name).open('xb') as f:
            f.write(payload);f.flush();os.fsync(f.fileno())
    sync_dir(root);sync_dir(root.parent)
    return sha(raw)


def load(root, *, expected_checkpoint, expected_standard, device):
    root=Path(root)
    require(isinstance(expected_checkpoint,str) and HEX.fullmatch(expected_checkpoint)
            and isinstance(expected_standard,str) and HEX.fullmatch(expected_standard),
            'MNLM_LOOP_CHECKPOINT_PIN', 'supply the independently retained publication and standard pins')
    require((root/'COMMITTED').is_file() and (root/'manifest.json').is_file(),
            'MNLM_LOOP_CHECKPOINT_INCOMPLETE', 'restore the last committed version')
    require((root/'manifest.json').stat().st_size<=1<<20 and (root/'COMMITTED').stat().st_size==64,
            'MNLM_LOOP_CHECKPOINT_BUDGET', 'invalid manifest size')
    raw=(root/'manifest.json').read_bytes();m=json.loads(raw)
    require(sha(raw)==expected_checkpoint==(root/'COMMITTED').read_text() and canonical(m)==raw,
            'MNLM_LOOP_CHECKPOINT_CORRUPT', 'publication differs from its retained pin')
    require(m['schema']==SCHEMA and m['standard_id']==expected_standard and m['code_sha256']==code_pins()
            and m['torch_version']==str(torch.__version__),
            'MNLM_LOOP_CHECKPOINT_CONTRACT', 'use the pinned standard, code and runtime')
    require(all(m.get(n) is False for n in ('scientific_admission','meaning_asserted','behavioural_compliance_established','optimizer_resume_supported')),
            'MNLM_LOOP_CHECKPOINT_SCOPE', 'engineering snapshot cannot authorize production')
    config=ModelConfig(**m['config']);config.validate()
    require(set(m['files'])=={'weights.pt','contexts.npy'} and sum(x['bytes'] for x in m['files'].values())<=config.checkpoint_bytes,
            'MNLM_LOOP_CHECKPOINT_BUDGET', 'payload budget or roster differs')
    payloads={}
    for n,pin in m['files'].items():
        require((root/n).is_file() and (root/n).stat().st_size==pin['bytes'], 'MNLM_LOOP_CHECKPOINT_CORRUPT', n)
        payloads[n]=(root/n).read_bytes()
        require(sha(payloads[n])==pin['sha256'], 'MNLM_LOOP_CHECKPOINT_CORRUPT', n)
    weights=torch.load(io.BytesIO(payloads['weights.pt']),map_location='cpu',weights_only=True)
    specs=tuple(Instrument(**s) for s in m['instruments'])
    refs={s.name:NativeInput(s.identity,tuple(m['reference_rows'][s.name]),weights[f'ref_values_{i}'],
                            weights[f'ref_padding_{i}'],weights[f'ref_absent_{i}']) for i,s in enumerate(specs)}
    model=IdentityLoopPredictor(config,specs,refs,source_id=m['source_id'])
    require(model.standard_id==expected_standard, 'MNLM_LOOP_STANDARD_DRIFT', 'reference tensors differ from the external standard pin')
    model.load_state_dict(weights,strict=True)
    model.check_reference()
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), 'MNLM_LOOP_CHECKPOINT_NONFINITE', 'invalid weights')
    model=model.to(device)
    array=np.load(io.BytesIO(payloads['contexts.npy']),allow_pickle=False)
    records=m['contexts']
    require(bool(records) and len({r['context'] for r in records})==len(records)
            and array.shape==(len(records),len(specs),config.width) and array.dtype==np.float32 and np.isfinite(array).all(),
            'MNLM_LOOP_CHECKPOINT_CONTEXT', 'context identity or native state shape differs')
    states=[]
    for i,r in enumerate(records):
        require(isinstance(r['context'],str) and bool(r['context']) and type(r['sequence']) is int and r['sequence']>=0
                and ((r['sequence']==0 and r['last_event'] is None) or
                     (r['sequence']>0 and isinstance(r['last_event'],str) and HEX.fullmatch(r['last_event']))),
                'MNLM_LOOP_CHECKPOINT_CONTEXT', 'invalid phase or event binding')
        states.append(ContextState(r['context'],expected_standard,r['sequence'],torch.from_numpy(array[i]).to(device),r['last_event']))
    return model,tuple(states),m

"""Pinned experimental protected decoder and cached ordered answer rendering."""
import json
from pathlib import Path

import torch
from model.answer_surface_input import predicted_atom
from model.contracts import ModelConfig
from model.native_population_training import fingerprint
from model.protected_surface import ProtectedSurfaceDecoder,versions
from model.protected_surface_state import read as read_state
from model.protected_surface_stream import render_atom
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp


def render_answer(events,model,*,region_pin,region_sizes,trace_pin):
    """Consume model-determined ordered constellations and END; never a target length."""
    completed=0;proposal=None;source=iter(events);surface=None
    try:
        for event in source:
            need(isinstance(event,dict) and type(event.get('proposal')) is str
                 and bool(event['proposal']) and event.get('kind') in ('end','constellation'),
                 'ANSWER_RENDER_EVENT')
            need(proposal is None or event['proposal']==proposal,'ANSWER_RENDER_PROPOSAL')
            proposal=event['proposal']
            if event['kind']=='end':
                need(set(event)=={'kind','proposal','generated','stop_logits','validity'}
                     and type(event['generated']) is int and event['generated']==completed
                     and event['validity']=='unassessed' and isinstance(event['stop_logits'],torch.Tensor)
                     and event['stop_logits'].dtype==torch.float32 and not event['stop_logits'].requires_grad
                     and event['stop_logits'].shape==(1,2) and bool(event['stop_logits'].isfinite().all())
                     and int(event['stop_logits'].argmax(-1))==1,'ANSWER_RENDER_END')
                sentinel=object();need(next(source,sentinel) is sentinel,'ANSWER_RENDER_AFTER_END')
                yield dict(kind='answer_end',atoms=completed,proposal=proposal,validity='unassessed',serving=False)
                return
            atom=predicted_atom(event,model.specs,model.config,region_pin=region_pin,
                                region_sizes=region_sizes,trace_pin=trace_pin,position=completed)
            yield dict(kind='atom_start',index=completed,proposal=proposal,regions=atom.regions,
                       validity='unassessed',serving=False)
            surface=render_atom(model,atom.slots);ended=False
            for item in surface:
                yield dict(item,atom=completed,proposal=proposal)
                if item['kind']=='refuse':return
                ended|=item['kind']=='atom_end'
            need(ended,'ANSWER_RENDER_MISSING_EOS');completed+=1;surface=None
        raise ValueError('MNLM_ANSWER_RENDER_MISSING_END')
    finally:
        if surface is not None:surface.close()
        if hasattr(source,'close'):source.close()


class TrainedProtectedDecoder:
    """Restore once; borrowed parent stays owned by its caller. CPU evidence only."""
    def __init__(self,parent,root,pin,cold,cold_pin,*,cpu_diagnostic=False):
        self.model=None;self.closed=False;self.files={};self.consumed={}
        parent.fresh()
        need(cpu_diagnostic is True,'PROTECTED_RENDER_EXPLICIT_DIAGNOSTIC')
        need(next(parent.model.parameters()).device.type=='cpu','PROTECTED_RENDER_CUDA_UNQUALIFIED')
        root=Path(root);cold=Path(cold)
        def read(path,p):
            before=stamp(path);need(not path.is_symlink() and file_sha(path)==p,'PROTECTED_RENDER_PIN')
            value=json.loads(path.read_bytes());need(stamp(path)==before,'PROTECTED_RENDER_CHANGED')
            self.files[path]=before;self.consumed[str(path)]=p;return value
        r=read(root/'report.json',pin);c=read(cold/'report.json',cold_pin)
        need(r['complete'] and c['complete'] and c['input_pin']==pin and r['binding']['parent']==parent.receipt
             and r['parameters']==c['parameters']==78_325_042 and c['updates']==len(r['updates'])==3
             and r['initialization_rng_preserved'] and not r['cuda_qualified'],'PROTECTED_RENDER_BINDING')
        need(all(c[k] for k in ('source_bytes_exact','all_zero_logits_and_free_exact','all_updates_exact',
             'all_final_moments_exact','midpoint_restart_exact','all_direct_free_logits_exact')),
             'PROTECTED_RENDER_COLD_EVIDENCE')
        config=ModelConfig(**r['config']);config.validate()
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
        need(r['parameters']*64<available//2,'PROTECTED_RENDER_PHYSICAL_MEMORY')
        base=Path(__file__).resolve().parents[1]
        for name in ('model/protected_surface.py','model/protected_surface_state.py','model/protected_surface_stream.py',
                     'model/stream_surface.py','model/stream_attention.py','model/answer_surface_stream.py'):
            path=base/name;before=stamp(path);need(file_sha(path)==r['code'][name],'PROTECTED_RENDER_IMPLEMENTATION')
            self.files[path]=before;self.consumed[str(path)]=r['code'][name]
        ref=r['checkpoints'][-1];state_root=Path(ref['root'])
        need(state_root.parent.resolve()==root.resolve() and not state_root.is_symlink(),'PROTECTED_RENDER_STATE_PATH')
        manifest=read(state_root/'manifest.json',ref['pin'])
        before={state_root/x['name']:stamp(state_root/x['name']) for x in manifest['shards']}
        payload,m=read_state(state_root,ref['pin'],binding=r['binding'],parent=parent.receipt['model'])
        need(all(stamp(p)==s for p,s in before.items()),'PROTECTED_RENDER_CHANGED')
        self.files.update(before)
        self.consumed.update({str(state_root/x['name']):x['sha256'] for x in m['shards']})
        need(m['step']==ref['step']==3 and m['state']==ref['state']==r['final'],'PROTECTED_RENDER_FINAL_STATE')
        self.model=ProtectedSurfaceDecoder(parent.model,config,seed=r['binding']['seed'])
        self.model.correction.load_state_dict(payload['correction'],strict=True)
        self.model.eval().requires_grad_(False)
        need(fingerprint(self.model.correction.state_dict())==fingerprint(payload['correction']),'PROTECTED_RENDER_WEIGHTS')
        self.receipt=dict(producer=pin,cold=cold_pin,checkpoint=ref['pin'],state=r['final'],parent=parent.receipt,
                          correction=fingerprint(payload['correction']),optimizer=fingerprint(payload['optimizer']),
                          optimizer_allocated=False,device='cpu',serving=False,semantic_validity=False)
        del payload
        self.weight_versions=versions(self.model);self.fresh()

    def fresh(self):
        need(not self.closed and self.model is not None,'PROTECTED_RENDER_CLOSED')
        need(all(stamp(p)==s for p,s in self.files.items()),'PROTECTED_RENDER_CHANGED')
        self.model.fresh()
        need(not self.model.training and all(not p.requires_grad for p in self.model.parameters())
             and versions(self.model)==self.weight_versions,'PROTECTED_RENDER_WEIGHTS_CHANGED')

    def render(self,events,**binding):
        self.fresh()
        try:yield from render_answer(events,self.model,**binding)
        finally:self.fresh()

    def close(self):
        self.model=None;self.closed=True

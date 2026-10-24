"""Real legal prediction/observed-surface joins and exact-byte measurement reuse."""
import json
from pathlib import Path

import torch

from audit import predicted_surface_source as source
from audit.answer_surface_stream import REFS
from audit.surface_device_authority import pinned
from model.stream_surface import StreamSurfaceDecoder, EOS
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, digest, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
PARENT = ('answer-breadth-cpu-v4-20261024','051f83dbd8cf3df36d8c4a76e3b45d7228d1904cdc86643c348d1c827b4e5f10')
COLD = ('answer-breadth-readback-v4-20261024','e5e97504b21b52a3a06b917c9be0444839fb9e6cd3ca08453101d3b6c568ed4b')
NEW = ('model/ordered_answer_guard.py','audit/ordered_answer_guard_source.py',
       'audit/ordered_answer_guard.py','audit/ordered_answer_guard_readback.py')


def load(root):
    state = source.load(root); parent = pinned(root/PARENT[0], PARENT[1]); cold = pinned(root/COLD[0], COLD[1])
    need(cold['source']['pin'] == PARENT[1], 'ORDERED_GUARD_SOURCE_COLD')
    path = root/PARENT[0]/'whole_proposals/learned_corpus/after.pt'
    rows = torch.load(path,map_location='cpu',weights_only=True)
    trace = next(r for r in rows if r['document']==58818 and r['proposal']=='finbert-mean' and r['mode']=='correct')
    need([e['kind'] for e in trace['events']] == ['constellation']*3+['end'], 'ORDERED_GUARD_REAL_END')
    examples = [e for e,m in zip(state['examples'],state['metadata']) if m['document']==58818]
    original = state['items'][58818]
    need(len(examples) == len(original['frames']), 'ORDERED_GUARD_DOCUMENT_JOIN')
    for e, frame in zip(examples,original['frames']):
        need(all(torch.equal(e.native[n].values[0],v) for n,v in frame.items()), 'ORDERED_GUARD_NATIVE_JOIN')
    examples = examples[2:5]; identities = {s.name:s.identity for s in state['specs']}; cache = {}
    for e in examples:
        native = {n:v.values[0].numpy().copy() for n,v in e.native.items()}
        receipt = dict(source_sha256=digest(e.raw),source_bytes=len(e.raw),instruments=identities,
            joins=[dict(source=source.authority.REFS['source'],occurrence=e.occurrence,uid=e.uid)],
            slots={n:dict(shape=list(v.shape),sha256=digest(v.astype('<f4').tobytes())) for n,v in native.items()})
        cache[digest(e.raw)] = dict(raw=e.raw,native=native,receipt=receipt)
    def measure(raw):
        need(digest(raw) in cache and cache[digest(raw)]['raw'] == raw, 'ANSWER_GUARD_MEASUREMENT_REQUIRED')
        row = cache[digest(raw)]
        return {n:v.copy() for n,v in row['native'].items()}, row['receipt']
    model = StreamSurfaceDecoder(state['config'],state['specs'])
    model.load_state_dict(state['checkpoint']['after'],strict=True); model.eval().requires_grad_(False)
    base = Path(__file__).resolve().parents[1]; code = dict(state['code'])
    for n in (*NEW,'model/capability_surface_guard.py','audit/capability_surface_guard_fsv.py',
              'audit/answer_surface_stream_readback.py'): code[n] = file_sha(base/n)
    def cache_pin():
        return fingerprint({k:dict(raw=digest(v['raw']),receipt=v['receipt'],
            native={n:torch.from_numpy(x) for n,x in v['native'].items()}) for k,v in cache.items()})
    seal = fingerprint((trace,cache_pin(),model.state_dict())); oldfresh = state['fresh']
    def fresh():
        oldfresh(); need(fingerprint((trace,cache_pin(),model.state_dict())) == seal, 'ORDERED_GUARD_SOURCE_MUTATED')
        for n,h in code.items(): need(file_sha(base/n) == h, 'ORDERED_GUARD_CODE_CHANGED')
        need(file_sha(root/PARENT[0]/'report.json') == PARENT[1] and file_sha(root/COLD[0]/'report.json') == COLD[1],
             'ORDERED_GUARD_PARENT_CHANGED')
    return dict(state=state,model=model,trace=trace,trace_pin=file_sha(path),examples=examples,measure=measure,
        cache=cache,code=code,fresh=fresh,binding=dict(region_pin=state['region_pin'],region_sizes=tuple(state['bundle']['sizes'])))


def observed_events(state):
    """Observed source replay control. These bytes are never called generated."""
    proposal = state['trace']['proposal']; output=[]
    for i,e in enumerate(state['examples']):
        codes = state['trace']['events'][i]['codes'][0].tolist()
        common = dict(proposal=proposal,validity='unassessed',serving=False)
        output.append(dict(common,kind='atom_start',index=i,
            regions=(state['binding']['region_pin']+':'+'.'.join(map(str,codes)),)))
        for position,value in enumerate(e.raw):
            output.append(dict(common,kind='byte',atom=i,position=position,token=value,value=value,utf8_prefix_valid=True))
        output.append(dict(common,kind='atom_end',atom=i,position=len(e.raw),token=EOS,bytes=len(e.raw),sha256=digest(e.raw)))
    output.append(dict(common,kind='answer_end',atoms=3))
    return output

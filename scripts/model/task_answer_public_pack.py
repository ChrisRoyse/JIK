"""Public task prompts for autonomous answer generation; no answer-store reader.

One immutable pool per instrument shares byte-identical observed inputs. Request
order and every native window survive. Identifiers/splits bind evidence; only
frames and NativeRequest features are supplied to the predictor.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import torch

from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import stamp

SCHEMA = 'task-public-native-pack-v1'
POLICY = dict(native_encoder_calls=0, model_loads=0, optimizer_updates=0,
    deduplication='exact_observed_input_identity', all_native_windows=True,
    answer_vectors=False, answer_lengths=False, answer_endpoints=False,
    query_status='previously_exposed_development', scientific_admission=False)
ROW_KEYS = {'task','split','group','family','population','fields','descriptor','state','pin'}


def seal(row):
    return digest(canonical({k:v for k,v in row.items() if k != 'pin'}))


def validate_row(row):
    need(set(row) == ROW_KEYS, 'TASK_PUBLIC_PRIVATE_FIELD')
    need(row['split'] in ('train','query') and all(isinstance(row[k],str) and row[k]
         for k in ('task','group','family','population')), 'TASK_PUBLIC_PARTITION')
    need(row['fields'] and all(set(f) == {'field','input'} for f in row['fields']),
         'TASK_PUBLIC_FIELDS')
    need(all(isinstance(f['field'],str) and f['field'] and re.fullmatch('[0-9a-f]{64}',f['input'])
             for f in row['fields']), 'TASK_PUBLIC_INPUT_ID')
    request = NativeRequest(row['descriptor'])
    need(request.observed == len(row['fields']), 'TASK_PUBLIC_FIELD_COUNT')
    need(row['pin'] == seal(row), 'TASK_PUBLIC_CASE_PIN')
    return request


class TaskPublicPack:
    """Read a public-only publication without opening its private source bindings."""
    def __init__(self, root, pin, expected_binding=None):
        self.root = Path(root); self.pin = pin; self.maps = []; self.closed = False
        raw = (self.root/'report.json').read_bytes()
        need(digest(raw) == pin, 'TASK_PUBLIC_REPORT_PIN')
        self.report = r = json.loads(raw)
        need(set(r) == {'schema','complete','policy','binding','roster','dimensions','files','summary'}
             and r['schema'] == SCHEMA and r['complete'] is True and r['policy'] == POLICY,
             'TASK_PUBLIC_SCHEMA')
        if expected_binding is not None:
            need(r['binding'] == expected_binding, 'TASK_PUBLIC_SOURCE_BINDING')
        self.roster = tuple(r['roster']); self.dimensions = dict(r['dimensions'])
        need(len(self.roster) == len(set(self.roster)) == 6 and
             all(re.fullmatch('[a-z0-9-]+',n) for n in self.roster)
             and set(self.dimensions) == set(self.roster)
             and all(type(v) is int and v > 0 for v in self.dimensions.values()), 'TASK_PUBLIC_ROSTER')
        expected = {'inputs.json','cases.jsonl'} | {n+s for n in self.roster
            for s in ('.offsets.npy','.windows.npy')}
        need(set(r['files']) == expected, 'TASK_PUBLIC_FILE_SET')
        self.stamps = {self.root/'report.json':stamp(self.root/'report.json')}
        for name,ref in r['files'].items():
            path = self.root/name; before = stamp(path)
            need(set(ref) == {'sha256','bytes'} and before[2] == ref['bytes']
                 and file_sha(path) == ref['sha256'] and stamp(path) == before, 'TASK_PUBLIC_FILE_PIN')
            self.stamps[path] = before
        self.ids = tuple(json.loads((self.root/'inputs.json').read_bytes()))
        need(self.ids and list(self.ids) == sorted(set(self.ids)) and
             all(re.fullmatch('[0-9a-f]{64}',v) for v in self.ids), 'TASK_PUBLIC_INPUTS')
        self.index = {key:i for i,key in enumerate(self.ids)}
        self.cases = [json.loads(line) for line in (self.root/'cases.jsonl').read_bytes().splitlines()]
        need(self.cases and [r['task'] for r in self.cases] == sorted({r['task'] for r in self.cases}),
             'TASK_PUBLIC_CASE_ORDER')
        self.by_task = {r['task']:r for r in self.cases}
        for row in self.cases: validate_row(row)
        need(set(self.ids) == {f['input'] for r in self.cases for f in r['fields']}, 'TASK_PUBLIC_INPUT_UNION')
        need(not {r['group'] for r in self.cases if r['split']=='train'} &
             {r['group'] for r in self.cases if r['split']=='query'}, 'TASK_PUBLIC_GROUP_LEAK')
        try:
            self.offsets = {n:self.array(n+'.offsets.npy') for n in self.roster}
            self.vectors = {n:self.array(n+'.windows.npy') for n in self.roster}
            for n in self.roster:
                o,v = self.offsets[n],self.vectors[n]
                need(o.dtype == np.dtype('<i8') and o.shape == (len(self.ids)+1,)
                     and int(o[0]) == 0 and bool((np.diff(o)>0).all())
                     and v.dtype == np.dtype('<f4') and v.shape == (int(o[-1]),self.dimensions[n]),
                     'TASK_PUBLIC_LAYOUT')
            windows = {n:int(self.offsets[n][-1]) for n in self.roster}
            need(r['summary'] == dict(tasks=len(self.cases),unique_inputs=len(self.ids),
                 context_fields=sum(len(x['fields']) for x in self.cases),
                 splits={s:sum(x['split']==s for x in self.cases) for s in ('train','query')},
                 windows=windows,payload_bytes=sum(windows[n]*self.dimensions[n]*4 for n in self.roster)),
                 'TASK_PUBLIC_SUMMARY')
            self.fresh()
        except BaseException:
            for a in self.maps:a._mmap.close()
            raise

    def array(self, name):
        a = np.load(self.root/name,mmap_mode='r',allow_pickle=False); self.maps.append(a); return a

    def load(self, task, pin):
        self.fresh(); row = self.by_task.get(task)
        need(row is not None and row['pin'] == pin, 'TASK_PUBLIC_REQUEST_PIN')
        request = validate_row(row); frames = []
        # Physical allocation check, never a maximum answer length or work limit.
        required = sum(int(self.offsets[n][self.index[f['input']]+1]-self.offsets[n][self.index[f['input']]])
                       *self.dimensions[n]*4 for f in row['fields'] for n in self.roster)
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:')))*1024
        need(required <= available//4, 'TASK_PUBLIC_REQUEST_MEMORY_REQUIRES_TILING')
        for f in row['fields']:
            i = self.index[f['input']]; frame = {}
            for n in self.roster:
                lo,hi = map(int,self.offsets[n][i:i+2]); v = np.array(self.vectors[n][lo:hi],copy=True)
                need(np.isfinite(v).all(), 'TASK_PUBLIC_NONFINITE')
                frame[n] = torch.from_numpy(v)
            frames.append(frame)
        public = dict(frames=tuple(frames),descriptor=deepcopy(row['descriptor']))
        need(fingerprint(public) == row['state'], 'TASK_PUBLIC_NATIVE_STATE')
        self.fresh()
        return dict(context=public['frames'],request=request)

    def fresh(self, physical=False):
        need(not self.closed and all(stamp(p)==s for p,s in self.stamps.items()), 'TASK_PUBLIC_CHANGED')
        if physical:
            need(file_sha(self.root/'report.json') == self.pin and all(
                file_sha(self.root/n)==ref['sha256'] for n,ref in self.report['files'].items()),
                'TASK_PUBLIC_FINAL_PIN')

    def close(self):
        try:self.fresh(physical=True)
        finally:
            for a in self.maps:a._mmap.close()
            self.closed = True

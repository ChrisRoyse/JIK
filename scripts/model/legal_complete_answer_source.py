"""Whole legal continuations backed by existing native readings and frozen labels.

Public requests read only their observed prefix. Future slots, codes and endings
belong to the private learner item. File readers are shared, never rebuilt per item.
"""
from pathlib import Path

import numpy as np
import torch

from model import legal_complete_answer_plan as plan
from audit.legal_grouped_io_authority import authenticate as io_authority
from model.legal_grouped_documents import read_document
from model.native_conditioned_answer import document_item, check
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need

PLAN = ('legal-complete-answer-plan-v2-20261024', '892de320b5b122235976ba4066ec4d9b4a1668a6261fd6ed1342e84e3cfae960')
COLD = ('legal-complete-answer-plan-cold-v2-20261024', 'ec9a122b7fb7f0ddbaddc45d9d529e5ff3180f56d24b0ccd49a0aee7dd54b740')


def capacity(payload, allowance):
    need(type(allowance) is int and allowance > 0, 'LEGAL_ANSWER_MEMORY_POLICY')
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    need(payload <= min(allowance, available//4), 'LEGAL_ANSWER_DOCUMENT_MEMORY')


class CompleteAnswerCorpus:
    def __init__(self, root, data, reader, cache, *, document_bytes=128 << 20):
        self.source = plan.Source(root); self.data, self.reader, self.cache = data, reader, cache
        self.allowance = document_bytes; self.rows = None; self.closed = False
        try:
            reports = [self.source.report(Path(root)/n, p) for n,p in (PLAN,COLD)]
            r,c = reports
            need(r['mode'] == 'publish' and c['mode'] == 'cold' and c['previous_pin'] == PLAN[1]
                 and r['binding'] == c['binding'] == self.source.binding and r['summary'] == c['summary']
                 and r['controls'] == c['controls'], 'LEGAL_ANSWER_PLAN_AUTHORITY')
            path = Path(root)/PLAN[0]/'plan.npy'; self.source.bind(path, r['files']['plan.npy'])
            self.rows = np.load(path, mmap_mode='r', allow_pickle=False)
            self.plan = plan.Plan(self.source, self.rows)
            need(data is reader.data and data is cache.data and data.binding == cache.binding['source']
                 == self.source.reports['cache']['binding']['source']
                 and cache.binding['publication'] == plan.REFS['cache'][1]
                 and cache.binding['cold'] == plan.REFS['cache_cold'][1]
                 and tuple(data.roster) == tuple(cache.roster) == self.source.roster,
                 'LEGAL_ANSWER_READER_AUTHORITY')
            need(np.array_equal(data.documents,self.source.documents)
                 and np.array_equal(data.mapping,self.source.mapping)
                 and np.array_equal(data.doc_sides,self.source.sides)
                 and np.array_equal(data.components,self.source.components), 'LEGAL_ANSWER_SOURCE_JOIN')
            self.roster = self.source.roster
            io,self.io_fresh = io_authority(root)
            need(io['source'] == data.binding and io['cache'] == cache.binding, 'LEGAL_ANSWER_GROUPED_AUTHORITY')
            self.binding = dict(plan=PLAN[1], cold=COLD[1], source=self.source.binding,
                                cache=cache.binding, code=file_sha(Path(__file__)), io=io)
            self.fresh()
        except BaseException:
            if self.rows is not None:self.rows._mmap.close()
            self.source.close(); raise

    def fresh(self):
        need(not self.closed, 'LEGAL_ANSWER_CLOSED')
        self.source.fresh(); self.cache.fresh(); self.io_fresh()
        need(file_sha(Path(__file__)) == self.binding['code'], 'LEGAL_ANSWER_CODE_CHANGED')

    def public(self, document):
        address = self.plan.request(document)
        ids = list(address['observed_uids'])
        payload = int(self.source.counts[ids].sum(dtype=np.int64))*768*4
        capacity(payload,self.allowance)
        frames = self.reader.frames(ids)
        request = NativeRequest(legal(len(ids)))
        self.fresh()
        return frames, request

    def example(self, document, *, fitting):
        value = self.plan._row(document)
        need(type(fitting) is bool and (not fitting or int(value['side']) == 0), 'LEGAL_ANSWER_NONTRAIN_FIT')
        capacity(int(value['windows'].sum())*768*4, self.allowance)
        side = plan.SIDES[int(value['side'])]
        original,timing = read_document(self.data,self.reader,self.cache,document,side=side,fit=fitting)
        need((original['source_start'],original['source_stop']) == (int(value['start']),int(value['stop'])),
             'LEGAL_ANSWER_SOURCE_BOUNDARY')
        original['component'] = int(value['component'])
        observed = int(value['answer_start']-value['start'])
        item = document_item(original,observed); request = NativeRequest(legal(observed))
        item['provenance'].update(request=request.pin, source_plan=PLAN[1], source_start=int(value['start']),
                                 source_stop=int(value['stop']), addresses=value['addresses'].tobytes().hex())
        item['binding'] = fingerprint((item['document'],item['side'],item['provenance']))
        self.validate(item,request,fitting=fitting)
        return item,request,timing

    def validate(self, item, request, *, fitting):
        value = self.plan._row(item['document']); observed = int(value['answer_start']-value['start'])
        need(type(fitting) is bool and (not fitting or int(value['side']) == 0), 'LEGAL_ANSWER_NONTRAIN_FIT')
        need(item['side'] == plan.SIDES[int(value['side'])]
             and len(item['context']) == observed and len(item['answers']) == int(value['stop']-value['answer_start']),
             'LEGAL_ANSWER_COMPLETE_BOUNDARY')
        p = item['provenance']
        need(p['source_plan'] == PLAN[1] and p['component'] == int(value['component'])
             and p['source_start'] == int(value['start']) and p['source_stop'] == int(value['stop'])
             and p['addresses'] == value['addresses'].tobytes().hex() and p['observed'] == observed,
             'LEGAL_ANSWER_PROVENANCE')
        expected = NativeRequest(legal(observed))
        need(request.pin == expected.pin == p['request'] and torch.equal(request.materialize('cpu'),expected.features),
             'LEGAL_ANSWER_PUBLIC_REQUEST')
        need(fingerprint((item['context'],item['answers'])) == item['native_pin']
             and fingerprint((item['codes'],item['counts'])) == item['label_pin']
             and fingerprint((item['document'],item['side'],p)) == item['binding'], 'LEGAL_ANSWER_ITEM_CHANGED')
        frames = (*item['context'],*item['answers'])
        need(all(set(f) == set(self.roster) for f in frames), 'LEGAL_ANSWER_ROSTER')
        for j,name in enumerate(self.roster):
            need(all(x[name].ndim == 2 and x[name].shape[1] == 768 and x[name].dtype == torch.float32 for x in frames)
                 and sum(len(f[name]) for f in frames) == int(value['windows'][j])
                 and sum(len(f[name]) for f in item['answers']) == int(value['answer_windows'][j]), 'LEGAL_ANSWER_WINDOWS')
        if fitting:check(item)
        self.fresh()

    def close(self):
        if self.closed:return
        try:self.fresh(); self.reader.finish(); self.source.close()
        finally:
            self.rows._mmap.close(); self.closed = True

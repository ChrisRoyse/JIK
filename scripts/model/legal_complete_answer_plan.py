"""Complete source continuations; private END labels never follow an interior tile.

Addresses route cached readings. They are not model features or outcome anchors.
PC-07: source bytes and counts are authenticated once; no native model is loaded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

SIDES = ('train', 'development', 'query')
REFS = {
    'source': ('legal-input-reuse-v1-20261022', '7dd93b1996f79768ead1bb419068b7eb62e3b15d3cf5f529e17c610d5279a02e'),
    'split': ('legal-case-split-v1-20261022', 'b8ef5070b4010aa35349f4b9eb7ff8c5013f239a2eb7d7622bacafba5f3b7085'),
    'split_cold': ('legal-case-split-readback-v1-20261022', 'ba7966c922613f26896ef5735ad979f33b766bfb1b4b9641fa532b16d4996eff'),
    'cache': ('legal-target-cache-full-full-v2-20261023', '77319fe3eaac3ea9db2ede167519a418151133b5c4641ef19ce6c901d2d5fc72'),
    'cache_cold': ('legal-target-cache-full-cold-v2-20261023', 'd4498a44902db8e642ab368e1e1ead435e7e87caed44cae274245215b2a9a77f'),
}
DTYPE = np.dtype([(n, '<i8') for n in
    ('document', 'start', 'answer_start', 'stop', 'side', 'component', 'collection')] +
    [('windows', '<i8', (6,)), ('answer_windows', '<i8', (6,)), ('addresses', 'V32')])
POLICY = dict(prefix='min(2, source_atom_count-1)', order='attested_document_order',
    scope='source_document_continuation_not_question_answering', terminal='after_last_source_atom',
    fit_split='train', native_encoder_calls=0, model_updates=0, native_dimensions_per_lane=768,
    source_claims_are_world_facts=False, untouched_test=False, scientific_admission=False)


class Source:
    def __init__(self, root):
        self.root = Path(root); self.files = {}; self.stamps = {}; self.maps = []; self.closed = False
        self.reports = {n:self.report(self.root/name, pin) for n,(name,pin) in REFS.items()}
        r, s, sc, c, cc = [self.reports[n] for n in REFS]
        need(s['inputs']['corpus'] == r['corpus'] and sc['source']['pin'] == REFS['split'][1]
             and sc['all_atom_sides_reconstructed'], 'ANSWER_PLAN_SPLIT_AUTHORITY')
        need(c['mode'] == 'publish' and c['scope'] == 'full' and cc['mode'] == 'cold'
             and cc['previous_pin'] == REFS['cache'][1] and cc['binding'] == c['binding']
             and all(cc['result'][n] for n in ('all_rows_exact', 'all_exports_exact',
                 'all_rows_canonical_singleton_exact')), 'ANSWER_PLAN_CACHE_AUTHORITY')
        source_binding = c['binding']['source']
        need(source_binding['source'] == REFS['source'][1] and source_binding['split'] == REFS['split'][1]
             and source_binding['cold_split'] == REFS['split_cold'][1], 'ANSWER_PLAN_SOURCE_BINDING')
        self.roster = tuple(c['binding']['roster']); need(len(self.roster) == 6, 'ANSWER_PLAN_ROSTER')
        corpus = self.report(Path(r['corpus']['path']), r['corpus']['pin'])
        metadata = Path(r['corpus']['path'])/'documents.jsonl'
        self.bind(metadata, corpus['files']['documents.jsonl'])
        with metadata.open('rb') as stream: self.metadata = [json.loads(line) for line in stream]
        self.collections = sorted({v['source_path'] for v in self.metadata})
        self.documents = self.array('source', 'documents.npy'); self.mapping = self.array('source', 'mapping.npy')
        self.sides = self.array('split', 'document-side.npy'); self.components = self.array('split', 'component.npy')
        self.atom_sides = self.array('split', 'atom-side.npy'); self.counts = self.array('cache', 'export/counts.npy')
        n, u, d = (int(r['counts'][k]) for k in ('occurrences', 'unique_inputs', 'documents'))
        need(self.documents.shape == (d,2) and self.mapping.shape == self.atom_sides.shape == (n,)
             and self.sides.shape == self.components.shape == (d,) and len(self.metadata) == d
             and self.counts.shape == (u,6) and self.counts.dtype == np.uint8,
             'ANSWER_PLAN_SOURCE_SHAPES')
        need(all(x.dtype.kind in 'iu' for x in self.maps) and self.documents[0,0] == 0
             and self.documents[-1,1] == n and np.array_equal(self.documents[1:,0], self.documents[:-1,1])
             and bool(((self.documents[:,1]-self.documents[:,0]) >= 2).all())
             and bool((self.sides < 3).all()), 'ANSWER_PLAN_DOCUMENT_PARTITION')
        components = {}
        for i,row in enumerate(self.metadata):
            lo,hi = map(int,self.documents[i]); side = int(self.sides[i]); component = int(self.components[i])
            need((row['atom_start'],row['atom_end']) == (lo,hi)
                 and bool((self.atom_sides[lo:hi] == side).all()), 'ANSWER_PLAN_METADATA_BOUNDARY')
            need(component >= 0 and components.setdefault(component,side) == side, 'ANSWER_PLAN_CASE_LEAK')
        for lo in range(0,n,16384):
            ids = self.mapping[lo:lo+16384]
            need(bool(((ids >= 0) & (ids < u)).all()), 'ANSWER_PLAN_SOURCE_ADDRESS')
        need(bool((self.counts > 0).all()), 'ANSWER_PLAN_ABSENT_WINDOW')
        self.binding = dict(sources={n:p for n,(_,p) in REFS.items()}, corpus=r['corpus'],
            target=c['binding']['target_checkpoint'], native_bank=source_binding['bank'], roster=list(self.roster),
            collections=self.collections, exposure=s['exposure'], support=s['support'], policy=POLICY)
        self.fresh()

    def bind(self, path, ref):
        path = Path(path); pin = ref if isinstance(ref,str) else ref['sha256']
        before = stamp(path)
        if path in self.stamps:
            need(self.files[str(path)]['sha256'] == pin and self.stamps[path] == before, 'ANSWER_PLAN_REBIND')
            return
        need((isinstance(ref,str) or before[2] == ref['bytes']) and file_sha(path) == pin
             and stamp(path) == before, 'ANSWER_PLAN_FILE_PIN')
        self.stamps[path] = before; self.files[str(path)] = dict(sha256=pin, bytes=before[2])

    def report(self, root, pin):
        path = root/'report.json'; self.bind(path,pin); value = json.loads(path.read_bytes())
        need(value['complete'], 'ANSWER_PLAN_INCOMPLETE_SOURCE'); return value

    def array(self, key, name):
        path = self.root/REFS[key][0]/name; self.bind(path,self.reports[key]['files'][name])
        value = np.load(path,mmap_mode='r',allow_pickle=False); self.maps.append(value); return value

    def fresh(self, physical=False):
        need(not self.closed, 'ANSWER_PLAN_SOURCE_CLOSED')
        if physical:
            need(all(file_sha(Path(p)) == v['sha256'] for p,v in self.files.items()), 'ANSWER_PLAN_FINAL_PIN')
        need(all(stamp(p) == s for p,s in self.stamps.items()), 'ANSWER_PLAN_SOURCE_CHANGED')

    def close(self):
        try: self.fresh(physical=True)
        finally:
            for a in self.maps: a._mmap.close()
            self.closed = True


def row(source, number):
    need(type(number) is int and 0 <= number < len(source.documents), 'ANSWER_PLAN_DOCUMENT')
    start,stop = map(int,source.documents[number]); answer = start+min(2,stop-start-1)
    uids = source.mapping[start:stop]; values = source.counts[uids]
    result = np.zeros((),dtype=DTYPE)
    for key,value in dict(document=number,start=start,answer_start=answer,stop=stop,
        side=int(source.sides[number]), component=int(source.components[number]),
        collection=source.collections.index(source.metadata[number]['source_path'])).items(): result[key] = value
    result['windows'] = values.sum(0,dtype=np.int64)
    result['answer_windows'] = values[answer-start:].sum(0,dtype=np.int64)
    result['addresses'] = hashlib.sha256(uids.tobytes()).digest()
    return result


def validate(source, value):
    need(value.dtype == DTYPE and np.array_equal(value,row(source,int(value['document']))), 'ANSWER_PLAN_ROW_CHANGED')


class Plan:
    def __init__(self, source, rows):
        need(rows.dtype == DTYPE and rows.shape == (len(source.documents),)
             and np.array_equal(rows['document'],np.arange(len(rows))), 'ANSWER_PLAN_DOCUMENT_COVERAGE')
        self.source,self.rows = source,rows

    def _row(self, document):
        self.source.fresh()
        need(type(document) is int and 0 <= document < len(self.rows), 'ANSWER_PLAN_DOCUMENT')
        value = self.rows[document]; validate(self.source,value); return value

    def request(self, document):
        value = self._row(document); lo,hi = int(value['start']),int(value['answer_start'])
        # No answer length, terminal position, component, class label or source ID enters a model request.
        return dict(operation='continue_observed_source', position_kind='attested_document_order',
                    observed_uids=tuple(map(int,self.source.mapping[lo:hi])))

    def supervision(self, document, *, fitting, tile=128):
        value = self._row(document)
        need(type(fitting) is bool and (not fitting or int(value['side']) == 0), 'ANSWER_PLAN_NONTRAIN_FIT')
        need(type(tile) is int and 0 < tile <= 128, 'ANSWER_PLAN_TILE_MEMORY')
        start,answer,stop = (int(value[k]) for k in ('start','answer_start','stop'))
        for lo in range(answer-1,stop,tile):
            hi = min(lo+tile,stop); current = np.arange(lo,hi,dtype='<i8')
            end = current == stop-1; targets = np.where(end,-1,current+1)
            uids = np.full(len(current),-1,dtype='<i8'); uids[~end] = self.source.mapping[targets[~end]]
            yield dict(source_start=start, decision_start=lo, current=current, target=targets,
                target_uid=uids, end=end, requires_carried_state=lo>answer-1)
        self.source.fresh()


def build(source, path):
    result = np.lib.format.open_memmap(path,mode='w+',dtype=DTYPE,shape=(len(source.documents),))
    for i in range(len(result)): result[i] = row(source,i)
    result.flush(); result._mmap.close(); source.fresh()

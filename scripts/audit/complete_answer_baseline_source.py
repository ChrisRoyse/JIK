"""Complete recorded answer spans and public requests, without native/model loads."""
from copy import deepcopy
from pathlib import Path

from model.capability_answer_targets import AnswerTargets
from claims.episodic_memory import digest as row_digest
from text.capability_answer_source import stamp
from text.constellation_corpus import digest, file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
REF = ('capability-answer-sequences-v1-20261023', 'ea03479741a29a5b97cdb0477ccb8f56bac87c9bf23c9f3d57559c87c509bef7')
NEW = ('model/complete_answer_baselines.py', 'audit/complete_answer_baseline_source.py',
       'audit/complete_answer_baselines.py', 'audit/complete_answer_baselines_readback.py')


def load(root):
    view = AnswerTargets(root/REF[0], REF[1]); bank = []; queries = []; truth = {}
    def answer(key, fitting):
        a = view.target(key, fitting=fitting); pop, _, _ = view.source.selected(key)
        raw = pop.fields[a['source_field']]['text'].encode(); lo, hi = a['reference_span']; text = raw[lo:hi]
        need(digest(text) == a['reference_sha256'], 'COMPLETE_BASELINE_SOURCE_SPAN')
        return dict(text=text.decode(), sha256=a['reference_sha256'], atoms=deepcopy(a['atoms']),
                    end=deepcopy(a['end']), reference_span=list(a['reference_span']), source_field=a['source_field'])
    for split, keys in (('train', view.source.train), ('query', view.source.query)):
        for key in keys:
            r = view.rows[key]; public = view.request(key)
            row = dict(task=key, split=split, group=r['split_group'], family=r['family'],
                       public=dict(operation=public['operation'], fields=[{n:f[n] for n in ('role','text')}
                                                                        for f in public['context']]))
            if split == 'train': bank.append(dict(row, answer=answer(key, True)))
            else: queries.append(row); truth[key] = answer(key, False)
    need(len(bank) == 319 and len(queries) == 170
         and not {r['group'] for r in bank} & {r['group'] for r in queries}, 'COMPLETE_BASELINE_POPULATION')
    base = Path(__file__).resolve().parents[1]
    names = (*NEW, 'model/capability_baselines.py', 'model/capability_answer_targets.py', 'text/capability_answer_source.py')
    code = {n:file_sha(base/n) for n in names}
    paths = [*view.source.paths, *(view.root/n for n in (*view.report['files'], 'report.json', 'COMMITTED'))]
    inputs = {str(p):dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in paths}
    def fresh():
        view.source.fresh()
        need(all(stamp(p) == value for p,value in view.stamps.items()), 'ANSWER_PUBLICATION_CHANGED')
        need(all(row_digest(row) == view.seals[key] for key,row in view.rows.items()), 'ANSWER_MUTATED_VIEW')
        for n,h in code.items(): need(file_sha(base/n) == h, 'COMPLETE_BASELINE_CODE_CHANGED')
        for n,ref in inputs.items(): need(file_sha(n) == ref['sha256'], 'COMPLETE_BASELINE_SOURCE_CHANGED')
    return dict(view=view, bank=bank, queries=queries, truth=truth, code=code, inputs=inputs, fresh=fresh)


def score(candidate, target):
    available = candidate['status'] == 'candidate'; length_only = candidate['status'] == 'length_only'
    n = len(candidate['answer']['atoms']) if available else candidate.get('atoms') if length_only else None
    return dict(available=available, proposed_atoms=n, expected_atoms=len(target['atoms']),
        exact_recorded_bytes=available and candidate['answer']['text'] == target['text'],
        exact_measurement_sequence=available and [x['measurement'] for x in candidate['answer']['atoms']] ==
                                               [x['measurement'] for x in target['atoms']],
        length_match=n == len(target['atoms']), correctness_established=False)


def summary(rows):
    result = {}
    for dimension in ('family', 'operation'):
        result[dimension] = {}
        for value in sorted({r[dimension] for r in rows}):
            group = [r for r in rows if r[dimension] == value]; cells = {}
            for method in ('bm25','modal','exact','length_mode'):
                predictions = [r['arms'][method] for r in group]
                cells[method] = dict(tasks=len(group), groups=len({r['group'] for r in group}),
                    **{n:sum(v['metrics'][n] for v in predictions) for n in
                       ('available','exact_recorded_bytes','exact_measurement_sequence','length_match')},
                    clarification_required=sum(v['prediction']['status'] == 'clarification_required' for v in predictions),
                    unavailable=sum(v['prediction']['status'] == 'unavailable' for v in predictions))
            result[dimension][value] = cells
    return result

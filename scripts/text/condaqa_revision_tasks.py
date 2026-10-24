"""Attested reading responses and counterfactual evidence revisions, never dialogue fiction."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.capability_tasks import TaskBuilder, check
from text.condaqa_source import FILES, REPO, REVISION, git_blob
from text.evaluation_multiscale import file_sha


TRAIN_PIN = 'f5f8a9d1b64a6c3b111b00d2e125f1d7f4da8b50acfdbc788d2901e2e43d79ec'
EDIT_KINDS = {0: 'original', 1: 'paraphrase', 2: 'negation_scope', 3: 'affirmative'}


def category(label):
    return label if label in ('YES', 'NO', "DON'T KNOW") else 'OTHER'


def index(rows):
    groups, cells, counts = defaultdict(dict), defaultdict(set), Counter()
    for i, row in enumerate(rows):
        key, edit = (row['PassageID'], row['QuestionID']), row['PassageEditID']
        check(edit not in groups[key] and edit in EDIT_KINDS, 'CONDA_EDIT_KEY')
        groups[key][edit] = (i, row)
    for (passage, _), group in groups.items():
        check(set(group) == set(EDIT_KINDS), 'CONDA_INCOMPLETE_EDITS')
        check(len({r['sentence2'] for _, r in group.values()}) == 1, 'CONDA_QUESTION_CHANGE')
        for edit in (1, 2, 3):
            cell = (edit, category(group[0][1]['label']), category(group[edit][1]['label']))
            cells[passage].add(cell)
            counts[cell] += 1
    return groups, cells, counts


def choose(cells, counts):
    """Coverage selection is exposed, deterministic and independent of model scores."""
    uncovered, selected = set(counts), []
    while uncovered:
        passage = min(cells, key=lambda p: (-len(cells[p] & uncovered), p))
        check(bool(cells[passage] & uncovered), 'CONDA_UNCOVERED_CELL')
        selected.append(passage)
        uncovered -= cells[passage]
    return selected


def build(raw_root, output):
    tick = start = time.perf_counter()
    pins, training_bytes = {}, None
    for name, (_, size, blob) in FILES.items():
        payload = (raw_root / name).read_bytes()
        check(len(payload) == size and git_blob(payload) == blob, 'CONDA_SOURCE_PIN')
        pins[name] = hashlib.sha256(payload).hexdigest()
        if name == 'train.json':
            training_bytes = payload
    check(pins['train.json'] == TRAIN_PIN, 'CONDA_TRAIN_PIN')
    rows = [json.loads(line) for line in training_bytes.splitlines()]
    groups, cells, counts = index(rows)
    selected = choose(cells, counts)
    phases = dict(source_parse_and_coverage=time.perf_counter() - tick)
    source = dict(source_kind='condaqa-evidence-revision-v1', root=str(raw_root), files=pins,
        repository=REPO, revision=REVISION, licence='Apache-2.0 provider release',
        producer_sha256=file_sha(Path(__file__)),
        source_schema_sha256=file_sha(Path(__file__).with_name('condaqa_source.py')),
        selected_passages=selected, encoder_exposure_certified=False,
        judgment_scope='training question-author answers; not majority-verifier judgments',
        real_world_events_established=False, actual_dialogue=False, actual_revision_chronology=False,
        selection=dict(rule='greedy uncovered transition cells; numeric passage ID tie-break',
                       categories=['YES', 'NO', "DON'T KNOW", 'OTHER'],
                       model_score_used=False, performance_sample=False))
    tick = time.perf_counter()
    builders = {family: TaskBuilder(family, source) for family in ('conversation', 'reasoning')}
    transitions, answers = Counter(), Counter()
    for (passage, question), variants in sorted(groups.items()):
        if passage not in selected:
            continue
        group_id = 'condaqa:' + TRAIN_PIN + ':passage:' + str(passage)
        field_sets = {}
        for family, out in builders.items():
            field_sets[family] = {}
            for edit, (row_index, row) in sorted(variants.items()):
                address = dict(artifact_sha256=TRAIN_PIN, row=row_index,
                    sample_id=row['SampleID'], passage_id=passage, question_id=question, edit_id=edit)
                field_sets[family][edit] = [out.field(row[name], role, dict(address, field=name))
                    for name, role in (('sentence1', 'source_passage'),
                                       ('sentence2', 'source_question'), ('label', 'source_answer'))]
        for edit, (row_index, row) in sorted(variants.items()):
            provenance = dict(artifact_sha256=TRAIN_PIN, row=row_index,
                original_row=variants[0][0], passage_id=passage, question_id=question,
                edit_id=edit, edit_kind=EDIT_KINDS[edit], temporal_order_asserted=False,
                source_judgment_is_world_fact=False, human_judgment_checked=False)
            out = builders['conversation']
            p, q, a = field_sets['conversation'][edit]
            for origin, kind in ((p, 'source_answer_from_passage'), (q, 'source_answer_for_question')):
                out.edge(origin, a, kind, provenance)
            out.task(context=[p, q], targets=[a], consequences=[], group=group_id,
                kind='respond_to_source_question', provenance=provenance,
                checks=[dict(scope='attested_question_author_answer', status='source_relative',
                             dialogue_history_available=False, answer_correctness_checked=False)])
            answers[row['label']] += 1
            if edit == 0:
                continue
            out = builders['reasoning']
            old_p, old_q, old_a = field_sets['reasoning'][0]
            new_p, new_q, new_a = field_sets['reasoning'][edit]
            out.edge(old_p, old_a, 'source_answer_from_passage', provenance)
            out.edge(old_q, old_a, 'source_answer_for_question', provenance)
            out.edge(new_p, new_a, 'source_answer_from_passage', provenance)
            out.edge(new_q, new_a, 'source_answer_for_question', provenance)
            out.edge(old_p, new_p, 'provider_passage_edit', provenance)
            out.edge(old_a, new_a, 'provider_answer_revision', provenance)
            out.task(context=[old_p, old_q, old_a, new_p], targets=[new_a], consequences=[],
                group=group_id, kind='revise_with_source_evidence', provenance=provenance,
                checks=[dict(scope='paired_original_and_provider_edited_passage', status='source_relative',
                    hidden_reasoning_fabricated=False, independent_entailment_checked=False)])
            transitions[(edit, category(variants[0][1]['label']), category(row['label']))] += 1
    phases['connected_fields_and_tasks'] = time.perf_counter() - tick
    coverage = dict(source_rows=len(rows), source_passages=len(cells), source_questions=len(groups),
        selected_passages=len(selected), selected_questions=sum(p in selected for p, _ in groups),
        source_transition_cells=[dict(cell=list(c), n=n) for c, n in sorted(counts.items())],
        selected_transition_cells=[dict(cell=list(c), n=n) for c, n in sorted(transitions.items())],
        observed_cells_covered=set(transitions) == set(counts), reading_answers=dict(answers),
        semantic_labels_invented=False, source_relative_abstention_only=True,
        cross_family_group_registration_required=True)
    check(coverage['observed_cells_covered'], 'CONDA_CELL_COVERAGE')
    tick = time.perf_counter()
    results = {family: out.publish(output / family,
        scope='author judgments over original/counterfactual passages; no event or world-truth claim',
        coverage=coverage) for family, out in builders.items()}
    phases['publication'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    return dict(populations=results, coverage=coverage, phases_seconds=phases,
                wall_seconds=wall, phase_coverage=sum(phases.values()) / wall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.raw_root, args.output)))

"""Group-held-out development readback of correct/shuffled kernel learning."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import numpy as np
import torch
from audit.capability_kernel_learning import system
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, predict, pinned
from model.capability_objective import next_constellation_loss, native_similarity
from model.loop_training_checkpoint import tensor_fingerprint


def read(path): return json.loads(Path(path).read_bytes())


def summary(rows):
    families = {}
    for family in sorted({r['family'] for r in rows}):
        selected = [r for r in rows if r['family'] == family]
        families[family] = dict(tasks=len(selected), groups=len({r['group'] for r in selected}), arms={})
        for arm in ('initial', 'correct', 'shuffled'):
            values = [r['arms'][arm] for r in selected]
            families[family]['arms'][arm] = dict(
                surface_loss=float(np.mean([v['surface_loss'] for v in values])),
                region_accuracy=float(np.mean([v['region_correct'] for v in values])),
                native_similarity={n:float(np.mean([v['native_similarity'][n] for v in values]))
                                   for n in values[0]['native_similarity']},
                reciprocal_loss={n:float(np.mean([v['reciprocal_loss'][n] for v in values]))
                                 for n in values[0]['reciprocal_loss']})
        for baseline in ('initial', 'shuffled', 'train_unigram'):
            differences = [r['arms']['correct']['surface_loss'] - (r['unigram_loss'] if baseline == 'train_unigram'
                           else r['arms'][baseline]['surface_loss']) for r in selected]
            families[family]['correct_vs_' + baseline] = dict(
                better=sum(v < 0 for v in differences), worse=sum(v > 0 for v in differences),
                ties=sum(v == 0 for v in differences), mean_delta=float(np.mean(differences)))
    return families


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--experiments', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'LEARNING_EVAL_EXISTS'); a.output.mkdir()
    started = time.perf_counter(); config = read(a.config); definitions = read(a.experiments)
    corpus = KernelLearningCorpus(config); reports = {n:pinned(v) for n,v in definitions.items()}
    need(set(reports) == {'correct', 'shuffled'} and reports['correct']['source'] == reports['shuffled']['source']
         and reports['correct']['initial_parameters'] == reports['shuffled']['initial_parameters'],
         'LEARNING_EVAL_MATCHED_START')
    roots = {n:Path(v['path']).parent for n,v in definitions.items()}
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-kernel-development-evaluation'); configure()
    model = system(config, corpus); world = model['world']
    checkpoints = {'initial':roots['correct'] / 'initial.pt',
                   **{n:roots[n] / 'final.pt' for n in reports}}
    weights = {}
    for arm, path in checkpoints.items():
        if arm != 'initial': need(sha(path.read_bytes()) == reports[arm]['final_checkpoint'], 'LEARNING_EVAL_PIN')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        need(saved['source'] == reports['correct']['source'], 'LEARNING_EVAL_CHECKPOINT_SOURCE')
        weights[arm] = saved['model']
    # Explicit baseline fitted on exactly the same complete source target bytes.
    counts = defaultdict(lambda: np.ones(258, np.int64))
    for key in corpus.train:
        task = corpus.tasks[key]; field = task['selections']['targets']['fields'][0]['id']
        raw = corpus.sources[task['population']].population.fields[field]['text'].encode()
        counts[task['kind']] += np.bincount([*raw, 257], minlength=258)
    rows = []
    with torch.no_grad(), (a.output / 'queries.jsonl').open('x') as output_file:
        for index, key in enumerate(corpus.query):
            task = corpus.tasks[key]; native = corpus.batch(key, 'context', world.config, 'cuda')
            raw, target = corpus.target(key, world.config, 'cuda', fitting=False)
            row = dict(task=key, family=task['family'], operation=task['kind'], group=task['split_group'],
                       source_sha256=sha(raw), source_bytes=len(raw), arms={})
            c = counts[task['kind']]; probability = c / c.sum()
            row['unigram_loss'] = float(-np.log(probability[[*raw, 257]]).mean())
            row['persistence_similarity'] = {s.name:float(native_similarity(native[s.name].values[0],
                                                        target[s.name].values[0])) for s in world.specs}
            for arm in weights:
                model.load_state_dict(weights[arm], strict=True)
                result = predict(world, model['request'], native, key=key, operation=task['kind'])
                _, meaning = next_constellation_loss(world, result, target)
                surface, _ = model['decoder'].render_loss(raw, result['predicted_constellation'])
                expected = meaning['label']; actual = int(result['constellation_regions'].argmax(-1))
                similarities = {s.name:float(native_similarity(
                    result['predicted_constellation'].slots[s.name].values[0], target[s.name].values[0]))
                    for s in world.specs}
                reciprocal = {}
                for spec in world.specs:
                    visible = {n:x for n,x in native.items() if n != spec.name}
                    prediction = model['reciprocal'](visible, hidden=spec.name, windows=native[spec.name].values.shape[1])
                    reciprocal[spec.name] = float(model['reciprocal'].loss(prediction['prediction'], native[spec.name].values))
                row['arms'][arm] = dict(surface_loss=surface['loss'], scored_positions=surface['scored_positions'],
                    region_correct=actual == expected, predicted_region=actual, target_region=expected,
                    native_similarity=similarities, reciprocal_loss=reciprocal)
            need(len({v['scored_positions'] for v in row['arms'].values()}) == 1,
                 'LEARNING_EVAL_SUPPORT')
            output_file.write(json.dumps(row, sort_keys=True) + '\n'); output_file.flush(); rows.append(row)
            print(json.dumps(dict(query=index, family=task['family'], bytes=len(raw))), flush=True)
    need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin'], 'LEARNING_EVAL_JOURNAL')
    result = dict(schema='capability-kernel-development-evaluation-v1', experiments=definitions,
        config_sha256=sha(a.config.read_bytes()), query_records_sha256=sha((a.output / 'queries.jsonl').read_bytes()),
        queries=len(rows), families=summary(rows), all_native_windows_and_surface_bytes_scored=True,
        correct_and_null_start_identical=True, gpu_before=gpu, seconds=time.perf_counter()-started,
        scientific_admission=False, learned_capability=False, serving=False,
        sealed_capability_evaluation=False, generated_surface_evaluated=False,
        exposure='development_exposed_group_holdout', number_status='measured', acceptance='author-checked',
        limits=['Native similarity is bidirectional coverage, not order equivalence or world truth.',
                'Region labels are frozen native geometry, not external semantic correctness.',
                'Complete teacher-forced surface loss does not demonstrate free generation.',
                'Operation comes from declared task metadata; parsing new user requests remains unproved.'])
    print(json.dumps(dict(pin=write_committed(a.output / 'report.json', result), families=result['families'])), flush=True)


if __name__ == '__main__': run_cli(main)

"""Capture only the pinned original human-written GSM8K training release."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.condaqa_source import git_blob
from text.connected_source_capture import capture, LIMITS
from text.evaluation_multiscale import file_sha

REPO = 'openai/grade-school-math'
REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
FILES = {
    'LICENSE': ('LICENSE', 1062, '9e84fcbc4d81a1f433c90caf9f1cef373c12edae'),
    'README.md': ('README.md', 6112, 'f5d12ef45cc7b78dbc148cbd060994decbc3159d'),
    'train.jsonl': ('grade_school_math/data/train.jsonl', 4166206,
                    '7d97154b91aef28d01c3301741c81ce90039a4b1'),
}


def check(ok, code):
    if not ok:
        raise ValueError('MNLM_GSM8K_SOURCE_' + code)


def inventory(payload):
    records = [json.loads(line) for line in payload.splitlines()]
    check(bool(records) and all(set(r) == {'question', 'answer'} for r in records), 'FIELDS')
    check(all(isinstance(r[k], str) and r[k].strip() for r in records for k in r), 'TEXT')
    check(all(r['answer'].count('####') == 1 for r in records), 'ANSWER_DELIMITER')
    return dict(rows=len(records), unique_questions=len({r['question'] for r in records}),
                answer_steps=dict(sorted(Counter(len(r['answer'].splitlines()) - 1
                                               for r in records).items())),
                max_question_bytes=max(len(r['question'].encode()) for r in records),
                max_answer_bytes=max(len(r['answer'].encode()) for r in records))


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    receipts, phases, errors, inv = {}, {}, [], None
    try:
        receipts['provider-tree.json'] = capture(
            f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1',
            output / 'provider-tree.json', LIMITS['metadata_bytes'])
        tree = json.loads((output / 'provider-tree.json').read_bytes())
        check(tree['sha'] == REVISION and tree['truncated'] is False, 'TREE_REVISION')
        by_path = {r['path']: r for r in tree['tree']}
        for path, size, blob in FILES.values():
            entry = by_path[path]
            check(entry['type'] == 'blob' and entry['size'] == size and entry['sha'] == blob,
                  'TREE_FILE_PIN')
        phases['provider_tree'] = time.perf_counter() - start
        tick = time.perf_counter()
        for name, (path, size, blob) in FILES.items():
            receipts[name] = capture(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}',
                                     output / name, size)
            raw = (output / name).read_bytes()
            check(len(raw) == size and git_blob(raw) == blob, 'FILE_PIN')
        phases['allowlisted_capture'] = time.perf_counter() - tick
        tick = time.perf_counter()
        licence = (output / 'LICENSE').read_text()
        check('MIT License' in licence and 'Copyright (c) 2021 OpenAI' in licence, 'LICENCE')
        readme = (output / 'README.md').read_text()
        check('human problem writers' in readme and 'train.jsonl' in readme, 'AUTHOR_RELEASE')
        inv = inventory((output / 'train.jsonl').read_bytes())
        publish(output / 'inventory.json', inv)
        phases['licence_and_inventory'] = time.perf_counter() - tick
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append(dict(type=type(exc).__name__, error=str(exc)))
    tick = time.perf_counter()
    publish(output / 'capture.json', dict(repository=REPO, revision=REVISION, receipts=receipts,
        selected_source_files={n: dict(path=p, bytes=s, git_blob_sha1=b)
                               for n, (p, s, b) in FILES.items()}, credentials_used=False))
    files = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
             for p in output.iterdir() if p.is_file()}
    phases['publication_and_readback'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    report = dict(schema='gsm8k-human-training-capture-v1', complete=not errors, errors=errors,
        source=REPO, revision=REVISION, files=files, licence='MIT', attribution='OpenAI, 2021',
        scope='original provider question and human answer; derivation not yet independently checked',
        encoder_exposure_certified=False, sealed_evaluation=False, training_ready=False,
        heldout_payloads_obtained=False, model_generated_solutions_obtained=False,
        scientific_admission=False, served=False, code_sha256=file_sha(Path(__file__)),
        phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values()) / wall)
    publish(output / 'report.json', report)
    print(json.dumps(dict(complete=not errors, errors=errors, inventory=inv,
                          report_sha256=file_sha(output / 'report.json'), wall_seconds=wall)))
    return 2 if errors else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(build(parser.parse_args().output))

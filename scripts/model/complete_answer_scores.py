"""Post-generation whole-answer accounting; all absent positions remain scored."""
import json
from pathlib import Path

import numpy as np

from model.complete_answer_trace import replay
from text.constellation_corpus import need


def native_similarity(x, y):
    a, b = x.double().numpy(), y.double().numpy()
    an, bn = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
    need(np.isfinite(a).all() and np.isfinite(b).all() and np.all(an > 0) and np.all(bn > 0),
         'ANSWER_SCORE_NATIVE')
    score = np.clip((a/an)@(b/bn).T, -1, 1)
    return float(min(score.max(1).mean(), score.max(0).mean()))


def score(output, pin, *, future, roster, copy_frame):
    need(len(future) > 0 and set(copy_frame) == set(roster), 'ANSWER_SCORE_REFERENCE')
    expected = len(future); generated = 0; ended = False
    slots = {n:[-1.]*expected for n in roster}
    copy = {n:[native_similarity(copy_frame[n], f[n]) for f in future] for n in roster}
    for event in replay(output, pin):
        if event['kind'] == 'end': ended = True; continue
        if generated < expected:
            for n in roster: slots[n][generated] = native_similarity(event['slots'][n], future[generated][n])
        generated += 1
    report = json.loads((Path(output)/'report.json').read_bytes())
    extra = max(0, generated-expected); covered = min(generated, expected)
    return dict(expected_atoms=expected, generated_atoms=generated, ended=ended,
        ended_at_reference_boundary=ended and generated == expected, censored=not ended,
        matched_positions=covered, missing=expected-covered, extra=extra, per_slot=slots, copy_current=copy,
        # This extra-penalized score is separate from the existing reference-denominator score.
        full_output_mean={n:(sum(v)-extra)/(expected+extra) for n,v in slots.items()},
        missing_similarity=-1., extra_similarity=-1., refusal=report['refusal'],
        reference_scored_after_generation=True, semantic_validity_established=False)

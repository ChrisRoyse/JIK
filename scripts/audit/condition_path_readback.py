"""Independent array-level readback of a sealed-input condition-path trace."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def require(value, code):
    if not value:
        raise ValueError('CONDITION_READBACK_'+code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, code, tolerance=2e-5):
    require(np.shape(a) == np.shape(b), code+'_SHAPE')
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    require(error <= tolerance, code)
    return error


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    value = np.exp(x-x.max(axis=-1, keepdims=True))
    return value/value.sum(axis=-1, keepdims=True)


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))


def normalize(value):
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    report = json.loads((args.root/'report.json').read_bytes())
    require(sha(args.root/'arrays.npz') == report['arrays']['sha256'], 'ARRAY_PIN')
    arrays = np.load(args.root/'arrays.npz', allow_pickle=False)
    source = Path(report['source']['root'])
    require(sha(source/'seal.json') == report['source']['seal'], 'SOURCE_SEAL')
    entries = json.loads((source/'seal.json').read_bytes())['files']
    for name, entry in entries.items():
        require(sha(source/name) == entry['sha256'], 'SOURCE_BYTES')
    require(sha(source/'experience.sqlite3') == report['source']['journal'], 'JOURNAL')
    numeric, comparisons = [], []
    upstream = json.loads((source/'source.json').read_bytes())
    base = Path(upstream['root'])
    require(sha(base/'seal.json') == upstream['seal'], 'UPSTREAM_SEAL')
    base_entries = json.loads((base/'seal.json').read_bytes())['files']
    require(sha(base/'summary.json') == base_entries['summary.json']['sha256'], 'UPSTREAM_CONFIG')
    config = json.loads((base/'summary.json').read_bytes())['config']
    parent = Path(config['parent'])
    require(sha(parent/'run.json') == config['parent_pin'], 'PARENT_RUN')
    parent_pin = json.loads((parent/'run.json').read_bytes())['initial']
    frozen_projectors = None

    def weights(root, pin):
        require(sha(root/'manifest.json') == pin == (root/'COMMITTED').read_text(), 'WORLD_PIN')
        world = json.loads((root/'manifest.json').read_bytes())
        require(sha(root/'world.pt') == world['payload_sha256'], 'WORLD_PAYLOAD')
        require(sha(root/'core/manifest.json') == world['core'], 'CORE_PIN')
        core = json.loads((root/'core/manifest.json').read_bytes())
        require(sha(root/'core/weights.pt') == core['files']['weights.pt']['sha256'], 'CORE_PAYLOAD')
        return [torch.load(p, weights_only=True, map_location='cpu') for p in
                (root/'core/weights.pt', root/'world.pt')]

    def read(trace, core, world, saved=None):
        saved = arrays if saved is None else saved
        lane_logits = []
        labels = np.array([r['outcome'] for r in trace['prediction']['bank']['rows']])
        errors = []
        for name, lane in trace['lanes'].items():
            t = {n:saved[k] for n,k in lane['arrays'].items()}
            require(t['native'].shape[-1] == 768 and t['query'].shape[-1] == 32, 'TYPED_WIDTH')
            require(np.isfinite(t['native']).all(), 'NATIVE_FINITE')
            projected = normalize(t['native'].astype(np.float64)@core['projectors.'+name+'.weight'].numpy().T
                                  +core['projectors.'+name+'.bias'].numpy())
            errors.append(close(projected, t['projected'], 'PROJECTOR'))
            raw_query = (t['context']+t['projected']).astype(np.float64)@world['query.'+name+'.weight'].numpy().T
            raw_query += world['query.'+name+'.bias'].numpy()
            errors.append(close(raw_query, t['raw_query'], 'QUERY'))
            errors.append(close(normalize(raw_query), t['query'], 'QUERY_NORMALIZATION'))
            scores = 10*np.einsum('bwd,rkd->bwrk', t['query'].astype(np.float64),
                                 t['keys'].astype(np.float64))
            scores = np.where(t['reference_padding'][None,None,:,:], -np.inf, scores)
            matches = scores.max(axis=-1)
            errors.append(close(matches, t['matches'], 'MAXSIM'))
            logits = np.where(t['padding'][:,:,None], 0, matches).sum(axis=1)/(~t['padding']).sum(axis=1)[:,None]
            errors.append(close(logits, t['regions'], 'REDUCTION'))
            lane_logits.append(t['regions'].astype(np.float64))
            probs = softmax(t['regions'])[0]
            errors.append(close(probs, trace['prediction']['instruments'][name]['region_probabilities'], 'LANE_PROB', 1e-12))
            masses = [probs[labels == i].sum() for i in range(3)]
            errors.append(close(masses, trace['prediction']['instruments'][name]['probabilities'], 'LANE_MASS', 1e-12))
        # The implementation sums per-lane log-softmax, a product of scalar
        # evidence. No native vectors from different instruments are pooled.
        common = sum(np.log(softmax(x)) for x in lane_logits)
        errors.append(close(common, saved[trace['arrays']['common']], 'COMMON_LOGITS'))
        probs = softmax(saved[trace['arrays']['common']])[0]
        errors.append(close(probs, trace['prediction']['common']['region_probabilities'], 'COMMON_PROB', 1e-12))
        errors.append(close([probs[labels == i].sum() for i in range(3)],
                            trace['prediction']['common']['probabilities'], 'COMMON_MASS', 1e-12))
        return max(errors)

    for model in report['models']:
        if model['name'] == 'parent':
            core, world = weights(parent/'initial', parent_pin)
        else:
            candidate = source/'candidates'/('rehearsal-'+str(['first','second'].index(model['name'])+1))
            require(sha(candidate/'manifest.json') == model['checkpoint'], 'CANDIDATE_PIN')
            m = json.loads((candidate/'manifest.json').read_bytes())
            core, world = weights(candidate/'model', m['model'])
        projectors = {n:v.numpy() for n,v in core.items() if n.startswith('projectors.')}
        if frozen_projectors is not None:
            require(all(np.array_equal(v, frozen_projectors[n]) for n,v in projectors.items()), 'FROZEN_PROJECTORS')
        frozen_projectors = projectors
        for trace in [*model['actual'], *model['counterfactuals'].values()]:
            numeric.append(read(trace, core, world))
        first, second = model['actual']
        lanes = {}
        for name in first['lanes']:
            x = {n:arrays[k] for n,k in first['lanes'][name]['arrays'].items()}
            y = {n:arrays[k] for n,k in second['lanes'][name]['arrays'].items()}
            sub = model['counterfactuals']['condition_only']['lanes'][name]
            z = {n:arrays[k] for n,k in sub['arrays'].items()}
            reverse = model['counterfactuals']['condition_reverse']['lanes'][name]
            rev = {n:arrays[k] for n,k in reverse['arrays'].items()}
            fields = [i['binding']['fields'][name] for i in report['inputs']]
            cond_counts = [max(f['window_end'] for f in group) for group in fields]
            action_counts = [x['native'].shape[1]-cond_counts[0], y['native'].shape[1]-cond_counts[1]]
            require(np.array_equal(x['native'][:,:action_counts[0]], z['native'][:,:action_counts[0]]), 'FIXED_ACTION')
            require(np.array_equal(y['native'][:,action_counts[1]:], z['native'][:,action_counts[0]:]), 'CONDITION_SWAP')
            require(np.array_equal(x['native'][:,action_counts[0]:], rev['native'][:,action_counts[1]:]), 'CONDITION_REVERSE')
            matches = {}
            for field in fields[0]:
                pair = next(v for v in fields[1] if v['text_sha256'] == field['text_sha256'])
                lo, hi = action_counts[0]+field['window_start'], action_counts[0]+field['window_end']
                a, b = action_counts[1]+pair['window_start'], action_counts[1]+pair['window_end']
                require(np.array_equal(x['native'][:,lo:hi], y['native'][:,a:b]), 'SHARED_FIELD_BYTES')
                matches[field['field']] = dict(windows=hi-lo, native_equal=True)
            new_fields = [f for f in fields[1] if f['text_sha256'] not in {v['text_sha256'] for v in fields[0]}]
            require(len(new_fields) == 1 and new_fields[0]['field'] == 'wheel_filename', 'ONE_ENVIRONMENT_FIELD')
            added = new_fields[0]
            at = action_counts[1]+added['window_start']
            original_wheel = action_counts[0]+next(v['window_start'] for v in fields[0] if v['field'] == 'wheel_filename')
            lanes[name] = dict(action_windows=action_counts, condition_windows=cond_counts,
                shared_fields=matches, added_native_vs_existing_wheel_cosine=cosine(y['native'][:,at], x['native'][:,original_wheel]),
                added_projected_vs_existing_wheel_cosine=cosine(y['projected'][:,at], x['projected'][:,original_wheel]),
                added_query_vs_existing_wheel_cosine=cosine(y['query'][:,at], x['query'][:,original_wheel]),
                condition_only_core_hidden_l2=float(np.linalg.norm(arrays[model['counterfactuals']['condition_only']['arrays']['core_hidden']].astype(np.float64)-arrays[first['arrays']['core_hidden']].astype(np.float64))))
        grads = {}
        for kind in ('brier', 'margin'):
            a, b = arrays[first[kind+'_gradient']], arrays[second[kind+'_gradient']]
            value = cosine(a,b)
            close(value, model['gradients'][kind]['cosine'], 'GRADIENT_COSINE', 1e-12)
            grads[kind] = value
        for i, actual in enumerate(model['actual']):
            probs = np.array(actual['prediction']['common']['probabilities'])
            truth = np.eye(3)[[1,0][i]]
            close(((probs-truth)**2).sum(), actual['brier'], 'BRIER', 1e-12)
        comparisons.append(dict(model=model['name'], lanes=lanes, gradients=grads,
            common_probabilities=[v['prediction']['common']['probabilities'] for v in model['actual']],
            substitutions={n:v['prediction']['common']['probabilities'] for n,v in model['counterfactuals'].items()}))
    # Run the actual reconstruction reader on corrupted copies, not on new data.
    original = report['models'][-1]['actual'][0]
    negatives = {}
    for name,key in [('common_logit', original['arrays']['common']),
                     ('native_input', next(iter(original['lanes'].values()))['arrays']['native'])]:
        corrupt = dict(arrays)
        corrupt[key] = arrays[key].copy()
        corrupt[key].flat[0] += .1
        try:
            read(original, core, world, corrupt)
        except ValueError as error:
            negatives[name] = str(error)
        else:
            raise ValueError('CONDITION_READBACK_NEGATIVE_CONTROL')
    result = dict(complete=True, acceptance='author-checked', report=sha(args.root/'report.json'),
        reader_sha256=sha(Path(__file__)),
        arrays=sha(args.root/'arrays.npz'), source_files=len(entries), trace_count=len(numeric),
        maximum_reconstruction_error=max(numeric), models=comparisons, negative_controls=negatives,
        source_unchanged=True, scientific_admission=False, serving=False, seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'models'}))


if __name__ == '__main__':
    main()

"""Categorical/native prediction diagnostics; no information-theoretic admission."""
from collections import defaultdict

import numpy as np
import torch

from audit.legal_predictor_training import distance
from model.legal_cached_documents import categorical_baselines
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need


def probabilities(logits):
    values = logits.detach().cpu().double().numpy()
    values -= values.max(1, keepdims=True)
    values = np.exp(values); values /= values.sum(1, keepdims=True)
    return values


def categorical(probability, codes):
    rows = np.arange(len(codes)); ce = []; brier = []
    for level, p in enumerate(probability):
        need(np.isfinite(p).all() and bool((p >= 0).all()) and np.allclose(p.sum(1), 1, rtol=0, atol=1e-12),
             'CACHED_PREDICTION_PROBABILITY')
        target = codes[:, level]; selected = p[rows, target]
        need(bool((selected > 0).all()), 'CACHED_PREDICTION_ZERO_TARGET_PROBABILITY')
        ce.append(-np.log(selected)); brier.append((p*p).sum(1)-2*selected+1)
    return np.stack(ce, axis=1), np.stack(brier, axis=1)


@torch.no_grad()
def document(model, target, item, cache, continuation, stats, tile):
    frames = item['frames']; device = model.encoder.query.device
    n = len(frames)-1; prediction = []; counts = {k:[] for k in model.roster}
    forced = [[] for _ in model.sizes]; wrong = [[] for _ in model.sizes]
    native_distances = {k:[] for k in model.roster}; state = None; stream = model.stream()
    # This is a target-permutation diagnostic. Autonomous forecasts are computed first.
    perm = np.roll(np.arange(n), 1); permuted = item['codes'][torch.from_numpy(perm)]
    for lo in range(0, len(frames), tile):
        hi = min(lo+tile, len(frames))
        native = tuple({k:x.to(device) for k, x in f.items()} for f in frames[lo:hi])
        hidden, state = stream.step(native, context='evaluate:'+str(item['document']), offset=lo, state=state, detach_history=True)
        stop = min(hi, n)
        if stop <= lo: continue
        hidden = hidden[:stop-lo]; out = model.forecast(hidden)
        truth = item['codes'][lo:stop].to(device); window_targets = {k:x[lo:stop].to(device) for k,x in item['counts'].items()}
        actual = model.loss(hidden, truth, window_targets)
        shuffled = model.loss(hidden, permuted[lo:stop].to(device), window_targets)
        for level in range(len(model.sizes)):
            p = probabilities(actual['logits'][level]); forced[level].append(p)
            wrong[level].append(probabilities(shuffled['logits'][level]))
            score = -np.log(p[np.arange(stop-lo), truth[:, level].cpu().numpy()]).mean()
            need(abs(score-float(actual['region'][level])) <= 5e-7, 'CACHED_PREDICTION_CE_RECOUNT')
        saved = cpu(out); rendered = model.render(saved, target)
        prediction.append(saved['codes'].numpy())
        for name in model.roster:
            counts[name].append(saved['counts'][name].numpy())
            for j in range(stop-lo):
                future = frames[lo+j+1][name]
                native_distances[name].append([distance(future, rendered[j][name]),
                    distance(future, frames[lo+j][name]), distance(future, stats['centers'][name].float()[None])])
    need(state.next_atom == len(frames), 'CACHED_PREDICTION_INCOMPLETE_HISTORY')
    baseline, modal = categorical_baselines(cache, continuation, item['uids'][:-1])
    correct = [np.concatenate(x) for x in forced]; incorrect = [np.concatenate(x) for x in wrong]
    truth = item['codes'].numpy(); ce, brier = categorical(correct, truth)
    shuffled_ce, _ = categorical(incorrect, permuted.numpy())
    uid = np.asarray(item['uids'], dtype=np.int64)
    masks = dict(all=np.ones(n, dtype=np.bool_), target_absent_from_train=(continuation.presence[uid[1:]] & 1) == 0,
                 both_inputs_absent_from_train=((continuation.presence[uid[:-1]] & 1) == 0) & ((continuation.presence[uid[1:]] & 1) == 0),
                 permuted_target_distinct=uid[1:][perm] != uid[1:], exact_input_successor_available=modal >= 0)
    result = dict(document=item['document'], side=item['side'], uids=uid, target=truth,
                  predictions=dict(model=np.concatenate(prediction), **baseline),
                  probabilities=correct, permuted_probabilities=incorrect, permutation=perm,
                  conditional_ce=ce, conditional_brier=brier, permuted_target_ce=shuffled_ce,
                  masks=masks, modal_successor_uids=modal,
                  native={name:dict(predicted_count=np.concatenate(counts[name]), target_count=item['counts'][name].numpy(),
                                   ordered_distances=np.asarray(native_distances[name])) for name in model.roster},
                  native_pin=item['native_pin'], label_pin=item['label_pin'],
                  autonomous_counts=True, metric_is_shannon_information=False)
    need(fingerprint(item['frames']) == item['native_pin'], 'CACHED_PREDICTION_NATIVE_CHANGED')
    return result


def summary(results, roster):
    groups = defaultdict(list)
    for result in results:
        groups[(result['side'], result['source_collection'])].append(result)
        groups[(result['side'], 'ALL_COLLECTIONS')].append(result)
    out = {}
    for (side, collection), documents in groups.items():
        cells = {}
        for subset in ['all', 'target_absent_from_train', 'both_inputs_absent_from_train']:
            masks = [r['masks'][subset] for r in documents]; total = sum(int(m.sum()) for m in masks)
            if total == 0:
                cells[subset] = dict(rows=0, status='unmeasured_empty_support'); continue
            labels = np.concatenate([r['target'][m] for r,m in zip(documents,masks)])
            values = dict(rows=total, documents=sum(bool(m.any()) for m in masks), accuracy={}, prefix_accuracy={})
            for name in documents[0]['predictions']:
                predicted = np.concatenate([r['predictions'][name][m] for r,m in zip(documents,masks)])
                equal = labels == predicted
                values['accuracy'][name] = equal.mean(0).tolist()
                values['prefix_accuracy'][name] = np.logical_and.accumulate(equal, axis=1).mean(0).tolist()
                if name == 'root_transition':
                    values['accuracy'][name][1:] = [None]*(labels.shape[1]-1)
                    values['prefix_accuracy'][name][1:] = [None]*(labels.shape[1]-1)
            for key in ['conditional_ce', 'conditional_brier']:
                values[key] = np.concatenate([r[key][m] for r,m in zip(documents,masks)]).mean(0).tolist()
            contrast_masks = [m & r['masks']['permuted_target_distinct'] for r,m in zip(documents,masks)]
            contrast_n = sum(int(m.sum()) for m in contrast_masks)
            values['permuted_target_rows'] = contrast_n
            values['correct_minus_permuted_log_score'] = (np.concatenate([
                (r['permuted_target_ce']-r['conditional_ce'])[m] for r,m in zip(documents,contrast_masks)]).mean(0).tolist()
                if contrast_n else None)
            values['exact_input_successor_coverage'] = sum(int((r['masks']['exact_input_successor_available'] & m).sum())
                                                         for r,m in zip(documents,masks))/total
            values['native'] = {}
            for name in roster:
                distances = np.concatenate([r['native'][name]['ordered_distances'][m] for r,m in zip(documents,masks)])
                correct = sum(int(((r['native'][name]['predicted_count'] == r['native'][name]['target_count']) & m).sum())
                              for r,m in zip(documents,masks))
                values['native'][name] = dict(count_accuracy=correct/total,
                    **dict(zip(['ordered_model', 'ordered_copy_current', 'ordered_train_mean'], distances.mean(0).tolist())))
            cells[subset] = values
        out[side+':'+collection] = cells
    return out

"""Paired exposed-query diagnostics; all bytes and per-instrument arrays retained."""
from collections import defaultdict
import json
import math
import os

import numpy as np
import torch

from audit.capability_rescore_metrics import array, coverage, trace, difference
from claims.capability_population import need, sha
from model.capability_structure import predict_structured


def summarize(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r['arm'], r['operation'], 'all')].append(r)
        cells[(r['arm'], r['operation'], r['group'])].append(r)
    result = []
    for (arm, operation, group), values in sorted(cells.items()):
        cell = dict(arm=arm, operation=operation, group=group, tasks=len(values),
                    groups=len({r['group'] for r in values}), eligible=sum(r['eligible'] for r in values),
                    region_hits=sum(r['label'] == r['selected'] for r in values), lanes={})
        cell['mean_nll'] = math.fsum(r['correct']['canonical_loss'] for r in values) / len(values)
        cell['conditioning_delta'] = math.fsum(r['delta']['mean_byte_delta'] for r in values) / len(values)
        for n in values[0]['lanes']:
            ls = [r['lanes'][n] for r in values]
            cell['lanes'][n] = dict(mean_target=math.fsum(x['target'] for x in ls)/len(ls),
                wins=sum(x['target'] > x['distractor'] for x in ls),
                ties=sum(x['target'] == x['distractor'] for x in ls),
                losses=sum(x['target'] < x['distractor'] for x in ls))
        result.append(cell)
    return result


@torch.no_grad()
def evaluate(model, contexts, targets, labels, plan, arm, root):
    corpus = contexts.corpus; world = model['world']; records = []; predictions = {}
    for key in dict.fromkeys([*corpus.query, *plan['donors'].values()]):
        output = predict_structured(model, contexts.get(key, world.config, 'cuda'), corpus.tasks[key]['kind'])
        predictions[key] = (output['predicted_constellation'].slots,
                            int(next(iter(output['world_owner'].values()))[0]))
    with (root/'queries.jsonl').open('x') as stream:
        for index, key in enumerate(corpus.query):
            raw, target = targets[key]; donor = plan['donors'][key]
            pred, selected = predictions[key]; wrong, _ = predictions[donor]
            correct_trace, correct = trace(model['decoder'], raw, pred)
            wrong_trace, shuffled = trace(model['decoder'], raw, wrong)
            arrays = {tag+'__'+n:array(s).cpu().numpy() for tag, slots in [('predicted', pred), ('wrong', wrong)]
                      for n, s in slots.items()}
            path = root/(key+'.npz')
            np.savez(path, **arrays, correct=correct_trace, shuffled=wrong_trace)
            task = corpus.tasks[key]
            row = dict(arm=arm, task=key, operation=task['kind'], family=task['family'], group=task['split_group'],
                eligible=plan['decisions'][key]['allowed'], purpose=plan['decisions'][key]['purpose'],
                label=labels[key]['label'], label_ties=labels[key]['ties'], selected=selected, donor=donor,
                source_sha256=sha(raw), source_bytes=len(raw), file=path.name, file_sha256=sha(path.read_bytes()),
                correct=correct, shuffled=shuffled, delta=difference(correct_trace, wrong_trace), lanes={})
            for name, slot in pred.items():
                row['lanes'][name] = dict(target=coverage(array(slot), array(target[name])),
                    distractor=coverage(array(slot), array(targets[donor][1][name])))
            need(all(math.isfinite(v) for x in row['lanes'].values() for v in x.values()), 'EXPERIENCE_EVAL_FINITE')
            stream.write(json.dumps(row, sort_keys=True)+'\n'); stream.flush(); records.append(row)
            if index % 40 == 0: print(json.dumps(dict(stage='evaluation', arm=arm, index=index)), flush=True)
        os.fsync(stream.fileno())
    return records

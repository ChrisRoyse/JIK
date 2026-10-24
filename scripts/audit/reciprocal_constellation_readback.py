"""NumPy reconstruction of saved reciprocal predictions, losses and selected derivatives.

This reads persisted arrays and independently implements the equations. It does
not call the candidate model or its loss, and does not rerun any embedder.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.capability_population import need, sha
from claims.process_conditions import write_committed


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k].astype(np.float64) for k in value.files}


def positions(length, width):
    phase = np.arange(length, dtype=np.float32)[:, None]
    frequency = np.exp(np.arange(0, width, 2, dtype=np.float32) * np.float32(-math.log(10000)/width))
    phase = phase * frequency
    result = np.empty((length, width), np.float32)
    result[:, 0::2], result[:, 1::2] = np.sin(phase), np.cos(phase)
    return result.astype(np.float64)


def predict(parameters, native, hidden, names, heads):
    def linear(x, prefix):
        return x @ parameters[prefix+'.weight'].T + parameters[prefix+'.bias']

    def attention(query, memory, prefix):
        b, length, width = query.shape
        tokens, depth = memory.shape[1], width//heads
        q = linear(query, prefix+'.q').reshape(b, length, heads, depth).transpose(0, 2, 1, 3)
        k = linear(memory, prefix+'.k').reshape(b, tokens, heads, depth).transpose(0, 2, 1, 3)
        v = linear(memory, prefix+'.v').reshape(b, tokens, heads, depth).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(depth)
        weights = np.exp(scores-scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        result = (weights @ v).transpose(0, 2, 1, 3).reshape(b, length, width)
        return linear(result, prefix+'.out')

    b, windows, _ = native[hidden].shape
    width = parameters['seed'].shape[1]
    query = parameters['seed'][names.index(hidden)][None, None, :] + positions(windows, width)[None]
    query = np.broadcast_to(query, (b, windows, width))
    visible = [name for name in names if name != hidden]
    branches = []
    for name in visible:
        memory = linear(native[name], 'projectors.'+name)
        memory += positions(memory.shape[1], width)[None]
        branches.append(attention(query, memory, 'readers.'+name))
    views = np.stack(branches, axis=2).reshape(b*windows, len(visible), width)
    mixed = attention(query.reshape(b*windows, 1, width), views, 'mix.'+hidden).reshape(b, windows, width)
    output = linear(mixed, 'output.'+hidden)
    return output/np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-12)


def loss(prediction, target):
    p = prediction/np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
    y = target/np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    return float((1-(p*y).sum(-1)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    need(not a.output.exists(), 'RECIPROCAL_READER_EXISTS')
    r = json.loads((a.root/'report.json').read_bytes())
    surface_raw = (a.surface/'report.json').read_bytes()
    need(sha(surface_raw) == '513944c0c193b7f7b156b487c222ce221ecad0762c4ccd5fb24d4837d7d66d29',
         'RECIPROCAL_READER_SOURCE_REPORT')
    surface = json.loads(surface_raw)
    need(sha(a.journal.read_bytes()) == r['journal'] and r['model_updated'] is False
         and r['optimizer_steps'] == 0 and r['original_flags_restored'], 'RECIPROCAL_READER_MUTATION')
    for name, record in r['files'].items():
        raw = (a.root/name).read_bytes()
        need(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'RECIPROCAL_READER_FILE:'+name)
    p = arrays(a.root/'initial.npz')
    names = sorted(s['name'] for s in surface['instruments'])
    need(len(r['cases']) == len(surface['cases'])*len(names) == 36
         and len(r['controls']) == len(r['cases']), 'RECIPROCAL_READER_ROSTER')
    sources = {}
    for i, entry in enumerate(surface['cases']):
        file = 'native-'+str(i)+'.npz'
        raw = arrays(a.root/file)
        for name in names:
            need(list(raw[name].shape) == entry['native'][name]['shape']
                 and sha(raw[name].astype('<f4').tobytes()) == entry['native'][name]['sha256'],
                 'RECIPROCAL_READER_SOURCE_VECTOR')
        sources[file] = (raw, entry)
    checked, derivatives, families = [], [], set()
    for c, control in zip(r['cases'], r['controls'], strict=True):
        native, entry = sources[c['native']]
        need(all(c[k] == entry[k] for k in ('task', 'field', 'family'))
             and c['visible'] == [n for n in names if n != c['hidden']], 'RECIPROCAL_READER_MASK')
        actual = predict(p, native, c['hidden'], names, surface['config']['heads'])
        saved = arrays(a.root/(c['prefix']+'-prediction.npz'))['prediction']
        difference = float(np.abs(actual-saved).max())
        need(difference < 5e-5 and abs(loss(actual, native[c['hidden']])-c['loss']) < 5e-5
             and abs(loss(actual, -native[c['hidden']])-c['changed_target_loss']) < 5e-5,
             'RECIPROCAL_READER_EQUATIONS')
        gradients = arrays(a.root/(c['prefix']+'-gradients.npz'))
        need(all(np.isfinite(g).all() for g in gradients.values()), 'RECIPROCAL_READER_NONFINITE')
        for name in names:
            prefix = 'projectors.'+name+'.'
            norm = sum(float(np.square(v).sum()) for k, v in gradients.items() if k.startswith(prefix))
            need(abs(norm-c['source_projector_gradients'][name]) <= 1e-12
                 and (norm == 0 if name == c['hidden'] else norm > 0), 'RECIPROCAL_READER_CONNECTION')
        need(control['case'] == c['prefix'] and all(control[k] for k in
             ('hidden_input_refused', 'missing_input_refused', 'swapped_instrument_refused',
              'target_only_changes_loss')), 'RECIPROCAL_READER_NEGATIVE')
        # One case per capability family: independent central differences in
        # every visible branch, at the largest saved weight derivative.
        if c['family'] not in families:
            families.add(c['family'])
            for name in c['visible']:
                key = 'projectors.'+name+'.weight'; g = gradients[key]
                index = np.unravel_index(np.abs(g).argmax(), g.shape)
                original, epsilon = p[key][index], 0.01
                p[key][index] = original+epsilon
                plus = loss(predict(p, native, c['hidden'], names, surface['config']['heads']), native[c['hidden']])
                p[key][index] = original-epsilon
                minus = loss(predict(p, native, c['hidden'], names, surface['config']['heads']), native[c['hidden']])
                p[key][index] = original
                finite = (plus-minus)/(2*epsilon); saved_gradient = float(g[index])
                need(abs(finite-saved_gradient) < max(5e-5, abs(saved_gradient)*0.02),
                     'RECIPROCAL_READER_DERIVATIVE')
                derivatives.append(dict(case=c['prefix'], parameter=key, index=list(map(int, index)),
                    finite_difference=finite, saved_gradient=saved_gradient))
        checked.append(dict(case=c['prefix'], maximum_prediction_error=difference))
    report = dict(schema='reciprocal-native-numpy-readback-v1', report=sha((a.root/'report.json').read_bytes()),
        source_report=sha(surface_raw), journal=r['journal'], cases=checked, derivatives=derivatives,
        equations_independently_reconstructed=True, measured_gradient_flow=True,
        scientific_admission=False, learned_capability=False, optimizer_steps=0,
        number_status='measured', acceptance='author-checked', metric_mapping=['T3/spec27 model connectivity'])
    pin = write_committed(a.output, report)
    print(json.dumps(dict(pin=pin, cases=len(checked), derivatives=len(derivatives),
                         max_error=max(c['maximum_prediction_error'] for c in checked))), flush=True)


if __name__ == '__main__': main()

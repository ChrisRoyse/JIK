"""Independent physical recount of transient energy VJPs and whole-answer updates."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from audit import captured_answer_readback as rd
from audit.energy_program_readback import program
from model.native_stochastic_placement import energy_loss
from model.native_population_training import cpu, fingerprint


def direct(root, arch, wrapper, source, source_report):
    info = wrapper['direct'][arch]; path = root/(arch+'-vjps.pt')
    rd.require(rd.sha(path) == info['pin'], 'TRANSIENT_READ_DIRECT_PIN')
    value = rd.load(root, path.name); actual_source = source/(arch+'-controls.pt')
    rd.require(rd.sha(actual_source) == value['source'] == source_report['control_files'][actual_source.name], 'TRANSIENT_READ_REAL_SOURCE')
    real = rd.load(source, actual_source.name); roster = tuple(real['variances'])
    wanted = ['loss','joint','all_outputs','empirical','noncontiguous','duplicate_samples','zero_distance','partial_inputs',
              *('lane:'+n for n in roster)]
    rd.require(info['cases'] == wanted == [r['case'] for r in value['rows']], 'TRANSIENT_READ_CASES')
    rd.exact(value['negative'], info['negative']); counts = dict(cases=0, tensors=0, bytes=0, peak_saved_bytes=0)
    pending = {r['case']:r for r in info['negative']['pending']}
    for row in value['rows']:
        case = row['case']; a, b = row['expected'], row['actual']; compared = rd.exact(a,b)
        for k in ('tensors','bytes'): counts[k] += compared[k]
        rd.exact(a['targets'], real['targets'])
        for r, original, target in zip(a['inputs'], real['samples'], real['targets']):
            for n in roster:
                expected = original[n]
                if case == 'duplicate_samples': expected = expected[:1].expand_as(expected)
                elif case == 'zero_distance': expected = target[n][None].expand_as(expected)
                rd.exact(r[n], expected)
                if case == 'noncontiguous': rd.require(not r[n].is_contiguous(), 'TRANSIENT_READ_CONTIGUOUS_CONTROL')
        inputs = tuple({n:x.detach().clone().requires_grad_(case != 'partial_inputs' or n == roster[0]) for n,x in r.items()}
                       for r in a['inputs'])
        result = energy_loss(inputs, a['targets'], real['variances'], empirical=case == 'empirical')
        outputs = (result['loss'], result['joint'], *(result['lanes'][n] for n in roster))
        selected = outputs if case == 'all_outputs' else (result['joint'],) if case == 'joint' else \
                   (result['lanes'][case[5:]],) if case.startswith('lane:') else (result['loss'],)
        gradients = rd.torch.autograd.grad(selected, tuple(x for r in inputs for x in r.values() if x.requires_grad),
                                           tuple(rd.torch.ones_like(x) for x in selected), allow_unused=True)
        rd.exact(cpu(result), a['value']); rd.exact(cpu(gradients), a['gradients'])
        if case in pending:
            rd.require(pending[case]['value'] == fingerprint(result) and pending[case]['gradient'] == fingerprint(gradients),
                       'TRANSIENT_READ_PENDING')
        rd.require(len(row['calls']) == 1 and row['calls'][0]['completed'] and row['calls'][0]['transient']
            and row['calls'][0]['backwards'] == 1, 'TRANSIENT_READ_DIRECT_CALL')
        program(json.loads(json.dumps(row['calls'][0]['program'])))
        storage = row['storage']
        rd.require(storage['remaining_saved_storage_bytes'] == storage['remaining_storages'] == 0, 'TRANSIENT_READ_DIRECT_STORAGE')
        counts['peak_saved_bytes'] = max(counts['peak_saved_bytes'], storage['peak_saved_storage_bytes']); counts['cases'] += 1
    negative = info['negative']
    rd.require(set(negative) == {'higher_order','mutation','nested','ownership','exception','closed','malformed','pending'}
        and negative['higher_order'] == 'MNLM_TEXT_CORPUS_ENERGY_TRANSIENT_HIGHER_ORDER_UNSUPPORTED'
        and negative['mutation'] == 'MNLM_TEXT_CORPUS_GRADIENT_STORAGE_MUTATED'
        and set(negative['malformed']) == {'empty','roster','dtype','shape','sample_nan','target_inf','variance','memory','target_grad'}
        and len(pending) == 3, 'TRANSIENT_READ_NEGATIVES')
    return counts


def run(a):
    began = time.perf_counter(); rd.torch.set_num_threads(1); a.output.mkdir(exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    roots = (a.producer,a.cold); wrappers = [rd.read(p/'report.json',h) for p,h in zip(roots,(a.producer_pin,a.cold_pin))]
    first,cold = wrappers
    rd.require(first['complete'] and cold['complete'] and cold['cold_reproduced'] and cold['previous'] == a.producer_pin
        and first['code'] == cold['code'] and first['direct'] == cold['direct'] and first['real_source'] == cold['real_source'],
        'TRANSIENT_READ_BINDING')
    for n,h in first['code'].items():
        rd.require(rd.sha(rd.BASE/n) == h and all(rd.sha(p/'source'/n) == h for p in roots), 'TRANSIENT_READ_SOURCE')
    source = a.root/first['real_source'][0]; source_report = rd.read(source/'report.json',first['real_source'][1])
    reports = [rd.read(p/'numerical/report.json',w['numerical_report']) for p,w in zip(roots,wrappers)]
    rd.require(reports[0]['binding'] == reports[1]['binding'] and reports[1]['previous_pin'] == first['numerical_report'], 'TRANSIENT_READ_COLD_BINDING')
    reference_root = a.root/rd.REFERENCE[0]; reference = rd.read(reference_root/'report.json', rd.REFERENCE[1]); rd.manifest(reference_root,reference)
    totals = dict(tensors=0,bytes=0); counts = {}; direct_counts = {}; updates = decisions = 0; performance = {}
    for root,wrapper,numerical in zip(roots,wrappers,reports):
        rd.manifest(root/'numerical',numerical)
        rd.require(wrapper['captured_backward_in_both_arms'] and not any(wrapper[k] for k in ('gpu_used','cuda_graph_created','promoted','performance_claim'))
            and numerical['all_updates_exact'] and numerical['restored_continuations_exact'], 'TRANSIENT_READ_SCOPE')
        observations = {(r['architecture'],r['document'],r['variant']):r for r in wrapper['observations']}
        rd.require(len(observations) == len(wrapper['observations']) == 30, 'TRANSIENT_READ_OBSERVATIONS')
        counts[root.name] = {}; direct_counts[root.name] = {}; performance[root.name] = {}
        for arch in rd.ARCHS:
            direct_counts[root.name][arch] = direct(root,arch,wrapper,source,source_report)
            counter = dict(calls=0,completed=0,backwards=0,native_elements=0,decisions=0)
            peaks = {m:0 for m in ('original','captured')}
            for index,doc in enumerate(rd.DOCS):
                step = index+6; expected = rd.load(reference_root,f'{arch}/update-{step}.pt')
                row = numerical['results'][arch]['updates'][index]
                other = reports[1 if numerical is reports[0] else 0]['results'][arch]['updates'][index]
                rd.require(row['order'] == other['order'][::-1], 'TRANSIENT_READ_ORDER')
                for k in ('document','step','numerical','chain','counts','restored_state'):
                    rd.require(row.get(k) == other.get(k), 'TRANSIENT_READ_COLD_RECORD')
                for mode,label in (('original','original'),('captured','transient')):
                    value = rd.load(root/'numerical',f'{arch}/{mode}/update-{step}.pt')
                    c = rd.exact(rd.numeric(value),rd.numeric(expected))
                    for k in totals: totals[k] += c[k]
                    obs = observations[arch,doc,label]; n = value['record']['decisions']
                    rd.require(obs['step'] == step-1 and obs['decisions'] == n and value['record']['document'] == doc
                        and sum(r['end'] for r in value['record']['records']) == 1, 'TRANSIENT_READ_DECISIONS')
                    if mode == 'original': rd.require(not obs['calls'], 'TRANSIENT_READ_ORIGINAL')
                    else:
                        proposals = 6 if arch == 'whole_proposals' else 1; calls = obs['calls']
                        rd.require(len(calls) == 2*(n-1)*proposals and all(r['transient'] for r in calls)
                            and sum(r['backwards'] for r in calls) == (n-1)*proposals
                            and all(r['backwards'] in (0,1) and (not r['backwards'] or r['completed']) for r in calls), 'TRANSIENT_READ_CALLS')
                        counter['calls'] += len(calls); counter['completed'] += sum(r['completed'] for r in calls)
                        counter['backwards'] += sum(r['backwards'] for r in calls)
                        counter['native_elements'] += sum(program(r['program']) for r in calls); counter['decisions'] += n
                    s = value['record']['storage']; peaks[mode] = max(peaks[mode],s['peak_saved_storage_bytes'])
                    rd.require(s['remaining_saved_storage_bytes'] == s['remaining_storages'] == 0
                        and all(float(v['step']) == step for v in value['optimizer']['state'].values()), 'TRANSIENT_READ_STORAGE_ADAM')
                    for branch in value['early'].values():
                        rd.require(branch['requires_grad'] and bool(branch['gradient'].abs().any()) and bool(branch['end_gradient'].abs().any()), 'TRANSIENT_READ_HISTORY')
                    if index in (1,4):
                        cp = rd.load(root/'numerical',f'{arch}/{mode}/checkpoint-{index+1}.pt')
                        rd.exact(cp['model'],value['model']); rd.exact(cp['optimizer'],value['optimizer'])
                        rd.require(cp['cursor'] == index+1 and cp['step'] == step and cp['chain'] == row['chain'] and cp['counts'] == row['counts'], 'TRANSIENT_READ_RESTART')
                    updates += 1; decisions += n
            counts[root.name][arch] = counter
            performance[root.name][arch] = dict(peak_saved_bytes=peaks, **{m+'_seconds':numerical['results'][arch][m+'_seconds'] for m in peaks})
    rd.exact(*counts.values()); rd.exact(*direct_counts.values())
    controls = rd.controls()
    rd.require(updates == 60 and decisions == 792 and not rd.torch.cuda.is_initialized(), 'TRANSIENT_READ_TOTAL')
    result = dict(complete=True, producer=a.producer_pin,cold=a.cold_pin,reference=rd.REFERENCE[1],
        updates_compared=updates, decisions=decisions, comparison=totals, counters=counts, direct=direct_counts, performance=performance,
        all_numerical_bytes_exact=True, checkpoints_exact=True, controls=controls, reader_source=rd.sha(__file__),
        helpers={n:rd.sha(rd.BASE/n) for n in ('audit/captured_answer_readback.py','audit/energy_program_readback.py','model/native_stochastic_placement.py','model/native_population_training.py')},
        gpu_used=False, cuda_graph_created=False, promoted=False, performance_claim=False, seconds=time.perf_counter()-began, acceptance='author-checked')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f: json.dump(result,f,sort_keys=True,indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=rd.sha(a.output/'report.json'),counters=counts,direct=direct_counts,performance=performance)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('root','producer','cold','output'): p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin'): p.add_argument('--'+n,required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): (a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise

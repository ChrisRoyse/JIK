"""Source-bound matched RQ collapse interventions, stage qualification and cold replay."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_rq_growth as growth
from audit import legal_rq_residual as residual
from audit.legal_rq_growth_readback import check_chain, recount
from audit.legal_rq_intervention_steps import step, loss_reference, fingerprint, frozen_check
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.native_rq_data import NativeRQData, training_components
from model.native_rq_interventions import InterventionNativeRQ, ARMS, SCHEDULE, QUALIFY
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

POLICY = dict(growth.POLICY, interventions=list(ARMS), schedule=SCHEDULE, qualification_updates=QUALIFY,
              initial='phase40_position_residual_initial_unchanged', baseline_first32_exact=False)


def pins():
    base = Path(__file__).resolve().parents[1]
    names = ['audit/legal_rq_interventions.py', 'audit/legal_rq_intervention_steps.py',
             'model/native_rq_interventions.py', 'model/text_sigreg.py', 'audit/text_sigreg_readback.py']
    return dict(residual.pins(), **{n:file_sha(base/n) for n in names})


def make(dimensions, stats, arm, initial):
    torch.manual_seed(POLICY['seed'])
    model = InterventionNativeRQ(dimensions, growth.CONFIG, POLICY['sizes'], arm, stats)
    model.load_state_dict(initial['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'], weight_decay=POLICY['weight_decay'],
                                 foreach=False, fused=False)
    optimizer.load_state_dict(initial['optimizer'])
    need(not optimizer.state and sum(p.numel() for p in model.parameters()) == 494208, 'RQ_INTERVENTION_INITIAL')
    return model, optimizer


def checkpoints(arm, count):
    points = {count-1}
    if arm == 'decoder_warmup': points.add(SCHEDULE['decoder_warmup'])
    if arm == 'continuous_then_rq': points.add(SCHEDULE['continuous_warmup'])
    need(all(i < count for i in points), 'RQ_INTERVENTION_BOUNDARY_NOT_EXERCISED')
    return sorted(points)


def load(path, pin, binding, number):
    need(file_sha(path) == pin, 'RQ_INTERVENTION_CHECKPOINT_PIN')
    r = torch.load(path, map_location='cpu', weights_only=True)
    need(r['binding'] == binding and r['step'] == number, 'RQ_INTERVENTION_CHECKPOINT_BINDING')
    return r


def loss_at(model, reader, uids, cases, root):
    root.mkdir(); return loss_reference(model, reader, uids, cases, root)


def eval_all(model, reader, data, selection, cases, stats, root):
    model.set_step(SCHEDULE['updates']); uids = selection['populations']['large']
    groups = dict(train=(uids, cases))
    for name, side in [('development',1),('query',2)]:
        ids = selection['held'][name]; groups[name] = (ids, growth.components_for(data, ids, side))
    result = {}
    for name, (ids, components) in groups.items():
        result[name] = growth.evaluate(model, reader, ids, components, stats, root/name)
    need(result['train']['native_sha256'] == stats['native_sha256'], 'RQ_INTERVENTION_TRAIN_BYTES')
    return result


def fit_arm(mode, root, arm, model, opt, reader, uids, cases, stats, probes, data, selection, binding, qualified):
    count = QUALIFY[arm] if mode == 'qualify' else SCHEDULE['updates']; points = checkpoints(arm, count)
    initial_probe, _ = residual.probe(model, reader, probes, stats, initial=True)
    if qualified is not None:
        need(initial_probe == qualified['initial'], 'RQ_INTERVENTION_INITIAL_REPLAY')
    losses = dict(initial=loss_at(model, reader, uids, cases, root/'initial-loss'))
    records = []; files = {}; began = time.perf_counter()
    with (root/'updates.jsonl').open('x') as stream:
        for i in range(count):
            if i in points:
                name = f'resume-{i}.pt'; residual.snapshot(root/name, model, opt, i, binding); files[name] = file_sha(root/name)
            row = step(model, opt, reader, uids, cases, i); records.append(row)
            stream.write(json.dumps(row, sort_keys=True)+'\n')
            if (i+1)%128 == 0:
                stream.flush(); print(json.dumps(dict(arm=arm, step=i+1, stage=row['stage'], loss=row['loss'],
                                                    seconds=time.perf_counter()-began)), flush=True)
    files['updates.jsonl'] = file_sha(root/'updates.jsonl')
    residual.snapshot(root/'final.pt', model, opt, count, binding); files['final.pt'] = file_sha(root/'final.pt')
    after, payload = residual.probe(model, reader, probes, stats)
    torch.save(payload, root/'probe.pt'); files['probe.pt'] = file_sha(root/'probe.pt')
    losses['final'] = loss_at(model, reader, uids, cases, root/'final-loss')
    measured = {}
    if mode == 'fit':
        measured = eval_all(model, reader, data, selection, cases, stats, root)
    negatives = dict(invalid_cursor=rejected(lambda:model.set_step(-1)))
    if arm == 'fixed_encoder':
        frozen = model.frozen_names(); before = fingerprint({n:p for n,p in model.named_parameters() if n in frozen})
        name, parameter = next((n,p) for n,p in model.named_parameters() if n in frozen)
        copy = parameter.detach().clone()
        with torch.no_grad(): parameter.view(-1)[0] += 1
        negatives['frozen_mutation'] = rejected(lambda:frozen_check(model,opt,frozen,before))
        with torch.no_grad(): parameter.copy_(copy)
        frozen_check(model,opt,frozen,before)
    with torch.no_grad(): model.position_0[0,0] += 1
    negatives['statistics_mutation'] = rejected(model.check_statistics)
    return dict(count=count, initial=initial_probe, after=after, checkpoints=points, files=files,
        last_update=records[-1], first32=[{k:r[k] for k in ['after','gradient','optimizer','loss']} for r in records[:32]],
        loss_reference=losses, controls=negatives,
        evaluation={n:dict(pin=v['pin'],rows=v['rows'],distinct_codes=v['distinct_codes'],slices=v['slices']) for n,v in measured.items()})


def cold_arm(root, previous_root, reference, arm, model, opt, reader, uids, cases, stats, probes, data, selection, binding, mode):
    for n, pin in reference['files'].items(): need(file_sha(previous_root/n) == pin, 'RQ_INTERVENTION_ARTIFACT_PIN')
    records = [json.loads(line) for line in (previous_root/'updates.jsonl').read_text().splitlines()]
    need(len(records) == reference['count'], 'RQ_INTERVENTION_LEDGER_COUNT')
    for i, row in enumerate(records):
        need(row['step'] == i+1 and row['ids'] == [uids[j] for j in growth.batch_indices(uids,i)], 'RQ_INTERVENTION_LEDGER_ORDER')
        if i: need(row['before'] == records[i-1]['after'], 'RQ_INTERVENTION_LEDGER_CHAIN')
    if mode == 'fit': check_chain(records,uids)
    losses = dict(initial=loss_at(model, reader, uids, cases, root/'initial-loss'))
    initial_probe, _ = residual.probe(model, reader, probes, stats, initial=True)
    need(initial_probe == reference['initial'], 'RQ_INTERVENTION_COLD_INITIAL')
    replays = []
    for i in reference['checkpoints']:
        filename = f'resume-{i}.pt'; saved = load(previous_root/filename, reference['files'][filename], binding, i)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        actual = step(model, opt, reader, uids, cases, i)
        need(actual == records[i], 'RQ_INTERVENTION_COLD_STEP')
        replays.append(dict(step=i+1, exact=True, transition=actual['transition']))
    final = load(previous_root/'final.pt', reference['files']['final.pt'], binding, reference['count'])
    need(fingerprint(model.state_dict()) == fingerprint(final['model'])
         and fingerprint(opt.state_dict()) == fingerprint(final['optimizer']), 'RQ_INTERVENTION_COLD_FINAL')
    after, payload = residual.probe(model, reader, probes, stats)
    need(after == reference['after'] and fingerprint(payload) == fingerprint(torch.load(previous_root/'probe.pt',map_location='cpu',weights_only=True)),
         'RQ_INTERVENTION_COLD_PROBE')
    losses['final'] = loss_at(model, reader, uids, cases, root/'final-loss')
    need(losses == reference['loss_reference'], 'RQ_INTERVENTION_COLD_LOSS')
    measured = {}
    if mode == 'fit':
        fresh = eval_all(model, reader, data, selection, cases, stats, root)
        for name, r in fresh.items():
            need(r['pin'] == reference['evaluation'][name]['pin'], 'RQ_INTERVENTION_COLD_OUTPUT')
            arrays = [np.load(root/name/n, allow_pickle=False) for n in ['codes.npy','errors.npy','eligible.npy','windows.npy']]
            measured[name] = dict(pin=r['pin'], **recount(r,*arrays,reader.roster))
    negatives = dict(wrong_checkpoint_binding=rejected(lambda:load(previous_root/'final.pt',reference['files']['final.pt'],dict(binding,arm='wrong'),reference['count'])),
        wrong_checkpoint_pin=rejected(lambda:load(previous_root/'final.pt','0'*64,binding,reference['count'])))
    return dict(replays=replays, final_state_exact=True, initial=initial_probe, after=after, loss_reference=losses,
                evaluation=measured, controls=negatives)


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    parent = growth.pinned(a.residual/'report.json',a.residual_pin)
    need(parent['complete'] and parent['mode'] == 'qualify' and parent['common']['code'] == residual.pins(), 'RQ_INTERVENTION_PARENT')
    prior = growth.pinned(a.growth/'report.json',a.growth_pin)
    need(prior['complete'] and prior['mode'] == 'fit' and prior['code'] == growth.pins()
         and a.growth_pin == parent['growth_pin'], 'RQ_INTERVENTION_GROWTH')
    baseline = growth.pinned(a.baseline/'report.json',a.baseline_pin)
    selection = growth.pinned(a.growth/'selection.json',prior['selection_pin'])
    q = Path(prior['qualification']['path']); q_report = growth.pinned(q/'report.json',prior['qualification']['pin'])
    stat_path = q/'large-statistics.pt'; need(file_sha(stat_path) == q_report['statistic_pins']['large'], 'RQ_INTERVENTION_STATISTICS')
    stats = torch.load(stat_path,map_location='cpu',weights_only=True)
    source = ReuseInputs(baseline['source']['path'],baseline['source']['pin'])
    continuation = Continuations(Path(baseline['continuations']['path']),baseline['continuations']['pin'])
    split,cold,bank = continuation.report['split'],continuation.report['cold'],baseline['bank']
    data = LegalLearningSource(source,split['path'],split['pin'],cold['path'],cold['pin'],bank['path'],bank['pin'])
    need(data.binding == prior['source_binding'] == selection['source_binding'], 'RQ_INTERVENTION_SOURCE')
    reader = NativeRQData(data,Path(baseline['proof']['path']),baseline['proof']['pin'])
    uids = selection['populations']['large']; probes = selection['populations']['small']
    need(stats['uids'] == uids, 'RQ_INTERVENTION_STATISTICS_SCOPE')
    _,cases = training_components(data,uids,continuation.presence)
    initial_path = a.residual/'position_residual/initial.pt'
    initial_pin = parent['results']['position_residual']['files']['initial.pt']
    initial = residual.load_snapshot(initial_path,initial_pin,dict(parent['common'],recipe='position_residual'),0)
    common = dict(source=data.binding, selection_pin=prior['selection_pin'],statistic_pin=file_sha(stat_path),
        initial_pin=initial_pin, parent_pin=a.residual_pin, code=pins(),policy=POLICY)
    previous = None
    if a.mode != 'qualify':
        need(a.previous is not None and a.previous_pin is not None, 'RQ_INTERVENTION_PREVIOUS_REQUIRED')
        previous = growth.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['common'] == common, 'RQ_INTERVENTION_PREVIOUS_BINDING')
    if a.mode == 'fit':
        need(previous['mode'] == 'qualify' and a.cold_proof is not None, 'RQ_INTERVENTION_QUALIFICATION_REQUIRED')
        readback = growth.pinned(a.cold_proof/'report.json',a.cold_pin)
        need(readback['complete'] and readback['mode'] == 'cold' and readback['previous_pin'] == a.previous_pin
             and readback['common'] == common, 'RQ_INTERVENTION_COLD_REQUIRED')
    if a.mode == 'cold': need(previous['mode'] in ['qualify','fit'], 'RQ_INTERVENTION_COLD_SOURCE')
    phases = dict(source_bindings=time.perf_counter()-tick); tick = time.perf_counter(); results = {}
    for arm in ARMS:
        start = time.perf_counter(); root = a.output/arm; root.mkdir(); binding = dict(common,arm=arm)
        model,opt = make(prior['dimensions'],stats,arm,initial)
        if a.mode == 'cold':
            value = cold_arm(root,a.previous/arm,previous['results'][arm],arm,model,opt,reader,uids,cases,stats,probes,
                             data,selection,binding,previous['mode'])
        else:
            value = fit_arm(a.mode,root,arm,model,opt,reader,uids,cases,stats,probes,data,selection,binding,
                            previous['results'][arm] if previous else None)
            need(value['initial'] == parent['results']['position_residual']['initial'], 'RQ_INTERVENTION_PARENT_INITIAL')
        results[arm] = dict(value,seconds=time.perf_counter()-start)
        write_committed(root/'report.json',results[arm])
        print(json.dumps(dict(mode=a.mode,arm=arm,complete=True,seconds=results[arm]['seconds'])),flush=True)
    if a.mode != 'cold':
        need(results['fixed_encoder']['first32'] == results['decoder_warmup']['first32'], 'RQ_INTERVENTION_SHARED_WARMUP')
    phases['qualification_fit_or_cold']=time.perf_counter()-tick;tick=time.perf_counter()
    source.fresh();data.fresh();continuation.fresh();phases['final_freshness']=time.perf_counter()-tick
    seconds=time.perf_counter()-began;need(sum(phases.values())/seconds>=.9,'RQ_INTERVENTION_PHASE_ATTRIBUTION')
    report=dict(schema='legal-rq-collapse-interventions-v1',complete=True,mode=a.mode,common=common,
        previous_pin=a.previous_pin,results=results,phases_seconds=phases,seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        maximum_cached_native_bytes=reader.peak_cache_bytes,maximum_read_native_bytes=reader.maximum_read_bytes,
        model_promotion=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds,complete=True)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['qualify','fit','cold'],required=True)
    for n in ['baseline','growth','residual']:
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--cold-proof',type=Path);p.add_argument('--cold-pin');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'failure.json').exists():
            write_committed(args.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise

"""Independent source, byte-schedule and gradient recount for legal inversion."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import legal_surface_decoder_proof as producer
from audit.capability_surface_fsv import check_schedule
from claims.process_conditions import write_committed
from model.contracts import Instrument, ModelConfig
from model.native import NativeInput
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder, batches
from text.constellation_corpus import file_sha, need


def run(a):
    began = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.input_pin, 'LEGAL_SURFACE_READBACK_PIN')
    r = json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['code'] == producer.code(), 'LEGAL_SURFACE_READBACK_CODE')
    for name,pin in r['files'].items(): need(file_sha(a.input/name) == pin, 'LEGAL_SURFACE_READBACK_FILE')
    source, bundle = producer.comparison.source_bundle(a.root)
    records, bank, data, c, _, _, _, _ = producer.statistics.authenticate(a.root)
    saved = torch.load(a.input/'observations.pt', map_location='cpu', weights_only=True)
    expected = {item['document']:item for item in bundle['items']}
    cursor = slots = windows = total_bytes = 0
    for document in saved:
        item = expected[document['document']]
        need((document['start'], document['stop'], document['side']) ==
             (item['start'], item['stop'], item['side']), 'LEGAL_SURFACE_READBACK_DOCUMENT')
        for k, value in enumerate(document['observations']):
            occurrence = item['start']+k; uid = int(data.mapping[occurrence])
            data.indices([occurrence], item['side'], fit=item['side']=='train')
            raw = bytes(value['raw'].tolist())
            need(value['uid'] == uid and value['occurrence'] == occurrence
                 and raw == data.source.input(uid)['text'].encode('utf-8'), 'LEGAL_SURFACE_READBACK_BYTES')
            need(value['train_present'] == bool(c.presence[uid]&1), 'LEGAL_SURFACE_READBACK_EXPOSURE')
            need(set(value['native']) == set(data.roster), 'LEGAL_SURFACE_READBACK_ROSTER')
            for name,v in value['native'].items():
                # Separate previously sealed source producer, not the new adapter.
                need(torch.equal(v, item['frames'][k][name]), 'LEGAL_SURFACE_READBACK_NATIVE_BYTES')
                slots += 1; windows += len(v)
            check_schedule(raw, value['schedule']); total_bytes += len(raw); cursor += 1
    need(cursor == r['observations'] and slots == r['slots'] and windows == r['native_windows']
         and total_bytes == r['surface_bytes'], 'LEGAL_SURFACE_READBACK_COUNTS')
    config = ModelConfig(**r['config']); specs = tuple(Instrument(**s).validate() for s in r['instruments'])
    model = StreamSurfaceDecoder(config, specs)
    state = torch.load(a.input/'initial.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True); model.train()
    examples = {v['occurrence']:v for d in saved for v in d['observations']}
    gradient_results = []
    for case in r['gradients']:
        value = examples[case['occurrence']]; raw = bytes(value['raw'].tolist())
        native = {s.name:NativeInput(s.identity, (value['row'],), value['native'][s.name][None],
            torch.zeros((1,len(value['native'][s.name])),dtype=torch.bool), torch.zeros(1,dtype=torch.bool))
                  for s in specs}
        model.zero_grad(set_to_none=True); total = 0.; positions = 0
        for prefix, labels, offsets, typed, items in batches(raw, native, config):
            logits = model(prefix, typed, offsets=offsets)['logits']
            loss = F.cross_entropy(logits.reshape(-1,config.vocabulary), labels.reshape(-1),
                                   ignore_index=-100,reduction='sum')/(len(raw)+1)
            loss.backward(); total += float(loss.detach()); positions += int((labels != -100).sum())
        gradients = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        prior = torch.load(a.input/case['file'], map_location='cpu', weights_only=True)
        need(positions == len(raw)+1 and total == case['loss']['loss']
             and fingerprint(gradients) == fingerprint(prior), 'LEGAL_SURFACE_READBACK_GRADIENTS')
        gradient_results.append(dict(occurrence=case['occurrence'], loss=total, positions=positions,
                                     gradient_state=fingerprint(gradients), exact=True))
    need(fingerprint(model.state_dict()) == fingerprint(state), 'LEGAL_SURFACE_READBACK_WEIGHTS_CHANGED')
    need(not torch.cuda.is_initialized(), 'LEGAL_SURFACE_READBACK_GPU')
    data.fresh(); data.source.fresh()
    need(producer.code() == r['code'], 'LEGAL_SURFACE_READBACK_CODE_CHANGED')
    out = dict(complete=True, input=str(a.input), input_pin=a.input_pin, source=source,
        observations=cursor, slots=slots, native_windows=windows, bytes=total_bytes,
        all_source_bytes_exact=True, all_native_arrays_exact=True, all_schedules_exact=True,
        gradients=gradient_results, optimizer_steps=0, gpu_initialized=False,
        seconds=time.perf_counter()-began, number_status='measured', acceptance='author-checked',
        learned_capability=False, scientific_admission=False)
    pin = write_committed(a.output/'report.json', out)
    print(json.dumps(dict(complete=True,pin=pin,observations=cursor,gradient_cases=len(gradient_results))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise

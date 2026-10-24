"""Replay generated tokens through the ordinary decoder and audit terminal bytes."""
import argparse
import codecs
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_prediction_readback import arrays, slots
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.capability_objective import PredictedSurfaceDecoder
from model.capability_surface_runtime import render
from model.constellation import Constellation
from model.contracts import Instrument, ModelConfig
from model.stream_surface import BOS, EOS


def read(path): return json.loads(Path(path).read_bytes())


@torch.no_grad()
def replay(decoder, predicted, trace, raw, report):
    """No cached projection path: call the ordinary decoder at every position."""
    prefix=[BOS]; actual=bytearray(); terminal=None; eos=False
    parser=codecs.getincrementaldecoder('utf-8')('strict')
    for index,row in enumerate(trace):
        need(terminal is None,'SURFACE_READBACK_AFTER_TERMINAL')
        offset=max(0,index+1-decoder.config.max_prefix)
        need(row['position']==index and row['offset']==offset,'SURFACE_READBACK_POSITION')
        tokens=torch.tensor([prefix[-decoder.config.max_prefix:]],device='cuda',dtype=torch.int64)
        output=decoder(tokens,predicted.slots,offsets=torch.tensor([offset],device='cuda'))['logits'][0,-1]
        need(sha(output.cpu().numpy().astype('<f4').tobytes())==row['logits_sha256']
             and int(output.argmax())==row['token'],'SURFACE_READBACK_LOGITS')
        token=row['token']
        if token==EOS:
            eos=True
            try: parser.decode(b'',final=True)
            except UnicodeDecodeError: terminal='SURFACE_UTF8_INCOMPLETE'
            else: terminal='COMPLETE' if actual else 'SURFACE_EMPTY'
        elif not 0<=token<256: terminal='SURFACE_INVALID_SYMBOL'
        else:
            piece=bytes([token]); actual.extend(piece); prefix.append(token)
            # Physical prefix memory remains fixed even on an unusually long output.
            prefix=prefix[-decoder.config.max_prefix:]
            try: parser.decode(piece,final=False)
            except UnicodeDecodeError: terminal='SURFACE_UTF8_INVALID'
    if terminal is None and max(0,len(trace)+1-decoder.config.max_prefix)>=2**24:
        terminal='SURFACE_POSITION_UNREPRESENTABLE'
    need(terminal==report['status'] and report['complete']==(terminal=='COMPLETE')
         and eos==report['eos'] and len(trace)==report['steps']
         and len(actual)==report['bytes'] and bytes(actual)==raw,'SURFACE_READBACK_TERMINAL')
    return dict(tokens=len(trace),bytes=len(actual),status=terminal,ordinary_forward_exact=True)


def refused(code, action):
    try: action()
    except (ValueError,RuntimeError) as exc:
        need(code in str(exc),'SURFACE_READBACK_WRONG_REFUSAL')
        return dict(expected=code,error=str(exc))
    raise ValueError('SURFACE_READBACK_NEGATIVE_ACCEPTED:'+code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); need(not a.output.exists(),'SURFACE_READBACK_EXISTS')
    need(sha((a.source/'report.json').read_bytes())==a.pin,'SURFACE_READBACK_REPORT')
    source=read(a.source/'report.json'); gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('29','surface-runtime-readback'); configure(); started=time.perf_counter(); results=[]
    for row in source['cases']:
        root=Path(row['source']['prediction']); case=root/str(row['source']['case'])
        need(sha((root/'report.json').read_bytes())==row['source']['report_pin'],'SURFACE_READBACK_SOURCE')
        need(sha((root/'decoder.pt').read_bytes())==row['decoder_sha256']
             and sha((case/'prediction.npz').read_bytes())==row['predicted_slots'],'SURFACE_READBACK_WEIGHTS')
        meta=read(case/'view.json'); config=ModelConfig(**meta['config'])
        specs=tuple(Instrument(**s) for s in meta['specs'])
        decoder=PredictedSurfaceDecoder(config,specs).to('cuda').eval()
        decoder.load_state_dict(torch.load(root/'decoder.pt',weights_only=True,map_location='cuda'),strict=True)
        predicted=Constellation(slots(arrays(case/'prediction.npz'),specs,'readback-predicted'),
                                'predicted',tuple(meta['prediction_regions']))
        predicted.validate(specs,config)
        folder=a.source/str(row['case'])
        for filename,entry in row['files'].items():
            value=(folder/filename).read_bytes()
            need(len(value)==entry['bytes'] and sha(value)==entry['sha256'],'SURFACE_READBACK_FILE')
        trace=[json.loads(line) for line in (folder/'tokens.jsonl').read_text().splitlines()]
        raw=(folder/'candidate.bin').read_bytes()
        result=replay(decoder,predicted,trace,raw,row)
        controls=[]; scratch=a.output.with_suffix('.negative-must-not-exist')
        controls.append(refused('PREDICTION_REQUIRED',lambda:render(decoder,
            replace(predicted,origin='observed',regions=None),scratch)))
        incomplete=dict(predicted.slots);incomplete.pop(specs[0].name)
        controls.append(refused('CONSTELLATION_ROSTER',lambda:render(decoder,replace(predicted,slots=incomplete),scratch)))
        decoder.train()
        controls.append(refused('EVAL_REQUIRED',lambda:render(decoder,predicted,scratch)))
        decoder.eval(); need(not scratch.exists(),'SURFACE_READBACK_REFUSAL_MUTATION')
        altered=[dict(x) for x in trace]
        if altered:
            altered[0]['token']=(altered[0]['token']+1)%config.vocabulary
            controls.append(refused('READBACK_LOGITS',lambda:replay(decoder,predicted,altered,raw,row)))
        results.append(dict(case=row['case'],family=row['family'],controls=controls,**result))
        del decoder,predicted
    report=dict(schema='surface-runtime-readback-v1',source=str(a.source),source_pin=a.pin,cases=results,
        seconds=time.perf_counter()-started,gpu_before=gpu,producer_sha256=sha(Path(__file__).read_bytes()),
        number_status='measured',acceptance='author-checked',learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,report),cases=results)),flush=True)


if __name__=='__main__':run_cli(main)

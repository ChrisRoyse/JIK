"""Replay saved predictor views and run the actual strict streaming surface decoder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.capability_prediction_readback import load_view,arrays,slots,equal
from audit.fsv_identity_loop import configure
from claims.capability_population import need,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease,run_cli
from model.capability_objective import PredictedSurfaceDecoder
from model.capability_surface_runtime import render


def read(path): return json.loads(Path(path).read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); config=read(a.config); need(not a.output.exists(),'SURFACE_RUNTIME_FSV_EXISTS')
    gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-surface-runtime'); configure()
    started=time.perf_counter(); a.output.mkdir(); records=[]
    for i,item in enumerate(config['cases']):
        root=Path(item['prediction']); need(sha((root/'report.json').read_bytes())==item['report_pin'],'SURFACE_RUNTIME_FSV_REPORT')
        report=read(root/'report.json'); record=report['cases'][item['case']]; case=root/str(item['case'])
        for name in ('context','prediction'):
            need(sha((case/record['files'][name]['name']).read_bytes())==record['files'][name]['sha256'],
                 'SURFACE_RUNTIME_FSV_INPUT')
        meta=read(case/'view.json'); view=load_view(case,meta)
        definition=read(case/'definition.json')
        native=slots(arrays(case/'context.npz'),view.specs,digest(definition['context']))
        allowed=torch.ones(len(definition['regions']),dtype=torch.bool,device='cuda')
        with torch.no_grad():
            predicted=view(native,(view.start('capability:'+record['key']),),allowed=allowed,phase=0,
                event_ids=[meta['event']],evidence_origin=['source_observation'],advance_phase=False)['predicted_constellation']
        equal({n:x.values for n,x in predicted.slots.items()},arrays(case/'prediction.npz'),'SURFACE_RUNTIME_FSV_FORWARD')
        decoder=PredictedSurfaceDecoder(view.config,view.specs).to('cuda').eval()
        decoder.load_state_dict(torch.load(root/'decoder.pt',map_location='cuda',weights_only=True),strict=True)
        result=render(decoder,predicted,a.output/str(i))
        result.update(case=i,family=record['family'],task=record['key'],source=item,
            view_sha256=sha((case/'view.pt').read_bytes()),decoder_sha256=sha((root/'decoder.pt').read_bytes()),
            definition_sha256=sha((case/'definition.json').read_bytes()),input_slots=record['files']['context']['sha256'],
            predicted_slots=record['files']['prediction']['sha256'])
        write_committed(a.output/str(i)/'report.json',result); records.append(result)
        print(json.dumps(dict(case=i,family=record['family'],status=result['status'],bytes=result['bytes'])),flush=True)
        del predicted,view,decoder; torch.cuda.empty_cache()
    need(len(records)==4 and {r['family'] for r in records}=={'coding','conversation','reasoning'},'SURFACE_RUNTIME_FSV_SCOPE')
    result=dict(schema='capability-surface-runtime-fsv-v1',cases=records,gpu_before=gpu,
        config_sha256=sha(a.config.read_bytes()),producer_sha256=sha(Path(__file__).read_bytes()),
        runtime_sha256=sha((Path(__file__).resolve().parents[1]/'model/capability_surface_runtime.py').read_bytes()),
        seconds=time.perf_counter()-started,number_status='measured',acceptance='author-checked',
        scientific_admission=False,learned_capability=False,serving=False,
        generated_measurements_pending=sum(r['complete'] for r in records))
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result))),flush=True)


if __name__=='__main__': run_cli(main)

"""Open the sealed feedback journal without rebuilding or rewriting its ancestry."""
import json
from pathlib import Path

from audit.fsv_field_world import authenticate
from claims.codata_source import check, sha
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from model.condition_catalog import NativeConditions
from model.condition_input import ConditionNative
from model.controlled_session import ControlledLoop
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs


def open_session(source_root, source_pin, root, saved):
    proof = authenticate(source_root, source_pin)
    source = json.loads((source_root/'source.json').read_bytes())
    c = source['config']
    field = Path(source['field']['root'])
    authenticate(field, source['field']['seal'])
    retained = json.loads((field/'source.json').read_bytes())
    native, grounded = Path(retained['root']), Path(retained['retained']['root'])
    for path, pin in ((Path(source['root']),source['seal']), (native,retained['seal']),
                      (grounded,retained['retained']['seal']),
                      (Path(retained['upstream']['root']),retained['upstream']['seal'])):
        authenticate(path, pin)
    raw = ToolSource(Path(c['raw']))
    inputs = ToolInputs(raw, Path(c['population']), Path(c['native']),
                        population_pin=c['population_pin'], native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']), inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'], 'CONTROLLED_PARENT_PIN')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs, Path(c['kernel']), (Path(c['kernel'])/'COMMITTED').read_text())
    session = ControlledLoop(root/'experience.sqlite3', raw, inputs, parent_root/'initial',
        model_pin=parent['initial'], standard=parent['standard'], world=parent['world'],
        contexts={'train:1':c['providers'][0]['task']}, device='cuda',
        condition_inputs=NativeConditions(providers), routing=routing,
        growth_root=Path(c['output'])/'growth', candidate_root=grounded/'candidates',
        native_root=native/'candidates', field_root=field/'candidates', controlled_root=root/'candidates',
        expected_head=saved['head'], expected_contract=saved['contract'])
    check(session.snapshot() == saved, 'CONTROLLED_PREFIX_REPLAY')
    return session, proof

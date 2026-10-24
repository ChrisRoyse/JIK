"""Real preparation and separate physical readback before any model forecast."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposed_preparation_readback import physical
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.remaining_attempt_source import RemainingAttemptSource
from claims.attempt_population import AttemptPopulation, build as population_build
from claims.process_conditions import write_committed
from claims.proposal_population import cached
from oracle.prepare_attempt import PreparedAttempt, build as prepare


def run(config, stage):
    started = time.perf_counter()
    source = RemainingAttemptSource(config['source']['root'],config['source']['pin'])
    raw = Path(config['source_readback']['path']).read_bytes()
    source_readback = json.loads(raw)
    check(sha(raw) == config['source_readback']['pin'] and source_readback['source'] == source.pin
          and source_readback['source_reference'] == source.reference()
          and len(source_readback['controls']) == 4
          and source_readback['immutable_before'] == source_readback['immutable_after'],
          'ATTEMPT_SETUP_SOURCE_READBACK')
    root,pop = Path(config['prepared']),Path(config['population'])
    if stage == 'build':
        prepared = prepare(source,Path(config['runtime']['root']),config['runtime']['pin'],root)
        population = population_build(prepared,Path(config['cache']['root']),config['cache']['pin'],pop)
        return dict(prepared_pin=prepared.conditions.pin,population_pin=population.pin,
                    population=population.report,wall_seconds=time.perf_counter()-started)
    before = physical(root)
    prepared = PreparedAttempt(root,(root/'conditions-COMMITTED').read_text())
    population = AttemptPopulation(pop,(pop/'COMMITTED').read_text())
    check(population.prepared.spec == prepared.spec, 'ATTEMPT_SETUP_JOIN')
    parsed = time.perf_counter()
    spec = json.loads((root/'prepared.json').read_bytes())
    routing = json.loads((root/'condition-commitment.json').read_bytes())['routing']
    public = json.loads((root/'condition-inputs.json').read_bytes())['inputs']
    checks = {}
    for name,item in spec['conditions'].items():
        program = (root/name/'work/program.py').read_bytes()
        check(program == source.action[slice(*source.row['utf8_span'])] == source.program, 'ATTEMPT_SETUP_PROGRAM')
        original = Path(spec['runtime_source']['root'])/name/'runtime'
        wheels = {p.name:sha(p.read_bytes()) for p in (root/name/'runtime').iterdir()}
        check(wheels == {p.name:sha(p.read_bytes()) for p in original.iterdir()} == item['wheels'],
              'ATTEMPT_SETUP_RUNTIME')
        p = public[routing[name]]
        check(p['program'].encode() == program and p['wheels'] == wheels and digest(p) == routing[name],
              'ATTEMPT_SETUP_PUBLIC')
        checks[name] = dict(program=sha(program),bytes=len(program),wheels=wheels,condition=routing[name])
    # Read the durable Parquet and field links, rather than trusting returned rows.
    rows = pq.read_table(pop/'proposal/chunks/atoms.parquet').to_pylist()
    fields = json.loads((pop/'fields.json').read_bytes())
    cache_rows,_,cache_probe = cached(population.cache_root,population.cache_pin)
    for item in fields:
        text = (rows if item['origin']=='new' else cache_rows)[item['unit']]['text']
        expected = source.action.decode() if item['role']=='action' else source.program.decode() if item['role']=='program' else sorted(public[item['condition']]['wheels'])[item['ordinal']-2]
        check(text == expected and sha(text.encode()) == item['text_sha256'], 'ATTEMPT_SETUP_FIELD')
        check(item['unit'] != (population.probe['unit'] if item['origin']=='new' else cache_probe['unit']),
              'ATTEMPT_SETUP_PROBE_LEAK')
    output = Path(config['readback'])
    check(not output.exists(), 'ATTEMPT_SETUP_READBACK_EXISTS')
    output.mkdir()
    scratch = output/'control-copy'
    shutil.copytree(root,scratch)
    controls = {}
    for relative,code in (('with_dependency/work/program.py','ATTEMPT_PREPARED_BYTES'),
                          ('without_dependency/runtime/packaging-24.1-py3-none-any.whl','ATTEMPT_PREPARED_BYTES'),
                          ('process_inside.py','ATTEMPT_PREPARED_BYTES'),('prepared.json','ATTEMPT_PREPARED_PIN'),
                          ('condition-inputs.json','CONDITION_INPUT_PIN')):
        path = scratch/relative
        old = path.read_bytes()
        try:
            path.write_bytes(old+b' ')
            try:
                PreparedAttempt(scratch,prepared.conditions.pin)
            except ValueError as error:
                check(code in str(error), 'ATTEMPT_SETUP_CONTROL_CODE:'+relative+':'+str(error))
            else:
                raise AssertionError('ATTEMPT_SETUP_CONTROL_ACCEPTED:'+relative)
        finally:
            path.write_bytes(old)
        controls[relative] = code
    check(physical(root) == before, 'ATTEMPT_SETUP_SOURCE_MUTATION')
    finished = time.perf_counter()
    return dict(schema='attempt-setup-readback-v1',acceptance='author-checked',number_status='measured',
        source=source.reference(),prepared_pin=prepared.conditions.pin,population_pin=population.pin,
        conditions=checks,inputs=len(rows),fields=len(fields),controls=controls,
        native_model_surfaces=population.report['new_model_surfaces'],cached_surfaces=population.report['cached_model_surfaces'],
        boundaries=['source_span_to_work_program','runtime_bytes_to_public_conditions','durable_fields_to_population'],
        phases=dict(authentication=parsed-started,physical_controls=finished-parsed),wall_seconds=finished-started,
        executed=False,forecast_present=False,serving=False,scientific_admission=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('build','read'),required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_bytes())
    result = run(config,a.stage)
    output = Path(config['build_report']) if a.stage == 'build' else Path(config['readback'])/'readback.json'
    pin = write_committed(output,result)
    print(json.dumps(dict(pin=pin,**{k:v for k,v in result.items() if k!='population'})),flush=True)

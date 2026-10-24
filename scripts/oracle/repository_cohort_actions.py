"""Execute remaining recorded actions against each task's already-witnessed oracle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle.bound_rows import BoundRows
from oracle.container_run import execute
from oracle.prepare_swe import (apply_patch,check_modes,executables,inventory,new_executables,
                                patch_paths,relative,reset_test_files,source_files)
from oracle.replay_swe import recount,verify_prepared
from oracle.scm_version import generated
from oracle.test_contract import junit_identity,selectors
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_COHORT_ACTION'
META = ['instance_id','base_commit','test_patch']
ATTEMPT = ['instance_id','generated_patch','target']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(execution,required,result):
    record = json.loads((execution/'execution.json').read_bytes())
    lines = (execution/'stdout.bin').read_text().splitlines()
    envelopes = [json.loads(v.split('=',1)[1]) for v in lines if v.startswith('MNLM_EXECUTION_RESULT=')]
    require(envelopes == [record['result']],CODE,'stdout/envelope mismatch')
    lost = {n for n in required if n.split('::',1)[0] in (record['result'].get('uncollectable') or {})}
    xml = record['result']['junit']
    cases = list(ET.fromstring(xml).iter('testcase')) if xml else []
    states = {(c.attrib['classname'],c.attrib['name']):'failed' if c.find('failure') is not None else 'passed'
              for c in cases}
    expected = {junit_identity(n):state for n,state in result['states'].items() if n not in lost}
    require(states == expected and len(cases)+len(lost) == len(required) and
            all(result['states'][n] == 'failed' for n in lost),CODE,'independent JUnit differs')
    return dict(tests=len(required),failed=sum(v == 'failed' for v in result['states'].values()),
                collection_failed=len(lost),exitcode=record['exitcode'])


def open_witness(root,item,witness,code):
    plan = read_pinned(root/witness['plan']['path'],witness['plan']['sha256'])
    prepared = Path(witness['prepared'])
    prepared_pin = digest(prepared/'prepared.json')
    spec,before = verify_prepared(prepared,prepared_pin)
    last = witness['runs'][-1]
    fsv = read_pinned(Path(item['path']).parent/'fsv'/last['tag']/'report.json',last['fsv_report_sha256'])
    require(before == fsv['before'] == fsv['after'] and spec['plan_sha256'] == witness['plan']['sha256'] and
            spec['task'] == witness['task'],CODE,'prepared state differs from pinned witness')
    execution = Path(witness['execution'])
    report = read_pinned(execution/'report.json',digest(execution/'report.json'))
    require(report['prepared_sha256'] == prepared_pin and report['before'] == report['after'] == before and
            all(code.get(n) == v for n,v in report['code_sha256'].items()),CODE,'witness/code changed')
    for name in ('base','reference','agent'):
        record = json.loads((execution/name/'execution.json').read_bytes())
        result = recount(record,spec['required_tests'],allow_uncollectable=name != 'reference')
        checked = independent(execution/name,spec['required_tests'],result)
        require(result == report['conditions'][name] and checked['failed'] == fsv['independent'][name]['failed'] and
                checked['collection_failed'] == fsv['independent'][name]['collection_failed'] and
                checked['tests'] == fsv['independent'][name]['tests']+checked['collection_failed'],
                CODE,'witness execution differs from pinned readback')
    require(report['conditions']['reference']['outcome'] == 1 and
            all(report['conditions']['base']['states'][n] == ('failed' if role == 'FAIL_TO_PASS' else 'passed')
                for n,role in spec['required_tests'].items()),CODE,'original environment witness')
    modes = {};files = source_files(root,plan,modes)
    shared = dict(path=item['path'],sha256=item['sha256'],prepared_sha256=prepared_pin,
                  execution_sha256=digest(execution/'report.json'),fsv_sha256=last['fsv_report_sha256'])
    return plan,prepared,prepared_pin,spec,before,modes,files,shared


def action(root,cohort,case,attempt,rows,plan,prepared,spec,files,modes,shared,output):
    start = time.perf_counter()
    output.mkdir()
    row = rows.read(attempt['binding'],ATTEMPT)
    meta = rows.read(case['benchmark'],META)
    text = row['generated_patch']
    require(row['instance_id'] == meta['instance_id'] == case['task'] and
            meta['base_commit'] == case['base_commit'] and
            text.encode() == (cohort.root/attempt['path']).read_bytes() and
            hashlib.sha256(text.encode()).hexdigest() == attempt['sha256'],CODE,'action/source binding')
    allowed = patch_paths(text)
    test_paths = patch_paths(meta['test_patch'])
    require(sorted(test_paths) == plan['allowed_paths']['test'],CODE,'test patch differs from witness')
    tree = output/'agent';tree.mkdir()
    for name,value in files.items():
        path = tree/relative(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    extras = generated(files) if plan.get('generated_files') else []
    require([dict(path=g['path'],rule=g['rule'],sha256=hashlib.sha256(g['content']).hexdigest()) for g in extras]
            == plan.get('generated_files',[]),CODE,'generated source declaration')
    for g in extras:
        path = tree/relative(g['path']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(g['content'])
    execs = sorted(n for n,m in modes.items() if m == '100755')
    executables(tree,execs)
    method = apply_patch(tree,output/'agent.patch',text,allowed,fallback=plan['agent_patch_fallback'])
    reset = reset_test_files(tree,files,test_paths)
    apply_patch(tree,output/'test.patch',meta['test_patch'],test_paths)
    expected_execs = sorted(set(execs)|new_executables(text)|new_executables(meta['test_patch']))
    check_modes(tree,expected_execs)
    before = inventory(tree)
    require(all(before.get(n) == spec['conditions']['reference'].get(n) for n in test_paths),
            CODE,'action used different test bytes')
    preparation = dict(schema='repository-recorded-action-preparation-v1',task=case['task'],
        partition=case['partition'],attempt=attempt,witness=shared,image=spec['image'],
        runtime=str(prepared/'runtime'),runtime_inventory=spec['runtime'],condition=before,
        executables=expected_execs,required_tests=spec['required_tests'],
        patch_method=method,reset_test_files=reset,provider_target=row['target'],admitted=False)
    prep_pin = publish(output/'prepared.json',preparation)
    built = time.perf_counter()
    execute(spec['image'],tree,prepared/'runtime',output/'execution',modules=selectors(spec['required_tests']))
    record = json.loads((output/'execution/execution.json').read_bytes())
    result = recount(record,spec['required_tests'],allow_uncollectable=True)
    checked = independent(output/'execution',spec['required_tests'],result)
    require(inventory(tree) == before,CODE,'execution changed action bytes')
    wall = time.perf_counter()-start
    status = 'observed' if result['outcome'] == int(row['target']) else 'disputed'
    report = dict(schema='repository-recorded-action-v1',task=case['task'],partition=case['partition'],
        status=status,preparation_sha256=prep_pin,execution_sha256=digest(output/'execution/execution.json'),
        witness=shared,result=result,independent_readback=checked,provider_target=row['target'],
        provider_agrees=status == 'observed',before=before,after=before,wall_seconds=wall,
        phases=dict(preparation=built-start,execution_and_readback=wall-(built-start)),
        learned_action=False,admitted=False,acceptance='author-checked')
    pin = publish(output/'report.json',report)
    return dict(status=status,path=str(output/'report.json'),sha256=pin,outcome=result['outcome'],
                wall_seconds=wall)


def run(witness_path,witness_pin,output):
    start = time.perf_counter()
    scope = read_pinned(witness_path,witness_pin)
    require(scope['schema'] == 'repository-cohort-witness-v1',CODE,'witness schema')
    root = Path(scope['root'])
    require(output.resolve().is_relative_to(root) and not output.exists(),CODE,'fresh MNLM output required')
    cohort = RepositoryCohort(scope['cohort']['path'],scope['cohort']['sha256'])
    cases = cohort.training_cases('learn_1')+cohort.training_cases('learn_2')
    requests = []
    for case in cases:
        requests.append((case['benchmark'],META))
        requests.extend((a['binding'],ATTEMPT) for a in case['attempts'][1:])
    rows = BoundRows(root,requests)
    output.mkdir(parents=True)
    publish(output/'bound-rows.json',rows.receipts)
    results = []
    code = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    binding_seconds = time.perf_counter()-start
    tick = time.perf_counter()
    for case,item in zip(cases,scope['results'],strict=True):
        require(case['task'] == item['task'] and case['partition'] == item['partition'],CODE,'witness selection')
        witness = read_pinned(Path(item['path']),item['sha256'])
        destination = output/str(case['index']).zfill(2);destination.mkdir()
        if witness['status'] != 'observed':
            result = dict(task=case['task'],partition=case['partition'],status='gap',
                          error='ORIGINAL_TASK_WITNESS_REQUIRED',source_error=witness.get('error'))
            publish(destination/'gap.json',result);results.append(result)
            continue
        try:
            plan,prepared,prepared_pin,spec,before,modes,files,shared = open_witness(root,item,witness,code)
        except (SenseError,OSError,ValueError,KeyError,TypeError) as exc:
            error = exc.envelope() if isinstance(exc,SenseError) else dict(
                code=CODE+'_WITNESS',message='Shared witness unavailable.',exception=type(exc).__name__)
            result = dict(task=case['task'],partition=case['partition'],status='gap',error=error)
            publish(destination/'gap.json',result);results.append(result)
            print(json.dumps(result),flush=True)
            continue
        actions = []
        for ordinal,attempt in enumerate(case['attempts'][1:],1):
            path = destination/str(ordinal)
            try:
                result = action(root,cohort,case,attempt,rows,plan,prepared,spec,files,modes,shared,path)
            except Exception as exc:
                error = exc.envelope() if isinstance(exc,SenseError) else dict(
                    code=CODE+'_EXECUTION',message='Action not evaluated; inspect retained evidence.',
                    exception=type(exc).__name__)
                path.mkdir(exist_ok=True)
                pin = publish(path/'gap.json',dict(task=case['task'],attempt=attempt,error=error))
                result = dict(status='gap',path=str(path/'gap.json'),sha256=pin,error=error)
            actions.append(dict(ordinal=ordinal,**result))
            print(json.dumps(dict(task=case['task'],ordinal=ordinal,**result)),flush=True)
        require(before == verify_prepared(prepared,prepared_pin)[1],CODE,'shared witness/runtime changed')
        result = dict(task=case['task'],partition=case['partition'],status='executed',actions=actions,witness=shared)
        publish(destination/'result.json',result);results.append(result)
    end = time.perf_counter()
    wall = end-start
    report = dict(schema='repository-cohort-actions-v1',witness=dict(path=str(witness_path),sha256=witness_pin),
        results=results,wall_seconds=wall,phases=dict(binding=binding_seconds,cases=end-tick,
                                                   other=wall-binding_seconds-(end-tick)),
        action_counts=dict(Counter(a['status'] for r in results for a in r.get('actions',[]))),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reserved_evaluation_used=False,learned_skill=False,admitted=False,acceptance='author-checked')
    pin = publish(output/'summary.json',report)
    print(json.dumps(dict(path=str(output/'summary.json'),sha256=pin,counts=report['action_counts'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--witness',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.witness,a.pin,a.output)

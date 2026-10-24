"""Cold readback of recorded actions, their original oracle witnesses and refusals."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle.container_run import validate_config
from oracle.prepare_swe import check_modes,inventory,patch_paths
from oracle.replay_swe import recount,verify_prepared
from oracle.repository_cohort_actions import digest,open_witness
from oracle.test_contract import junit_identity,selectors
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_ACTION_READBACK'


def payload(tree,expected,execs):
    require(inventory(tree) == expected,CODE,'action source bytes differ')
    check_modes(tree,execs)


def independent(execution,required,states):
    record = json.loads((execution/'execution.json').read_bytes())
    markers = [v.split('=',1)[1] for v in (execution/'stdout.bin').read_text().splitlines()
               if v.startswith('MNLM_EXECUTION_RESULT=')]
    require(len(markers) == 1 and json.loads(markers[0]) == record['result'],CODE,'stdout binding')
    r = record['result']
    lost = {n for n in required if n.split('::',1)[0] in (r.get('uncollectable') or {})}
    cases = list(ET.fromstring(r['junit']).iter('testcase')) if r['junit'] else []
    require(len(cases)+len(lost) == len(required),CODE,'independent test denominator')
    seen = {}
    for case in cases:
        key = (case.attrib['classname'],case.attrib['name'])
        require(key not in seen and case.find('error') is None and case.find('skipped') is None,
                CODE,'abnormal or duplicated JUnit test')
        seen[key] = 'failed' if case.find('failure') is not None else 'passed'
    require(seen == {junit_identity(n):v for n,v in states.items() if n not in lost} and
            all(states[n] == 'failed' for n in lost),CODE,'independent states differ')
    return dict(tests=len(required),passed=sum(v == 'passed' for v in seen.values()),
                failed=sum(v == 'failed' for v in seen.values())+len(lost),uncollectable=len(lost))


def read_action(item,case,cohort,spec,prepared,shared):
    report_path = Path(item['path'])
    report = read_pinned(report_path,item['sha256'])
    directory = report_path.parent
    prep = read_pinned(directory/'prepared.json',report['preparation_sha256'])
    attempt = case['attempts'][item['ordinal']]
    require(report['task'] == prep['task'] == case['task'] and
            report['partition'] == prep['partition'] == case['partition'] and
            case['partition'] in ('learn_1','learn_2') and prep['attempt'] == attempt and
            report['witness'] == prep['witness'] == shared,CODE,'task/action/partition binding')
    require((directory/'agent.patch').read_bytes() == (cohort.root/attempt['path']).read_bytes() and
            digest(directory/'agent.patch') == attempt['sha256'],CODE,'original recorded action')
    require(prep['image'] == spec['image'] and prep['runtime'] == str(prepared/'runtime') and
            prep['runtime_inventory'] == spec['runtime'] and prep['required_tests'] == spec['required_tests'],
            CODE,'shared oracle differs')
    expected = prep['condition']
    payload(directory/'agent',expected,prep['executables'])
    require(report['before'] == report['after'] == expected,CODE,'action before/after differs')
    test_paths = patch_paths((directory/'test.patch').read_text())
    require(all(expected.get(n) == spec['conditions']['reference'].get(n) for n in test_paths),
            CODE,'protected test bytes differ')
    execution = directory/'execution'
    raw = read_pinned(execution/'execution.json',report['execution_sha256'])
    result = recount(raw,spec['required_tests'],allow_uncollectable=True)
    require(result == report['result'] and item['outcome'] == result['outcome'],CODE,'outcome differs')
    checked = independent(execution,spec['required_tests'],result['states'])
    before = json.loads((execution/'before.json').read_bytes())
    after = json.loads((execution/'after.json').read_bytes())
    mounts = {'/work':str(directory/'agent'),'/runtime':str(prepared/'runtime'),
              '/harness.py':str(Path(__file__).with_name('pytest_inside.py'))}
    label = before['Config']['Labels']['mnlm.oracle.run']
    work = (prepared/'runtime/mnlm-runtime.json').exists()
    for config in (before,after):
        validate_config(config,spec['image'],label,mounts,work)
        require(config['Config']['Cmd'] == ['-I','-B','/harness.py','tests',*selectors(spec['required_tests'])],
                CODE,'execution command differs')
    require(before['Id'] == after['Id'] == raw['container_id'] and not before['State']['Running'] and
            not after['State']['Running'] and not after['State']['OOMKilled'] and
            after['State']['ExitCode'] == raw['exitcode'] and
            after['State']['StartedAt'] < after['State']['FinishedAt'],CODE,'container lifecycle differs')
    agrees = result['outcome'] == int(prep['provider_target'])
    require(report['provider_target'] == prep['provider_target'] and report['provider_agrees'] == agrees and
            report['status'] == item['status'] == ('observed' if agrees else 'disputed'),CODE,'provider comparison')
    return dict(ordinal=item['ordinal'],outcome=result['outcome'],status=item['status'],**checked)


def controls(cohort,prepared,spec,execution,output,summary_path):
    checks = []

    def refuses(name,action,code):
        try:
            action()
        except (SenseError,ValueError) as exc:
            actual = exc.code if isinstance(exc,SenseError) else str(exc)
            require(actual == code,CODE,name+' wrong refusal')
            checks.append(dict(name=name,code=actual))
        else:
            raise SenseError(CODE,name+' accepted broken state')

    good = json.loads((execution/'reference/execution.json').read_bytes())
    for name,edit in (
        ('missing-test',lambda r:r['result']['collected'].pop()),
        ('duplicate-test',lambda r:r['result']['collected'].append(r['result']['collected'][0])),
        ('missing-phase',lambda r:r['result']['phases'].pop()),
        ('failed-setup',lambda r:r['result']['phases'][0].update(outcome='failed')),
        ('junit-mismatch',lambda r:r['result'].update(junit='<testsuites/>')),
        ('timeout-is-not-outcome',lambda r:r.update(status='MNLM_EXECUTION_TIMEOUT'))):
        changed = deepcopy(good);edit(changed)
        publish(output/(name+'.json'),changed)
        refuses(name,lambda:recount(changed,spec['required_tests']),'MNLM_EXECUTION_EVIDENCE')
    refuses('reserved-is-not-training',lambda:cohort.training_cases('reserved_eval'),
            'MNLM_CODATA_COHORT_TRAINING_PARTITION')
    refuses('wrong-summary-digest',lambda:read_pinned(summary_path,'0'*64),'MNLM_UCR_INPUT_DIGEST')
    refuses('path-traversal',lambda:patch_paths('diff --git a/../escape b/../escape\n'),
            'MNLM_EXECUTION_PREPARATION')
    copied = output/'mutated-source';shutil.copytree(prepared/'agent',copied)
    original = spec['conditions']['agent']
    selected = copied/sorted(original)[0]
    selected.write_bytes(selected.read_bytes()+b'\n# readback control\n')
    refuses('source-changed',lambda:payload(copied,original,spec.get('executables',[])),CODE)
    info = json.loads((execution/'probe/before.json').read_bytes())
    mounts = {m['Destination']:m['Source'] for m in info['Mounts']}
    changed = deepcopy(info);changed['HostConfig']['NetworkMode'] = 'host'
    publish(output/'network-enabled.json',changed)
    refuses('network-enabled',lambda:validate_config(changed,spec['image'],
        info['Config']['Labels']['mnlm.oracle.run'],mounts,'/work-rw' in info['HostConfig']['Tmpfs']),
        'MNLM_EXECUTION_SANDBOX')
    return checks


def run(path,pin,output):
    start = time.perf_counter()
    report = read_pinned(path,pin)
    scope = read_pinned(Path(report['witness']['path']),report['witness']['sha256'])
    require(report['schema'] == 'repository-cohort-actions-v1' and
            digest(Path(__file__).with_name('repository_cohort_actions.py')) == report['code_sha256'],
            CODE,'action writer changed')
    cohort = RepositoryCohort(scope['cohort']['path'],scope['cohort']['sha256'])
    cases = cohort.training_cases('learn_1')+cohort.training_cases('learn_2')
    output.mkdir(parents=True,exist_ok=False)
    code = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    results,checks,control_target = [],[],None
    for case,item,first in zip(cases,report['results'],scope['results'],strict=True):
        require(case['task'] == item['task'] == first['task'] and case['partition'] == item['partition'],
                CODE,'cohort selection differs')
        witness = read_pinned(Path(first['path']),first['sha256'])
        if witness['status'] != 'observed':
            require(item['status'] == 'gap' and not item.get('actions'),CODE,'unwitnessed task evaluated')
            results.append(dict(task=case['task'],partition=case['partition'],status='gap',actions=[]))
            continue
        plan,prepared,prep_pin,spec,before,modes,files,shared = open_witness(Path(scope['root']),first,witness,code)
        require(item['witness'] == shared and [a['ordinal'] for a in item['actions']] == [1,2,3],
                CODE,'action coverage differs')
        actions = [dict(ordinal=0,outcome=witness['agent_outcome'],status='observed')]
        for action in item['actions']:
            if action['status'] == 'gap':
                gap = read_pinned(Path(action['path']),action['sha256'])
                require(gap['task'] == case['task'] and gap['attempt'] == case['attempts'][action['ordinal']],
                        CODE,'gap action differs')
                actions.append(dict(ordinal=action['ordinal'],status='gap',error=gap['error']))
            else:
                actions.append(read_action(action,case,cohort,spec,prepared,shared))
        if control_target is None:
            control_target = (prepared,spec,Path(witness['execution']))
        require(verify_prepared(prepared,prep_pin)[1] == before,CODE,'shared bytes changed during readback')
        results.append(dict(task=case['task'],partition=case['partition'],status='readback',actions=actions))
    require(control_target is not None,CODE,'no observed witness')
    checks = controls(cohort,*control_target,output,path)
    value = dict(schema='repository-cohort-actions-readback-v1',source=dict(path=str(path),sha256=pin),
        results=results,checks=checks,negative_controls=len(checks),
        outcomes=dict(Counter(a.get('outcome','gap') for r in results for a in r['actions'])),
        wall_seconds=time.perf_counter()-start,code_sha256=digest(Path(__file__)),
        learned_skill=False,admitted=False,acceptance='author-checked')
    result = publish(output/'report.json',value)
    print(json.dumps(dict(path=str(output/'report.json'),sha256=result,outcomes=value['outcomes'],
                         negative_controls=len(checks))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.summary,a.pin,a.output)

"""Independent bytes, native windows and journal-state readback for proposal feedback."""
import argparse
import base64
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from claims.fresh_echo_population import FreshEchoPopulation
from model.fresh_echo_input import FreshEchoInputs


def decode(value):
    raw = base64.b64decode(value['base64'],validate=True)
    check(sha(raw) == value['sha256'] and value['dtype'] == 'float32-le','PROPOSAL_ECHO_READBACK_TENSOR')
    return np.frombuffer(raw,dtype='<f4').reshape(value['shape']).copy()


def distance(matrix):
    n,m = matrix.shape
    costs = np.zeros((n+1,m+1),dtype=np.float64)
    costs[:,0], costs[0,:] = np.arange(n+1),np.arange(m+1)
    for i in range(n):
        for j in range(m):
            costs[i+1,j+1] = min(costs[i,j+1]+1,costs[i+1,j]+1,costs[i,j]+1-matrix[i,j])
    return float(costs[n,m]/max(n,m))


def readback(root, output):
    check(not output.exists(),'PROPOSAL_ECHO_READBACK_EXISTS')
    output.mkdir()
    config = json.loads((root/'config.json').read_bytes())
    inputs = FreshEchoInputs(**config['inputs'])
    population = inputs.population
    arrays = {}
    for spec in inputs.specs:
        with np.load(inputs.native/spec.name/'slots.npz',allow_pickle=False) as archive:
            vectors,offsets = archive['vectors'],archive['offsets']
            arrays[spec.name] = {}
            for key in inputs.keys:
                units = [r['unit'] for r in population.refs if r['observation'] == key and r['unit'] is not None]
                check(population.probe['unit'] not in units,'PROPOSAL_ECHO_READBACK_PROBE')
                actual = np.concatenate([vectors[offsets[u]:offsets[u+1]] for u in units])
                i = inputs.lookup[key]
                adapter, ptr = inputs.arrays[spec.name]
                check(np.array_equal(actual,adapter[ptr[i]:ptr[i+1]]),'PROPOSAL_ECHO_READBACK_WINDOWS')
                arrays[spec.name][key] = actual
    controls = []

    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'PROPOSAL_ECHO_READBACK_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('PROPOSAL_ECHO_READBACK_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))

    refuse('wrong_population_pin','PROPOSAL_ECHO_POPULATION_PIN',
           lambda:FreshEchoPopulation(population.root,'0'*64))
    bad = dict(config['inputs'],native_pin='0'*64)
    refuse('wrong_native_pin','PROPOSAL_NATIVE_PIN',lambda:FreshEchoInputs(**bad))
    refs = copy.deepcopy(population.refs)
    refs[1]['unit'] = population.probe['unit']
    refuse('probe_as_output','PROPOSAL_ECHO_PROBE_LEAK',
           lambda:inputs.materialize(inputs.keys[0],inputs.specs[0].name,refs=refs))
    refuse('stream_order_swapped','PROPOSAL_ECHO_STREAM_ORDER',
           lambda:inputs.materialize(inputs.keys[0],inputs.specs[0].name,refs=list(reversed(population.refs))))
    corrupted = output/'corrupt-native'
    shutil.copytree(inputs.native,corrupted)
    p = corrupted/inputs.specs[0].name/'slots.npz'
    p.write_bytes(p.read_bytes()+b'changed')
    refuse('changed_persisted_windows','MNLM_WINDOW_READBACK_FILE_HASH',
           lambda:FreshEchoInputs(**dict(config['inputs'],native=str(corrupted))))
    source = json.loads((Path(config['source'])/'snapshot-final.json').read_bytes())
    final = json.loads((root/'snapshot-final.json').read_bytes())
    enrollment = json.loads((root/'enrollment.json').read_bytes())
    begin,finish,cold = [json.loads((root/(s+'-report.json')).read_bytes()) for s in ('begin','finish','cold')]
    contexts = copy.deepcopy(enrollment['payload']['contexts'])
    check(len(contexts) == 4 and enrollment['body']['payload'] == digest(enrollment['payload']),
          'PROPOSAL_ECHO_READBACK_ENROLLMENT')
    head,count,transitions = ZERO,0,[]
    with sqlite3.connect((root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous,raw,actual in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body)
                  and actual == digest(dict(seq=seq,previous=previous,body=body)),'PROPOSAL_ECHO_READBACK_CHAIN')
            if seq == source['events']-1:
                check(actual == source['head'],'PROPOSAL_ECHO_READBACK_PREFIX')
            payload = None
            if body.get('payload') is not None:
                data, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(data) == body['payload'],'PROPOSAL_ECHO_READBACK_PAYLOAD')
                payload = json.loads(data)
            if body['kind'] == 'fresh_echo_enrollment':
                check(body == enrollment['body'] and payload == enrollment['payload'],'PROPOSAL_ECHO_READBACK_ENROLLMENT')
            if body['kind'] == 'field_echo_transition' and body['request']['context'] in contexts:
                request = body['request']
                ctx = contexts[request['context']]
                check(request['observation'] == ctx['observation'] and payload['before'] == ctx['state']
                      and request['phase'] == (3 if ctx['status'] == 'awaiting_echo' else 4)
                      and payload['after']['sequence'] == payload['before']['sequence']+1,
                      'PROPOSAL_ECHO_READBACK_STATE')
                old,new = decode(payload['before']['hidden']),decode(payload['after']['hidden'])
                check(np.all(np.any(old != new,axis=1)),'PROPOSAL_ECHO_READBACK_UNCHANGED_LANE')
                forecasts = inputs.observations.forecasts['predictions'][ctx['condition']]['models'][ctx['variant']]
                errors = {}
                for i,spec in enumerate(inputs.specs):
                    actual_values = arrays[spec.name][ctx['observation']]
                    check(payload['observed'][spec.name] == dict(shape=list(actual_values.shape),
                          dtype=actual_values.dtype.str,sha256=sha(actual_values.tobytes())),
                          'PROPOSAL_ECHO_READBACK_OBSERVED')
                    check(forecasts['constellation'][spec.name]['instrument'] == spec.identity,
                          'PROPOSAL_ECHO_READBACK_INSTRUMENT')
                    predicted = decode(forecasts['constellation'][spec.name]).astype(np.float64)
                    actual_values = actual_values.astype(np.float64)
                    predicted /= np.linalg.norm(predicted,axis=1,keepdims=True)
                    actual_values /= np.linalg.norm(actual_values,axis=1,keepdims=True)
                    cosines = np.clip(actual_values@predicted.T,-1,1)
                    expected = payload['prediction_error'][spec.name]
                    check(np.array_equal(cosines,np.asarray(expected['cosines']))
                          and distance(cosines) == expected['ordered_distance'],'PROPOSAL_ECHO_READBACK_ERROR')
                    delta = float(np.linalg.norm(new[i]-old[i]))
                    check(abs(delta-payload['state_delta_by_instrument'][spec.name]) < 1e-6,
                          'PROPOSAL_ECHO_READBACK_DELTA')
                    errors[spec.name] = expected['ordered_distance']
                check(payload['frozen'] == ctx['fork']['candidate']['fingerprint']
                      and all(payload[k] is False for k in ('learned_understanding','new_weight_update',
                          'correction_effectiveness_established','scientific_admission','serving')),
                      'PROPOSAL_ECHO_READBACK_SCOPE')
                ctx.update(state=payload['after'],status='awaiting_correction' if request['phase']==3 else 'corrected_state',
                           last_transition=body['payload'])
                transitions.append(dict(context=request['context'],variant=ctx['variant'],condition=ctx['condition'],
                                        phase=request['phase'],errors=errors))
            head,count = actual,count+1
    check(count == source['events']+9 == final['events'] and head == final['head'] and len(transitions) == 8,
          'PROPOSAL_ECHO_READBACK_COUNTS')
    check(all(final['native_feedback_contexts'][k] == v for k,v in {**source['native_feedback_contexts'],**contexts}.items()),
          'PROPOSAL_ECHO_READBACK_CONTEXTS')
    check(cold['snapshot'] == digest(final) and cold['head'] == head,'PROPOSAL_ECHO_READBACK_RESTART')
    report = dict(schema='fresh-echo-readback-v1',author_checked=True,head=head,events=count,
        source_head=source['head'],population=population.pin,native=inputs.identity['native'],
        windows={n:{k:len(v) for k,v in rows.items()} for n,rows in arrays.items()},
        contexts=len(contexts),transitions=transitions,controls=controls+begin['controls']+finish['controls'],
        isolation=begin['isolation']+finish['isolation'],cold_snapshot=cold['snapshot'],
        boundaries=['captured_bytes_to_native_windows','sealed_action_to_observed_echo','correction_state_and_restart'],
        scientific_admission=False,serving=False,new_weight_update=False,correction_effectiveness_established=False)
    pin = write_committed(output/'readback.json',report)
    return dict(pin=pin,events=count,contexts=len(contexts),transitions=len(transitions),controls=len(report['controls']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    print(json.dumps(readback(a.root,a.output)))

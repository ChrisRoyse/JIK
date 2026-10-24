"""Pinned create-only graph publication; a recall score cannot authorize meaning."""
from pathlib import Path

from kernel.capsule import LANES, load as load_capsule, read, sha, write
from kernel.derive import derive
from kernel.native_route import Router, check


def code_pins():
    root = Path(__file__).parent
    return {name: sha(root/name) for name in
            ('native_route.py', 'derive.py', 'capsule.py', 'artifact.py')}


def build(capsule, capsule_pin, output):
    check(not output.exists(), 'ARTIFACT_EXISTS')
    meta, lanes, queries = load_capsule(capsule, capsule_pin)
    output.mkdir()
    reports = {}
    for name in LANES:
        lane = lanes[name]
        edges, work = lane.weave(meta['plan'])
        graph = derive(lane.ids, edges, meta['plan'])
        write(output/(name+'.json'), graph)
        reports[name] = dict(file_sha256=sha(output/(name+'.json')),
                             corpus=len(lane.ids), edges=len(edges),
                             candidate_nodes=len(graph['candidate_graph']),
                             members=len(graph['members']), entries=len(graph['entries']),
                             ann_work=work, lens_id=meta['lanes'][name]['lens_id'])
    manifest = dict(schema='native-kernel-route-v1', capsule=capsule_pin,
                    code=code_pins(), lanes=reports, production_admission=False,
                    world_truth_established=False, meaning_asserted=False)
    write(output/'manifest.json', manifest)
    pin = sha(output/'manifest.json')
    (output/'COMMITTED').write_text(pin, encoding='ascii')
    return pin


def load(capsule, capsule_pin, output, expected_pin):
    check((output/'COMMITTED').is_file(), 'ARTIFACT_INCOMPLETE')
    check((output/'COMMITTED').read_text() == expected_pin
          and sha(output/'manifest.json') == expected_pin, 'ARTIFACT_PIN')
    manifest = read(output/'manifest.json')
    check(manifest['code'] == code_pins() and manifest['capsule'] == capsule_pin
          and tuple(sorted(manifest['lanes'])) == LANES
          and all(manifest[k] is False for k in
                  ('production_admission','world_truth_established','meaning_asserted')),
          'ARTIFACT_CONTRACT')
    meta, lanes, queries = load_capsule(capsule, capsule_pin)
    graphs, routers = {}, {}
    for name in LANES:
        path = output/(name+'.json')
        check(sha(path) == manifest['lanes'][name]['file_sha256'], 'GRAPH_PIN')
        check(manifest['lanes'][name]['lens_id'] == meta['lanes'][name]['lens_id'],
              'INSTRUMENT_JOIN')
        graphs[name] = read(path)
        routers[name] = Router(lanes[name], graphs[name], meta['plan'])
    return meta, lanes, queries, graphs, routers

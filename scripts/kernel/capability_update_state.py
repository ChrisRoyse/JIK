"""Canonical sparse-index checkpoints for connected-task generations."""
import json
import os
from pathlib import Path

from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from kernel.capability_kernel import TaskCompanion, TaskKernel, code_pins
from kernel.capability_updates import TaskUpdates
from kernel.capability_lane import lanes
from kernel.repository_update_state import load_index, validate, flush_directory
from model.tool_kernel import PLAN


def producers():
    return dict(kernel=code_pins(), checkpoint=sha(Path(__file__).read_bytes()))


def save(companion, root):
    root = Path(root); value = companion.last
    need(not root.exists() and value is not None and value['recall_passes'], 'TASK_STATE_UNREADY')
    lane_map = lanes(companion.population, value['ids'])
    checks = {n: validate(companion.updates[n], lane, value['graphs'][n]) for n, lane in lane_map.items()}
    root.mkdir(); files, canonical = {}, {}
    for name, u in sorted(companion.updates.items()):
        binary = bytes(u.index.save())
        with (root/(name+'.usearch')).open('xb') as stream:
            stream.write(binary); stream.flush(); os.fsync(stream.fileno())
        index = load_index(root/(name+'.usearch'), lane_map[name].unit.shape[1])
        need(bytes(index.save()) == binary, 'TASK_STATE_ROUNDTRIP')
        canonical[name] = index
        metadata = dict(instrument=u.instrument, plan=u.plan, keys=u.keys,
            owners=[[k, v] for k, v in sorted(u.owners.items())], signatures=u.signatures,
            next_key=u.next_key, components=u.components, previous_edges=u.previous_edges)
        write_committed(root/(name+'.json'), metadata)
        for ext in ('.json', '.usearch'):
            path = root/(name+ext)
            files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    manifest = dict(schema='capability-updater-state-v1', continuation='canonical-reload-each-commit-v1',
        kernel=companion.previous, source=value['source'], code=producers(), files=files, checks=checks,
        cache_sha256=value['cache_sha256'], scientific_admission=False, serving=False)
    pin = write_committed(root/'manifest.json', manifest)
    flush_directory(root); write_committed(root/'COMMITTED', dict(manifest=pin)); flush_directory(root)
    flush_directory(root.parent)
    for name, index in canonical.items(): companion.updates[name].index = index
    return pin, checks


def restore(population, root, pin):
    root = Path(root)
    raw = (root/'manifest.json').read_bytes()
    need(sha(raw) == pin and json.loads((root/'COMMITTED').read_bytes()) == dict(manifest=pin), 'TASK_STATE_PIN')
    m = json.loads(raw)
    need(m['code'] == producers() and m['schema'] == 'capability-updater-state-v1'
         and m['continuation'] == 'canonical-reload-each-commit-v1'
         and not m['scientific_admission'] and not m['serving'], 'TASK_STATE_CONTRACT')
    accepted = TaskKernel(population, **m['kernel'], source=m['source'])
    v = accepted.value
    need(m['cache_sha256'] == v['cache_sha256'], 'TASK_STATE_CACHE')
    cache = json.loads((Path(m['kernel']['path'])/'cache.json').read_bytes())
    need(set(m['files']) == {n+ext for n in population.specs for ext in ('.json', '.usearch')}, 'TASK_STATE_FILES')
    for name, info in m['files'].items():
        path = root/name
        need(path.stat().st_size == info['bytes'] and sha(path.read_bytes()) == info['sha256'], 'TASK_STATE_BYTES')
    companion = TaskCompanion(population); checks = {}
    for name, lane in accepted.lanes.items():
        data = json.loads((root/(name+'.json')).read_bytes())
        need(data['instrument'] == population.specs[name].identity and data['plan'] == PLAN, 'TASK_STATE_INSTRUMENT')
        u = TaskUpdates(data['instrument'], PLAN, cache[name])
        u.index = load_index(root/(name+'.usearch'), lane.unit.shape[1])
        u.keys, u.signatures, u.owners = data['keys'], data['signatures'], dict(data['owners'])
        need(len(u.owners) == len(data['owners']), 'TASK_STATE_OWNER_DUPLICATE')
        u.next_key, u.components, u.previous_edges = data['next_key'], data['components'], data['previous_edges']
        checks[name] = validate(u, lane, v['graphs'][name]); companion.updates[name] = u
    need(checks == m['checks'], 'TASK_STATE_READBACK')
    companion.last, companion.previous = v, m['kernel']
    return companion, accepted

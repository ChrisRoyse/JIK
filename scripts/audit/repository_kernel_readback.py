"""Cold, physical readback of repository graph growth and all retained edge scores."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import networkx as nx
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.derive import derive,topology
from kernel.component_kernel import component_derive,exact_feedback,minimum_readback
from kernel.derive import feedback_vertices
from kernel.native_route import check
from kernel.repository_population import RepositoryPopulation
from kernel.repository_kernel import RepositoryKernel,lanes,assess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); start = time.perf_counter()
    config = json.loads((args.root/'config.json').read_bytes())
    witness_raw = (args.root/'witness.json').read_bytes()
    witness = json.loads(witness_raw)
    check(json.loads((args.root/'finished.json').read_bytes())['pin'] == sha(witness_raw),'REPOSITORY_READER_WITNESS')
    population = RepositoryPopulation(config['journal'],config['pin'])
    excluded,queries = population.holdout(config['initial_groups'])
    check(witness['split']['excluded_tasks'] == excluded and witness['split']['queries'] == queries,
          'REPOSITORY_READER_SPLIT')
    records,controls,score_cache = [],[],{}
    previous = None
    def refuse(name,code,fn):
        try:fn()
        except ValueError as error:check(code in str(error),'REPOSITORY_READER_CONTROL_CODE')
        else:raise AssertionError('MNLM_REPOSITORY_READER_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    def exact(name,a,b):
        signatures = sorted(population.records[k]['receipt']['slots'][name]['sha256'] for k in (a,b))
        signature = name,tuple(signatures)
        if signature not in score_cache:
            x,y = [population.values[k][name].astype(np.float64) for k in (a,b)]
            x /= np.sqrt((x*x).sum(axis=1))[:,None]
            y /= np.sqrt((y*y).sum(axis=1))[:,None]
            matrix = np.einsum('ik,jk->ij',x,y)
            score_cache[signature] = float(min(matrix.max(axis=0).mean(),matrix.max(axis=1).mean()))
        return score_cache[signature]
    for stage in witness['stages']:
        root = Path(stage['root'])
        raw = (root/'manifest.json').read_bytes(); v = json.loads(raw)
        check(sha(raw) == stage['pin'] and v['parent'] == previous,'REPOSITORY_READER_LINEAGE')
        previous = dict(path=str(root),pin=stage['pin'])
        ids = population.prefix(stage['groups'],excluded,v['withdrawn'])
        check(v['ids'] == ids and v['slots'] == {k:population.records[k]['receipt']['slots'] for k in ids},
              'REPOSITORY_READER_COVERAGE')
        cache_raw = (root/'cache.json').read_bytes(); cache = json.loads(cache_raw)
        check(sha(cache_raw) == v['cache_sha256'],'REPOSITORY_READER_CACHE')
        edge_count,max_error = 0,0.
        for name,graph in v['graphs'].items():
            present = [k for k in ids if len(population.values[k][name])]
            topology_graph = topology(present,graph['edges'])
            expected_groups,expected_scores = {n:[] for n in
                ('candidate_graph','members','graph_only_support','entries','groundedness_gaps')},{}
            for component in nx.weakly_connected_components(topology_graph):
                local = component_derive(sorted(component),[e for e in graph['edges'] if e[0] in component],v['plan'])
                selected = topology_graph.subgraph(local['candidate_graph'])
                if len(selected) <= 20:
                    minimum_readback(selected,local['members'])
                    small = selected.subgraph(sorted(selected)[:9])
                    check(exact_feedback(small) == feedback_vertices(small),'REPOSITORY_READER_EXACT_PARITY')
                for key in expected_groups:expected_groups[key].extend(local[key])
                expected_scores.update(local['scores'])
            check(all(graph[k] == sorted(x) for k,x in expected_groups.items()) and graph['scores'] == expected_scores,
                  'REPOSITORY_READER_DERIVATION')
            for a,b,score in graph['edges']:
                direct = exact(name,a,b)
                max_error = max(max_error,abs(score-direct));edge_count += 1
                pa,pb = sorted(population.records[k]['receipt']['slots'][name]['sha256'] for k in (a,b))
                key = digest(dict(instrument=v['instruments'][name],a=pa,b=pb,metric='min-mean-MaxSim-v1'))
                check(cache[name][key] == score and abs(direct-score) <= 2e-12,'REPOSITORY_READER_EDGE_SCORE')
        actual = assess(population,lanes(population,ids),v['graphs'],queries)
        check(actual == v['calibration'] and v['recall_passes'] == all(r['passes'] for r in actual.values()),
              'REPOSITORY_READER_CALIBRATION')
        if v['recall_passes']:
            kernel = RepositoryKernel(population,root,stage['pin'],expected_source=v['source'])
            expected = next(r['trace'] for r in witness['routes'] if r['root'] == str(root))
            cutoff = max(r['available'] for r in population.records.values())
            check(kernel.route(queries[0],source=v['source'],cutoff=cutoff) == expected,'REPOSITORY_READER_COLD_ROUTE')
            absent = next(k for k,r in population.records.items() if r['absence'])
            refuse('empty_observation_is_not_zero_vector','REPOSITORY_ABSENT_QUERY',lambda:
                kernel.route(absent,source=v['source'],cutoff=cutoff))
        else:
            refuse('failed_recall_is_unservable','REPOSITORY_RECALL_GATE',lambda:
                RepositoryKernel(population,root,stage['pin'],expected_source=v['source']))
        timing = json.loads((root/'timings.json').read_bytes())
        check(abs(sum(timing['phases'].values())-timing['wall_seconds']) < 1e-8,'REPOSITORY_READER_TIMINGS')
        records.append(dict(pin=stage['pin'],edges=edge_count,max_score_error=max_error,
            nodes=len(ids),absent=len(v['absent']),recall={n:r['recall'] for n,r in actual.items()}))
    good_stage = next((s for s in witness['stages'] if s['passes']),None)
    if good_stage is not None:
        with tempfile.TemporaryDirectory(prefix='repository-kernel-faults-',dir=args.root.parent) as tmp:
            clone = Path(tmp)/'artifact';shutil.copytree(good_stage['root'],clone)
            v = json.loads((clone/'manifest.json').read_bytes())
            source = copy.deepcopy(v['source']);source['head'] = '0'*64
            refuse('head_changed_after_restart','REPOSITORY_STALE_HEAD',lambda:
                RepositoryKernel(population,clone,good_stage['pin'],expected_source=source))
            # Re-hash the semantic defect, so this is not merely a checksum test.
            v['ids'] = v['ids'][1:]
            pin = write_committed(clone/'bad-manifest.json',v)
            (clone/'bad-manifest.json').replace(clone/'manifest.json')
            (clone/'COMMITTED').write_text(json.dumps(dict(manifest=pin)))
            refuse('omitted_constellation_with_valid_hash','REPOSITORY_ARTIFACT_COVERAGE',lambda:
                RepositoryKernel(population,clone,pin,expected_source=v['source']))
            shutil.copyfile(Path(good_stage['root'])/'manifest.json',clone/'manifest.json')
            shutil.copyfile(Path(good_stage['root'])/'COMMITTED',clone/'COMMITTED')
            v = json.loads((clone/'manifest.json').read_bytes())
            name = sorted(v['calibration'])[0]
            v['calibration'][name]['rows'][0]['recall'] = 0.123
            pin = write_committed(clone/'forged-calibration.json',v)
            (clone/'forged-calibration.json').replace(clone/'manifest.json')
            (clone/'COMMITTED').write_text(json.dumps(dict(manifest=pin)))
            refuse('forged_calibration_with_valid_hash','REPOSITORY_RECALL_REPLAY',lambda:
                RepositoryKernel(population,clone,pin,expected_source=v['source']))
            shutil.copyfile(Path(good_stage['root'])/'manifest.json',clone/'manifest.json')
            shutil.copyfile(Path(good_stage['root'])/'COMMITTED',clone/'COMMITTED')
            (clone/'cache.json').write_text('{}')
            refuse('persisted_cache_corruption','REPOSITORY_CACHE_PIN',lambda:
                RepositoryKernel(population,clone,good_stage['pin'],expected_source=v['source']))
    final = witness['stages'][-1]
    withdrawn = final['withdrawn']
    after = json.loads((Path(final['root'])/'manifest.json').read_bytes())
    check(not set(withdrawn) & set(after['ids']) and all(not set(withdrawn) & {x for a,b,_ in g['edges'] for x in (a,b)}
          for g in after['graphs'].values()),'REPOSITORY_READER_WITHDRAWAL')
    population.fresh()
    result = dict(schema='repository-kernel-readback-v1',witness=sha(witness_raw),source=config,
        records=records,controls=controls,unique_edge_scores=len(score_cache),withdrawn=withdrawn,
        old_source_unchanged=sha(Path(config['journal']).read_bytes()) == config['pin'],
        wall_seconds=time.perf_counter()-start,acceptance='author-checked',number_status='measured',
        scientific_admission=False,learned_capability=False,serving=False)
    pin = write_committed(args.output,result)
    print(json.dumps(dict(pin=pin,controls=len(controls),edges=sum(r['edges'] for r in records),
        wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':main()

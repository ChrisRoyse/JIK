"""Extend the frozen group registry with context revisions of the same coding episodes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import CapabilityPopulation
from claims.capability_context_population import ContextPopulation
from claims.capability_revision_population import RevisionPopulation
from claims.process_conditions import write_committed
from kernel.capability_registry import freeze_together
from senses.trajectory_population import read_pinned
from text.evaluation_multiscale import file_sha


def sources(previous, previous_pin, population, pin):
    history = read_pinned(previous, previous_pin)
    pops = {}
    for key, path in history['population_paths'].items():
        report = read_pinned(Path(path) / 'report.json', key)
        cls = RevisionPopulation if report['source'].get('source_kind') in (
            'ami-chronological-continuation-v1', 'condaqa-evidence-revision-v1') else CapabilityPopulation
        pops[key] = cls(path, key)
    pops[pin] = ContextPopulation(population, pin)
    return pops, history['splits']


def build(a):
    pops, history = sources(a.previous, a.previous_pin, a.population, a.population_pin)
    result = freeze_together(pops, history)
    result.update(previous=dict(path=str(a.previous), pin=a.previous_pin),
        population_paths={pin:str(pop.root) for pin,pop in pops.items()},
        producer_sha256=file_sha(Path(__file__)),
        added_source_scope='context revisions of existing episodes; not new independent experiences')
    pin = write_committed(a.output,result)
    print(json.dumps(dict(registry_sha256=pin, counts=result['counts'],
        new_split={k:len(result['splits'][a.population_pin][k]) for k in ('train','queries')})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','population','output'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('previous-pin','population-pin'): p.add_argument('--'+name,required=True)
    build(p.parse_args())

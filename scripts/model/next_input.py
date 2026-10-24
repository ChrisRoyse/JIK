"""Use the established typed native batching with an explicit fresh-source reader."""
from pathlib import Path

from audit.window_readback import readback
from claims.codata_source import check
from claims.episodic_memory import digest
from claims.next_population import NextPopulation
from claims.proposal_population import cached
from model.proposal_input import ProposalNative, load


class NextNative(ProposalNative):
    def __init__(self, population, population_pin, native, native_pin, cache_native, cache_native_pin):
        self.population = p = NextPopulation(population,population_pin)
        self.roots = dict(new=Path(native),cache=Path(cache_native))
        self.pins = dict(population=population_pin,native=native_pin,cached_population=p.cache_pin,
                         cached_native=cache_native_pin,conditions=p.prepared.conditions.pin)
        self.readbacks = dict(new=readback(self.roots['new'],p.root),cache=readback(self.roots['cache'],p.cache_root))
        new,specs,new_files = load(self.roots['new'],native_pin,population_pin)
        old,old_specs,old_files = load(self.roots['cache'],cache_native_pin,p.cache_pin)
        check(specs == old_specs, 'NEXT_NATIVE_INSTRUMENT_CHANGED')
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.arrays,self.files = dict(new=new,cache=old),dict(new=new_files,cache=old_files)
        old_rows,_,old_probe = cached(p.cache_root,p.cache_pin)
        self.rows = dict(new=p.rows,cache=old_rows)
        self.probes = dict(new=p.probe['unit'],cache=old_probe['unit'])
        self.fields,self.field_pin = p.fields,digest(p.fields)

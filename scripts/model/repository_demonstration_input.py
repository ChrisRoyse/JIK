"""Use the existing typed constellation consumer for reference demonstration occurrences."""
from pathlib import Path

from audit.window_readback import readback
from claims.codata_source import check
from claims.episodic_memory import digest
from claims.repository_demonstration_population import DemonstrationPopulation
from model.proposal_input import load
from model.repository_cohort_input import CohortNative


class DemonstrationNative(CohortNative):
    def __init__(self,population,population_pin,native,native_pin,partition):
        check(partition in ('learn_1','learn_2'),'COHORT_NATIVE_TRAINING_PARTITION')
        self.partition=partition;self.population=DemonstrationPopulation(population,population_pin)
        self.native=Path(native);self.readback=readback(self.native,self.population.root)
        self.arrays,specs,self.files=load(self.native,native_pin,population_pin)
        self.specs=tuple(specs[n] for n in sorted(specs))
        self.identity=dict(population=population_pin,native=native_pin,
                           observations=self.population.observations.pin,partition=partition)
        self.memory_pin=self.memory_hash();self.fields_pin=digest(self.population.fields)

"""Append immutable execution batches without reindexing or copying native rows."""
from claims.codata_source import check
from claims.episodic_memory import digest
from claims.condition_catalog import ConditionCatalog


class ObservationBatches:
    def __init__(self, batches):
        self.batches = tuple(batches)
        self.source = batches[0].source
        self.entries, self.providers = {}, {}
        for batch in batches:
            check(batch.source is self.source, "PROCESS_BATCH_SOURCE")
            check(not self.entries.keys() & batch.entries.keys(), "PROCESS_BATCH_COLLISION")
            self.entries.update(batch.entries)
            self.providers.update((key, batch) for key in batch.entries)
        self.pin = digest(dict(schema="process-observation-batches-v1",
                               batches=[b.pin for b in batches]))

    def for_observation(self, key):
        check(key in self.providers, "PROCESS_BATCH_UNKNOWN")
        return self.providers[key]

    def read(self, key):
        return self.for_observation(key).read(key)

    def execution_conditions(self, key):
        return self.for_observation(key).execution_conditions(key)

    def condition_inputs(self):
        providers = {}
        for batch in self.batches:
            pin = batch.manifest.get("condition_commitment")
            if pin not in providers:
                providers[pin] = batch.condition_inputs()
        return next(iter(providers.values())) if len(providers) == 1 else ConditionCatalog(providers.values())

    def episode(self, key, evidence=()):
        return self.for_observation(key).episode(key, evidence)

    def validate_episode(self, episode):
        self.for_observation(episode["source"]["observation"]).validate_episode(episode)

    def assess(self, episode):
        return self.for_observation(episode["source"]["observation"]).assess(episode)

    def compare_report(self, key):
        return self.for_observation(key).compare_report(key)


class ProcessBatches:
    def __init__(self, batches):
        self.batches = tuple(batches)
        self.base, self.specs = batches[0].base, batches[0].specs
        check(all(b.base is self.base and b.specs == self.specs for b in batches),
              "PROCESS_BATCH_INSTRUMENTS")
        self.observations = ObservationBatches([b.observations for b in batches])
        self.providers = {key: b for b in batches for key in b.observations.entries}
        self.identity = dict(schema="process-native-batches-v1",
                             batches=[b.identity for b in batches])

    def for_observation(self, key):
        check(key in self.providers, "PROCESS_BATCH_UNKNOWN")
        return self.providers[key]

    def checked(self, key):
        # Preserve the original batch identity and its original local unit index.
        return self.for_observation(key).checked(key)

    def observed(self, key, config, device):
        return self.for_observation(key).observed(key, config, device)


def append(existing, new):
    if existing is None:
        return new
    return ProcessBatches((*getattr(existing, "batches", (existing,)), new))

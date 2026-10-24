"""Roster-bound resident access to complete, cold-read typed association graphs."""
from kernel.published_associations import PublishedAssociations, need, read_json, unchanged
from kernel.typed_association_paths import TypedPaths


class AssociationPanel:
    def __init__(self, manifest, pin, *, expected_roster, cache_shards=8):
        self.lanes = {}; self.closed = False; self.seals = {}
        self.manifest = read_json(manifest, pin, self.seals); self.pin = pin
        refs = self.manifest['graphs']; names = [r['name'] for r in refs]
        expected = list(expected_roster)
        need(expected and len(set(expected)) == len(expected) and names == sorted(expected), 'PANEL_ROSTER')
        need(self.manifest['schema'] == 'complete-legal-association-inputs-v1'
             and self.manifest['all_pairs_exhaustive'] is False
             and self.manifest['scientific_admission'] is False, 'PANEL_SCOPE')
        signatures = {(r['binding']['source_pin'], r['binding']['bank_pin'], r['population']) for r in refs}
        need(len(signatures) == 1 and len({r['binding']['lens_id'] for r in refs}) == len(refs), 'PANEL_IDENTITIES')
        need(self.manifest['population'] == refs[0]['population']
             and self.manifest['total_edges'] == sum(r['population']*10 for r in refs)
             and self.manifest['total_payload_bytes'] == sum(r['payload_bytes'] for r in refs), 'PANEL_TOTALS')
        # Validate every cold authority before opening any payload store.
        for ref in refs:
            cold = read_json(ref['cold']['path'], ref['cold']['pin'], self.seals)
            plan = cold.get('contract', cold.get('plan'))
            need(cold['complete'] and cold['mode'] == 'cold' and cold['scope'] == 'full'
                 and cold['full_graph_complete'] and cold['all_rows_structurally_read']
                 and not cold['scientific_admission'] and cold['previous_pin'] == ref['pin']
                 and plan['binding'] == ref['binding'], 'PANEL_COLD_AUTHORITY')
            coverage = cold.get('coverage', cold)
            need(coverage['queries'] == ref['population'] and coverage['edges'] == ref['edges'], 'PANEL_COLD_COVERAGE')
        try:
            for ref in refs:
                b = ref['binding']
                lane = PublishedAssociations(ref['path'], ref['pin'], source_pin=b['source_pin'],
                    bank_pin=b['bank_pin'], lens_id=b['lens_id'], cache_shards=cache_shards)
                self.lanes[ref['name']] = lane
                need(lane.binding == b and lane.population == ref['population']
                     and lane.stats['shards'] == ref['shards']
                     and lane.stats['stored_edges'] == ref['edges']
                     and lane.stats['payload_bytes_authenticated'] == ref['payload_bytes'], 'PANEL_PHYSICAL_CENSUS')
            self.walker = TypedPaths(self.lanes); self.fresh()
        except BaseException:
            self.close(); raise

    def fresh(self):
        need(not self.closed, 'PANEL_CLOSED'); unchanged(self.seals)
        for lane in self.lanes.values(): lane.fresh()

    def read(self, instrument, uids):
        self.fresh(); need(instrument in self.lanes, 'PANEL_MISSING_INSTRUMENT')
        lane = self.lanes[instrument]
        return lane.read(uids, lens_id=lane.binding['lens_id'])

    def paths(self, origin, instruments):
        self.fresh()
        for path in self.walker.paths(origin, instruments):
            unchanged(self.seals); yield path

    def training_authority(self):
        self.fresh(); need(False, 'PANEL_TRANSDUCTIVE_NOT_TRAINING_AUTHORITY')

    def close(self):
        for lane in self.lanes.values(): lane.close()
        self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

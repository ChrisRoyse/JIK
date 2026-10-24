"""Full finite-population channel variation audit, never an admission assay.

Count physical observations separately from units and measured states. Constant
claims use exact arrays, not hashes alone. Context channels are only certified
absent when their join preconditions prove there is no eligible observation.
O(input bytes + distinct payload token rows * DIM + channel references).
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1]/'canon')]
from native_population import Population, file_hash
from senses.contract import CHANNELS, SenseError, canonical, require
from senses.instruments import contracts, tokens


def summarize(counts, observations, constant):
    """Empirical state entropy is a diagnostic, never a population MI estimate."""
    require(sum(counts.values()) == observations, 'MNLM_VARIATION_DENOMINATOR', 'state count does not bind its source')
    empirical = -math.fsum(n/observations * math.log2(n/observations) for n in counts.values())
    return dict(observations=observations, states=len(counts), exact_constant=constant,
                empirical_state_entropy_bits=empirical, shannon_information_certified=False)


def run(a):
    started, phases = time.perf_counter(), {}
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = file_hash(a.population/'manifest.json')
    require(manifest == a.manifest_sha256, 'MNLM_NATIVE_MANIFEST', 'caller pin differs')
    original = file_hash(a.population/'native.sqlite')
    pop = Population(a.population, manifest)
    roster = contracts()
    (a.output/'contracts.json').write_bytes(canonical(roster))
    try:
        t = time.perf_counter()
        sources, units, refs, clocks = collections.Counter(), collections.defaultdict(set), {}, 0
        record_hash = hashlib.sha256()
        for record in pop.records():
            source = record['locator']['source']
            sources[source] += 1
            units[source].add(record['unit'])
            clocks += int(any(f['type'] == 'clock' and f['clock_role'] == 'observed'
                              for f in record['channels']['fields']['items']))
            for ref in record['referents']:
                refs.setdefault(ref['key'], set()).add((source, record['unit']))
            record_hash.update(canonical(record)+b'\n')
        # A single label unit per referent is excluded by Store.context itself.
        multi_units = sum(len({u for _, u in group}) > 1 for group in refs.values())
        cross_units = sum(len({u for _, u in group}) > 1 and len({s for s, _ in group}) > 1
                          for group in refs.values())
        context_absent = {'history': clocks == 0 or multi_units == 0, 'co_reference': cross_units == 0}
        context_proof = dict(observations_with_observed_clock=clocks, referents=len(refs),
                             referents_with_multiple_units=multi_units,
                             referents_with_cross_source_distinct_units=cross_units,
                             proved_absent=context_absent,
                             rule='history requires earlier observed time and different unit; co-reference requires different source and unit')
        phases['source_and_context_readback'] = time.perf_counter()-t
        t = time.perf_counter()
        memberships = collections.defaultdict(list)
        query = ('SELECT c.channel,c.digest,o.source,count(*) FROM channels c '
                 'JOIN observations o ON o.atom=c.atom GROUP BY c.channel,c.digest,o.source')
        for channel, ph, source, n in pop.db.execute(query):
            memberships[channel, ph].append((source, n))
        counts, nonempty = collections.defaultdict(collections.Counter), collections.Counter()
        first, varied, payload_count = {}, set(), 0
        with (a.output/'payload-measurements.jsonl').open('xb') as out:
            for channel, ph, body in pop.db.execute('SELECT channel,digest,body FROM payloads ORDER BY channel,digest'):
                require(hashlib.sha256(body).hexdigest() == ph, 'MNLM_NATIVE_PAYLOAD_DIGEST', ph)
                payload = json.loads(body)
                if channel in context_absent and not context_absent[channel]:
                    receipt = dict(channel=channel, payload_sha256=ph, measured=False,
                                   reason='MNLM_VARIATION_CONTEXT_UNMEASURED')
                else:
                    values = tokens(payload['items'], channel)
                    state = hashlib.sha256(canonical(dict(absent=not len(values), shape=list(values.shape), dtype='<f4'))
                                           + values.astype('<f4').tobytes()).hexdigest()
                    receipt = dict(channel=channel, payload_sha256=ph, measured=True,
                                   shape=list(values.shape), state_sha256=state, absent=not len(values))
                    for source, n in memberships[channel, ph]:
                        key = source, channel
                        counts[key][state] += n
                        nonempty[key] += n if len(values) else 0
                        if key not in varied:
                            if key not in first:
                                first[key] = values.copy()
                            elif not np.array_equal(first[key], values):
                                varied.add(key)
                                del first[key]
                receipt['source_multiplicities'] = memberships[channel, ph]
                out.write(canonical(receipt)+b'\n')
                payload_count += 1
        phases['distinct_payload_measurement'] = time.perf_counter()-t
        t = time.perf_counter()
        profiles = {}
        for source, n in sorted(sources.items()):
            channels = {}
            for channel in CHANNELS:
                key = source, channel
                if channel in context_absent and not context_absent[channel]:
                    channels[channel] = dict(status='unmeasured_context', informative=False)
                    continue
                channels[channel] = {**summarize(counts[key], n, key not in varied),
                                     'nonempty_observations': nonempty[key], 'status': 'measured'}
                require((channels[channel]['states'] == 1) == channels[channel]['exact_constant'],
                        'MNLM_VARIATION_STATE_COLLISION', 'digest and exact equality disagree')
            variable = [c for c, r in channels.items() if r['status'] == 'measured' and not r['exact_constant']]
            unknown = [c for c, r in channels.items() if r['status'] != 'measured']
            profiles[source] = dict(observations=n, units=len(units[source]), channels=channels,
                                    variable_measured_channels=variable, unresolved_channels=unknown,
                                    empirical_possible_informative_channel_ceiling=len(variable)+len(unknown),
                                    admission=False)
        require(file_hash(a.population/'native.sqlite') == original, 'MNLM_NATIVE_DATABASE_DIGEST', 'source changed')
        phases['aggregation_and_source_exit'] = time.perf_counter()-t
        report = dict(acceptance='author-checked only', admission=False,
                      scope='finite retained population under current frozen channel instruments; no population MI or IID claim',
                      observations=sum(sources.values()), distinct_units=len(set().union(*units.values())),
                      payloads=payload_count, profiles=profiles, context_proof=context_proof,
                      source_manifest_sha256=manifest, source_database_sha256=original,
                      records_sha256=record_hash.hexdigest(),
                      instrument_code_sha256=file_hash(Path(__file__).parents[1]/'senses/instruments.py'),
                      audit_code_sha256=file_hash(Path(__file__)), phases_seconds=phases,
                      seconds=time.perf_counter()-started)
        report['timing_coverage'] = sum(phases.values())/report['seconds']
        (a.output/'report.json').write_bytes(canonical(report))
        (a.output/'SHA256.json').write_bytes(canonical({p.name:file_hash(p) for p in a.output.iterdir() if p.is_file()}))
        print(json.dumps(dict(observations=report['observations'], units=report['distinct_units'],
                              sources={s:{k:p[k] for k in ('units','variable_measured_channels','unresolved_channels',
                                        'empirical_possible_informative_channel_ceiling')} for s,p in profiles.items()},
                              seconds=report['seconds'], timing_coverage=report['timing_coverage'])))
        return report
    finally:
        pop.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population', required=True, type=Path)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    try:
        run(p.parse_args())
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)

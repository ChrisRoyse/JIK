"""Cold persisted-native readback, independent traversal and changed-state controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.constellation_corpus import canonical, file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank, ROSTER, native


def run(a):
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    phase = dict(cold_pin_seconds=time.perf_counter()-start)
    tick = time.perf_counter()
    # Independent columnar iterator, deliberately not Bank.row's row-group cache.
    readers = {n: (r for batch in pq.ParquetFile(a.bank/n/"readings.parquet").iter_batches(batch_size=73)
                   for r in batch.to_pylist()) for n in ROSTER}
    counts = {n: dict(atoms=0, windows=0, multiple=0, floats=0) for n in ROSTER}
    fingerprints = {n: hashlib.sha256() for n in ROSTER}
    samples, index, transitions = {}, 0, 0
    for d in range(len(corpus.documents)):
        sources = corpus.document(d)
        for source in sources:
            measured = bank.constellation(source, index)
            for name, reader in readers.items():
                row = next(reader)
                independent = np.asarray(row["vectors"], np.float32)
                values, trace = measured[name]
                need(row["atom_id"] == source["atom_id"] and row["index"] == index
                     and np.array_equal(independent, values), "COLD_NATIVE_EQUAL")
                c = counts[name]
                c["atoms"] += 1
                c["windows"] += len(values)
                c["multiple"] += len(values) > 1
                c["floats"] += values.size
                fingerprints[name].update(values.tobytes())
                if len(values) > 1 and name not in samples:
                    samples[name] = (row, source, index)
            index += 1
        transitions += sum(1 for _ in corpus.next_pairs(d))
    need(all(next(reader, None) is None for reader in readers.values()), "COLD_EXTRA_ROW")
    for name, c in counts.items():
        r = bank.reports[name]
        need(c["atoms"] == r["inputs"] and c["windows"] == r["windows"]
             and c["multiple"] == r["multiwindow_atoms"] and name in samples, "COLD_COUNTS")
    need(transitions == corpus.report["counts"]["transitions"], "COLD_TRANSITIONS")
    phase["independent_native_consumer_seconds"] = time.perf_counter()-tick
    tick = time.perf_counter()
    controls = []
    def deny(label, fn, prefix):
        try:
            fn()
        except Exception as exc:
            code = getattr(exc, "code", str(exc))
            need(code.startswith(prefix), "COLD_CONTROL_CODE:"+label+":"+code)
            controls.append(dict(name=label, error=code))
        else:
            raise ValueError("MNLM_TEXT_CORPUS_CONTROL_ACCEPTED:"+label)
    for name, (row, source, index) in samples.items():
        tests = []
        bad = copy.deepcopy(row)
        bad["vectors"].pop()
        tests.append(("missing_window", bad, "MNLM_LENS_WINDOW_PROVENANCE"))
        bad = copy.deepcopy(row)
        bad["atom_id"] = "0"*64
        tests.append(("wrong_source_atom", bad, "MNLM_TEXT_CORPUS_READING_ALIGNMENT"))
        bad = copy.deepcopy(row)
        bad["vectors"][0] = [0.]*len(bad["vectors"][0])
        tests.append(("zero_native_vector", bad, "MNLM_TEXT_CORPUS_READING_VALUES"))
        for label, bad, code in tests:
            path = a.output/(name+"-"+label+".json")
            path.write_bytes(canonical(bad))
            deny(name+":"+label, lambda p=path: native(json.loads(p.read_bytes()), source, index,
                                                       bank.reports[name]), code)
    deny("unadmitted_training", bank.training_authority, "MNLM_TEXT_CORPUS_ADMISSION_REQUIRED")
    phase["persisted_negative_controls_seconds"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema="text-constellation-bank-readback-v1", corpus=a.corpus_pin, bank=a.bank_pin,
        counts=counts, source_transitions=transitions, controls=controls,
        native_hashes={n: h.hexdigest() for n, h in fingerprints.items()},
        complete=True, scientific_admission=False, served=False, optimizer_steps=0,
        code_sha256=file_sha(Path(__file__)), phases_seconds=phase, wall_seconds=wall,
        phase_coverage=sum(phase.values())/wall, acceptance="author-checked only")
    publish(a.output/"report.json", report)
    print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), counts=counts,
                         controls=len(controls), seconds=wall)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())

"""Read-only, bounded native window bank aligned by exact source atom IDs."""
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
import text_windows
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import stamp

SCHEMA = "text-constellation-window-bank-v1"
ROSTER = ("all-mpnet-base-v2", "codebert-mean", "e5-base-v2", "finbert-mean",
          "pubmedbert-embeddings", "specter2-base")
BATCH = 32


def native(row, source, index, report):
    need(row["index"] == index and row["atom_id"] == source["atom_id"]
         and row["text_sha256"] == source["text_sha256"], "READING_ALIGNMENT")
    values = np.asarray(row["vectors"], dtype=np.float32)
    need(values.ndim == 2 and values.shape[1] == report["dimensions"] and len(values) > 0
         and np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1., atol=2e-6),
         "READING_VALUES")
    trace = json.loads(row["trace"])
    need(trace["prefix"] == report["contract"].get("prefix", ""), "READING_PREFIX")
    text_windows.validate(trace, "dense", len(values), text=source["text"])
    values.flags.writeable = False
    return values, trace


class Bank:
    def __init__(self, root, pin, corpus):
        self.root, self.pin, self.corpus = Path(root), pin, corpus
        need(file_sha(self.root/"bank.json") == pin, "BANK_PIN")
        self.manifest = m = json.loads((self.root/"bank.json").read_bytes())
        need(m["schema"] == SCHEMA and m["complete"] and not m["errors"]
             and m["scientific_admission"] is False and m["served"] is False
             and m["corpus"] == corpus.pin and m["roster"] == list(ROSTER)
             and set(m["instruments"]) == set(ROSTER), "BANK_CONTRACT")
        self.stamps = {self.root/"bank.json": stamp(self.root/"bank.json")}
        self.tables, self.reports, self.cache = {}, {}, {}
        for name in ROSTER:
            path = self.root/name
            need(file_sha(path/"report.json") == m["instruments"][name]["report_sha256"], "INSTRUMENT_PIN")
            r = self.reports[name] = json.loads((path/"report.json").read_bytes())
            need(r["schema"] == SCHEMA and r["corpus"] == corpus.pin and r["name"] == name
                 and r["lens_id"] == digest(canonical(r["contract"])) == m["instruments"][name]["lens_id"]
                 and r["inputs"] == corpus.report["counts"]["atoms"] and r["frozen_unchanged"]
                 and r["determinism"]["exact"] and r["determinism"]["overflow_windows"] > 1
                 and r["scientific_admission"] is False and r["served"] is False, "INSTRUMENT_CONTRACT")
            self.stamps[path/"report.json"] = stamp(path/"report.json")
            for filename, f in r["files"].items():
                need(Path(filename).name == filename and "\\" not in filename and ":" not in filename,
                     "BANK_PATH")
                target = path/filename
                need(target.stat().st_size == f["bytes"] and file_sha(target) == f["sha256"], "READING_FILE_PIN")
                self.stamps[target] = stamp(target)
            table = self.tables[name] = pq.ParquetFile(path/"readings.parquet")
            need(table.metadata.num_rows == r["inputs"] and table.num_row_groups == (r["inputs"]+BATCH-1)//BATCH,
                 "READING_ROW_COVERAGE")
            need(all(table.metadata.row_group(i).num_rows == min(BATCH, r["inputs"]-i*BATCH)
                     for i in range(table.num_row_groups)), "READING_BATCH_GEOMETRY")
        self.fresh()

    def fresh(self):
        self.corpus.fresh()
        need(all(stamp(path) == expected for path, expected in self.stamps.items()), "BANK_CHANGED")

    def _row(self, name, source, index):
        need(name in self.tables and type(index) is int and 0 <= index < self.reports[name]["inputs"], "READING_INDEX")
        group = index//BATCH
        cached = self.cache.get(name)
        if cached is None or cached[0] != group:
            cached = self.cache[name] = (group, self.tables[name].read_row_group(group).to_pylist())
        result = native(cached[1][index%BATCH], source, index, self.reports[name])
        return result

    def row(self, name, source, index):
        self.fresh()
        result = self._row(name, source, index)
        self.fresh()
        return result

    def constellation(self, source, index):
        # A mapping of typed complete windows; never average/concatenate lanes.
        self.fresh()
        result = {name: self._row(name, source, index) for name in ROSTER}
        self.fresh()
        return result

    def training_authority(self):
        raise ValueError("MNLM_TEXT_CORPUS_ADMISSION_REQUIRED")

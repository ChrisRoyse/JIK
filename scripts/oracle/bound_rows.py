"""One verified hash pass per Parquet source, bounded selected-row projection."""
from copy import deepcopy
from pathlib import Path
import time

import pyarrow.parquet as pq

from senses.contract import digest, require
from senses.trajectory_population import open_bound

CODE = "MNLM_EXECUTION_ROW_BINDING"


class BoundRows:
    def __init__(self, root, requests):
        require(0 < len(requests) <= 256, CODE, "explicit bounded row requests required")
        self.root = Path(root).resolve()
        self.rows, self.receipts = {}, []
        groups = {}
        for binding, columns in requests:
            require(type(binding["row"]) is int and binding["row"] >= 0
                    and columns and len(set(columns)) == len(columns), CODE, "row/column contract")
            source = {k: v for k, v in binding.items() if k != "row"}
            entry = groups.setdefault(binding["path"], (source, []))
            require(entry[0] == source, CODE, "conflicting source bindings")
            entry[1].append((binding, columns))
        for source, wanted in groups.values():
            started = time.perf_counter()
            with open_bound(self.root, source) as stream:
                hashed = time.perf_counter()
                parquet = pq.ParquetFile(stream)
                require("rows" not in source or source["rows"] == parquet.metadata.num_rows,
                        CODE, "source row count changed")
                require(all(b["row"] < parquet.metadata.num_rows for b, _ in wanted), CODE,
                        "requested row outside source")
                columns = sorted({c for _, cols in wanted for c in cols})
                indices = {b["row"] for b, _ in wanted}
                selected, offset, read_rows, read_groups = {}, 0, 0, 0
                for group in range(parquet.num_row_groups):
                    count = parquet.metadata.row_group(group).num_rows
                    if any(offset <= i < offset + count for i in indices):
                        read_groups += 1
                        position = offset
                        for batch in parquet.iter_batches(batch_size=32, row_groups=[group], columns=columns):
                            for row in batch.to_pylist():
                                if position in indices:
                                    selected[position] = row
                                position += 1
                            read_rows += batch.num_rows
                        require(position == offset + count, CODE, "row group census changed")
                    offset += count
                require(selected.keys() == indices, CODE, "selected rows missing")
                for binding, cols in wanted:
                    self.rows[digest([binding, cols])] = {c: selected[binding["row"]][c] for c in cols}
            finished = time.perf_counter()
            self.receipts.append({"source": source, "hash_passes": 1, "columns": columns,
                                  "selected_rows": len(indices), "decoded_rows": read_rows,
                                  "row_groups": read_groups, "hash_seconds": hashed - started,
                                  "projection_seconds": finished - hashed})

    def read(self, binding, columns):
        key = digest([binding, columns])
        require(key in self.rows, CODE, "row/projection was not bound in this batch")
        return deepcopy(self.rows[key])

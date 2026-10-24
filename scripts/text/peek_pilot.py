"""Print the field names and one truncated record of each acquired pilot shard (inspection only)."""
import gzip
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

root = Path(sys.argv[1])
for f in sorted(root.rglob("*")):
    if not f.is_file() or f.name.endswith((".json", ".lock", ".metadata")) or ".cache" in f.parts:
        continue
    if f.suffix == ".parquet":
        pf = pq.ParquetFile(f)
        row = pf.read_row_group(0, columns=None).slice(0, 1).to_pylist()[0]
        n = pf.metadata.num_rows
    else:
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt", encoding="utf-8") as h:
            row = json.loads(h.readline())
        n = "?"
    print(f.relative_to(root), "rows", n)
    for k, v in row.items():
        print("   ", k, repr(str(v)[:120]))

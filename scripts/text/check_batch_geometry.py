"""Does any fixed batch of the text-pilot measurement exceed measure_pilot.TOKEN_BUDGET? (CPU; tokenizers only)

If none does, slots measured before the budget split was added used exactly the geometry the current
runtime would use, so they remain one runtime. Prints, per lens, the largest padded batch in tokens.
"""
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lens.catalogue import by_name
from lens.learned import snapshot_dir
from text.measure_pilot import BATCH, BLOCK, TOKEN_BUDGET

atoms, lenses = Path(sys.argv[1]), sys.argv[2].split(",")
texts = pq.read_table(atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
out = {}
for name in lenses:
    spec = by_name(name)
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(spec["repo"])), trust_remote_code=False)
    lens_ = [min(len(x), spec["max_len"]) for x in tok([spec.get("prefix", "") + t for t in texts], truncation=True, max_length=spec["max_len"])["input_ids"]]
    worst, over = 0, 0
    for b0 in range(0, len(texts), BLOCK):
        idx = sorted(range(b0, min(b0 + BLOCK, len(texts))), key=lambda i: (len(texts[i]), i))
        for j in range(0, len(idx), BATCH):
            part = idx[j:j + BATCH]
            padded = len(part) * max(lens_[i] for i in part)
            worst = max(worst, padded); over += padded > TOKEN_BUDGET
    out[name] = {"max_padded_tokens": worst, "batches_over_budget": over}
    print(name, out[name], flush=True)
print(json.dumps({"budget": TOKEN_BUDGET, "all_within": all(v["batches_over_budget"] == 0 for v in out.values())}))

"""Census of Wikidata5m P31 classes (CPU): entity counts for declared domain classes and the top 200 overall."""
import collections
import io
import json
from pathlib import Path
import sys

import zstandard

canon = Path(sys.argv[1])
DOMAIN = {"biomedical": ["Q12136", "Q11173", "Q8054", "Q7187", "Q12140", "Q169872", "Q18123741", "Q2996394", "Q4936952"],
          "finance": ["Q4830453", "Q6881511", "Q891723", "Q783794", "Q43229", "Q2085381", "Q1664720"],
          "legal": ["Q2334719", "Q7748", "Q820655", "Q1137809", "Q131569", "Q93288", "Q1006079", "Q1810437"],
          "science": ["Q13442814", "Q11862829", "Q1936384", "Q2465832", "Q7187", "Q11173", "Q12136"],
          "code": ["Q7397", "Q9143", "Q341", "Q166142", "Q1639024", "Q7889"]}
count = collections.Counter()
for sh in ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test"):
    with (canon / f"{sh}.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        for line in io.TextIOWrapper(zr, encoding="utf-8"):
            if '"p": "P31"' in line:
                f = json.loads(line)["f"]
                count[f["o"]] += 1
print(json.dumps({d: {c: count[c] for c in cs} for d, cs in DOMAIN.items()}, indent=1))
print(json.dumps(count.most_common(200)))

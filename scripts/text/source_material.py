"""Source material availability, not a quality rater, sense or truth oracle.

Retain exact bytes and offsets. A missing image's filename is not its content.
The conservative syntax recognizer is deliberately narrower than a Markdown
parser; unrecognized defects remain possible and never earn a clean-data claim.
"""
import bisect
from collections import Counter
import hashlib
import re

CONTRACT = "source-material-v1"
IMAGE = re.compile(r'!\[[^\]\r\n]*\]\([^\r\n)]*\)(?:\{[^}\r\n]*\})?|<img\b[^>]*>', re.I)
DAMAGE = re.compile(r'[\ufffd\x00-\x08\x0b\x0e-\x1f\x7f]+')


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_MATERIAL_"+code)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def material(field):
    text = field["text"]
    check(len(text) == field["characters"] and len(text.encode()) == field["utf8_bytes"]
          and digest(text) == field["sha256"], "SOURCE_HASH")
    images = [dict(start=m.start(), end=m.end(), sha256=digest(m.group()),
                   code="IMAGE_CONTENT_UNAVAILABLE", scanned_page="scanned-page" in m.group().lower())
              for m in IMAGE.finditer(text)]
    damage = [dict(start=m.start(), end=m.end(), sha256=digest(m.group()),
                   code="ENCODING_DAMAGE") for m in DAMAGE.finditer(text)]
    # The PMC exporter sometimes emits only its carrier label and image links.
    residual = IMAGE.sub("", text).strip()
    carrier_only = bool(images) and residual.lower() in ("", "pmc")
    obstructions = sorted(images+damage, key=lambda r: (r["start"], r["end"], r["code"]))
    if carrier_only:
        obstructions = [dict(start=0, end=len(text), sha256=field["sha256"],
                             code="IMAGE_ONLY_SOURCE")]
    if not text.strip():
        status = "EMPTY_SOURCE"
    elif carrier_only:
        status = "IMAGE_ONLY_SOURCE"
    elif obstructions:
        status = "PARTIAL_TEXT_AVAILABLE"
    else:
        status = "TEXT_AVAILABLE_UNASSESSED"
    return dict(contract=CONTRACT, field_name=field["name"], text_sha256=field["sha256"],
                source_characters=len(text), status=status, image_references=images,
                encoding_damage=damage, obstructions=obstructions,
                native_panel_requirements={m: 5 for m in (("text", "image") if images else ("text",))},
                source_modified=False, quality_assessed=False, grounded=False)


def obstruction_index(annotation):
    """Merge overlapping unavailable intervals; per-node lookup O(log intervals)."""
    spans = []
    for r in annotation["obstructions"]:
        a, b = r["start"], r["end"]
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(b, spans[-1][1])
        else:
            spans.append([a, b])
    return [a for a, _ in spans], [b for _, b in spans]


def intersects(start, end, index):
    starts, ends = index
    i = bisect.bisect_right(ends, start)
    return i < len(starts) and starts[i] < end


def project(field, graph, annotation):
    check(annotation == material(field), "ANNOTATION")
    if graph is None:
        check(not field["text"].strip(), "MISSING_GRAPH")
        return dict(ready_node_ids=[], deferred_nodes=[], next_edges=[], counts={})
    check(graph["source_text_sha256"] == field["sha256"], "GRAPH_SOURCE")
    index = obstruction_index(annotation); ready, deferred, counts = [], [], Counter()
    for node in graph["nodes"]:
        a, b = node["start"], node["end"]
        check(0 <= a < b <= len(field["text"]), "SPAN")
        check(digest(field["text"][a:b]) == node["content_sha256"], "NODE_HASH")
        code = "UNAVAILABLE_MATERIAL" if intersects(a, b, index) else (
            "EMPTY_TEXT" if not field["text"][a:b].strip() else None)
        if code:
            deferred.append(dict(id=node["id"], code=code)); counts["deferred:"+node["level"]] += 1
        else:
            ready.append(node["id"]); counts["available:"+node["level"]] += 1
    valid = set(ready)
    # Preserve existing adjacency. Filtering must never connect across a gap.
    edges = [e for e in graph["edges"] if e["relation"] == "next"
             and e["from"] in valid and e["to"] in valid]
    counts["retained_next_edges"] = len(edges)
    return dict(ready_node_ids=ready, deferred_nodes=deferred, next_edges=edges, counts=dict(counts))


class AvailableField:
    """Validate once at field load; causal queries cannot promote deferred parents."""
    def __init__(self, field, graph, annotation, projection):
        check(projection == project(field, graph, annotation), "PROJECTION")
        self.graph = graph
        self.ready = set(projection["ready_node_ids"])
        self.status = annotation["status"]

    def view(self, cutoff, split_group):
        check(self.graph is not None, "EMPTY_SOURCE")
        check(split_group == self.graph["split_group"], "GROUP")
        check(type(cutoff) is int and 0 <= cutoff <= self.graph["source_characters"], "CUTOFF")
        nodes = [n for n in self.graph["nodes"] if n["id"] in self.ready and n["end"] <= cutoff]
        ids = {n["id"] for n in nodes}
        return dict(nodes=nodes, edges=[e for e in self.graph["edges"]
                    if e["from"] in ids and e["to"] in ids],
                    material_status=self.status, meaning_asserted=False, grounded=False)

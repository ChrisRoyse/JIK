"""Exact published scientific quantities, with scope and display precision preserved.

The checker attests the pinned publication. It does not replicate its experiments
or promote a reported estimate to certain world truth. Explicit numbers are
encoded scalars, never invented senses.
"""
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from claims.codata_capture import SOURCES, LIMIT
from claims.episodic_memory import KINDS, digest

DATES = {"2018": "2019-05-20", "2022": "2024-05-09"}
SCOPE = "pinned_scientific_reference_contents"


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_CODATA_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def visible(raw):
    parser = Text()
    parser.feed(raw.decode("utf-8"))
    return " ".join(" ".join(parser.parts).split())


def number(text, *, ellipsis=False):
    compact = text.replace(" ", "")
    check(bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:\.\.\.)?(?:e[+-]?\d+)?", compact)), "NUMBER_SYNTAX")
    truncated = "..." in compact
    check(not truncated or ellipsis, "UNCERTAINTY_TRUNCATED")
    try:
        value = Decimal(compact.replace("...", ""))
    except InvalidOperation:
        check(False, "NUMBER_SYNTAX")
    check(value.is_finite(), "NUMBER_FINITE")
    # A printed nonterminating exact value is not a completely specified Decimal.
    return dict(display=text, compact=compact, decimal=None if truncated else str(value),
                display_truncated=truncated)


def parse(raw, edition):
    check(edition in DATES, "EDITION")
    lines = raw.decode("ascii").splitlines()
    check(len(lines) > 11 and lines[1].strip() == "Fundamental Physical Constants --- Complete Listing"
          and lines[2].strip() == edition+" CODATA adjustment", "TABLE_EDITION")
    check(lines[9].split() == ["Quantity", "Value", "Uncertainty", "Unit"]
          and set(lines[10]) == {"-"}, "TABLE_HEADER")
    rows = {}
    for line_number, line in enumerate(lines[11:], 12):
        check(len(line) >= 110 and line[59] == line[84] == line[109] == " ", "TABLE_COLUMNS")
        quantity, value, uncertainty, unit = (
            line[:60].strip(), line[60:85].strip(), line[85:110].strip(), line[110:].strip())
        check(quantity and quantity not in rows, "DUPLICATE_QUANTITY")
        exact = uncertainty == "(exact)"
        n = number(value, ellipsis=exact)
        u = None if exact else number(uncertainty)
        check(exact or Decimal(u["decimal"]) > 0, "UNCERTAINTY_POSITIVE")
        rows[quantity] = dict(edition=edition, quantity=quantity, value=n,
                              uncertainty=u, exact_as_published=exact,
                              unit=unit, unit_status="printed" if unit else "dimensionless_table_entry",
                              line=line_number, raw_line=line,
                              publication_date=DATES[edition], measurement_date=None)
    check(0 < len(rows) <= 512, "ROW_BOUND")
    return rows


class CodataSource:
    def __init__(self, root, expected_pin):
        root = Path(root)
        check(all((root/n).is_file() for n in (*SOURCES, "manifest.json", "COMMITTED")), "SOURCE_INCOMPLETE")
        check((root/"manifest.json").stat().st_size <= LIMIT
              and all((root/n).stat().st_size <= LIMIT for n in SOURCES), "SOURCE_SIZE")
        raw = (root/"manifest.json").read_bytes()
        check(sha(raw) == expected_pin and (root/"COMMITTED").read_text() == expected_pin, "SOURCE_PIN")
        manifest = json.loads(raw)
        check(manifest["schema"] == "codata-source-snapshot-v1" and manifest["complete"]
              and manifest["errors"] == [] and set(manifest["files"]) == set(SOURCES), "SOURCE_MANIFEST")
        self.raw, self.manifest, self.pin = {}, manifest, expected_pin
        for name, url in SOURCES.items():
            receipt = manifest["files"][name]
            payload = (root/name).read_bytes()
            host = urlsplit(receipt["final_url"]).hostname
            check(receipt["url"] == url and urlsplit(receipt["final_url"]).scheme == "https"
                  and host and host.endswith(".nist.gov"), "SOURCE_ORIGIN")
            check(0 < len(payload) == receipt["bytes"] <= LIMIT and sha(payload) == receipt["sha256"], "SOURCE_BYTES")
            self.raw[name] = payload
        history = visible(self.raw["history.html"])
        check(re.search(r"9 May 2024.*?Values updated to the 2022 CODATA recommended values\."
                        r".*?2022 values replace the previously recommended 2018 CODATA", history)
              and re.search(r"20 May 2019.*?Values updated to the 2018 CODATA recommended values", history),
              "RELEASE_HISTORY")
        check("historical reference" in visible(self.raw["archive2018.html"])
              and "2022 values" in visible(self.raw["overview.html"]), "PROVIDER_EDITIONS")
        check("Standard Reference Data Act" in visible(self.raw["rights.html"]), "RIGHTS_POLICY")
        self.rows = {year: parse(self.raw[year+".txt"], year) for year in DATES}

    def row(self, edition, quantity):
        check(edition in self.rows and quantity in self.rows[edition], "QUANTITY")
        return self.rows[edition][quantity]

    def episode(self, edition, quantity, kind="source_statement", evidence=()):
        row = self.row(edition, quantity)
        ref = dict(snapshot=self.pin, artifact_sha256=self.manifest["files"][edition+".txt"]["sha256"],
                   edition=edition, quantity=quantity, line=row["line"])
        body = dict(claim_kind=kind, content=row["raw_line"], source=ref,
                    evidence=list(evidence), event_time=None)
        return dict(id=digest(body), **body)

    def validate_episode(self, episode):
        check(type(episode) is dict and set(episode) == {
            "id", "claim_kind", "content", "source", "evidence", "event_time"}, "EPISODE_SCHEMA")
        check(episode["claim_kind"] in KINDS, "CLAIM_KIND")
        ref, evidence = episode["source"], episode["evidence"]
        check(type(ref) is dict and {"edition", "quantity"} <= set(ref), "SOURCE_ADDRESS")
        check(type(evidence) is list and all(type(x) is str for x in evidence)
              and len(set(evidence)) == len(evidence), "EVIDENCE_SCHEMA")
        check(episode == self.episode(ref["edition"], ref["quantity"], episode["claim_kind"], evidence),
              "EPISODE_CONTENT")

    def assess(self, episode):
        self.validate_episode(episode)
        check(episode["claim_kind"] == "source_statement", "NO_WORLD_CHECKER")
        receipt = dict(checker="codata-publication-v1", episode=episode["id"], claim_kind="source_statement",
                       scope=SCOPE, source=episode["source"], world_validity="not_independently_replicated")
        return dict(id=digest(receipt), **receipt)

    def changes(self, old, new):
        check(old == "2018" and new == "2022", "REVISION_ORDER")
        left, right = self.rows[old], self.rows[new]
        fields = ("value", "uncertainty", "exact_as_published", "unit")
        return dict(added=sorted(right.keys()-left.keys()), removed=sorted(left.keys()-right.keys()),
                    changed=sorted(q for q in left.keys() & right.keys()
                                   if any(left[q][f] != right[q][f] for f in fields)),
                    unchanged=sorted(q for q in left.keys() & right.keys()
                                     if all(left[q][f] == right[q][f] for f in fields)))

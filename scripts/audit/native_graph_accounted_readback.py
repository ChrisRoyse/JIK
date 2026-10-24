"""Read numeric graph state and independently enforce persisted timing contracts."""
import json
import math
from pathlib import Path

from audit.native_graph_coverage import read_shard
from kernel.native_graph_accounted import validate_timing
from kernel.native_graph_accounted_store import TIMING_CONTRACT
from text.constellation_corpus import need


def timing_row(r):
    need('timing' in r,'GRAPH_ACCOUNTED_TIMING_MISSING')
    t=r['timing'];validate_timing(t)
    need(t['seconds']==r['seconds'] and t['phases_seconds']==r['phases_seconds']
         and type(r['cpu_seconds']) in (int,float) and math.isfinite(r['cpu_seconds']) and r['cpu_seconds']>=0,
         'GRAPH_ACCOUNTED_TIMING_BINDING')


def read_accounted(root,pin,plan_pin,lane,expected,*,committed=True):
    root=Path(root)
    need(not committed or (root/'COMMITTED').is_file(),'GRAPH_ACCOUNTED_SEAL_MISSING')
    r,a=read_shard(root,pin,plan_pin,lane,expected,committed=committed)
    need(r.get('timing_contract')==TIMING_CONTRACT,'GRAPH_ACCOUNTED_TIMING_CONTRACT');timing_row(r)
    query_seconds=0.;batches=0
    with (root/'batches.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);timing_row(row);query_seconds+=row['seconds'];batches+=1
    # The outer query phases also include function entry/return and clock bookkeeping.
    need(batches==(len(expected)+63)//64 and query_seconds<=r['phases_seconds']['query'],
         'GRAPH_ACCOUNTED_QUERY_WALL')
    return r,a

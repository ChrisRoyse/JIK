"""Compact disjoint query ranges and immutable graph publication receipts."""
import json
from pathlib import Path

from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from kernel.certified_index_graph import pinned
from text.constellation_corpus import digest,file_sha,need


def once(path,value):
    pin=digest(canonical(value))
    if path.exists():need(pinned(path,pin)==value,'GRAPH_CONTINUATION_METADATA_CHANGED')
    else:need(write_committed(path,value)==pin,'GRAPH_CONTINUATION_METADATA_PIN')
    return pin


def tasks(prefix,population,selection,scope):
    need(type(prefix) is int and type(population) is int and 0<=prefix<population,
         'GRAPH_CONTINUATION_POPULATION')
    need(scope in ['qualify','full'],'GRAPH_CONTINUATION_SCOPE')
    if scope=='full':return [dict(start=lo,stop=min(lo+16384,population)) for lo in range(prefix,population,16384)]
    stop=min(prefix+16384,population-1)
    rows=[dict(start=prefix,stop=stop)] if stop>prefix else []
    # The full-sized first tile exercises publication at its production shape.
    # Other intervals cover the frozen numerical controls; the last UID stays singleton.
    for uid in sorted({int(u) for u in selection if stop<=u<population-1}):
        if rows and rows[-1]['stop']==uid and rows[-1]['stop']-rows[-1]['start']<16384:rows[-1]['stop']=uid+1
        else:rows.append(dict(start=uid,stop=uid+1))
    rows.append(dict(start=population-1,stop=population))
    return rows


def check_ranges(rows,prefix,population,scope):
    cursor=prefix;count=0
    for row in rows:
        need(set(row)=={'start','stop'} and all(type(row[k]) is int for k in row)
             and cursor<=row['start']<row['stop']<=population and row['stop']-row['start']<=16384,
             'GRAPH_CONTINUATION_RANGE')
        if scope=='full':need(row['start']==cursor,'GRAPH_CONTINUATION_RANGE_GAP')
        count+=row['stop']-row['start'];cursor=row['stop']
    if scope=='full':need(cursor==population and prefix+count==population,'GRAPH_CONTINUATION_FULL_GAP')
    return count


def validate_records(imports,records,ranges,population,windows,scope):
    cursor=old_count=native=0
    for row in imports:
        need(row['start']==cursor and row['queries']==row['stop']-row['start'] and row['stop']>cursor,
             'GRAPH_CONTINUATION_PREFIX_GAP')
        cursor=row['stop'];old_count+=row['queries'];native+=row['native_windows']
    count=check_ranges(ranges,cursor,population,scope)
    need(len(records)==len(ranges),'GRAPH_CONTINUATION_SHARD_COUNT')
    for row,expected in zip(records,ranges):
        need(row['start']==expected['start'] and row['stop']==expected['stop']
             and row['queries']==expected['stop']-expected['start'],'GRAPH_CONTINUATION_RECORD_SCOPE')
        native+=row['native_windows']
    if scope=='full':need(old_count+count==population and native==windows,'GRAPH_CONTINUATION_NATIVE_COVERAGE')
    return dict(imported_queries=old_count,new_queries=count,queries=old_count+count,
                edges=10*(old_count+count),native_windows=native,population=population,
                full_graph_complete=scope=='full',all_pairs_exhaustive=False)


def read_publication(root):
    root=Path(root);need((root/'COMMITTED').is_file(),'GRAPH_CONTINUATION_PUBLICATION_UNSEALED')
    marker=json.loads((root/'COMMITTED').read_bytes());relative=Path(marker['report_path'])
    need(len(relative.parts)==3 and relative.parts[0]=='runs' and relative.parts[1].isdigit()
         and len(relative.parts[1])==6 and relative.parts[2]=='report.json',
         'GRAPH_CONTINUATION_REPORT_ADDRESS')
    r=pinned(root/relative,marker['report_sha256'])
    need(r['complete'] and r['schema']=='native-graph-continuation-v1','GRAPH_CONTINUATION_REPORT')
    return r,marker['report_sha256']

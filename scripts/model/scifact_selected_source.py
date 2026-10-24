"""Pinned prior selector and native observations; no new source/evaluation claims."""
import argparse
from pathlib import Path
from audit.scifact_rationale_result_readback import context,inspect_training,read_completed
from audit.scifact_rationale_analysis_readback import derive
from text.evaluation_multiscale import file_sha
from model.scifact_selected import selections,plans,donors

PINS=dict(population="5d054e0f8f0cadb6827f79f1f44cbec6f23b78a4c7a144b066db99fe4e435c47",
          rationale="c529b4bdf0d0eadd8f5610ee272159c19c2945fce244dfda334c7b6d3002bb77",
          baseline="2f6024ffaa96cff2cfd0dc1e2cf5f88548809f13b3372017a1eec775b8e98bb0",
          capture="fba87c09ca247767f36f70b8294cd614e082a7fba71b2c7a94e7092f741e2c58",
          completed="5d3153e5fa281f7b1ddb22434dcc1502078ed4bfbda9ca18eb09673b681b1f35")


def arguments(extra=()):
    p=argparse.ArgumentParser()
    for name in (*PINS,"inventory","output",*extra):p.add_argument("--"+name,type=Path,required=True)
    return p.parse_args()


def load(a):
    c=context(a.population,PINS["population"],a.inventory,a.rationale,PINS["rationale"],a.baseline,PINS["baseline"])
    inspected=inspect_training(a.capture,PINS["capture"],c);ctx=c["native"];truth=c["truth"]
    read_completed(a.completed,PINS["completed"],inspected,derive(ctx,truth,inspected),PINS["population"])
    selection=selections(ctx,truth,inspected);gathers=plans(ctx,truth,selection)
    before=dict(ctx["sources"])
    for name in PINS:
        before.update({str(p):file_sha(p) for p in getattr(a,name).rglob('*') if p.is_file()})
    before[str(a.inventory)]=file_sha(a.inventory)
    return dict(ctx=ctx,truth=truth,selection=selection,plans=gathers,donors=donors(ctx["rows"]),before=before,baseline=a.baseline,
                selector_paths={str(s):inspected["outputs"][f"{s}/rationale"]["paths"] for s in (25,26,27)})

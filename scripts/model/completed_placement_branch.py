"""Retain a completed branch's measured evaluations across a placement handoff."""
from model import placement_resume_publication as publication
from model import legal_placement_comparison as measure
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def describe(root, branch, name, census, items, roster):
    need(branch['step']==len(census.documents), 'COMPLETED_PLACEMENT_CURSOR')
    need('tail' not in branch or branch['tail']['records']==0, 'COMPLETED_PLACEMENT_PENDING_TAIL')
    rows,paths=publication.evaluation(root/name/'final',items,name,roster)
    expected=sum(len(i['codes']) for i in items)*(len(roster) if name=='whole_proposals' else 1)
    need(len(rows)==expected,'COMPLETED_PLACEMENT_COVERAGE')
    return dict(schema='completed-placement-branch-v1',step=branch['step'],rows=len(rows),
        checkpoint=branch['checkpoints'][str(branch['step'])],summary=measure.summary(rows,roster),
        row_state=fingerprint(rows),files={name+'/final/'+n:file_sha(root/name/'final'/n) for n in paths})


def validate(root, branch, name, census, items, roster, files):
    need(branch['completed']==describe(root,branch,name,census,items,roster),'COMPLETED_PLACEMENT_CHANGED')
    expected=branch['completed']['files']
    need({n:h for n,h in files.items() if n.startswith(name+'/final/')}==expected,
         'COMPLETED_PLACEMENT_FILE_MANIFEST')
    return branch['completed']


def import_final(source, destination, branch):
    for rel,pin in branch['completed']['files'].items():
        publication.copy_file(source/rel,destination/rel,pin)

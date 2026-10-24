"""Authenticate complete answers, explicit controls and original execution evidence once."""
import json
from pathlib import Path

from audit.complete_answer_baseline_source import load as answers, ROOT
from model.complete_answer_supervision import CompleteAnswerSupervision
from text.constellation_corpus import file_sha,need

BASE = ('complete-answer-baselines-v2-20261024','8ca6330abe6f7fbcb51918e91846d27d079fd81ad0f23012e55084514d9695ca')
COLD = ('complete-answer-baselines-cold-v2-20261024','f7f05f3bb5b545cfb1d3e159a1f1f94c730616e4bf1e687116bfaa36d1738955')
NEW = ('model/complete_answer_supervision.py','audit/complete_supervision_source.py',
       'audit/complete_supervision.py','audit/complete_supervision_readback.py')


def load(root):
    state = answers(root); first,second = [json.loads((root/n/'report.json').read_bytes()) for n,_ in (BASE,COLD)]
    need(all(file_sha(root/n/'report.json')==h for n,h in (BASE,COLD)) and
         second['source']==dict(path=str(root/BASE[0]),pin=BASE[1]) and second['all_selections_exact'], 'COMPLETE_SUPERVISION_BASELINE')
    for n,h in first['files'].items():need(file_sha(root/BASE[0]/n)==h,'COMPLETE_SUPERVISION_BASELINE_FILE')
    model = CompleteAnswerSupervision(state['view']); paths = set(map(Path,state['inputs']))
    for pop in state['view'].source.populations.values():
        if hasattr(pop,'parent'):
            paths.update(pop.parent.root/n for n in (*pop.parent.report['files'],'report.json','COMMITTED'))
    for pop in model.single.populations.values():
        paths.update(pop.root/n for n in (*pop.report['files'],'report.json','COMMITTED'))
        paths.update(map(Path,pop.observations.files))
        paths.update(Path(pop.observations.binding[n]) for n in ('path','proof'))
    paths.update(root/BASE[0]/n for n in (*first['files'],'report.json'))
    paths.add(root/COLD[0]/'report.json')
    inputs = {str(p):dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
    base = Path(__file__).resolve().parents[1]; code = dict(state['code'])
    for n in (*NEW,'model/capability_supervision.py','text/arithmetic_tasks.py','audit/arithmetic_readback.py'):
        code[n] = file_sha(base/n)
    def fresh():
        state['fresh']()
        for n,h in code.items():need(file_sha(base/n)==h,'COMPLETE_SUPERVISION_CODE_CHANGED')
        for n,r in inputs.items():need(file_sha(n)==r['sha256'],'COMPLETE_SUPERVISION_SOURCE_CHANGED')
    return dict(state=state,model=model,inputs=inputs,code=code,fresh=fresh,
        predictions=json.loads((root/BASE[0]/'predictions.json').read_bytes()))


def summary(rows):
    result = {}
    for family in sorted({r['family'] for r in rows}):
        result[family] = {}
        for arm in sorted({r['arm'] for r in rows}):
            selected = [r for r in rows if r['family']==family and r['arm']==arm]; assessed = [r['assessment'] for r in selected if r['assessment']]
            result[family][arm] = dict(tasks=len(selected),available=len(assessed),source_matches=sum(r['source_agreement'] for r in assessed),
                arithmetic_passes=sum(r.get('arithmetic',{}).get('all_recognized_equalities_pass',False) for r in assessed),
                passing_execution_matches=sum(bool(r.get('execution')) and r['execution']['outcome']==1 for r in assessed),
                failed_execution_matches=sum(bool(r.get('execution')) and r['execution']['outcome']==0 for r in assessed),
                new_execution_required=sum(r.get('requires_new_execution',False) for r in assessed),semantic_accepts=0)
    return result

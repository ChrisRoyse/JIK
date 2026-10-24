"""Compare real uninterrupted kernel growth with a cold-restored sparse updater."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_kernel import AssociationCompanion
from kernel.repository_population import RepositoryPopulation
from kernel.repository_update_state import save,restore,producers


def read(path):return json.loads(Path(path).read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('warm','cold'),required=True)
    a=p.parse_args();started=time.perf_counter()
    source=read(a.source/'witness.json');config=read(a.source/'config.json')
    population=RepositoryPopulation(config['journal'],config['pin'])
    split=source['split'];excluded,queries=split['excluded_tasks'],split['queries']
    bound=dict(source=str(a.source),witness=sha((a.source/'witness.json').read_bytes()),code=producers())
    loaded=time.perf_counter()
    if a.stage=='warm':
        check(not a.root.exists(),'UPDATE_WITNESS_EXISTS');a.root.mkdir()
        write_committed(a.root/'config.json',bound)
        companion=AssociationCompanion(population,excluded,queries)
        value,kernel,timing=companion.sync(12,a.root/'warm-prefix-12')
        pin,checks=save(companion,a.root/'checkpoint-12')
        write_committed(a.root/'checkpoint-12-pin.json',dict(pin=pin,checks=checks,kernel=kernel))
    else:
        check(read(a.root/'config.json')==bound,'UPDATE_WITNESS_CONFIG')
        checkpoint=read(a.root/'checkpoint-12-pin.json')
        companion,checks=restore(population,a.root/'checkpoint-12',checkpoint['pin'])
        check(checks==checkpoint['checks'],'UPDATE_WITNESS_RESTORE')
    ready=time.perf_counter();rows=[]
    withdrawn=read(a.source/'withdrawn'/'manifest.json')['withdrawn']
    for label,groups,withdrawals in (('13',13,()),('14',14,()),
                                     ('withdrawn',14,withdrawn),('reinstated',14,())):
        target=a.root/(a.stage+'-prefix-'+label)
        value,pin,timing=companion.sync(groups,target,withdrawn=withdrawals)
        if a.stage=='cold':
            old=read(a.root/('warm-prefix-'+label)/'manifest.json')
            checks={name:digest(value[name])==digest(old[name]) for name in
                    ('ids','absent','slots','graphs','calibration','work','cache_sha256')}
            write_committed(a.root/('cold-compare-'+label+'.json'),checks)
            check(all(checks.values()),'UPDATE_WITNESS_RESUME_DIFFERENCE:'+str(checks))
        else:checks={}
        state_pin,state_checks=save(companion,a.root/(a.stage+'-checkpoint-'+label))
        rows.append(dict(groups=groups,label=label,pin=pin,timing=timing,work=value['work'],equal=checks,
                         state_pin=state_pin,state_checks=state_checks,
                         recall={n:r['recall'] for n,r in value['calibration'].items()}))
        print(json.dumps(dict(stage=a.stage,label=label,equal=checks,seconds=timing['wall_seconds'])),flush=True)
    ended=time.perf_counter()
    result=dict(stage=a.stage,rows=rows,checks=checks,source=config['pin'],
        phases=dict(source=loaded-started,prepare=ready-loaded,updates=ended-ready),wall_seconds=ended-started,
        measured=True,acceptance='author-checked',scientific_admission=False,learned_capability=False)
    write_committed(a.root/(a.stage+'-stage.json'),result)
    print(json.dumps(dict(stage=a.stage,wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            if root.is_dir():write_committed(root/('failure-'+sys.argv[sys.argv.index('--stage')+1]+'.json'),dict(traceback=traceback.format_exc()))
        raise

"""Read persisted native updater generations and exercise physically damaged copies."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from usearch.index import Index

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.episodic_memory import canonical,digest
from claims.process_conditions import write_committed
from kernel.repository_population import RepositoryPopulation
from kernel.repository_update_state import restore


def read(path):return json.loads(Path(path).read_bytes())


def alter(path,value):path.write_bytes(canonical(value))


def inspect(population,path):
    manifest=read(path/'manifest.json');kernel=Path(manifest['kernel']['path'])
    check(read(path/'COMMITTED')==dict(manifest=sha((path/'manifest.json').read_bytes())),
          'UPDATER_READBACK_COMMIT')
    check(sha((kernel/'manifest.json').read_bytes())==manifest['kernel']['pin'],'UPDATER_READBACK_KERNEL')
    value=read(kernel/'manifest.json');cache=read(kernel/'cache.json');rows={}
    for file,info in manifest['files'].items():
        raw=(path/file).read_bytes()
        check(len(raw)==info['bytes'] and sha(raw)==info['sha256'],'UPDATER_READBACK_FILE')
    for name,spec in population.specs.items():
        index=Index.restore(path/(name+'.usearch'));metadata=read(path/(name+'.json'))
        owners=dict(metadata['owners']);keys=[];windows=0
        ids=sorted(k for k in value['ids'] if len(population.values[k][name]))
        check(set(metadata['keys'])==set(ids) and metadata['instrument']==spec.identity,
              'UPDATER_READBACK_ROSTER')
        for atom in ids:
            raw=population.values[atom][name]
            unit=raw.astype(np.float64)
            unit/=np.sqrt(np.einsum('ij,ij->i',unit,unit))[:,None]
            atom_keys=metadata['keys'][atom]
            check(len(atom_keys)==len(raw) and all(owners[k]==atom for k in atom_keys),
                  'UPDATER_READBACK_OWNERS')
            actual=np.stack(index.get(np.asarray(atom_keys,np.uint64),dtype='f32'))
            check(np.array_equal(actual,unit.astype(np.float32)),'UPDATER_READBACK_NATIVE')
            keys.extend(atom_keys);windows+=len(raw)
        check(sorted(keys)==sorted(index.keys[:].tolist()) and len(keys)==len(set(keys)),
              'UPDATER_READBACK_KEYS')
        check(metadata['signatures']=={k:sha(population.values[k][name].astype('<f4').tobytes()) for k in ids},
              'UPDATER_READBACK_SIGNATURES')
        check([e for k in ids for e in metadata['previous_edges'][k]]==value['graphs'][name]['edges'],
              'UPDATER_READBACK_NEIGHBORS')
        check(manifest['checks'][name]['cache_entries']==len(cache[name]),'UPDATER_READBACK_CACHE')
        rows[name]=dict(atoms=len(ids),windows=windows,index_bytes=(path/(name+'.usearch')).stat().st_size,
                        next_key=metadata['next_key'],native_equal=True)
    return dict(pin=sha((path/'manifest.json').read_bytes()),lanes=rows,
                bytes=sum(p.stat().st_size for p in path.iterdir()),kernel=manifest['kernel']['pin'])


def reseal(root):
    manifest=read(root/'manifest.json')
    for name in manifest['files']:
        raw=(root/name).read_bytes();manifest['files'][name]=dict(bytes=len(raw),sha256=sha(raw))
    alter(root/'manifest.json',manifest);pin=sha((root/'manifest.json').read_bytes())
    alter(root/'COMMITTED',dict(manifest=pin));return pin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();start=time.perf_counter()
    config=read(a.root/'config.json');source=Path(config['source']);original=read(source/'config.json')
    population=RepositoryPopulation(original['journal'],original['pin'])
    warm,cold=(read(a.root/(s+'-stage.json')) for s in ('warm','cold'))
    labels=[r['label'] for r in warm['rows']]
    check(labels==[r['label'] for r in cold['rows']]==['13','14','withdrawn','reinstated'],
          'UPDATER_READBACK_STAGES')
    checkpoints={'checkpoint-12':inspect(population,a.root/'checkpoint-12')};successors={}
    for label in labels:
        manifests=[]
        for stage in ('warm','cold'):
            name=stage+'-checkpoint-'+label
            checkpoints[name]=inspect(population,a.root/name)
            manifests.append(read(a.root/name/'manifest.json'))
        check(manifests[0]['files']==manifests[1]['files'],'UPDATER_READBACK_STATE_DIFFERENCE')
        before,after=(read(a.root/(s+'-prefix-'+label)/'manifest.json') for s in ('warm','cold'))
        same={k:before[k]==after[k] for k in ('ids','absent','slots','graphs','calibration','work','cache_sha256')}
        check(all(same.values()) and same==read(a.root/('cold-compare-'+label+'.json')),
              'UPDATER_READBACK_CONTINUATION')
        successors[label]=dict(equal=same,recall={n:r['recall'] for n,r in after['calibration'].items()})
    controls=[];base=a.root/'checkpoint-12';control_root=a.root/'controls'
    check(not control_root.exists(),'UPDATER_READBACK_CONTROL_EXISTS');control_root.mkdir()
    name=sorted(population.specs)[0]
    def control(label,code,mutate,*,seal=True):
        root=control_root/label;shutil.copytree(base,root)
        before=digest({p.name:sha(p.read_bytes()) for p in root.iterdir()})
        mutate(root)
        pin=reseal(root) if seal else read(base/'COMMITTED')['manifest']
        after=digest({p.name:sha(p.read_bytes()) for p in root.iterdir()})
        try:restore(population,root,pin)
        except ValueError as error:check(code in str(error),'UPDATER_READBACK_CONTROL_CODE:'+str(error))
        else:raise AssertionError('UPDATER_READBACK_ACCEPTED:'+label)
        check(before!=after,'UPDATER_READBACK_CONTROL_NO_CHANGE')
        controls.append(dict(name=label,code=code,before=before,after=after,resealed=seal))
    control('missing_commit','UPDATE_STATE_UNCOMMITTED',lambda r:(r/'COMMITTED').unlink(),seal=False)
    def corrupt(root):
        with (root/(name+'.usearch')).open('ab') as stream:stream.write(b'corrupt')
    control('corrupt_index','UPDATE_STATE_BYTES',corrupt,seal=False)
    def metadata_change(root,change):
        path=root/(name+'.json');value=read(path);change(value);alter(path,value)
    control('owner','UPDATE_STATE_OWNER',lambda r:metadata_change(r,lambda v:
        v['owners'][0].__setitem__(1,next(k for k in v['keys'] if k!=v['owners'][0][1]))))
    control('plan','UPDATE_STATE_INSTRUMENT',lambda r:metadata_change(r,lambda v:
        v['plan'].__setitem__('ann_candidates',v['plan']['ann_candidates']+1)))
    def vector(root):
        path=root/(name+'.usearch');index=Index.restore(path);keys=index.keys[:]
        other=index.get(int(keys[-1]),dtype='f32')
        index.remove(np.asarray([keys[0]],np.uint64),threads=1)
        index.add(np.asarray([keys[0]],np.uint64),np.asarray(other).reshape(1,-1),threads=1)
        path.write_bytes(bytes(index.save()))
    control('native_vector','UPDATE_STATE_NATIVE_VECTOR',vector)
    def index_plan(root):
        path=root/(name+'.usearch');old=Index.restore(path);keys=old.keys[:]
        changed=Index(ndim=old.ndim,metric='cos',dtype='f32',connectivity=8)
        changed.add(keys,np.stack(old.get(keys,dtype='f32')),threads=1)
        path.write_bytes(bytes(changed.save()))
    control('index_plan','UPDATE_STATE_INDEX_PLAN',index_plan)
    def source_change(root):
        path=root/'manifest.json';value=read(path);value['source']['head']='0'*64
        alter(path,value)
    control('source','REPOSITORY_STALE_HEAD',source_change)
    def component(value):
        scores=next(iter(value['components'].values()))['scores']
        scores.pop(next(iter(scores)))
    control('component','UPDATE_STATE_COMPONENT_SCORES',lambda r:metadata_change(r,component))
    population.fresh()
    check(checkpoints['checkpoint-12']==inspect(population,base),'UPDATER_READBACK_ORIGINAL_MUTATED')
    result=dict(schema='repository-updater-readback-v1',checkpoints=checkpoints,successors=successors,
        controls=controls,source=original['pin'],number_status='measured',acceptance='author-checked',
        continuation='canonical-reload-each-commit-v1',scientific_admission=False,learned_capability=False,
        boundary_checks=['native_source_to_index_and_owners','committed_generation_to_cold_successor',
                         'withdrawal_and_reinstatement','physically_damaged_resealed_state_refuses'],
        total_checkpoint_bytes=sum(v['bytes'] for v in checkpoints.values()),
        control_bytes=sum(p.stat().st_size for p in control_root.rglob('*') if p.is_file()),
        wall_seconds=time.perf_counter()-start)
    pin=write_committed(a.root/'readback.json',result)
    print(json.dumps(dict(pin=pin,checkpoints=len(checkpoints),controls=len(controls),seconds=result['wall_seconds'])))


if __name__=='__main__':main()

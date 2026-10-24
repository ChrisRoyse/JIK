"""Restored source capture corruptions at provenance, bytes and claim boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.condaqa_source_readback import read,need
from senses.trajectory_population import publish

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    digest=lambda b:hashlib.sha256(b).hexdigest()
    original={p.name:p.read_bytes() for p in a.source.iterdir() if p.is_file()}
    publish(a.output/'cold.json',read(a.source));shadow=a.output/'owned-shadow';shutil.copytree(a.source,shadow)
    phases['cold_source_and_owned_copy']=time.perf_counter()-start;tick=time.perf_counter();controls=[]
    def control(name,file,mutation,code,binary=False):
        changed=mutation(original[file]) if binary else json.dumps(mutation(json.loads(original[file])),sort_keys=True).encode()
        (shadow/file).write_bytes(changed)
        if file!='report.json':
            h=json.loads(original['report.json']);h['files'][file]=dict(bytes=len(changed),sha256=digest(changed))
            (shadow/'report.json').write_text(json.dumps(h),encoding='utf-8')
        try:read(shadow)
        except ValueError as exc:
            need(str(exc)=='MNLM_CONDAQA_READBACK_'+code,'CONTROL_WRONG_REFUSAL')
            controls.append(dict(name=name,expected=code,observed=str(exc),refused=True))
        else:raise ValueError('MNLM_CONDAQA_READBACK_CONTROL_ACCEPTED')
        finally:
            for n,b in original.items():(shadow/n).write_bytes(b)
    def change(value,path,new):
        target=value
        for key in path[:-1]:target=target[key]
        target[path[-1]]=new;return value
    control('provider revision invented','provider-tree.json',lambda v:change(v,['sha'],'0'*40),'TREE')
    control('training bytes changed','train.json',lambda b:b[:-1]+bytes([b[-1]^1]),'BLOB_PIN',True)
    control('licence substituted','LICENSE',lambda b:b.replace(b'Apache',b'XXXXXX',1),'BLOB_PIN',True)
    control('heldout payload requested','capture.json',lambda v:change(v,['receipts','train.json','url'],v['receipts']['train.json']['url'].replace('_train.','_test.')),'URL')
    control('allowlist path changed','capture.json',lambda v:change(v,['selected_source_files','train.json','path'],'data/condaqa_test.json'),'ALLOWLIST')
    control('row count invented','inventory.json',lambda v:change(v,['rows'],5833),'INVENTORY')
    control('answer count invented','inventory.json',lambda v:change(v,['labels','YES'],v['labels']['YES']+1),'INVENTORY')
    control('world facts asserted','report.json',lambda v:change(v,['flags','source_judgments_are_world_facts'],True),'FLAGS')
    control('training admission asserted','report.json',lambda v:change(v,['flags','training_ready'],True),'FLAGS')
    for name,action,restore in [('extra heldout file',lambda:(shadow/'test.json').write_bytes(b'{}'),lambda:(shadow/'test.json').unlink()),
                                ('training file omitted',lambda:(shadow/'train.json').unlink(),lambda:(shadow/'train.json').write_bytes(original['train.json']))]:
        action()
        try:read(shadow)
        except ValueError as exc:
            need(str(exc)=='MNLM_CONDAQA_READBACK_FILES','CONTROL_WRONG_REFUSAL');controls.append(dict(name=name,expected='FILES',observed=str(exc),refused=True))
        else:raise ValueError('MNLM_CONDAQA_READBACK_CONTROL_ACCEPTED')
        finally:restore()
    phases['semantic_corruptions_and_restoration']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'restored.json',read(shadow))
    for root in (a.source,shadow):need(original=={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()},'SOURCE_DRIFT')
    phases['final_readback_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(source_report_sha256=digest(original['report.json']),controls=controls,originals_unchanged=True,shadow_restored=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked',served=False)
    publish(a.output/'report.json',report)
    print(json.dumps(dict(report_sha256=digest((a.output/'report.json').read_bytes()),controls=len(controls),wall_seconds=wall)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())

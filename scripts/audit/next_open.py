"""Open a real immutable learning prefix beside its ledgered candidate successor."""
from functools import partial
from pathlib import Path
import shutil

from audit.corrective_open import open_session
from audit.fresh_learning_open import open_fresh, prepare as prepare_learning, sources
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from model.brier_region_session_v3 import BrierData, BrierLoop
from model.next_experiment import NextLoop


def recipe(config):
    source = Path(config['source'])
    seal = authenticate(Path(config['proof']['root']),config['proof']['pin'])
    for name in ('config.json','snapshot-final.json','experience.sqlite3','candidates/brier-1/manifest.json'):
        check(evidence(source/name) == seal['files']['session/'+name],'NEXT_SOURCE_BYTES')
    previous = read(source/'config.json')
    spec = config['experiment']
    check(previous['root'] == str(source) and spec['learning'] == previous['spec']
          and spec['before'] == previous['spec']['checkpoint'],'NEXT_SOURCE_RECIPE')
    corrected = spec['corrected']
    check(corrected['root'] == str(source/'candidates/brier-1') and
          sha((Path(corrected['root'])/'manifest.json').read_bytes()) == corrected['pin'],
          'NEXT_CANDIDATE_CHECKPOINT')
    native = spec['native']
    check(sha((Path(native['native'])/'report.json').read_bytes()) == native['native_pin'],'NEXT_NATIVE_PIN')
    check(sha(Path(config['native_readback']['path']).read_bytes()) == config['native_readback']['pin']
          and read(Path(config['native_readback']['path']))['native'] == native['native_pin'],
          'NEXT_NATIVE_READBACK')
    return previous


def prepare(config, previous, root):
    check(not root.exists(),'NEXT_OUTPUT_EXISTS')
    root.mkdir()
    prepare_learning(previous,root/'learning')
    source = Path(config['source'])
    shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
    shutil.copytree(source/'candidates/brier-1',root/'candidates/brier-1')


def open_next(config, previous, root, saved):
    resolved = sources(previous)
    prefix,base,sessions = open_fresh(previous,root/'learning',loop_type=BrierLoop,sources_value=resolved)
    data = BrierData(prefix,base,previous['spec'])
    prospective = resolved[2]
    ancestor = Path(prospective['source'])
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(NextLoop,learning_base=base,increment_base=prefix.increment_base,transfer_data=data))
    return session,(session,*sessions)

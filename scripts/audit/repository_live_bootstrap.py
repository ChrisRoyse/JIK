"""Restore the live journal without requiring a separately saved latest snapshot."""
from pathlib import Path

from audit.corrective_open import authenticate
from audit.fresh_learning_open import sources
from audit.repository_memory_readback import chain,read
from audit import successor_residual_open as prior_open
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from model.condition_catalog import NativeConditions
from model.condition_input import ConditionNative
from model.repository_live_memory import LiveRepositoryMemoryLoop
from model.successor_residual_data import ResidualData
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs


def open_current(recipes,root,population,*,loop_type=LiveRepositoryMemoryLoop):
    """A frozen ancestor pins the prefix; SQLite's committed tail supplies its head.

    Derived model/loop state is reconstructed by replay, not supplied by an
    unauthenticated snapshot. This does not claim resistance to a privileged
    actor truncating and rewriting all unsealed successor history.
    """
    base=population.base
    check(sha(Path(base['path']).read_bytes())==base['sha256'],'LIVE_BOOTSTRAP_BASE')
    base_contract,base_rows=chain(Path(base['path']))
    contract,rows=chain(root/'experience.sqlite3')
    check(contract==base_contract and len(base_rows)==base['events'] and
          base_rows[-1][2]==base['head'] and rows[:len(base_rows)]==base_rows,'LIVE_BOOTSTRAP_PREFIX')
    import json
    contract_pin=digest(json.loads(contract[0][0]));head=rows[-1][2]
    del rows,base_rows
    learning,previous=recipes
    original,sessions=prior_open.open_successor(learning,previous,root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    try:
        data=ResidualData(original,original.transfer_data,learning['spec'])
        prospective=sources(previous)[2];source_root=Path(prospective['source'])
        authenticate(source_root,prospective['source_pin'])
        source=read(source_root/'source.json');c=source['config'];field=Path(source['field']['root'])
        authenticate(field,source['field']['seal']);retained=read(field/'source.json')
        native,grounded=Path(retained['root']),Path(retained['retained']['root'])
        for path,pin in ((Path(source['root']),source['seal']),(native,retained['seal']),
                         (grounded,retained['retained']['seal']),
                         (Path(retained['upstream']['root']),retained['upstream']['seal'])):
            authenticate(path,pin)
        raw=ToolSource(Path(c['raw']))
        inputs=ToolInputs(raw,Path(c['population']),Path(c['native']),
                          population_pin=c['population_pin'],native_pin=c['native_pin'])
        providers=[ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']),inputs,
            Path(v['condition_population']),Path(v['condition_native']),
            population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
        parent_root=Path(c['parent'])
        check(sha((parent_root/'run.json').read_bytes())==c['parent_pin'],'LIVE_BOOTSTRAP_MODEL_PARENT')
        parent=read(parent_root/'run.json')
        routing=ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
        session=loop_type(root/'experience.sqlite3',raw,inputs,parent_root/'initial',
            model_pin=parent['initial'],standard=parent['standard'],world=parent['world'],
            contexts={'train:1':c['providers'][0]['task']},device='cuda',
            condition_inputs=NativeConditions(providers),routing=routing,
            growth_root=Path(c['output'])/'growth',candidate_root=grounded/'candidates',
            native_root=native/'candidates',field_root=field/'candidates',controlled_root=root/'candidates',
            expected_head=head,expected_contract=contract_pin,repository_population=population,
            learning_base=original.learning_base,increment_base=original.increment_base,
            transfer_data=original.transfer_data,attempt_data=data,attempt_spec=learning['spec'])
        check(session.head==head,'LIVE_BOOTSTRAP_HEAD')
        return session,(session,*sessions)
    except BaseException:
        for prior in sessions:prior.close()
        raise

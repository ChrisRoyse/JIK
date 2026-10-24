"""Independent source membership and full persisted paired-schedule validation."""
from collections import defaultdict
import numpy as np
from model.multirc_context import check

def validate(grouped,selected,ctx):
    rr=ctx['rows'];y=ctx['labels'];train=set(ctx['train'].tolist());byq=defaultdict(list)
    for i in sorted(train):byq[rr[i]['case']].append(i)
    within=[]
    for case,ii in sorted(byq.items()):
        positive=[i for i in ii if y[i]==1];negative=[i for i in ii if y[i]==0]
        if positive and negative:within.append(dict(key=case,positive=positive,negative=negative,possible_pairs=len(positive)*len(negative)))
    lookup={(r['case'],r['option']):i for i,r in enumerate(rr)};across=[]
    for record in sorted(ctx['contrasts'],key=lambda r:(r['document'],r['option_normalized'])):
        if record['members'][0]['partition']!='train':continue
        pos=[];neg=[]
        for member in record['members']:
            i=lookup[member['case'],member['option']];check(i in train and int(y[i])==member['label'],'CONTRAST_RAW_LABEL')
            (pos if member['label']==1 else neg).append(i)
        across.append(dict(key=[record['document'],record['option_normalized']],positive=sorted(pos),negative=sorted(neg),possible_pairs=len(pos)*len(neg)))
    check(grouped==dict(within_question=within,across_question=across),'CONTRAST_SOURCE_GROUPS')
    check(set(selected)=={'rows','pairs','ordinary','groups'},'CONTRAST_SCHEDULE_FIELDS')
    ids=selected['rows'];pairs=selected['pairs'];ordinary=selected['ordinary'];gi=selected['groups']
    check(ids.shape==(2000,64) and pairs.shape==(2000,24,2) and ordinary.shape==(2000,16) and gi.shape==(2000,24),'CONTRAST_SCHEDULE_SHAPES')
    check(all(np.issubdtype(v.dtype,np.integer) for v in selected.values()),'CONTRAST_SCHEDULE_TYPES')
    check(set(ids.reshape(-1).tolist())<=train,'CONTRAST_TRAIN_SCOPE')
    check(np.all((pairs>=0)&(pairs<64)) and np.all((ordinary>=0)&(ordinary<64)),'CONTRAST_PAIR_ADDRESS')
    coverage=np.sort(np.concatenate((ordinary,pairs.reshape(2000,48)),axis=1),axis=1)
    check(np.all(coverage==np.arange(64)[None]),'CONTRAST_POSITION_COVERAGE')
    families=[within,across];membership=[[(set(g['positive']),set(g['negative'])) for g in family] for family in families]
    for step in range(2000):
        check(len(set(ids[step,ordinary[step]].tolist()))==16,'CONTRAST_ORDINARY_DUPLICATE')
        for j,(left,right) in enumerate(pairs[step]):
            family=j//12;g=int(gi[step,j]);check(0<=g<len(families[family]),'CONTRAST_GROUP_ADDRESS')
            a=int(ids[step,left]);b=int(ids[step,right]);pos,neg=membership[family][g]
            check(a in pos and b in neg and y[a]==1 and y[b]==0,'CONTRAST_PAIR_SOURCE_MEMBERSHIP')
    return dict(updates=2000,ordinary_options_per_update=16,pairs_per_family=12,all_48000_pairs_source_matched=True,
                all_128000_training_positions_clean=True,pair_groups=[len(within),len(across)],possible_pairs=[25234,1747],
                labels_and_group_identifiers_are_model_inputs=False,world_truth_assessed=False)

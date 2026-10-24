"""TRAIN-only categorical support and explicit successor baselines; no sense claims."""
import numpy as np

from text.constellation_corpus import need


def empty(sizes, roster):
    return dict(rows=0, joint=np.zeros(tuple(sizes), dtype=np.int64),
                root_transition=np.zeros((sizes[0],sizes[0]), dtype=np.int64),
                copy_level=np.zeros(len(sizes), dtype=np.int64),
                copy_prefix=np.zeros(len(sizes), dtype=np.int64),
                copy_counts=np.zeros(len(roster), dtype=np.int64),
                count_histogram=np.zeros((len(roster),256), dtype=np.int64))


def add(result, current, target, current_counts, target_counts):
    need(len(current) == len(target) == len(current_counts) == len(target_counts), 'TARGET_BASELINE_ROWS')
    result['rows'] += len(target)
    np.add.at(result['joint'], tuple(target.T), 1)
    np.add.at(result['root_transition'], (current[:,0],target[:,0]), 1)
    equal = current == target
    result['copy_level'] += equal.sum(0)
    result['copy_prefix'] += np.logical_and.accumulate(equal,axis=1).sum(0)
    result['copy_counts'] += (current_counts == target_counts).sum(0)
    for lane in range(target_counts.shape[1]):
        result['count_histogram'][lane] += np.bincount(target_counts[:,lane], minlength=256)


def summarize(result):
    total = result['rows']; joint = result['joint']
    need(total > 0 and int(joint.sum()) == total == int(result['root_transition'].sum()),
         'TARGET_BASELINE_DENOMINATOR')
    levels = [joint.sum(axis=tuple(j for j in range(joint.ndim) if j != i)) for i in range(joint.ndim)]
    prefixes = [joint.sum(axis=tuple(range(i+1,joint.ndim))) for i in range(joint.ndim)]
    out = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    out.update(level_histograms=[x.tolist() for x in levels], prefix_histograms=[x.tolist() for x in prefixes],
               majority_codes=[int(x.argmax()) for x in levels],
               majority_hits=[int(x.max()) for x in levels],
               occupied_prefixes=[int(np.count_nonzero(x)) for x in prefixes],
               support='all_TRAIN_within_document_successor_occurrences',
               fit_uses_held_out=False, scientific_admission=False)
    return out


def derive(data, codes, counts, sizes, roster):
    result = empty(sizes, roster)
    for document,(start,stop) in enumerate(data.documents):
        if data.doc_sides[document] != 0: continue
        for lo in range(int(start),int(stop)-1,16384):
            hi = min(lo+16384,int(stop)-1)
            left = data.mapping[lo:hi]; right = data.mapping[lo+1:hi+1]
            need(bool((data.sides[lo:hi+1] == 0).all()), 'TARGET_BASELINE_HELD_FIT')
            add(result,codes[left],codes[right],counts[left],counts[right])
    need(result['rows'] == data.report['support']['train']['successors'], 'TARGET_BASELINE_COMPLETE_TRAIN')
    data.fresh()
    return summarize(result)


def recount(data, codes, counts, sizes, roster):
    """Separate path over persisted successor addresses, using bincount not add.at."""
    result = empty(sizes,roster); successors = data.source.arrays['successors']
    for lo in range(0,len(successors),16384):
        all_left = successors[lo:lo+16384]
        left = all_left[data.sides[all_left] == 0]; right = left+1
        if not len(left): continue
        data.validate_next(left,right)
        need(bool((data.sides[right] == 0).all()), 'TARGET_BASELINE_RECOUNT_HELD')
        u,v = data.mapping[left],data.mapping[right]
        a,b = codes[u].astype(np.int64),codes[v].astype(np.int64)
        packed = np.ravel_multi_index(tuple(b.T),tuple(sizes))
        result['joint'] += np.bincount(packed,minlength=int(np.prod(sizes))).reshape(sizes)
        result['root_transition'] += np.bincount(a[:,0]*sizes[0]+b[:,0],minlength=sizes[0]**2).reshape(sizes[0],sizes[0])
        result['rows'] += len(left)
        for i in range(len(sizes)):
            result['copy_level'][i] += np.count_nonzero(a[:,i] == b[:,i])
            result['copy_prefix'][i] += np.count_nonzero((a[:,:i+1] == b[:,:i+1]).all(1))
        for i in range(len(roster)):
            result['copy_counts'][i] += np.count_nonzero(counts[u,i] == counts[v,i])
            bins,n = np.unique(counts[v,i],return_counts=True)
            result['count_histogram'][i,bins] += n
    need(result['rows'] == data.report['support']['train']['successors'], 'TARGET_BASELINE_RECOUNT_DENOMINATOR')
    data.fresh()
    return summarize(result)

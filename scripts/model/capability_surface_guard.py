"""Typed native-window coverage guard. Cold-start geometry does not admit a meaning."""
import numpy as np

from claims.capability_population import need, sha
from claims.episodic_memory import digest


POLICY=dict(schema='native-surface-guard-policy-v1',metric='min-mean-MaxSim-v1',
    threshold=0.7,threshold_source='MNLM doctrine #1 section 10 cold-start default',
    calibration='cold_start_not_calibrated',semantic_false_accept_rate=None,
    serving=False,scientific_admission=False)


def coverage(a,b):
    """All windows participate; the physical tile never limits total work."""
    need(a.ndim==b.ndim==2 and len(a)>0 and len(b)>0 and a.shape[1]==b.shape[1]
         and np.isfinite(a).all() and np.isfinite(b).all(),'SURFACE_GUARD_SHAPE')
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    an,bn=np.linalg.norm(a,axis=1,keepdims=True),np.linalg.norm(b,axis=1,keepdims=True)
    need((an>1e-10).all() and (bn>1e-10).all(),'SURFACE_GUARD_ZERO_NORM')
    a,b=a/an,b/bn; left,right=np.full(len(a),-np.inf),np.full(len(b),-np.inf)
    for i in range(0,len(a),16):
        for j in range(0,len(b),16):
            dot=a[i:i+16]@b[j:j+16].T
            left[i:i+16]=np.maximum(left[i:i+16],dot.max(axis=1))
            right[j:j+16]=np.maximum(right[j:j+16],dot.max(axis=0))
    return dict(candidate_window_maxima=left.tolist(),predicted_window_maxima=right.tolist(),
        candidate_coverage=float(left.mean()),predicted_coverage=float(right.mean()),
        score=float(min(left.mean(),right.mean())),window_pairs=len(a)*len(b))


def evaluate(raw,measured,predicted,identities,receipt,*,policy):
    """Receipt binds the entire rendered bytes to separately attested native readings."""
    need(policy==POLICY,'SURFACE_GUARD_POLICY')
    need(isinstance(raw,bytes) and raw,'SURFACE_GUARD_EMPTY')
    try: raw.decode('utf-8',errors='strict')
    except UnicodeDecodeError as exc: raise ValueError('SURFACE_GUARD_UTF8') from exc
    need(set(measured)==set(predicted)==set(identities) and len(identities)>=5,'SURFACE_GUARD_MISSING_SLOT')
    need(receipt['source_sha256']==sha(raw) and receipt['source_bytes']==len(raw),
         'SURFACE_GUARD_SOURCE')
    need(receipt['instruments']==identities and set(receipt['slots'])==set(identities),
         'SURFACE_GUARD_INSTRUMENT')
    results={}
    for name in sorted(identities):
        a,b=measured[name],predicted[name]
        need(a.dtype==b.dtype==np.float32 and receipt['slots'][name]==dict(
            shape=list(a.shape),sha256=sha(a.astype('<f4',copy=False).tobytes())),
            'SURFACE_GUARD_NATIVE_BYTES')
        row=coverage(a,b); row.update(threshold=policy['threshold'],passes=row['score']>=policy['threshold'])
        results[name]=row
    return dict(schema='native-surface-guard-v1',policy=digest(policy),source_sha256=sha(raw),
        receipt=digest(receipt),instruments=identities,slots=results,
        geometric_conjunction=all(v['passes'] for v in results.values()),
        accepted=False,reasons=['PANEL_NOT_ADMITTED','SURFACE_GUARD_NOT_CALIBRATED','ANCHOR_CHECK_REQUIRED'],
        calibration=policy['calibration'],semantic_false_accept_rate=None,
        scientific_admission=False,serving=False)

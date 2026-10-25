"""Numerical runtime and immutable ancestry for protected-decoder run segments."""
import json
import os
from pathlib import Path
import subprocess

import torch
from senses.contract import digest
from text.constellation_corpus import file_sha,need


def runtime(device):
    device=torch.device(device);gpu=None
    if device.type=='cuda':
        lease=Path.home()/'mnlm/.lease/gpu'
        need(lease.is_file() and json.loads(lease.read_bytes()).get('pid')==os.getpid(),'PROTECTED_RUNTIME_LEASE')
        p=torch.cuda.get_device_properties(device)
        gpu=dict(name=p.name,capability=[p.major,p.minor],memory=p.total_memory,
                 driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    return dict(device=str(device),torch=str(torch.__version__),cuda=torch.version.cuda,gpu=gpu,
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def origin(ref,binding,seen=None):
    """Check explicit ancestry without loading model tensors or inferring a parent."""
    seen=set() if seen is None else seen
    need(isinstance(ref,dict) and {'origin','root','step','cursor','ledger','after'}.issubset(ref),
         'PROTECTED_ORIGIN_REQUIRED')
    item=ref['origin'];path=Path(item['path'])
    need(path.name=='origin.json' and path.parent.resolve()==Path(ref['root']).resolve().parent,
         'PROTECTED_ORIGIN_PATH')
    key=str(path.resolve());need(key not in seen,'PROTECTED_ORIGIN_CYCLE');seen.add(key)
    need(file_sha(path)==item['pin'],'PROTECTED_ORIGIN_PIN');v=json.loads(path.read_bytes())
    need(v['schema']=='protected-corpus-origin-v1' and v['program']==binding,'PROTECTED_ORIGIN_BINDING')
    start=v['start'];previous=v['resume']
    if previous is None:
        need(start==dict(cursor=0,step=0,ledger=digest(binding)),'PROTECTED_ORIGIN_INITIAL')
    else:
        source=Path(previous['path']);p=previous['reference']
        need(file_sha(source)==previous['pin'] and json.loads(source.read_bytes())==p,'PROTECTED_ORIGIN_REFERENCE')
        need(source.resolve().parent==Path(p['root']).resolve().parent,'PROTECTED_ORIGIN_REFERENCE_PATH')
        need(start=={k:p[k] for k in ('cursor','step','ledger')} and v['initial_correction']==p['after'],
             'PROTECTED_ORIGIN_CONTINUATION')
        need(p['step']<ref['step'],'PROTECTED_ORIGIN_PROGRESS');origin(p,binding,seen)
    need(start['step']<ref['step'] and start['cursor']<ref['cursor'],'PROTECTED_ORIGIN_EMPTY_SEGMENT')
    return v

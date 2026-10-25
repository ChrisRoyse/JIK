"""Real complete-answer update comparison for native-only scalar checkpoints."""
import argparse
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import answer_scalar_probe as probe
from claims.process_conditions import write_committed
from model import scalar_placement_checkpoint as candidate

NEW=('model/scalar_placement_checkpoint.py','audit/answer_placement_probe.py',
     'model/scalar_answer_checkpoint.py','audit/answer_scalar_probe.py')


def run(a):
    original,code=probe.candidate,probe.NEW
    probe.candidate,probe.NEW=candidate,NEW
    try:probe.run(a)
    finally:probe.candidate,probe.NEW=original,code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=probe.proof.source.ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise

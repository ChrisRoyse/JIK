"""Execute recurrent checkpoint selection and actual retained behavior across restart."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import torch

from audit.text_learning_support import build
from audit.text_selection_controls import controls, selected_predictions
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.text_candidate_evidence import CandidateEvidence
from model.text_candidate_session import TextCandidateSession
from model.text_dynamics import DynamicsData
from model.text_sequence_data import FLAGS, PURPOSE
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def run(a, *, data_type=DynamicsData):
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    phases = {}
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = data_type(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    phases["sources"] = time.perf_counter()-start
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "TEXT_SELECTION_GPU_BUSY")
    take_lease("28", "text-diagnostic-candidate-lifecycle")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build(data, a)
        phases["model"] = time.perf_counter()-tick
        tick = time.perf_counter()
        evidence = CandidateEvidence(a.source_run, a.source_pin, model, data)
        phases["actual_checkpoint_and_tape_replay"] = time.perf_counter()-tick
        if a.cold:
            check(file_sha(a.cold/"report.json") == a.cold_pin, "TEXT_SELECTION_COLD_PIN")
            previous = json.loads((a.cold/"report.json").read_bytes())
            tick = time.perf_counter()
            session = TextCandidateSession(a.cold/"selection.db", evidence, create=False, retained=previous["retained"])
            check(session.retained() == previous["retained"], "TEXT_SELECTION_COLD_POINTER")
            selected = selected_predictions(session, a.output/"selected.db")
            check(selected == previous["selected_predictions"], "TEXT_SELECTION_COLD_BEHAVIOR")
            session.close()
            phases["selected_behavior_after_restart"] = time.perf_counter()-tick
            result = dict(exact=True, retained=previous["retained"], selected_predictions=selected)
        else:
            tick = time.perf_counter()
            session = TextCandidateSession(a.output/"selection.db", evidence, create=True)
            initialized = session.retained()
            decisions = []
            for name in ("cycle-1", "cycle-2"):
                decisions.append(session.review(name, expected_head=session.journal.head, purpose=PURPOSE))
                check(evidence.fingerprint() == session.records[session.selected]["fingerprint"], "TEXT_SELECTION_ACTUAL_SELECTED")
            retained = session.retained()
            phases["decision_and_physical_restore"] = time.perf_counter()-tick
            tick = time.perf_counter()
            negative = controls(session, a.output)
            phases["negative_controls"] = time.perf_counter()-tick
            tick = time.perf_counter()
            selected = selected_predictions(session, a.output/"selected.db")
            session.close()
            session = TextCandidateSession(a.output/"selection.db", evidence, create=False, retained=retained)
            check(session.retained() == retained and evidence.fingerprint() == selected["fingerprint"], "TEXT_SELECTION_RESTART")
            session.close()
            phases["selected_behavior_and_restart"] = time.perf_counter()-tick
            result = dict(initialized=initialized, retained=retained, decisions=decisions,
                controls=negative, selected_predictions=selected,
                successful_acquisition_demonstrated=False, positive_promotion_exercised=False)
        result.update(schema="text-candidate-lifecycle-witness-v1", source_run=a.source_pin,
            evidence=evidence.pin, flags=FLAGS, acceptance="author-checked only", phases_seconds=phases,
            main_function_wall_seconds=time.perf_counter()-start, process_startup_included=False,
            driver_sha256=file_sha(Path(__file__)), controls_sha256=file_sha(Path(__file__).with_name("text_selection_controls.py")))
        result["phase_coverage"] = sum(phases.values())/result["main_function_wall_seconds"]
        check(.90 <= result["phase_coverage"] <= 1.001, "TEXT_SELECTION_PHASE_COVERAGE")
        publish(a.output/"report.json", result)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), selected=result["retained"]["selected"],
                              seconds=result["main_function_wall_seconds"], cold=result.get("exact"))), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("corpus", "bank", "identity-population", "identity-native", "identity-source", "source-run", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin", "source-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    p.add_argument("--causal-regions", action="store_true")
    args = p.parse_args()
    if args.causal_regions:
        from model.text_causal_regions import CausalRegionData
        run(args, data_type=CausalRegionData)
    else:
        run(args)

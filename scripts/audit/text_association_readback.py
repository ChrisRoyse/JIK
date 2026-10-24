"""Cold checkpoint inference; independent NumPy association scoring and bytes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch

from audit.text_dynamics_checks import checkpoint_model
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.native_world_checkpoint import load
from model.text_associations import AssociationData
from model.text_semantic_data import build_semantic
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def logprob(scores):
    x = scores.astype(np.float64)/.1
    x -= x.max(-1, keepdims=True)
    return x-np.log(np.exp(x).sum(-1, keepdims=True))


def statistics(teacher, student):
    target, predicted = logprob(teacher), logprob(student)
    bits = (np.exp(target)*(target-predicted)).sum(-1)/np.log(2)
    a, b = np.argsort(-teacher, axis=-1, kind="stable")[..., :5], np.argsort(-student, axis=-1, kind="stable")[..., :5]
    recall = (a[..., :, None] == b[..., None, :]).any(-1).mean(-1)
    return bits, recall


def run(a):
    need(file_sha(a.run/"report.json") == a.pin, "ASSOCIATION_READBACK_PIN")
    expected = json.loads((a.run/"report.json").read_bytes())
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    data_type, builder = AssociationData, build_semantic
    if "content_input" in expected["data"]:
        from model.text_content_scale import ContentData, build_content
        data_type, builder = ContentData, build_content
    if "association_representation" in expected["data"]:
        from model.text_atom_associations import AtomAssociationData
        data_type = AtomAssociationData
    extra = []
    if "regularizer" in expected["data"]:
        from model.text_sigreg import SigregData
        data_type = SigregData
        extra = [expected["data"]["regularizer"]["strength"]]
    if "geometry" in expected["data"]:
        from model.text_region_geometry import GeometryData, build_geometry
        data_type, builder = GeometryData, build_geometry
        extra = [expected["data"]["geometry"]["mode"]]
    data = data_type(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig(), *extra)
    need(data.bind == expected["data"], "ASSOCIATION_READBACK_BINDING")
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "ASSOCIATION_GPU_BUSY")
    take_lease("28", "association-independent-scoring")
    try:
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = builder(data, a)
        readback, files, maximum_error, regularization = {}, {}, 0., {}
        for stage in ("baseline", "cycle-1", "cycle-2"):
            cp = a.run/"initial-model" if stage == "baseline" else a.run/stage/"model"
            pin = expected["initial_model"] if stage == "baseline" else checkpoint_model(a.run/stage)
            load(cp, model, expected=pin, binding=data.bind)
            with torch.no_grad():
                hidden, ids, _ = data.hidden(model, data.groups)
                positions = {i: j for j, i in enumerate(ids)}
                # Save actual network outputs; subsequent scoring is independent
                # NumPy arithmetic, not the training/evaluation helper.
                vectors = torch.stack([torch.nn.functional.normalize(model.query[name](hidden[:, j]), dim=-1)
                    for j, name in enumerate(data.lanes)]).cpu().numpy()
            path = a.output/(stage+"-typed-internal-states.npz")
            np.savez(path, values=vectors, nodes=np.array(ids, np.int64))
            stored = np.load(path, allow_pickle=False)
            need(np.array_equal(stored["values"], vectors) and np.array_equal(stored["nodes"], ids),
                 "ASSOCIATION_READBACK_INTERNAL_BYTES")
            files[path.name] = file_sha(path)
            refs = vectors[:, [positions[i] for i in data.train_nodes]]
            if "regularizer" in expected["data"]:
                from audit.text_sigreg_readback import inspect
                regularization[stage] = inspect(model, data, hidden, ids, stage, a.output)
            readback[stage] = {}
            for group in data.groups:
                if group["split"] != "development":
                    continue
                nodes = group["indices"][:-1]
                query = vectors[:, [positions[i] for i in nodes]]
                recomputed = query.astype(np.float64)@refs.astype(np.float64).transpose(0, 2, 1)
                with torch.no_grad():
                    raw_scores = []
                    for lane, name in enumerate(data.lanes):
                        q = torch.nn.functional.normalize(model.query[name](hidden[[positions[i] for i in nodes], lane]), dim=-1)
                        r = torch.nn.functional.normalize(model.query[name](hidden[[positions[i] for i in data.train_nodes], lane]), dim=-1)
                        raw_scores.append((q@r.T).cpu().numpy())
                scores = np.stack(raw_scores)
                score_error = float(np.abs(scores-recomputed).max())
                need(score_error < 1e-6, "ASSOCIATION_READBACK_PAIR_ARITHMETIC")
                score_path = a.output/(stage+"-"+group["domain"]+"-scores.npy")
                np.save(score_path, scores, allow_pickle=False)
                scores = np.load(score_path, allow_pickle=False)
                files[score_path.name] = file_sha(score_path)
                target = data.teacher[:, [data.node_pos[i] for i in nodes]]
                bits, recall = statistics(target, scores)
                double_recall = statistics(target, recomputed)[1]
                shuffled = statistics(np.roll(target, 1, -1), scores)[0]
                uniform = statistics(target, np.zeros_like(scores))[0]
                measured = {}
                for lane, name in enumerate(data.lanes):
                    values = dict(kl_bits=float(bits[lane].mean()), neighbor_recall_at_5=float(recall[lane].mean()),
                        shuffled_correspondence_kl_bits=float(shuffled[lane].mean()))
                    old = expected["association_metrics"][stage][group["domain"]][name]
                    error = max(abs(values[k]-old[k]) for k in values)
                    maximum_error = max(maximum_error, error)
                    # Float64 CPU readback versus declared fp32 CUDA scores.
                    need(error < 3e-6, "ASSOCIATION_READBACK_METRICS")
                    values["uniform_kl_bits"] = float(uniform[lane].mean())
                    values["gain_over_uniform_bits"] = values["uniform_kl_bits"]-values["kl_bits"]
                    values["float64_recomputed_recall_at_5"] = float(double_recall[lane].mean())
                    values["pair_score_numeric_difference"] = score_error
                    values["student_similarity_mean"] = float(scores[lane].mean())
                    measured[name] = values
                readback[stage][group["domain"]] = measured
        publish(a.output/"report.json", dict(schema="association-independent-scoring-readback-v1", source=a.pin,
            metrics=readback, files=files, regularization=regularization,
            max_numeric_difference=maximum_error, scoring_tolerance=3e-6,
            acceptance="author-checked only", scientific_admission=False, conditional_lens_information=False))
        print(json.dumps(dict(sha256=file_sha(a.output/"report.json"), max_numeric_difference=maximum_error)))
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("corpus", "bank", "identity-population", "identity-native", "identity-source", "run", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin", "pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())

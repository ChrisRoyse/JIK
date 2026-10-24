"""Independent NumPy characteristic-function integration on persisted real latents."""
import numpy as np
import torch

from model.text_sigreg import directions, projected_states, statistic
from text.constellation_corpus import file_sha, need


def integral(values, projection):
    grid = np.linspace(0., 3., 17)
    reference = np.exp(-.5*grid**2)
    output = []
    for slot, matrix in zip(values.astype(np.float64), projection.astype(np.float64)):
        phase = (slot@matrix)[..., None]*grid
        characteristic = np.exp(1j*phase).mean(axis=0)
        difference = np.abs(characteristic-reference)**2
        output.append(2*len(slot)*np.trapezoid(difference*reference, grid, axis=-1).mean())
    return np.asarray(output)


def inspect(model, data, hidden, ids, stage, root):
    positions = {i: j for j, i in enumerate(ids)}
    chosen = hidden[[positions[i] for i in data.train_nodes]]
    with torch.no_grad():
        values = projected_states(model, data, chosen)
        projection = directions(values, 0)  # fixed audit directions at every checkpoint
        measured = statistic(values, projection).cpu().numpy()
        permutation = torch.arange(values.shape[1]-1, -1, -1, device=values.device)
        permuted = statistic(values[:, permutation], projection).cpu().numpy()
    path = root/(stage+"-sigreg-inputs.npz")
    np.savez(path, values=values.cpu().numpy(), directions=projection.cpu().numpy(),
             statistic=measured, nodes=np.array(data.train_nodes, np.int64))
    stored = np.load(path, allow_pickle=False)
    raw, matrix = stored["values"], stored["directions"]
    recomputed = integral(raw, matrix)
    need(np.allclose(recomputed, stored["statistic"], atol=3e-5, rtol=1e-6), "SIGREG_INDEPENDENT_ARITHMETIC")
    need(np.allclose(permuted, measured, atol=3e-5, rtol=1e-6), "SIGREG_SAMPLE_ORDER")
    centered = raw.astype(np.float64)-raw.mean(axis=1, keepdims=True, dtype=np.float64)
    normalized = raw.astype(np.float64)/np.linalg.norm(raw.astype(np.float64), axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1)
    collapsed = np.broadcast_to(raw.mean(1, keepdims=True), raw.shape)
    collapse_statistic = integral(collapsed, matrix)
    mask = ~np.eye(raw.shape[1], dtype=bool)
    return dict(file=path.name, sha256=file_sha(path), independent_statistic=recomputed.tolist(),
        max_numeric_difference=float(np.abs(recomputed-measured).max()), sample_permutation_invariant=True,
        coordinate_variance=(centered**2).mean((1, 2)).tolist(),
        centered_norm_min=norms.min(1).tolist(),
        mean_off_diagonal_cosine=[float((x@x.T)[mask].mean()) for x in normalized],
        deliberately_collapsed_statistic=collapse_statistic.tolist(),
        names=list(data.lanes), scientific_admission=False)

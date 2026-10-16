"""Select critical states and freeze simulator perturbations plus local geometry."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from flow_rl.factory import instantiate
from flow_rl.manifold.artifact import hash_arrays, load_bundle, save_bundle
from flow_rl.manifold.config import PreparationConfig
from flow_rl.manifold.geometry import (
    full_chunk_starts,
    nearest_action_subspaces,
    select_critical_states,
)

PAPER_URL = (
    "https://proceedings.iclr.cc/paper_files/paper/2026/file/"
    "9299ca9ed58731945e934adb5b71728c-Paper-Conference.pdf"
)
CODE_COMMIT = "9f28318b6f913dda9a63aa65cdab2f1bf4c2a0e4"
CODE_URL = (
    "https://github.com/simchowitzlabpublic/much-ado-about-noising/blob/"
    f"{CODE_COMMIT}/toyexp/toyexp/toyexp/scripts/train_proj.py"
)


def _validate_data(data, name):
    for key in ("observations", "actions", "terminals"):
        if key not in data:
            raise ValueError(f"{name} dataset is missing {key}")
    obs, actions, terminals = (
        np.asarray(data[key]) for key in ("observations", "actions", "terminals")
    )
    if (
        obs.ndim != 2
        or actions.ndim != 2
        or not len(obs)
        or obs.shape[1] < 1
        or actions.shape[1] < 1
    ):
        raise ValueError(f"{name} observations/actions must be nonempty [N, dimension] arrays")
    if len(obs) != len(actions) or terminals.shape != (len(obs),):
        raise ValueError(f"{name} dataset lengths differ")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError(f"{name} data must be finite")
    if not np.isin(terminals, [0, 1]).all():
        raise ValueError(f"{name} terminals must contain only 0 or 1")
    return obs, actions, terminals


def _observation(value, dim):
    value = np.asarray(value, dtype=np.float32).copy()
    if value.shape != (dim,) or not np.isfinite(value).all():
        raise ValueError(f"Simulator must return a finite observation of shape ({dim},)")
    return value


def generate_samples(reference, evaluation, adapter, config: PreparationConfig, progress=None):
    """Model-independent generation. Every rollout starts from the same saved state.

    ``adapter`` supplies physical action bounds, rollout(index, physical_action),
    and optionally restore_observation(index), metadata, restoration_statistics.
    Reference data must come from the training split; selection uses evaluation.
    """
    emit = progress or (lambda message: None)
    ref_obs, ref_actions, ref_terminals = _validate_data(reference, "Reference")
    eval_obs, eval_actions, eval_terminals = _validate_data(evaluation, "Evaluation")
    observation_dim, action_dim = ref_obs.shape[1], ref_actions.shape[1]
    if eval_obs.shape[1] != observation_dim or eval_actions.shape[1] != action_dim:
        raise ValueError("Reference and evaluation dimensions differ")
    low, high = np.asarray(adapter.action_low), np.asarray(adapter.action_high)
    if low.shape != (action_dim,) or high.shape != low.shape:
        raise ValueError("Simulator action bounds do not match dataset")
    if not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("Simulator action bounds must be finite and nondegenerate")
    for name, actions in (("Reference", ref_actions), ("Evaluation", eval_actions)):
        if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
            raise ValueError(f"{name} actions exceed physical action bounds")
    starts = full_chunk_starts(ref_terminals, config.chunk_size)
    if len(starts) < config.neighbors:
        raise ValueError("Not enough full training action chunks for the requested neighbors")
    selection = select_critical_states(
        eval_actions, eval_terminals, config.n_critical, config.window_before, config.window_after
    )
    indices = selection["indices"]
    n, m = config.n_critical, config.perturbations_per_state
    action_mean = ref_actions.mean(axis=0, dtype=np.float64)
    action_std = ref_actions.std(axis=0, dtype=np.float64).clip(config.std_floor)
    observation_mean = ref_obs.mean(axis=0, dtype=np.float64)
    observation_std = ref_obs.std(axis=0, dtype=np.float64).clip(config.std_floor)
    rng = np.random.default_rng(config.seed)
    noise = rng.normal(0, config.noise_scale, (n, m, action_dim))
    # zscore^-1(zscore(a) + eta) = a + std * eta. The mean cancels.
    requested_actions = eval_actions[indices, None, :] + noise * action_std
    applied_actions = np.clip(requested_actions, low, high).astype(np.float32)
    observations = np.empty((n, m, observation_dim), dtype=np.float32)
    nominal = np.empty((n, observation_dim), dtype=np.float32)
    restored = np.empty_like(nominal) if hasattr(adapter, "restore_observation") else None
    emit(
        f"Selected {n} critical states from {len(eval_obs)} evaluation transitions; rolling out {n * m} perturbations"
    )
    for position, index in enumerate(indices):
        if restored is not None:
            restored[position] = _observation(
                adapter.restore_observation(int(index)), observation_dim
            )
        nominal[position] = _observation(
            adapter.rollout(int(index), eval_actions[index]), observation_dim
        )
        for perturbation in range(m):
            observations[position, perturbation] = _observation(
                adapter.rollout(int(index), applied_actions[position, perturbation]),
                observation_dim,
            )
        if (position + 1) % max(1, n // 10) == 0:
            emit(f"Simulator rollouts: {position + 1}/{n} critical states complete")
    query = observations.reshape(n * m, observation_dim).astype(np.float64)
    knn_obs = ref_obs[starts].astype(np.float64)
    if config.observation_metric == "standardized":
        query = (query - observation_mean) / observation_std
        knn_obs = (knn_obs - observation_mean) / observation_std
    chunks = ref_actions[starts[:, None] + np.arange(config.chunk_size)[None, :]].astype(np.float64)
    if config.action_metric == "standardized":
        chunks = (chunks - action_mean) / action_std
    emit(f"Computing exact neighbors and SVD bases against {len(starts)} full training chunks")
    last_progress = -1

    def neighbor_progress(done, total):
        nonlocal last_progress
        fraction = done * 10 // total
        if fraction != last_progress:
            emit(f"Nearest-neighbor queries: {done}/{total}")
            last_progress = fraction

    geometry = nearest_action_subspaces(
        query,
        knn_obs,
        chunks,
        config.neighbors,
        rcond=config.svd_rcond,
        query_batch_size=config.query_batch_size,
        reference_batch_size=config.reference_batch_size,
        progress=neighbor_progress,
    )
    arrays = {
        "observations": observations.reshape(n * m, observation_dim),
        "critical_indices": indices,
        "critical_scores": selection["scores"],
        "selection_scores": selection["all_scores"],
        "critical_window_start": selection["window_start"][indices],
        "critical_window_end": selection["window_end"][indices],
        "critical_observations": eval_obs[indices],
        "critical_actions": eval_actions[indices],
        "nominal_next_observations": nominal,
        "critical_slot": np.repeat(np.arange(n), m),
        "perturbation_index": np.tile(np.arange(m), n),
        "normalized_action_noise": noise.reshape(n * m, action_dim),
        "requested_actions": requested_actions.reshape(n * m, action_dim),
        "applied_actions": applied_actions.reshape(n * m, action_dim),
        "inference_seeds": rng.integers(0, 2**31 - 1, n * m, dtype=np.int64),
        "action_low": low,
        "action_high": high,
        "action_mean": action_mean,
        "action_std": action_std,
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "basis": geometry["basis"],
        "ranks": geometry["ranks"],
        "singular_values": geometry["singular_values"],
        "neighbor_indices": starts[geometry["neighbor_indices"]],
        "neighbor_distances": geometry["neighbor_distances"],
    }
    if restored is not None:
        arrays["restored_observations"] = restored
        arrays["restore_observation_error"] = np.linalg.norm(restored - eval_obs[indices], axis=1)
    if "next_observations" in evaluation:
        original_next = np.asarray(evaluation["next_observations"])
        if original_next.shape != eval_obs.shape or not np.isfinite(original_next).all():
            raise ValueError("Evaluation next_observations must match observations and be finite")
        arrays["dataset_next_observations"] = original_next[indices]
        arrays["nominal_next_observation_error"] = np.linalg.norm(
            nominal - original_next[indices], axis=1
        )
    versions = {}
    for package in ("numpy", "ogbench", "mujoco"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    metadata = {
        "preparation": config.to_dict(),
        "reference_sha256": hash_arrays(reference),
        "evaluation_sha256": hash_arrays(evaluation),
        "reference_transitions": len(ref_obs),
        "evaluation_transitions": len(eval_obs),
        "reference_full_chunks": len(starts),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "action_vector_dim": action_dim * config.chunk_size,
        "sample_count": n * m,
        "versions": versions,
        "source": getattr(adapter, "metadata", {}),
        "restoration_statistics": getattr(adapter, "restoration_statistics", {}),
        "clipped_perturbation_count": int(
            np.any((requested_actions < low) | (requested_actions > high), axis=-1).sum()
        ),
        "selection": "sum of per-coordinate population action variances; trajectory-clipped inclusive window; descending score, ascending index ties",
        "perturbation": "Gaussian in training action z-score units; clip to physical bounds; restore selected state and advance one control step",
        "geometry": "uncentered column span of k training neighbor action chunks; Euclidean KNN; SVD numerical rank",
        "metric_definitions": {
            "off_manifold_norm": "||a - U U^T a||_2",
            "on_manifold_norm": "||U U^T a||_2",
            "total_norm": "||a||_2",
        },
        "references": {"paper": PAPER_URL, "code": CODE_URL, "code_commit": CODE_COMMIT},
    }
    return arrays, metadata


def prepare_from_config(config, progress=None):
    """Reuse an identical existing artifact before constructing any simulator."""
    path = Path(config["samples"])
    generation = {"source": config["source"], "preparation": config["preparation"]}
    if path.exists():
        bundle = load_bundle(path)
        if bundle.metadata.get("generation_config") != generation:
            raise ValueError(
                "Existing samples use a different generation configuration; choose a new samples path"
            )
        return bundle, True
    adapter = instantiate(config["source"]["target"], config["source"].get("kwargs", {}))
    try:
        reference, evaluation = adapter.load_datasets()
        arrays, metadata = generate_samples(
            reference, evaluation, adapter, PreparationConfig(**config["preparation"]), progress
        )
    finally:
        adapter.close()
    metadata["generation_config"] = generation
    return save_bundle(path, arrays, metadata), False

"""Evaluate policies or externally supplied predictions on frozen sample bundles."""

import csv
import json
from pathlib import Path

import numpy as np

from flow_rl.manifold.artifact import file_sha256
from flow_rl.manifold.geometry import decompose_norms

METRICS = ("off_manifold_norm", "on_manifold_norm", "total_norm")


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def evaluate_predictions(predictions, bundle, *, action_space="normalized"):
    """Score complete [N*m,H,A] predictions in the artifact's fixed coordinate system.

    normalized means the policy's [-1,1] actuator coordinates, NOT dataset
    z-scores. physical means environment action units. Metric standardization,
    if requested at preparation, is applied identically to data and predictions.
    """
    arrays, metadata = bundle.arrays, bundle.metadata
    prep = metadata["preparation"]
    expected = (metadata["sample_count"], prep["chunk_size"], metadata["action_dim"])
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != expected or not np.isfinite(predictions).all():
        raise ValueError(f"Predictions must be finite and have shape {expected}")
    if action_space == "normalized":
        if np.any(np.abs(predictions) > 1 + 1e-6):
            raise ValueError(
                "Normalized policy outputs must be in [-1,1]; use physical for external actions"
            )
        low, high = arrays["action_low"], arrays["action_high"]
        physical = low + (predictions + 1) * 0.5 * (high - low)
    elif action_space == "physical":
        physical = predictions.copy()
    else:
        raise ValueError("action_space must be normalized or physical")
    metric_actions = physical
    if prep["action_metric"] == "standardized":
        metric_actions = (physical - arrays["action_mean"]) / arrays["action_std"]
    flat = metric_actions.reshape(len(predictions), -1)
    norms = decompose_norms(flat, arrays["basis"])
    ranks = arrays["ranks"]
    dimension = metadata["action_vector_dim"]
    identity_error = np.abs(
        norms["total_norm"] ** 2 - norms["off_manifold_norm"] ** 2 - norms["on_manifold_norm"] ** 2
    )
    result = {
        "sample_id": bundle.sample_id,
        "sample_count": len(predictions),
        "critical_states": prep["n_critical"],
        "perturbations_per_state": prep["perturbations_per_state"],
        "chunk_size": prep["chunk_size"],
        "action_dim": metadata["action_dim"],
        "observation_metric": prep["observation_metric"],
        "action_metric": prep["action_metric"],
        "metric_definitions": metadata["metric_definitions"],
        "metrics": {key: _statistics(norms[key]) for key in METRICS},
        "geometry": {
            "vector_dimension": dimension,
            "neighbors": prep["neighbors"],
            "rank_min": int(ranks.min()),
            "rank_max": int(ranks.max()),
            "rank_mean": float(ranks.mean()),
            "full_rank_fraction": float(np.mean(ranks == dimension)),
            "zero_rank_fraction": float(np.mean(ranks == 0)),
            "pythagorean_max_absolute_error": float(identity_error.max()),
        },
        "notes": [
            "on_manifold_norm is the projected action magnitude, not error against an expert action",
            "zero vectors in metric coordinates have zero off-manifold norm; interpret together with total norm and task performance",
        ],
    }
    if np.any(ranks == dimension):
        result["notes"].append(
            "Some spans fill the entire action space; off-manifold norm is uninformative for those samples"
        )
    details = dict(norms, physical_predictions=physical, metric_predictions=metric_actions)
    return result, details


def predict_policy(policy, bundle, *, batch_size=256, deterministic=True, seed=0):
    """Use the existing torch actor API; preserve training mode and global RNG state.

    Stochastic inference uses a persisted per-sample seed and single-sample
    calls, so batching and model order cannot change a sample's random seed.
    """
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or seed < 0 or type(deterministic) is not bool:
        raise ValueError("seed must be nonnegative and deterministic a boolean")
    metadata = bundle.metadata
    for name, expected in (
        ("observation_dim", metadata["observation_dim"]),
        ("action_dim", metadata["action_dim"]),
        ("chunk_size", metadata["preparation"]["chunk_size"]),
    ):
        if getattr(policy, name, None) != expected:
            raise ValueError(f"Policy {name} must be {expected} for these samples")
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Reproducible built-in inference currently supports cpu or cuda")
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    observations = bundle.arrays["observations"]
    was_training = policy.training
    output = []
    policy.eval()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            if deterministic:
                for start in range(0, len(observations), batch_size):
                    tensor = torch.as_tensor(
                        observations[start : start + batch_size], device=device, dtype=dtype
                    )
                    output.append(policy.sample(tensor, deterministic=True).cpu().numpy())
            else:
                for index, observation in enumerate(observations):
                    sample_seed = (int(bundle.arrays["inference_seeds"][index]) + seed) % (
                        2**63 - 1
                    )
                    # Seed only the devices whose state is saved by fork_rng.
                    torch.random.default_generator.manual_seed(sample_seed)
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(sample_seed)
                    tensor = torch.as_tensor(observation[None], device=device, dtype=dtype)
                    output.append(policy.sample(tensor, deterministic=False).cpu().numpy())
    finally:
        policy.train(was_training)
    return np.concatenate(output, axis=0)


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(output_dir, bundle, results):
    """Save summaries plus auditable per-sample, per-state and prediction records.

    results: iterable of (label, summary, detail_arrays). All use one sample_id.
    Refuses to replace an existing result directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = bundle.arrays
    sample_rows, state_rows, prediction_arrays, models = [], [], {}, {}
    for model_index, (label, summary, details) in enumerate(results):
        if label in models or summary["sample_id"] != bundle.sample_id:
            raise ValueError(
                "Model labels must be unique and all results must use the same sample artifact"
            )
        models[label] = summary
        prefix = f"model_{model_index}"
        summary["prediction_key"] = prefix
        prediction_arrays[f"{prefix}_physical"] = details["physical_predictions"]
        prediction_arrays[f"{prefix}_metric"] = details["metric_predictions"]
        for index in range(bundle.metadata["sample_count"]):
            slot = int(arrays["critical_slot"][index])
            sample_rows.append(
                {
                    "model": label,
                    "sample_index": index,
                    "critical_slot": slot,
                    "critical_index": int(arrays["critical_indices"][slot]),
                    "perturbation_index": int(arrays["perturbation_index"][index]),
                    "rank": int(arrays["ranks"][index]),
                    **{key: float(details[key][index]) for key in METRICS},
                }
            )
        for slot, index in enumerate(arrays["critical_indices"]):
            mask = arrays["critical_slot"] == slot
            state_rows.append(
                {
                    "model": label,
                    "critical_slot": slot,
                    "critical_index": int(index),
                    "selection_score": float(arrays["critical_scores"][slot]),
                    **{key: float(details[key][mask].mean()) for key in METRICS},
                }
            )
    report = {"sample_id": bundle.sample_id, "sample_metadata": bundle.metadata, "models": models}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_csv(
        output_dir / "per_sample.csv",
        [
            "model",
            "sample_index",
            "critical_slot",
            "critical_index",
            "perturbation_index",
            "rank",
            *METRICS,
        ],
        sample_rows,
    )
    _write_csv(
        output_dir / "per_state.csv",
        ["model", "critical_slot", "critical_index", "selection_score", *METRICS],
        state_rows,
    )
    np.savez_compressed(
        output_dir / "predictions.npz", sample_id=np.array(bundle.sample_id), **prediction_arrays
    )
    return report


def compare_checkpoints(
    bundle,
    checkpoints,
    output_dir,
    *,
    device="cpu",
    batch_size=256,
    deterministic=True,
    seed=0,
    progress=None,
):
    """Inference-only comparison; neither data loading nor environment creation occurs."""
    import torch

    from flow_rl.training.checkpoints import load_policy

    if not checkpoints or any(not isinstance(key, str) or not key for key in checkpoints):
        raise ValueError("Supply at least one named checkpoint")
    if Path(output_dir).exists():
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    for checkpoint in checkpoints.values():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    results = []
    expected_env = bundle.metadata.get("source", {}).get("environment")
    for label, checkpoint in checkpoints.items():
        if progress:
            progress(f"Evaluating {label}: {checkpoint}")
        with torch.random.fork_rng(devices=[]):
            policy, config = load_policy(checkpoint, device=device)
        if expected_env is not None and config.to_dict()["environment"] != expected_env:
            raise ValueError(
                f"Checkpoint {label} environment does not match the frozen sample environment"
            )
        predictions = predict_policy(
            policy, bundle, batch_size=batch_size, deterministic=deterministic, seed=seed
        )
        summary, details = evaluate_predictions(predictions, bundle)
        summary.update(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "model_target": config.model.target,
                "device": str(device),
                "deterministic": deterministic,
                "seed": seed,
                "batch_size": batch_size,
                "torch_version": torch.__version__,
            }
        )
        results.append((label, summary, details))
    return write_results(output_dir, bundle, results)

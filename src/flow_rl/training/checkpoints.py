"""Atomic checkpoints with resolved configuration and offline continuation state."""

from pathlib import Path

import torch

from flow_rl.config import Config, from_dict
from flow_rl.training.build import build_actor


def save_checkpoint(path, config, algorithm, step, observation_dim, action_dim, dataset=None):
    payload = {
        "format_version": 1,
        "config": config.to_dict(),
        "step": step,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "algorithm": algorithm.state_dict(),
        "actor": algorithm.actor.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataset_rng_state": dataset.rng.bit_generator.state if dataset is not None else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return payload


def restore_training(path, config: Config, algorithm, dataset, resume: bool) -> int:
    saved = read_checkpoint(path)
    previous = from_dict(saved["config"])
    for key in ("model", "value", "algorithm", "environment"):
        if getattr(config, key) != getattr(previous, key):
            raise ValueError(f"Checkpoint {key} configuration does not match")
    algorithm.load_state_dict(saved["algorithm"])
    if not resume:
        return 0
    if config.training.mode == "online_rl":
        raise ValueError(
            "Online continuation uses --init-from: a live environment and replay state cannot be exactly resumed"
        )
    if config.data != previous.data or config.training.gamma != previous.training.gamma:
        raise ValueError("Resume requires unchanged dataset configuration and gamma")
    if config.experiment.seed != previous.experiment.seed:
        raise ValueError("Resume requires the original experiment seed")
    torch.set_rng_state(saved["torch_rng_state"])
    if saved["cuda_rng_state"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    if dataset is not None and saved["dataset_rng_state"] is not None:
        dataset.rng.bit_generator.state = saved["dataset_rng_state"]
    return saved["step"]


def load_policy(path, device="cpu"):
    saved = read_checkpoint(path)
    config = from_dict(saved["config"])
    actor = build_actor(config, saved["observation_dim"], saved["action_dim"], device)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    return actor, config

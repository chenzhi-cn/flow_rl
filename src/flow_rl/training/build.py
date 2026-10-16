"""Construct components and data from one resolved experiment configuration."""

import numpy as np
import torch

from flow_rl.config import Config
from flow_rl.data import TransitionDataset, collect_random_dataset, load_npz, load_ogbench_dataset
from flow_rl.factory import instantiate, resolve
from flow_rl.models.budget import build_model


def setup(config: Config) -> torch.device:
    torch.set_num_threads(config.experiment.torch_threads)
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    device = config.experiment.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def make_env(config: Config):
    env = instantiate(config.environment.target, config.environment.kwargs)
    try:
        if (
            env.observation_space.shape is None
            or env.action_space.shape is None
            or len(env.observation_space.shape) != 1
            or len(env.action_space.shape) != 1
            or not hasattr(env.action_space, "low")
        ):
            raise ValueError(
                "The built-in trainers require vector observations and continuous actions"
            )
        low, high = env.action_space.low, env.action_space.high
        if not np.all(np.isfinite(low) & np.isfinite(high) & (low < high)):
            raise ValueError("Action space must have finite, nondegenerate bounds")
    except Exception:
        env.close()
        raise
    return env


def normalize_action(action, env):
    low, high = env.action_space.low, env.action_space.high
    return np.clip(2 * (np.asarray(action) - low) / (high - low) - 1, -1, 1).astype(np.float32)


def denormalize_action(action, env):
    low, high = env.action_space.low, env.action_space.high
    return (low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)).astype(np.float32)


def load_data(config: Config, env) -> TransitionDataset | None:
    cfg = config.data
    if cfg.source == "none":
        return None
    if cfg.source == "ogbench":
        data, _ = load_ogbench_dataset(
            config.environment.kwargs["dataset_name"], cfg.dataset_dir, cur_env=env
        )
    elif cfg.source == "npz":
        data = load_npz(cfg.path)
    else:
        print(
            "Collecting RANDOM simulator transitions for a smoke test; this is not the OGBench play dataset.",
            flush=True,
        )
        data = collect_random_dataset(env, cfg.random_steps, config.experiment.seed)
    if cfg.max_transitions is not None:
        data = {key: value[: cfg.max_transitions] for key, value in data.items()}
    data = dict(data)
    data["actions"] = normalize_action(data["actions"], env)
    dataset = TransitionDataset(
        data, config.model.chunk_size, config.training.gamma, config.experiment.seed
    )
    if (
        dataset.observation_dim != env.observation_space.shape[0]
        or dataset.action_dim != env.action_space.shape[0]
    ):
        raise ValueError("Dataset dimensions do not match the environment")
    return dataset


def build_actor(config: Config, observation_dim: int, action_dim: int, device, build_spec=None):
    return build_model(
        config.model.target,
        config.model.kwargs,
        parameter_budget=config.model.parameter_budget,
        build_spec=build_spec,
        observation_dim=observation_dim,
        action_dim=action_dim,
        chunk_size=config.model.chunk_size,
    ).to(device)


def build_algorithm(
    config: Config, observation_dim: int, action_dim: int, device, dataset=None, build_specs=None
):
    build_specs = build_specs or {}
    actor = build_actor(config, observation_dim, action_dim, device, build_specs.get("actor_spec"))
    algorithm_cls = resolve(config.algorithm.target)
    requires_critic = getattr(algorithm_cls, "requires_critic", True)
    if (config.training.mode == "supervised") == requires_critic:
        raise ValueError(
            "supervised mode requires a supervised algorithm (e.g. BC); RL requires a critic algorithm (e.g. AWAC)"
        )
    components = {"actor": actor}
    if requires_critic:
        components["critic"] = build_model(
            config.value.target,
            config.value.kwargs,
            parameter_budget=config.value.parameter_budget,
            build_spec=build_specs.get("critic_spec"),
            observation_dim=observation_dim,
            action_dim=action_dim,
            chunk_size=config.model.chunk_size,
        ).to(device)
    if dataset is not None and config.data.normalize_observations:
        observations = dataset.data["observations"]
        mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = observations.std(axis=0, dtype=np.float64).clip(1e-3).astype(np.float32)
        for model in components.values():
            if not hasattr(model, "set_observation_stats"):
                raise ValueError(
                    "Model must implement set_observation_stats or disable data.normalize_observations"
                )
            model.set_observation_stats(mean, std)
    return instantiate(config.algorithm.target, config.algorithm.kwargs, **components)


def to_torch(batch: dict, device) -> dict:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }

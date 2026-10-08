"""Online AWAC collection and replay, optionally mixed with offline data."""

from collections import deque

import numpy as np
import torch

from flow_rl.data import ReplayBuffer
from flow_rl.training.build import denormalize_action, normalize_action, to_torch


def mixed_batch(replay, offline, batch_size, offline_ratio):
    offline_count = round(batch_size * offline_ratio) if offline is not None else 0
    batches = []
    if offline_count:
        batches.append(offline.sample(offline_count))
    if batch_size - offline_count:
        batches.append(replay.sample(batch_size - offline_count))
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def train_online(algorithm, dataset, env, config, device, on_step):
    cfg = config.training
    replay = ReplayBuffer(
        cfg.replay_capacity, config.model.chunk_size, cfg.gamma, config.experiment.seed + 1
    )
    obs, _ = env.reset(seed=config.experiment.seed)
    env.action_space.seed(config.experiment.seed)
    pending = deque()
    episode_return, episode_steps = 0.0, 0
    for step in range(1, cfg.steps + 1):
        if step <= cfg.warmup_steps:
            action = normalize_action(env.action_space.sample(), env)
        else:
            if not pending:
                with torch.no_grad():
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    chunk = algorithm.actor.sample(tensor)[0].cpu().numpy()
                pending.extend(chunk)
            action = pending.popleft()
        previous_obs = np.array(obs, copy=True)
        next_obs, reward, terminated, truncated, info = env.step(denormalize_action(action, env))
        episode_steps += 1
        # A runner-imposed horizon is a timeout: retain Bellman bootstrapping.
        truncated = bool(truncated or episode_steps >= config.evaluation.max_steps)
        replay.add(previous_obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        episode_return += float(reward)
        metrics = {"replay_transitions": replay.size, "replay_windows": len(replay)}
        if step >= cfg.warmup_steps and len(replay):
            for _ in range(cfg.updates_per_step):
                batch = mixed_batch(replay, dataset, cfg.batch_size, cfg.offline_ratio)
                metrics.update(algorithm.update(to_torch(batch, device)))
        if terminated or truncated:
            metrics.update(
                episode_return=episode_return, episode_success=float(info.get("success", False))
            )
            obs, _ = env.reset()
            episode_return, episode_steps = 0.0, 0
            pending.clear()
        on_step(step, metrics)
    return cfg.steps

"""Roll out a chunk policy with the same configurable inference controller."""

import time

import numpy as np

from flow_rl.factory import instantiate
from flow_rl.inference import AsyncChunkRunner, ChunkController, SynchronousChunkRunner
from flow_rl.training.build import denormalize_action


def make_runner(policy, config, action_dim):
    cfg = config.inference
    # Custom strategies are zero-argument factories, or configure an adapter factory.
    stitch = instantiate(cfg.stitching) if ":" in cfg.stitching else cfg.stitching
    smooth = instantiate(cfg.smoothing) if ":" in cfg.smoothing else cfg.smoothing
    controller = ChunkController(
        action_dim,
        control_dt=cfg.control_dt,
        stitching=stitch,
        smoothing=smooth,
        fallback=cfg.fallback,
        ema_alpha=cfg.ema_alpha,
        ensemble_decay=cfg.ensemble_decay,
        action_low=-1.0,
        action_high=1.0,
    )
    cls = AsyncChunkRunner if cfg.asynchronous else SynchronousChunkRunner
    return cls(
        policy,
        controller,
        refresh_steps=cfg.refresh_steps,
        time_mode="realtime" if cfg.realtime else "virtual",
    )


def evaluate(policy, env, config) -> dict:
    was_training = getattr(policy, "training", None)
    if hasattr(policy, "eval"):
        policy.eval()
    try:
        return _rollout(policy, env, config)
    finally:
        if was_training is not None and hasattr(policy, "train"):
            policy.train(was_training)


def _rollout(policy, env, config) -> dict:
    returns, successes, lengths, latencies = [], [], [], []
    with make_runner(policy, config, env.action_space.shape[0]) as runner:
        for episode in range(config.evaluation.episodes):
            obs, _ = env.reset(seed=config.experiment.seed + 10000 + episode)
            runner.reset()
            if isinstance(runner, AsyncChunkRunner):
                runner.warm_start(obs)
            total, success = 0.0, False
            origin = time.monotonic()
            for step in range(config.evaluation.max_steps):
                if config.inference.realtime:
                    wait = origin + step * config.inference.control_dt - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                started = time.perf_counter()
                action = (
                    runner.act(obs) if config.inference.realtime else runner.act(obs, step=step)
                )
                latencies.append((time.perf_counter() - started) * 1000)
                obs, reward, terminated, truncated, info = env.step(denormalize_action(action, env))
                total += float(reward)
                success = success or bool(info.get("success", False))
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(success)
            lengths.append(step + 1)
    return {
        "episodes": len(returns),
        "return_mean": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "episode_length_mean": float(np.mean(lengths)),
        "control_call_ms_mean": float(np.mean(latencies)),
        "control_call_ms_p95": float(np.percentile(latencies, 95)),
    }

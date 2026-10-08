"""Small random rollout data for integration smoke tests, not benchmark results."""

import numpy as np

from flow_rl.data.dataset import REQUIRED_KEYS
from flow_rl.envs.base import Environment


def collect_random_dataset(
    env: Environment, num_steps: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Record real simulator transitions without downloading official play data.

    Random actions are intended to check the training/inference plumbing only.
    A successful smoke run is not evidence that the policy has learned the task.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    records: dict[str, list] = {key: [] for key in REQUIRED_KEYS}
    for step in range(num_steps):
        # Drivers may return a view of a sensor buffer that step mutates in place.
        previous_observation = np.array(observation, copy=True)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        transition = {
            "observations": previous_observation,
            "actions": action,
            "rewards": reward,
            "next_observations": next_observation,
            # The final recorded transition ends this data segment, but may bootstrap.
            "terminals": float(terminated or truncated or step == num_steps - 1),
            "masks": float(not terminated),
        }
        for key, value in transition.items():
            records[key].append(np.array(value, copy=True))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    return {key: np.asarray(values, dtype=np.float32) for key, values in records.items()}

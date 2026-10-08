"""Adapter for simulators that already implement the Gymnasium interface."""

import gymnasium as gym


def make_gym_env(env_id: str, **kwargs) -> gym.Env:
    """Create any registered Gymnasium environment, preserving its spaces."""
    return gym.make(env_id, **kwargs)

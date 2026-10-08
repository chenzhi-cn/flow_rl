"""Replaceable reducers for predictions of the same control action."""

from typing import Protocol

import numpy as np


class ChunkStitcher(Protocol):
    def combine(self, actions: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """Combine [N, action_dim] candidates, ordered oldest to newest.

        ``ages`` is the number of control frames since each inference request.
        """
        ...


class ReplaceStitcher:
    """Use the newest available chunk at every control frame."""

    retain_history = False

    def combine(self, actions: np.ndarray, ages: np.ndarray) -> np.ndarray:
        return actions[-1].copy()


class TemporalEnsembleStitcher:
    """Exponentially favor predictions made closer to the current frame."""

    retain_history = True

    def __init__(self, decay: float = 0.5):
        if not np.isfinite(decay) or decay < 0:
            raise ValueError("ensemble decay must be finite and nonnegative")
        self.decay = decay

    def combine(self, actions: np.ndarray, ages: np.ndarray) -> np.ndarray:
        # Subtract the youngest age to keep the exponential numerically stable.
        weights = np.exp(-self.decay * (ages - ages.min()))
        return np.average(actions, axis=0, weights=weights).astype(np.float32)

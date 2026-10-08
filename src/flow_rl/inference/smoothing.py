"""Stateful, replaceable action smoothers."""

from typing import Protocol

import numpy as np


class ActionSmoother(Protocol):
    def reset(self) -> None: ...

    def apply(self, action: np.ndarray) -> np.ndarray: ...


class NoSmoothing:
    def reset(self) -> None:
        pass

    def apply(self, action: np.ndarray) -> np.ndarray:
        return action.copy()


class EMASmoother:
    """``alpha`` is the weight of the new action; 1 disables smoothing."""

    def __init__(self, alpha: float = 0.5):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self._previous is None:
            self._previous = action.copy()
        else:
            self._previous = self.alpha * action + (1 - self.alpha) * self._previous
        return self._previous.copy()

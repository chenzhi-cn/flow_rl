"""Deterministic timestamp-based execution of action chunks."""

from dataclasses import dataclass

import numpy as np

from .smoothing import ActionSmoother, EMASmoother, NoSmoothing
from .stitching import ChunkStitcher, ReplaceStitcher, TemporalEnsembleStitcher


@dataclass
class _Chunk:
    actions: np.ndarray
    requested_at: float
    available_at: float
    skipped: int


class ChunkController:
    """Align chunks to request time, stitch overlap, then smooth and clip.

    All timestamps must use the same clock and units (seconds). At receipt,
    ``ceil((received_at - requested_at) / control_dt)`` leading actions are
    discarded. Calling ``action`` once per control frame is the intended use.
    This object is driven by the caller and does not sleep or start threads.

    Custom stitchers implement ``combine(actions, ages)``; custom smoothers
    implement ``apply(action)`` and ``reset()``. They can be supplied as objects.
    """

    def __init__(
        self,
        action_dim: int,
        control_dt: float = 0.02,
        stitching: str | ChunkStitcher = "replace",
        smoothing: str | ActionSmoother = "none",
        fallback: str = "hold",
        action_low: np.ndarray | float | None = None,
        action_high: np.ndarray | float | None = None,
        ema_alpha: float = 0.5,
        ensemble_decay: float = 0.5,
    ):
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if fallback not in ("hold", "zeros"):
            raise ValueError("fallback must be 'hold' or 'zeros'")
        if isinstance(stitching, str):
            if stitching == "replace":
                stitching = ReplaceStitcher()
            elif stitching == "ensemble":
                stitching = TemporalEnsembleStitcher(ensemble_decay)
            else:
                raise ValueError(f"unknown chunk stitching: {stitching}")
        if isinstance(smoothing, str):
            if smoothing == "none":
                smoothing = NoSmoothing()
            elif smoothing == "ema":
                smoothing = EMASmoother(ema_alpha)
            else:
                raise ValueError(f"unknown action smoothing: {smoothing}")
        self.action_dim = action_dim
        self.control_dt = float(control_dt)
        self.stitcher = stitching
        self.smoother = smoothing
        self.fallback = fallback
        self.low = self._bound(action_low, -np.inf)
        self.high = self._bound(action_high, np.inf)
        if np.any(self.low > self.high):
            raise ValueError("action_low must not exceed action_high")
        self.reset()

    def _bound(self, bound: np.ndarray | float | None, default: float) -> np.ndarray:
        value = np.asarray(default if bound is None else bound, dtype=np.float32)
        try:
            value = np.broadcast_to(value, (self.action_dim,)).copy()
        except ValueError as exc:
            raise ValueError("action bounds must broadcast to [action_dim]") from exc
        if np.any(np.isnan(value)) or (bound is not None and not np.all(np.isfinite(value))):
            raise ValueError("explicit action bounds must be finite")
        return value

    def reset(self) -> None:
        self._chunks: list[_Chunk] = []
        self._last = np.clip(np.zeros(self.action_dim, dtype=np.float32), self.low, self.high)
        self._last_timestamp: float | None = None
        self.smoother.reset()

    def _elapsed_steps(self, elapsed: float) -> int:
        # A tiny relative tolerance prevents a floating point boundary such as
        # 0.3 / 0.1 = 3.0000000000000004 from accidentally skipping frame 3.
        ratio = elapsed / self.control_dt
        return int(np.ceil(ratio - 1e-9))

    def submit(self, chunk: np.ndarray, requested_at: float, received_at: float) -> bool:
        """Add a completed prediction; return False if latency exhausted it."""
        if not np.isfinite(requested_at) or not np.isfinite(received_at):
            raise ValueError("chunk timestamps must be finite")
        if received_at < requested_at:
            raise ValueError("received_at must not precede requested_at")
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"policy must return a nonempty [horizon, {self.action_dim}] action chunk"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("policy action chunk contains nonfinite values")
        skipped = max(0, self._elapsed_steps(received_at - requested_at))
        if skipped >= len(actions):
            return False
        if not getattr(self.stitcher, "retain_history", True):
            if any(item.requested_at > requested_at for item in self._chunks):
                return False
            self._chunks.clear()
        actions = np.clip(actions[skipped:], self.low, self.high).copy()
        self._chunks.append(_Chunk(actions, float(requested_at), float(received_at), skipped))
        self._chunks.sort(key=lambda item: item.requested_at)
        return True

    def action(self, timestamp: float) -> np.ndarray:
        """Return one bounded action, using fallback when no prediction remains."""
        if not np.isfinite(timestamp):
            raise ValueError("action timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("action timestamps must be nondecreasing; reset between episodes")
        self._last_timestamp = float(timestamp)
        candidates, ages, retained = [], [], []
        for item in self._chunks:
            if timestamp < item.available_at:
                retained.append(item)
                continue
            age = max(item.skipped, self._elapsed_steps(timestamp - item.requested_at))
            index = age - item.skipped
            if index < len(item.actions):
                candidates.append(item.actions[index])
                ages.append(age)
                retained.append(item)
        self._chunks = retained
        if candidates:
            action = self.stitcher.combine(np.stack(candidates), np.asarray(ages))
            action = self.smoother.apply(self._validate_action(action))
            action = self._validate_action(action)
            self._last = np.clip(action, self.low, self.high).astype(np.float32)
        elif self.fallback == "zeros":
            # A fallback is immediate; EMA must not leak old motion into zero mode.
            self.smoother.reset()
            self._last = np.clip(np.zeros(self.action_dim, dtype=np.float32), self.low, self.high)
        return self._last.copy()

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError("stitcher and smoother must return finite [action_dim] actions")
        return action

"""Synchronous and asynchronous inference with explicit control timebases."""

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy

import numpy as np

from flow_rl.models.base import ChunkPolicy, Observation

from .controller import ChunkController


class _RunnerBase:
    def __init__(
        self,
        policy: ChunkPolicy,
        controller: ChunkController,
        refresh_steps: int = 1,
        time_mode: str = "virtual",
        clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(refresh_steps, int) or refresh_steps <= 0:
            raise ValueError("refresh_steps must be a positive integer")
        if time_mode not in ("virtual", "realtime"):
            raise ValueError("time_mode must be 'virtual' or 'realtime'")
        self.policy = policy
        self.controller = controller
        self.refresh_steps = refresh_steps
        self.time_mode = time_mode
        self.clock = clock
        self._closed = False
        self._last_requested_at: float | None = None
        self._next_timestamp = 0.0
        self.controller.reset()

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("runner is closed")
        self._last_requested_at = None
        self._next_timestamp = 0.0
        self.controller.reset()

    def _timestamp(self, timestamp: float | None, step: int | None) -> float:
        if timestamp is not None and step is not None:
            raise ValueError("pass timestamp or step, not both")
        if step is not None:
            if not isinstance(step, int) or step < 0:
                raise ValueError("step must be a nonnegative integer")
            if self.time_mode != "virtual":
                raise ValueError("step is only supported with virtual time")
            now = step * self.controller.control_dt
        elif timestamp is not None:
            now = float(timestamp)
        elif self.time_mode == "realtime":
            now = float(self.clock())
        else:
            now = self._next_timestamp
        if not np.isfinite(now):
            raise ValueError("inference timestamps must be finite")
        if self.time_mode == "virtual":
            self._next_timestamp = now + self.controller.control_dt
        return now

    def _due(self, now: float) -> bool:
        interval = self.refresh_steps * self.controller.control_dt
        return self._last_requested_at is None or now - self._last_requested_at >= interval - 1e-9

    def close(self, wait: bool = True) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SynchronousChunkRunner(_RunnerBase):
    """Blocking inference for deterministic simulation and simple deployments.

    In virtual mode the environment is paused during prediction, so compute time
    does not age the chunk. In realtime mode measured compute time does age it.
    The runner never sleeps; a real robot control loop must supply its own pacing.
    """

    def act(
        self, observation: Observation, timestamp: float | None = None, step: int | None = None
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("runner is closed")
        now = self._timestamp(timestamp, step)
        if self._due(now):
            started = self.clock() if self.time_mode == "realtime" else 0.0
            chunk = self.policy.predict(deepcopy(observation))
            received_at = now + (self.clock() - started) if self.time_mode == "realtime" else now
            self.controller.submit(chunk, now, received_at)
            self._last_requested_at = now
            now = received_at
        return self.controller.action(now)


class AsyncChunkRunner(_RunnerBase):
    """Predict in one background worker while executing existing chunks.

    ``virtual`` measures prediction age in simulated frames: pass ``step`` or
    ``timestamp`` to ``act``; without either, calls advance from the previous
    supplied or implicit timestamp by ``control_dt``.
    ``realtime`` uses ``time.monotonic()`` (or the injected clock). Explicit
    timestamps override either clock mode and must share the request timebase.
    Neither mode sleeps. The real robot loop is responsible for control pacing.

    Call ``warm_start`` before fast simulated rollouts if the first frame should
    use a prediction. Subsequent empty chunks use the controller's fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flow-rl-policy")
        self._pending: Future | None = None
        self.reset()

    def reset(self) -> None:
        """Forget prior episode work, including any still-running prediction."""
        super().reset()
        if self._pending is not None:
            self._pending.cancel()
        # A running future cannot be cancelled. Discard its handle: only the
        # newly scheduled episode's future can ever reach controller.submit.
        self._pending = None
        self._pending_requested_at = None

    def warm_start(self, observation: Observation, timestamp: float | None = None) -> bool:
        """Compute the first chunk synchronously without advancing virtual time.

        Realtime warm starts still drop actions consumed by actual compute time.
        Call this directly after reset, before the first call to act.
        """
        if self._closed:
            raise RuntimeError("runner is closed")
        if self._pending is not None or self._last_requested_at is not None:
            raise RuntimeError("warm_start must be called directly after reset")
        started = float(self.clock()) if self.time_mode == "realtime" else 0.0
        requested_at = started if timestamp is None else float(timestamp)
        if not np.isfinite(requested_at):
            raise ValueError("inference timestamps must be finite")
        # Use the same executor as asynchronous calls: an old episode's running
        # request may still be finishing, and policies need not be thread-safe.
        chunk = self._executor.submit(self.policy.predict, deepcopy(observation)).result()
        received_at = (
            requested_at + float(self.clock()) - started
            if self.time_mode == "realtime"
            else requested_at
        )
        accepted = self.controller.submit(chunk, requested_at, received_at)
        self._last_requested_at = requested_at
        if self.time_mode == "virtual":
            self._next_timestamp = requested_at
        return accepted

    def act(
        self, observation: Observation, timestamp: float | None = None, step: int | None = None
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("runner is closed")
        now = self._timestamp(timestamp, step)
        if self._pending is not None and self._pending.done():
            pending, requested_at = self._pending, self._pending_requested_at
            self._pending = None
            self._pending_requested_at = None
            # Exceptions are raised in the control loop, never silently ignored.
            chunk = pending.result()
            self.controller.submit(chunk, requested_at, now)
        if self._pending is None and self._due(now):
            self._pending = self._executor.submit(self.policy.predict, deepcopy(observation))
            self._pending_requested_at = now
            self._last_requested_at = now
        return self.controller.action(now)

    def close(self, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=wait, cancel_futures=True)

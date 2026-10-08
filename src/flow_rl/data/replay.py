"""Bounded online replay with the same chunk-sampling contract as offline data."""

import numpy as np

from flow_rl.data.dataset import sample_windows, validate_horizon


class ReplayBuffer:
    """A ring buffer of actual executed actions and their observed transitions.

    Insertion updates only affected chunk starts. Sampling costs O(batch * chunk)
    and does not rebuild the whole replay dataset on every online update.
    """

    def __init__(self, capacity: int, chunk_size: int = 1, gamma: float = 0.99, seed: int = 0):
        validate_horizon(chunk_size, gamma)
        if not isinstance(capacity, int) or capacity < chunk_size:
            raise ValueError("capacity must be an integer at least as large as chunk_size")
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self._cursor = 0
        self._data: dict[str, np.ndarray] = {}
        self._valid_starts = np.empty(capacity, dtype=np.int64)
        self._positions = np.full(capacity, -1, dtype=np.int64)
        self._num_windows = 0

    def __len__(self) -> int:
        return self._num_windows

    @property
    def data(self) -> dict[str, np.ndarray]:
        """Return a chronological snapshot, for saving or inspection."""
        indices = (np.arange(self.size) + self._oldest) % self.capacity
        return {key: value[indices] for key, value in self._data.items()}

    @property
    def _oldest(self) -> int:
        return (self._cursor - self.size) % self.capacity

    def add(self, obs, action, reward, next_obs, terminated: bool, truncated: bool) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if obs.ndim != 1 or action.ndim != 1 or next_obs.shape != obs.shape:
            raise ValueError(
                "Replay expects vector observations/actions with matching next observation shape"
            )
        if not (
            np.all(np.isfinite(obs))
            and np.all(np.isfinite(next_obs))
            and np.all(np.isfinite(action))
            and np.isfinite(reward)
        ):
            raise ValueError("Replay transitions must be finite")
        if not self._data:
            self.observation_dim, self.action_dim = len(obs), len(action)
            self._data = {
                "observations": np.empty((self.capacity, len(obs)), dtype=np.float32),
                "next_observations": np.empty((self.capacity, len(obs)), dtype=np.float32),
                "actions": np.empty((self.capacity, len(action)), dtype=np.float32),
                **{
                    key: np.empty(self.capacity, dtype=np.float32)
                    for key in ("rewards", "terminals", "masks")
                },
            }
        if len(obs) != self.observation_dim or len(action) != self.action_dim:
            raise ValueError("Replay observation/action dimensions cannot change")
        position = self._cursor
        transition = {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "next_observations": next_obs,
            "terminals": float(terminated or truncated),
            "masks": float(not terminated),
        }
        for key, value in transition.items():
            self._data[key][position] = value
        self._cursor = (position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        for offset in range(self.chunk_size):
            self._refresh_start((position - offset) % self.capacity)

    def _refresh_start(self, start: int) -> None:
        offset = (start - self._oldest) % self.capacity
        valid = offset <= self.size - self.chunk_size
        if valid and self.chunk_size > 1:
            indices = (start + np.arange(self.chunk_size - 1)) % self.capacity
            valid = not np.any(self._data["terminals"][indices])
        position = self._positions[start]
        if valid and position < 0:
            self._valid_starts[self._num_windows] = start
            self._positions[start] = self._num_windows
            self._num_windows += 1
        elif not valid and position >= 0:
            self._num_windows -= 1
            last_start = self._valid_starts[self._num_windows]
            self._valid_starts[position] = last_start
            self._positions[last_start] = position
            self._positions[start] = -1

    def sample(self, batch_size: int, indices=None) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._num_windows:
            raise ValueError("Replay does not yet contain a full, same-trajectory action chunk")
        if indices is None:
            positions = self.rng.integers(self._num_windows, size=batch_size)
            starts = self._valid_starts[positions]
        else:
            offsets = np.asarray(indices)
            if offsets.shape != (batch_size,) or not np.issubdtype(offsets.dtype, np.integer):
                raise ValueError("indices must be an integer vector of length batch_size")
            starts = (offsets + self._oldest) % self.capacity
            if (
                np.any(offsets < 0)
                or np.any(offsets >= self.size)
                or np.any(self._positions[starts] < 0)
            ):
                raise ValueError("indices must start valid, full, same-trajectory windows")
        windows = (starts[:, None] + np.arange(self.chunk_size)[None, :]) % self.capacity
        return sample_windows(self._data, windows, self.gamma)

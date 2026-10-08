"""Trajectory-aware sampling of full action chunks and their Bellman targets."""

from collections.abc import Mapping

import numpy as np

REQUIRED_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "masks",
)


def validate_horizon(chunk_size: int, gamma: float) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")


def sample_windows(
    data: Mapping[str, np.ndarray], windows: np.ndarray, gamma: float
) -> dict[str, np.ndarray]:
    """windows contains transition indices, including the final valid transition.

    A zero mask includes the current reward but stops all subsequent rewards and
    bootstrapping. Trajectory boundaries are checked by the callers separately.
    """
    batch_size, horizon = windows.shape
    rewards = np.zeros(batch_size, dtype=np.float32)
    discounts = np.ones(batch_size, dtype=np.float32)
    for step in range(horizon):
        indices = windows[:, step]
        rewards += discounts * data["rewards"][indices]
        discounts *= gamma * data["masks"][indices]
    return {
        "observations": data["observations"][windows[:, 0]],
        "actions": data["actions"][windows],
        "rewards": rewards,
        "discounts": discounts,
        "next_observations": data["next_observations"][windows[:, -1]],
    }


class TransitionDataset:
    """Vector observations and actions from complete, explicit transitions.

    ``terminals`` marks an episode boundary, including a timeout. ``masks`` is
    the bootstrap mask: zero at task termination and one at a timeout. Keeping
    these fields separate is essential for OGBench's relabeled play datasets.
    Samples contain ``[batch, chunk_size, action_dim]`` actions. Full chunks
    never cross trajectory boundaries; short tails are excluded without padding.
    ``len(dataset)`` counts available windows; ``size`` counts transitions.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        chunk_size: int = 1,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        validate_horizon(chunk_size, gamma)
        missing = set(REQUIRED_KEYS) - data.keys()
        if missing:
            raise ValueError(f"Missing transition fields: {sorted(missing)}")
        self.data = {key: np.asarray(data[key], dtype=np.float32) for key in REQUIRED_KEYS}
        observations = self.data["observations"]
        actions = self.data["actions"]
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                "TransitionDataset expects vector observations/actions shaped [N, dimension]"
            )
        self.size = len(observations)
        if self.data["next_observations"].shape != observations.shape:
            raise ValueError("observations and next_observations must have matching shapes")
        if len(actions) != self.size:
            raise ValueError("actions must have the same length as observations")
        for key in ("rewards", "terminals", "masks"):
            if self.data[key].shape != (self.size,):
                raise ValueError(f"{key} must have shape [N]")
        for key in ("terminals", "masks"):
            if not np.all((self.data[key] == 0) | (self.data[key] == 1)):
                raise ValueError(f"{key} must contain only 0 and 1")
        if not all(np.all(np.isfinite(value)) for value in self.data.values()):
            raise ValueError("Transition fields must be finite")
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        starts = np.arange(max(0, self.size - chunk_size + 1))
        prefix = np.concatenate(([0], np.cumsum(self.data["terminals"], dtype=np.int64)))
        # The last transition in a window may end the trajectory.
        self.valid_starts = starts[prefix[starts + chunk_size - 1] == prefix[starts]]
        if len(self.valid_starts) == 0:
            raise ValueError(f"Dataset has no full, same-trajectory windows of length {chunk_size}")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def sample(self, batch_size: int, indices=None) -> dict[str, np.ndarray]:
        """Sample with replacement; explicit indices are raw transition starts."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if indices is None:
            starts = self.rng.choice(self.valid_starts, size=batch_size)
        else:
            starts = np.asarray(indices)
            if starts.shape != (batch_size,) or not np.issubdtype(starts.dtype, np.integer):
                raise ValueError("indices must be an integer vector of length batch_size")
            positions = np.searchsorted(self.valid_starts, starts)
            if np.any(positions >= len(self.valid_starts)) or np.any(
                self.valid_starts[np.minimum(positions, len(self.valid_starts) - 1)] != starts
            ):
                raise ValueError("indices must start valid, full, same-trajectory windows")
        windows = starts[:, None] + np.arange(self.chunk_size)[None, :]
        return sample_windows(self.data, windows, self.gamma)

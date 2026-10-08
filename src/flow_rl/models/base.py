"""Small contracts for trainable policies, remote VLA policies and world models."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

Observation = np.ndarray | Mapping[str, Any]


@runtime_checkable
class ChunkPolicy(Protocol):
    """Inference needs only this interface, so a policy may live in another process.

    Actions are normalized to [-1, 1] with shape [chunk_size, action_dim].
    A VLA adapter can accept a mapping containing images, state and instructions.
    """

    def predict(self, observation: Observation) -> np.ndarray: ...


class TrainableChunkPolicy(Protocol):
    """AWAC/BC additionally require differentiable likelihoods and model state."""

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor: ...

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor: ...

    def parameters(self, recurse: bool = True): ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]): ...


class WorldModel(Protocol):
    """Optional planning interface; the baseline AWAC learner does not need it."""

    def rollout(self, observation: Observation, actions: np.ndarray) -> list[Observation]: ...

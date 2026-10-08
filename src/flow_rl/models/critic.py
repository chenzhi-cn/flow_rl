"""Independent Q networks over observations and action chunks."""

from collections.abc import Sequence

import torch
from torch import nn

from flow_rl.models.mlp import NormalizedObservationModule, make_mlp


class EnsembleQ(NormalizedObservationModule):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__(observation_dim)
        if action_dim < 1 or chunk_size < 1 or num_qs < 1:
            raise ValueError("action_dim, chunk_size and num_qs must be positive")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_qs = num_qs
        self.qs = nn.ModuleList(
            make_mlp(observation_dim + action_dim * chunk_size, 1, hidden_dims, layer_norm)
            for _ in range(num_qs)
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        expected = (observations.shape[0], self.chunk_size, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        inputs = torch.cat(
            [self.normalize_observations(observations), actions.flatten(start_dim=1)],
            dim=-1,
        )
        return torch.stack([q(inputs).squeeze(-1) for q in self.qs], dim=0)

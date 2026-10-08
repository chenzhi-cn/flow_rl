"""Small Gaussian action-chunk policy for state observations."""

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from flow_rl.models.mlp import NormalizedObservationModule, make_mlp


class GaussianChunkPolicy(NormalizedObservationModule):
    """GELU MLP producing a diagonal Gaussian over the entire action chunk.

    The 256x2 defaults follow OGBench SAC's small state MLP sizes. This PyTorch
    implementation uses a tanh transform for bounded actions. It is an
    architecture adaptation, not an imported OGBench checkpoint. Widths and
    the standard-deviation parameterization are configurable.

    Reference: https://github.com/seohongpark/ogbench/blob/master/impls/agents/sac.py
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        initial_log_std: float = -0.5,
        state_dependent_std: bool = False,
        const_std: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__(observation_dim)
        if action_dim < 1 or chunk_size < 1:
            raise ValueError("action_dim and chunk_size must be positive")
        if not log_std_min <= initial_log_std <= log_std_max:
            raise ValueError("initial_log_std must be between log_std_min and log_std_max")
        if state_dependent_std and const_std:
            raise ValueError("state_dependent_std and const_std cannot both be true")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        flat_dim = action_dim * chunk_size
        self.net = make_mlp(
            observation_dim,
            flat_dim * (2 if state_dependent_std else 1),
            hidden_dims,
            layer_norm,
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)
        if state_dependent_std:
            with torch.no_grad():
                self.net[-1].bias[flat_dim:].fill_(initial_log_std)
        elif const_std:
            self.register_buffer("log_std", torch.full((flat_dim,), initial_log_std))
        else:
            self.log_std = nn.Parameter(torch.full((flat_dim,), initial_log_std))

    def distribution(self, observations: torch.Tensor) -> Normal:
        outputs = self.net(self.normalize_observations(observations))
        if self.state_dependent_std:
            mean, log_std = outputs.chunk(2, dim=-1)
        else:
            mean, log_std = outputs, self.log_std.expand_as(outputs)
        return Normal(mean, log_std.clamp(self.log_std_min, self.log_std_max).exp())

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self.distribution(observations)
        latent = distribution.mean if deterministic else distribution.rsample()
        return latent.tanh().reshape(-1, self.chunk_size, self.action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.sample(observations, deterministic=True)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        expected = (observations.shape[0], self.chunk_size, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        distribution = self.distribution(observations)
        # Datasets often contain exactly +/-1. Inverse tanh at those endpoints
        # is infinite, so evaluate the likelihood at the nearest interior point.
        epsilon = max(torch.finfo(actions.dtype).eps, 1e-6)
        bounded = actions.flatten(start_dim=1).clamp(-1 + epsilon, 1 - epsilon)
        latent = torch.atanh(bounded)
        log_jacobian = 2 * (math.log(2.0) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=-1)

    @torch.inference_mode()
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(
            np.asarray(observation),
            device=self.observation_mean.device,
            dtype=self.observation_mean.dtype,
        )
        if tensor.shape != (self.observation_dim,):
            raise ValueError(f"Expected one observation with shape ({self.observation_dim},)")
        return self.sample(tensor.unsqueeze(0), deterministic=deterministic)[0].cpu().numpy()

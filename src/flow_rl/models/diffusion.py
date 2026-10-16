"""A compact state-conditioned flow-matching action expert.

The time convention, interpolation objective and explicit Euler integration
follow OpenPI's flow head: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
This small transformer takes vector states; it does not contain OpenPI's vision
or language backbone and cannot load a PI0.5 checkpoint.
"""

import math

import numpy as np
import torch
from torch import nn

from flow_rl.models.mlp import NormalizedObservationModule


class _AdaptiveRMSNorm(nn.Module):
    """RMS normalization with a learned scale conditioned on diffusion time."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(tokens.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        normalized = tokens * inverse_rms.to(tokens.dtype)
        return normalized * (1 + self.modulation(condition).unsqueeze(1))


class _Attention(nn.Module):
    """Explicit attention keeps every projection replaceable by budget tooling."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        attended = (
            (weights @ value).transpose(1, 2).reshape(batch_size, sequence_length, hidden_dim)
        )
        return self.output(attended)


class _FlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int):
        super().__init__()
        self.attention_norm = _AdaptiveRMSNorm(hidden_dim)
        self.attention = _Attention(hidden_dim, num_heads)
        self.ffn_norm = _AdaptiveRMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens, condition))
        return tokens + self.ffn(self.ffn_norm(tokens, condition))


class DiffusionChunkPolicy(NormalizedObservationModule):
    """PI0.5-inspired tiny action expert sharing the Gaussian policy's I/O.

    One observation token and H noisy action tokens use bidirectional attention.
    Sinusoidal token positions distinguish the actions within a chunk, and time
    embeddings condition every RMS normalization. Training minimizes velocity
    MSE, not a Gaussian log likelihood. ``t=1`` is noise and ``t=0`` is clean.

    Deterministic sampling uses one persistent, fixed Gaussian noise chunk.
    Stochastic sampling draws a fresh chunk for every observation. No dropout is
    used. Ordinary Linear modules allow active parameter-budget replacements.
    """

    training_objective = "flow_matching"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        time_embed_dim: int = 32,
        num_inference_steps: int = 10,
    ):
        dimensions = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_multiplier": ff_multiplier,
            "time_embed_dim": time_embed_dim,
            "num_inference_steps": num_inference_steps,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads or hidden_dim % 2:
            raise ValueError("hidden_dim must be divisible by num_heads and by 2")
        if time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even")
        super().__init__(observation_dim)
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps
        self.state_embedding = nn.Linear(observation_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _FlowBlock(hidden_dim, num_heads, ff_multiplier) for _ in range(num_layers)
        )
        self.output_norm = _AdaptiveRMSNorm(hidden_dim)
        self.action_output = nn.Linear(hidden_dim, action_dim)

        # Fixed sine/cosine features carry no trainable parameter budget.
        periods = torch.logspace(math.log10(4e-3), math.log10(4.0), time_embed_dim // 2)
        self.register_buffer("time_frequencies", 2 * math.pi / periods)
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, hidden_dim, 2) / hidden_dim)
        positions = torch.arange(chunk_size + 1).unsqueeze(1) * frequencies.unsqueeze(0)
        self.register_buffer(
            "position_embedding", torch.cat([positions.sin(), positions.cos()], dim=-1)
        )
        # An explicit CPU generator avoids resetting or drawing from global RNG.
        # The final move also supports constructors run under torch.device('meta').
        generator = torch.Generator(device="cpu").manual_seed(0)
        fixed_noise = torch.randn(chunk_size, action_dim, generator=generator, device="cpu")
        self.register_buffer("deterministic_noise", fixed_noise.to(self.observation_mean.device))

    def budget_layer_names(self) -> tuple[str, ...]:
        """Allow budget tools to replace feedforward layers, keeping attention dense."""
        return tuple(
            f"blocks.{index}.ffn.{projection}"
            for index in range(len(self.blocks))
            for projection in (0, 2)
        )

    def _check_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> None:
        expected = (observations.shape[0], self.chunk_size, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError("Actions and noise must be finite floating point tensors")

    def _check_observations(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_observations(observations)
        if not observations.is_floating_point() or not torch.isfinite(observations).all():
            raise ValueError("Observations must be finite floating point tensors")
        return normalized

    def velocity(
        self, observations: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Predict dx/dt at noise level ``time`` shaped [batch], in [0, 1]."""
        normalized = self._check_observations(observations)
        self._check_actions(observations, noisy_actions)
        if time.shape != (observations.shape[0],):
            raise ValueError("time must have shape [batch]")
        if (
            not time.is_floating_point()
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("time must be finite floating point values in [0, 1]")
        phase = time.unsqueeze(1) * self.time_frequencies.unsqueeze(0)
        condition = self.time_embedding(torch.cat([phase.sin(), phase.cos()], dim=-1))
        state_token = self.state_embedding(normalized).unsqueeze(1)
        tokens = torch.cat([state_token, self.action_embedding(noisy_actions)], dim=1)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.action_output(self.output_norm(tokens[:, 1:], condition))

    def per_sample_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow-matching MSE [batch], with optional fixed noise/time for tests.

        x_t = t * noise + (1 - t) * action, so dx/dt = noise - action.
        Inverse-CDF sampling of U**(1/1.5) gives OpenPI's Beta(1.5, 1) times.
        """
        self._check_observations(observations)
        self._check_actions(observations, actions)
        if noise is None:
            noise = torch.randn_like(actions)
        self._check_actions(observations, noise)
        if time is None:
            time = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
            time = time.pow(1 / 1.5) * 0.999 + 0.001
        if time.shape != (observations.shape[0],):
            raise ValueError("time must have shape [batch]")
        interpolation = time[:, None, None]
        noisy_actions = interpolation * noise + (1 - interpolation) * actions
        target = noise - actions
        predicted = self.velocity(observations, noisy_actions, time)
        return (predicted - target).square().mean(dim=(1, 2))

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self._check_observations(observations)
        shape = (observations.shape[0], self.chunk_size, self.action_dim)
        if deterministic:
            actions = self.deterministic_noise.unsqueeze(0).expand(shape).clone()
        else:
            actions = torch.randn(shape, device=observations.device, dtype=observations.dtype)
        # Explicit Euler with negative time steps integrates noise at t=1 to t=0.
        dt = -1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            time = observations.new_full((shape[0],), 1 - step / self.num_inference_steps)
            actions = actions + dt * self.velocity(observations, actions, time)
        return actions.clamp(-1, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.sample(observations, deterministic=True)

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

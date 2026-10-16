"""Advantage Weighted Actor Critic for offline and replay-based online learning.

Reference: Nair et al., https://arxiv.org/abs/2006.09359
"""

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from flow_rl.algorithms.base import (
    Batch,
    actor_training_loss,
    optimizer_step,
    policy_objective,
    validate_checkpoint_objective,
)


class AWAC:
    """Double-Q TD learning with an advantage-weighted policy objective.

    Gaussian policies use the original AWAC maximum-likelihood objective. Flow
    policies use advantage-weighted flow matching, a regression surrogate rather
    than exact likelihood AWAC. The Q target and advantage computation are shared.

    `discounts` is already gamma**H times the bootstrap mask, and `rewards`
    is the discounted H-step return. Neither is discounted a second time.
    The baseline V(s) is estimated by sampling this actor and evaluating Q;
    there is no separately fitted IQL value network and no entropy bonus.
    """

    requires_critic = True

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature: float = 1.0,
        tau: float = 0.005,
        max_weight: float = 100.0,
        num_value_samples: int = 1,
        max_grad_norm: float = 10.0,
    ):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if not math.isfinite(max_weight) or max_weight <= 0:
            raise ValueError("max_weight must be finite and positive")
        if num_value_samples < 1 or max_grad_norm <= 0:
            raise ValueError("num_value_samples and max_grad_norm must be positive")
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.temperature = temperature
        self.tau = tau
        self.max_weight = max_weight
        self.num_value_samples = num_value_samples
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0

    @torch.no_grad()
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        batch_size = batch["observations"].shape[0]
        if batch["rewards"].shape != (batch_size,) or batch["discounts"].shape != (batch_size,):
            raise ValueError("rewards and discounts must have shape [batch]")
        next_actions = self.actor.sample(batch["next_observations"])
        next_q = self.target_critic(batch["next_observations"], next_actions).min(dim=0).values
        return batch["rewards"] + batch["discounts"] * next_q

    @torch.no_grad()
    def compute_weights(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        observations = batch["observations"]
        data_q = self.critic(observations, batch["actions"]).min(dim=0).values
        values = torch.stack(
            [
                self.critic(observations, self.actor.sample(observations)).min(dim=0).values
                for _ in range(self.num_value_samples)
            ]
        ).mean(dim=0)
        advantages = data_q - values
        # Clamp before exponentiating, including when advantages are large.
        weights = (advantages / self.temperature).clamp(max=math.log(self.max_weight)).exp()
        return weights, advantages

    @torch.no_grad()
    def _update_target(self) -> None:
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, self.tau)
        for target, source in zip(self.target_critic.buffers(), self.critic.buffers()):
            target.copy_(source)

    def update(self, batch: Batch) -> dict[str, float]:
        self.actor.train()
        self.critic.train()
        targets = self.compute_targets(batch)
        qs = self.critic(batch["observations"], batch["actions"])
        critic_loss = (qs - targets.unsqueeze(0)).square().mean()
        optimizer_step(critic_loss, self.critic_optimizer, self.critic, self.max_grad_norm)

        weights, advantages = self.compute_weights(batch)
        actor_loss, actor_metrics = actor_training_loss(
            self.actor, batch["observations"], batch["actions"], weights
        )
        optimizer_step(actor_loss, self.actor_optimizer, self.actor, self.max_grad_norm)
        self._update_target()
        self.num_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "q_mean": float(qs.detach().mean()),
            "target_q_mean": float(targets.mean()),
            "advantage_mean": float(advantages.detach().mean()),
            "weight_mean": float(weights.detach().mean()),
            "weight_max": float(weights.detach().max()),
            **actor_metrics,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "awac",
            "actor_objective": policy_objective(self.actor),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_updates": self.num_updates,
            "hyperparameters": {
                "temperature": self.temperature,
                "tau": self.tau,
                "max_weight": self.max_weight,
                "num_value_samples": self.num_value_samples,
                "max_grad_norm": self.max_grad_norm,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["algorithm"] != "awac":
            raise ValueError("Expected an AWAC checkpoint")
        validate_checkpoint_objective(self.actor, state)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.num_updates = state["num_updates"]
        for name, value in state["hyperparameters"].items():
            if name not in {
                "temperature",
                "tau",
                "max_weight",
                "num_value_samples",
                "max_grad_norm",
            }:
                raise ValueError(f"Unknown checkpoint hyperparameter: {name}")
            setattr(self, name, value)

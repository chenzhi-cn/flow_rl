"""Maximum-likelihood behavioral cloning with the same policy contract as AWAC."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from flow_rl.algorithms.base import Batch, optimizer_step


class BC:
    requires_critic = False

    def __init__(self, actor: nn.Module, actor_lr: float = 3e-4, max_grad_norm: float = 10.0):
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0

    def update(self, batch: Batch) -> dict[str, float]:
        self.actor.train()
        log_probs = self.actor.log_prob(batch["observations"], batch["actions"])
        loss = -log_probs.mean()
        optimizer_step(loss, self.actor_optimizer, self.actor, self.max_grad_norm)
        self.num_updates += 1
        return {
            "actor_loss": float(loss.detach()),
            "log_prob_mean": float(log_probs.detach().mean()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "bc",
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "max_grad_norm": self.max_grad_norm,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["algorithm"] != "bc":
            raise ValueError("Expected a BC checkpoint")
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.max_grad_norm = state["max_grad_norm"]
        self.num_updates = state["num_updates"]

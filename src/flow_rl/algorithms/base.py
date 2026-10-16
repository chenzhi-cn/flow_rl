"""Shared update and checkpoint contract."""

from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch import nn

Batch = Mapping[str, torch.Tensor]


class Algorithm(Protocol):
    actor: nn.Module

    def update(self, batch: Batch) -> dict[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


def policy_objective(actor: nn.Module) -> str:
    """Legacy policies exposing only log_prob retain their likelihood objective."""
    objective = getattr(actor, "training_objective", "negative_log_likelihood")
    if not isinstance(objective, str) or not objective:
        raise ValueError("Policy training_objective must be a nonempty string")
    return objective


def actor_training_loss(
    actor: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a per-example policy objective, optionally with detached weights.

    Flow matching supplies a regression surrogate, not an action log likelihood.
    The legacy log_prob fallback is therefore allowed only for likelihood policies.
    """
    objective = policy_objective(actor)
    per_sample = getattr(actor, "per_sample_loss", None)
    if callable(per_sample):
        losses = per_sample(observations, actions)
    elif objective == "negative_log_likelihood" and callable(getattr(actor, "log_prob", None)):
        losses = -actor.log_prob(observations, actions)
    else:
        raise TypeError("Policy must implement per_sample_loss(observations, actions)")
    expected = (observations.shape[0],)
    if not isinstance(losses, torch.Tensor) or tuple(losses.shape) != expected:
        raise ValueError(f"Policy per_sample_loss must return shape {expected}")
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Non-finite per-example policy loss")
    if weights is None:
        loss = losses.mean()
    else:
        if tuple(weights.shape) != expected:
            raise ValueError(f"Policy weights must have shape {expected}")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Policy weights must be finite and nonnegative")
        loss = (weights.detach() * losses).mean()
    if objective == "negative_log_likelihood":
        metrics = {"log_prob_mean": float((-losses.detach()).mean())}
    elif objective == "flow_matching":
        metrics = {"flow_matching_loss_mean": float(losses.detach().mean())}
    else:
        metrics = {"policy_loss_mean": float(losses.detach().mean())}
    return loss, metrics


def validate_checkpoint_objective(actor: nn.Module, state: Mapping[str, Any]) -> None:
    """Check before loading tensors; checkpoints predating this field used NLL."""
    saved = state.get("actor_objective", "negative_log_likelihood")
    current = policy_objective(actor)
    if saved != current:
        raise ValueError(f"Checkpoint actor objective {saved!r} does not match policy {current!r}")


def optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: float,
) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss; check the dataset and learning rate")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()

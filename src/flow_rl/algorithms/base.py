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

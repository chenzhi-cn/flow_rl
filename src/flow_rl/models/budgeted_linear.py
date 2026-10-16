"""A linear layer whose parameter count equals its connected trainable edges."""

import torch
import torch.nn.functional as F
from torch import nn


class BudgetedLinear(nn.Module):
    """An affine layer with a fixed sparse connectivity pattern.

    Only retained weights are Parameters. Missing connections are zero, not
    frozen or unused trainable padding. Every input and output retains at least
    one connection. The dense materialization favors a small, portable baseline
    over a dependency on a device-specific sparse kernel.
    """

    def __init__(self, linear: nn.Linear, weight_count: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        size = self.in_features * self.out_features
        minimum = max(self.in_features, self.out_features)
        if isinstance(weight_count, bool) or not isinstance(weight_count, int):
            raise TypeError("weight_count must be an integer")
        if not minimum <= weight_count <= size:
            raise ValueError(f"weight_count must be between {minimum} and {size}")
        # First connect every input and output; then choose additional unique
        # edges with a local generator, independent of the experiment RNG.
        k = torch.arange(minimum, device="cpu")
        coverage = (k % self.out_features) * self.in_features + k % self.in_features
        available = torch.ones(size, dtype=torch.bool, device="cpu")
        available[coverage] = False
        candidates = available.nonzero(as_tuple=True)[0]
        generator = torch.Generator(device="cpu").manual_seed(0)
        order = torch.randperm(len(candidates), generator=generator, device="cpu")
        indices = torch.cat((coverage, candidates[order[: weight_count - minimum]])).sort().values
        indices = indices.to(linear.weight.device)
        self.register_buffer("indices", indices)
        self.values = nn.Parameter(linear.weight.detach().flatten()[indices].clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat_weight = self.values.new_zeros(self.out_features * self.in_features)
        weight = flat_weight.scatter(0, self.indices, self.values).view(
            self.out_features, self.in_features
        )
        return F.linear(inputs, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, weights={self.values.numel()}, bias={self.bias is not None}"

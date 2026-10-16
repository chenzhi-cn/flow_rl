"""Numerical primitives for plasticity diagnostics, independent of training code."""

import math
from collections.abc import Callable
from numbers import Real

import torch
from torch import nn

_ACTIVATION_METRICS = (
    "dead_neuron_ratio",
    "dormant_neuron_ratio",
    "stable_rank",
    "effective_rank",
)


def activation_matrix(activations: torch.Tensor, feature_dim: int = -1) -> torch.Tensor:
    """Detach activations and flatten every non-feature axis into sample rows.

    No centering or normalization is applied. For example, a ``[B, H, A]``
    action chunk with ``feature_dim=-1`` produces ``B * H`` rows and ``A``
    feature columns. The returned dense matrix is float64 on CPU.
    """
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")
    if activations.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    if not activations.is_floating_point():
        raise TypeError("activations must have a real floating-point dtype")
    if activations.layout != torch.strided:
        raise ValueError("activations must be a dense tensor")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise TypeError("feature_dim must be an integer")
    if not -activations.ndim <= feature_dim < activations.ndim:
        raise ValueError("feature_dim is outside the activation dimensions")
    if activations.numel() == 0:
        raise ValueError("activations must contain at least one sample and one feature")
    matrix = activations.detach().to(device="cpu", dtype=torch.float64)
    matrix = matrix.movedim(feature_dim, -1).reshape(-1, activations.shape[feature_dim])
    if not torch.isfinite(matrix).all():
        raise ValueError("activations must contain only finite values")
    return matrix


def _matrix(matrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError("matrix must have exactly two dimensions: samples, features")
    return activation_matrix(matrix)


def _rank_metrics(matrix: torch.Tensor) -> tuple[float, float]:
    scale = matrix.abs().max().item()
    if scale == 0.0:
        return 0.0, 0.0
    # Scaling preserves both ranks while avoiding overflow in SVD or sigma**2.
    singular_values = torch.linalg.svdvals(matrix / scale)
    relative = singular_values / singular_values[0]
    stable_rank = relative.square().sum().item()
    probabilities = relative / relative.sum()
    probabilities = probabilities[probabilities > 0]
    effective_rank = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    return stable_rank, effective_rank


def rank_metrics(matrix: torch.Tensor) -> tuple[float, float]:
    """Return conventional stable rank and entropy effective rank.

    ``stable_rank = sum(sigma**2) / max(sigma)**2``;
    ``effective_rank = exp(-sum(p * log(p)))``, where ``p = sigma / sum(sigma)``.
    The input is the uncentered sample-by-feature activation matrix. Both
    statistics are defined as zero for an all-zero matrix.
    """
    return _rank_metrics(_matrix(matrix))


def mean_absolute_weight(
    model: nn.Module,
    *,
    include_bias: bool = False,
    parameter_filter: Callable[[str, nn.Parameter], bool] | None = None,
) -> float | None:
    """Return an element-weighted mean of selected absolute parameter values.

    By default, parameter leaf names ``weight``, ``*_weight`` and ``values``
    are included. ``values`` covers the active edges of ``BudgetedLinear``;
    its absent connections are not parameters. Normalization weights and
    frozen weights are included, while biases and buffers are excluded.
    ``include_bias`` also selects ``bias`` and ``*_bias``. A custom filter
    replaces these name rules (e.g. ``lambda name, parameter: True`` selects
    all parameters). Shared parameters are counted once, even with aliases.
    Return ``None`` if no nonempty parameters are selected.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(include_bias, bool):
        raise TypeError("include_bias must be a bool")
    if parameter_filter is not None and not callable(parameter_filter):
        raise TypeError("parameter_filter must be callable or None")
    count = 0
    scale = 0.0
    scaled_sum = 0.0
    seen: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        leaf = name.rsplit(".", 1)[-1]
        selected = (
            bool(parameter_filter(name, parameter))
            if parameter_filter is not None
            else (
                leaf in {"weight", "values"}
                or leaf.endswith("_weight")
                or (include_bias and (leaf == "bias" or leaf.endswith("_bias")))
            )
        )
        if not selected or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.numel() == 0:
            continue
        if parameter.is_complex() or parameter.layout != torch.strided:
            raise ValueError(f"selected parameter {name!r} must be a dense real tensor")
        values = parameter.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError(f"selected parameter {name!r} contains non-finite values")
        absolute = values.abs()
        parameter_scale = absolute.max().item()
        new_scale = max(scale, parameter_scale)
        if new_scale > 0.0:
            scaled_sum = scaled_sum * (scale / new_scale)
            scaled_sum += (absolute / new_scale).sum().item()
        scale = new_scale
        count += parameter.numel()
    if count == 0:
        return None
    # The normalized mean is in [0, 1]; clamp harmless summation roundoff.
    return scale * min(1.0, scaled_sum / count)


def _threshold(value: float, name: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        interval = "[0, 1]" if maximum == 1 else "[0, infinity)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return value


class ActivationAccumulator:
    """Accumulate exact activity counts and an uncentered activation spectrum.

    A streaming reduced QR factorization retains at most a ``D x D`` matrix
    for ``D`` features, rather than all ``N x D`` samples. The singular values
    of its R factor equal those of the full dataset matrix. Each update costs
    a QR on at most ``(D + batch_rows) x D`` values plus a batch SVD. Reporting
    full-dataset statistics additionally computes an SVD of R. Using QR
    avoids squaring the condition number, as forming a Gram matrix would.

    ``compute()`` reports statistics of the whole accumulated dataset.
    ``batch_mean()`` instead returns the sample-row-weighted mean of each
    batch's statistics; these differ for nonlinear rank and activity metrics.
    """

    def __init__(
        self,
        *,
        dormant_threshold: float = 0.01,
        activation_threshold: float = 0.0,
        activity_mode: str = "nonzero",
    ) -> None:
        self.dormant_threshold = _threshold(dormant_threshold, "dormant_threshold", 1.0)
        self.activation_threshold = _threshold(activation_threshold, "activation_threshold")
        if activity_mode not in {"nonzero", "positive"}:
            raise ValueError("activity_mode must be 'nonzero' or 'positive'")
        self.activity_mode = activity_mode
        self.reset()

    def reset(self) -> None:
        """Clear all samples while preserving the diagnostic configuration."""
        self.num_samples = 0
        self.num_batches = 0
        self.num_features: int | None = None
        self._nonzero_counts: torch.Tensor | None = None
        self._active_counts: torch.Tensor | None = None
        self._r: torch.Tensor | None = None
        self._scale = 0.0
        self._batch_only = False
        self._weighted_sums = {name: 0.0 for name in _ACTIVATION_METRICS}
        self._latest: dict[str, float] | None = None
        self._computed: dict[str, float] | None = None

    @property
    def latest(self) -> dict[str, float] | None:
        """A copy of the most recent batch's metrics, or None after reset."""
        return None if self._latest is None else dict(self._latest)

    def _activity_metrics(
        self, nonzero_counts: torch.Tensor, active_counts: torch.Tensor, rows: int
    ) -> dict[str, float]:
        frequencies = active_counts.to(dtype=torch.float64) / rows
        return {
            "dead_neuron_ratio": (nonzero_counts == 0).to(dtype=torch.float64).mean().item(),
            "dormant_neuron_ratio": (
                (frequencies < self.dormant_threshold).to(dtype=torch.float64).mean().item()
            ),
        }

    def update(self, matrix: torch.Tensor, *, batch_only: bool = False) -> None:
        """Add a batch represented as rows of samples and columns of neurons.

        ``batch_only=True`` skips building the QR state when only one training
        batch is needed. It requires an empty accumulator, and another update
        requires ``reset()`` first. This saves a factorization without changing
        the returned metrics or retaining the activation matrix.
        """
        if not isinstance(batch_only, bool):
            raise TypeError("batch_only must be a bool")
        if self._batch_only or (batch_only and self.num_batches):
            raise ValueError("batch_only accumulation requires reset before another update")
        matrix = _matrix(matrix)
        rows, features = matrix.shape
        if self.num_features is not None and features != self.num_features:
            raise ValueError(
                f"feature count changed from {self.num_features} to {features}; reset first"
            )
        nonzero_counts = (matrix != 0).sum(dim=0)
        active_values = matrix.abs() if self.activity_mode == "nonzero" else matrix
        active_counts = (active_values > self.activation_threshold).sum(dim=0)
        metrics = self._activity_metrics(nonzero_counts, active_counts, rows)
        metrics["stable_rank"], metrics["effective_rank"] = _rank_metrics(matrix)

        batch_scale = matrix.abs().max().item()
        scale = max(self._scale, batch_scale)
        r = self._r
        if not batch_only and batch_scale > 0.0:
            normalized = matrix / scale
            if r is not None:
                normalized = torch.cat((r * (self._scale / scale), normalized), dim=0)
            # A common scaling factor retains relative magnitude across batches.
            # R has min(total_nonzero_batch_rows, features) rows and is bounded
            # in storage independently of the number of accumulated samples.
            r = torch.linalg.qr(normalized, mode="r").R

        # Commit only after numerical operations succeed, preserving state if
        # invalid input or a factorization failure raises above.
        self.num_features = features
        self.num_samples += rows
        self.num_batches += 1
        self._nonzero_counts = (
            nonzero_counts
            if self._nonzero_counts is None
            else self._nonzero_counts + nonzero_counts
        )
        self._active_counts = (
            active_counts if self._active_counts is None else self._active_counts + active_counts
        )
        self._r = r
        self._scale = scale
        self._batch_only = batch_only
        self._latest = metrics
        self._computed = None
        for name, value in metrics.items():
            self._weighted_sums[name] += value * rows

    def _require_samples(self) -> None:
        if self.num_samples == 0:
            raise ValueError("no activation samples have been recorded")

    def compute(self) -> dict[str, float]:
        """Compute four activation metrics, caching until the next update."""
        self._require_samples()
        if self.num_batches == 1:
            # One batch already is the entire dataset; reuse its SVD result.
            return dict(self._latest)
        if self._computed is not None:
            return dict(self._computed)
        metrics = self._activity_metrics(
            self._nonzero_counts, self._active_counts, self.num_samples
        )
        metrics["stable_rank"], metrics["effective_rank"] = (
            (0.0, 0.0) if self._r is None else _rank_metrics(self._r)
        )
        self._computed = dict(metrics)
        return metrics

    def batch_mean(self) -> dict[str, float]:
        """Return batch metrics weighted by their number of sample rows."""
        self._require_samples()
        return {name: total / self.num_samples for name, total in self._weighted_sums.items()}

"""Opt-in plasticity probes for any eager PyTorch model."""

import copy
import weakref
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from .metrics import ActivationAccumulator, activation_matrix, mean_absolute_weight


class PlasticityMonitor:
    """Observe a chosen representation without changing outputs or gradients.

    Attach with ``probe = PlasticityMonitor(model, layer="net.3")``, or insert
    ``hidden = probe(hidden)`` immediately after a layer in your forward method.
    The monitor follows ``model.training``: each training call replaces the
    previous batch; evaluation calls accumulate until ``reset()`` or a mode
    change. Call ``reset()`` before each new evaluation dataset.

    ``compute()`` measures the combined representation. ``compute("batch_mean")``
    averages per-call metrics weighted by representation rows. These operations
    give different answers for nonlinear statistics such as rank and dormancy.
    Statistics are detached, computed on CPU in float64, and kept outside model
    parameters, buffers, and state_dict. Exact rank measurement can be expensive;
    set ``enabled = False`` when measurement is not needed.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: str | nn.Module | None = None,
        *,
        feature_dim: int = -1,
        dormant_threshold: float = 0.01,
        activation_threshold: float = 0.0,
        activity_mode: Literal["nonzero", "positive"] = "nonzero",
        include_bias: bool = False,
        parameter_filter: Callable[[str, nn.Parameter], bool] | None = None,
        output_selector: Callable[[object], torch.Tensor] | None = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer")
        if not isinstance(include_bias, bool):
            raise TypeError("include_bias must be a bool")
        if parameter_filter is not None and not callable(parameter_filter):
            raise TypeError("parameter_filter must be callable")
        if output_selector is not None and not callable(output_selector):
            raise TypeError("output_selector must be callable")
        self._model = weakref.ref(model)
        self.feature_dim = feature_dim
        self.include_bias = include_bias
        self.parameter_filter = parameter_filter
        self.output_selector = output_selector
        self.enabled = True
        self._settings = {
            "dormant_threshold": dormant_threshold,
            "activation_threshold": activation_threshold,
            "activity_mode": activity_mode,
        }
        self._accumulator = ActivationAccumulator(**self._settings)
        self._hook = None
        self.layer_name = "manual"
        self.reset()
        if layer is not None:
            if isinstance(layer, str):
                selected = model.get_submodule(layer)
                self.layer_name = layer or "<model output>"
            elif isinstance(layer, nn.Module):
                names = [name for name, module in model.named_modules() if module is layer]
                if not names:
                    raise ValueError("layer must belong to model")
                selected = layer
                self.layer_name = names[0] or "<model output>"
            else:
                raise TypeError("layer must be a module path, nn.Module, or None")
            self._hook = selected.register_forward_hook(self._observe_output)

    def _observe_output(self, _module, _inputs, output) -> None:
        if self.enabled:
            selected = self.output_selector(output) if self.output_selector else output
            self(selected)
        # A hook returning None preserves the original output (including tuples).

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        """Record activations and return the exact input tensor, with its graph."""
        if not self.enabled:
            return activations
        model = self._model()
        if model is None:
            raise RuntimeError("The monitored model no longer exists")
        matrix = activation_matrix(activations, feature_dim=self.feature_dim)
        weight_mean = mean_absolute_weight(
            model, include_bias=self.include_bias, parameter_filter=self.parameter_filter
        )
        new_window = model.training or self._training != model.training
        accumulator = ActivationAccumulator(**self._settings) if new_window else self._accumulator
        accumulator.update(matrix, batch_only=model.training)
        # Commit only after validation and factorization succeed. Training
        # needs one batch SVD, with neither QR nor retained dataset spectrum.
        self._accumulator = accumulator
        if new_window:
            self._weight_rows = 0
            self._weight_mean = 0.0
        self._training = model.training
        self._last_weight_mean = weight_mean
        if weight_mean is not None:
            # Online weighted mean avoids overflow from multiplying by row count.
            previous_rows = self._weight_rows
            self._weight_rows += matrix.shape[0]
            if previous_rows == 0:
                self._weight_mean = weight_mean
            else:
                fraction = matrix.shape[0] / self._weight_rows
                self._weight_mean = self._weight_mean * (1.0 - fraction) + weight_mean * fraction
        return activations

    def reset(self) -> None:
        """Start a fresh measurement window without removing the layer hook."""
        self._accumulator.reset()
        self._training: bool | None = None
        self._last_weight_mean: float | None = None
        self._weight_mean = 0.0
        self._weight_rows = 0

    def compute(
        self, aggregation: Literal["dataset", "batch_mean"] = "dataset"
    ) -> dict[str, float | None]:
        """Return five metrics; no matching model weights gives a None weight mean.

        Dataset weight magnitude is the snapshot at the last observed forward.
        Batch mean uses the row-weighted mean of forward-time snapshots. Keep
        model weights fixed while measuring an evaluation dataset.
        """
        if aggregation == "dataset":
            result = self._accumulator.compute()
            weight_mean = self._last_weight_mean
        elif aggregation == "batch_mean":
            result = self._accumulator.batch_mean()
            weight_mean = self._weight_mean if self._weight_rows else None
        else:
            raise ValueError("aggregation must be 'dataset' or 'batch_mean'")
        return {**result, "mean_absolute_weight": weight_mean}

    @property
    def latest(self) -> dict[str, float | None] | None:
        """Most recent observed batch, including when evaluating a dataset."""
        result = self._accumulator.latest
        if result is None:
            return None
        return {**result, "mean_absolute_weight": self._last_weight_mean}

    def summary(self) -> dict:
        """Return JSON-ready metrics plus representation/call counts."""
        return {
            "mode": "training" if self._training else "evaluation",
            "layer": self.layer_name,
            "num_samples": self._accumulator.num_samples,
            "num_batches": self._accumulator.num_batches,
            "num_features": self._accumulator.num_features,
            "dataset": self.compute(),
            "batch_mean": self.compute("batch_mean"),
        }

    def close(self) -> None:
        """Remove the hook; existing results remain available. Safe to repeat."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __deepcopy__(self, memo):
        """Bind a probe copied as part of a model to that copied model."""
        model = self._model()
        if model is None or id(model) not in memo:
            raise TypeError("Copy the model containing the probe, or construct a new probe")
        duplicate = type(self).__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__.update(copy.deepcopy(self.__dict__, memo))
        duplicate._model = weakref.ref(memo[id(model)])
        return duplicate

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

"""Build policies/value models with an exact, active parameter budget."""

import copy
import inspect
import math

import torch
from torch import nn

from flow_rl.factory import instantiate, resolve
from flow_rl.models.budgeted_linear import BudgetedLinear


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _width_configuration(target: str, kwargs: dict):
    parameters = inspect.signature(resolve(target)).parameters
    if "hidden_dims" in parameters:
        hidden_dims = kwargs.get("hidden_dims", parameters["hidden_dims"].default)
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError("Automatic parameter budgets need at least one hidden layer")
        depth = len(hidden_dims)
        return lambda width: {**kwargs, "hidden_dims": [width] * depth}, 1
    if "hidden_dim" in parameters:
        heads = kwargs.get("num_heads", parameters.get("num_heads"))
        heads = heads.default if isinstance(heads, inspect.Parameter) else heads
        heads = 1 if heads is None else heads
        if not isinstance(heads, int) or isinstance(heads, bool) or heads < 1:
            raise ValueError("num_heads must be a positive integer")
        return lambda width: {**kwargs, "hidden_dim": width}, math.lcm(heads, 2)
    raise ValueError(
        f"{target} does not expose hidden_dims or hidden_dim for automatic parameter budgeting"
    )


def _linear_candidates(model: nn.Module):
    # MultiheadAttention reads out_proj.weight directly instead of calling its
    # module, so leave those projections dense. Ordinary feedforward modules
    # call Linear.forward and can safely use BudgetedLinear.
    protected = [
        name for name, layer in model.named_modules() if isinstance(layer, nn.MultiheadAttention)
    ]
    designated = getattr(model, "budget_layer_names", None)
    allowed = set(designated()) if callable(designated) else None
    candidates = []
    for name, layer in model.named_modules():
        if not name or type(layer) is not nn.Linear:
            continue
        if allowed is not None and name not in allowed:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in protected):
            continue
        capacity = layer.weight.numel() - max(layer.in_features, layer.out_features)
        if capacity > 0:
            candidates.append((name, layer, capacity))
    return sorted(candidates, key=lambda item: (-item[2], item[0]))


def _replace_linear(model: nn.Module, name: str, weight_count: int) -> None:
    parent_path, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    layer = getattr(parent, leaf)
    if type(layer) is not nn.Linear:
        raise ValueError(f"Budget layer {name!r} is not a plain Linear")
    setattr(parent, leaf, BudgetedLinear(layer, weight_count))


def _restore_spec(target: str, runtime: dict, spec: dict, expected_budget: int | None):
    if spec.get("version") != 1 or spec.get("target") != target:
        raise ValueError("Checkpoint model specification does not match the target")
    if spec.get("parameter_budget") != expected_budget:
        raise ValueError("Checkpoint model specification does not match the parameter budget")
    model = instantiate(target, spec["resolved_kwargs"], **runtime)
    for entry in spec["sparse_layers"]:
        _replace_linear(model, entry["name"], entry["weight_count"])
    counts = parameter_counts(model)
    if counts["total"] != spec["total_parameters"]:
        raise ValueError("Checkpoint model specification has an inconsistent parameter count")
    if expected_budget is not None and counts["total"] != expected_budget:
        raise ValueError("Checkpoint model specification does not meet its parameter budget")
    model.build_spec = copy.deepcopy(spec)
    return model


def build_model(
    target: str,
    kwargs: dict,
    *,
    parameter_budget: int | None = None,
    build_spec: dict | None = None,
    **runtime,
):
    """Preserve native shapes by default; enforce exact parameter counts on opt-in.

    With a budget, retain depth/heads and choose a uniform width. Resolve the
    smallest dense width reaching the budget, then remove only the excess
    trainable affine connections while maintaining every input/output path.
    Invalid or structurally unattainable budgets fail rather than approximate.
    Resolution uses meta tensors and does not consume the experiment RNG.
    """
    if parameter_budget is not None and (
        isinstance(parameter_budget, bool)
        or not isinstance(parameter_budget, int)
        or parameter_budget <= 0
    ):
        raise ValueError("parameter_budget must be a positive integer or null")
    if build_spec is not None:
        return _restore_spec(target, runtime, build_spec, parameter_budget)
    resolved_kwargs = copy.deepcopy(kwargs)
    sparse_layers = []
    if parameter_budget is not None:
        configure, width_step = _width_configuration(target, kwargs)

        def count_at(units):
            # Also isolate user-defined constructors that explicitly allocate on CPU.
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                candidate = instantiate(target, configure(units * width_step), **runtime)
                counts = parameter_counts(candidate)
            if counts["trainable"] != counts["total"]:
                raise ValueError("Automatic budgeting requires fully trainable models")
            return counts["total"]

        minimum = count_at(1)
        if parameter_budget < minimum:
            raise ValueError(
                f"parameter_budget={parameter_budget} is too small; minimum dense model has {minimum} parameters"
            )
        lower, upper = 1, 1
        while count_at(upper) < parameter_budget:
            upper *= 2
        while lower < upper:
            middle = (lower + upper) // 2
            if count_at(middle) < parameter_budget:
                lower = middle + 1
            else:
                upper = middle
        resolved_kwargs = configure(lower * width_step)
    model = instantiate(target, resolved_kwargs, **runtime)
    dense_parameters = parameter_counts(model)["total"]
    if parameter_budget is not None:
        excess = dense_parameters - parameter_budget
        candidates = _linear_candidates(model)
        if excess > sum(capacity for _, _, capacity in candidates):
            raise ValueError(
                f"Cannot meet exact parameter_budget={parameter_budget} while preserving layer connectivity; use a larger budget"
            )
        capacity_total = sum(capacity for _, _, capacity in candidates)
        # Spread removed connections across eligible layers, avoiding a single
        # severely thinned feedforward layer in a multi-block action expert.
        removals = (
            [excess * capacity // capacity_total for _, _, capacity in candidates]
            if excess
            else [0] * len(candidates)
        )
        remainder = excess - sum(removals)
        order = (
            sorted(
                range(len(candidates)),
                key=lambda index: (-(excess * candidates[index][2] % capacity_total), index),
            )
            if remainder
            else []
        )
        for index in order[:remainder]:
            removals[index] += 1
        for (name, layer, _), removed in zip(candidates, removals):
            if not removed:
                continue
            weight_count = layer.weight.numel() - removed
            _replace_linear(model, name, weight_count)
            sparse_layers.append(
                {
                    "name": name,
                    "weight_count": weight_count,
                    "dense_weight_count": layer.weight.numel(),
                }
            )
    actual = parameter_counts(model)["total"]
    if parameter_budget is not None and actual != parameter_budget:
        raise RuntimeError(
            f"Parameter budget mismatch: expected {parameter_budget}, built {actual}"
        )
    model.build_spec = {
        "version": 1,
        "target": target,
        "parameter_budget": parameter_budget,
        "resolved_kwargs": copy.deepcopy(resolved_kwargs),
        "dense_parameters": dense_parameters,
        "total_parameters": actual,
        "sparse_layers": sparse_layers,
    }
    return model


def algorithm_parameter_report(algorithm) -> dict:
    actor = parameter_counts(algorithm.actor)
    critic = (
        parameter_counts(algorithm.critic)
        if hasattr(algorithm, "critic")
        else {"total": 0, "trainable": 0}
    )
    target = (
        parameter_counts(algorithm.target_critic)
        if hasattr(algorithm, "target_critic")
        else {"total": 0, "trainable": 0}
    )
    return {
        "actor_parameters": actor["total"],
        "actor_trainable_parameters": actor["trainable"],
        "critic_parameters": critic["total"],
        "critic_trainable_parameters": critic["trainable"],
        "target_critic_parameters": target["total"],
        "total_trainable_parameters": actor["trainable"] + critic["trainable"],
        "total_parameters_with_targets": actor["total"] + critic["total"] + target["total"],
        "actor_build": getattr(algorithm.actor, "build_spec", None),
        "critic_build": getattr(getattr(algorithm, "critic", None), "build_spec", None),
        "actor_objective": getattr(
            algorithm.actor, "training_objective", "negative_log_likelihood"
        ),
    }

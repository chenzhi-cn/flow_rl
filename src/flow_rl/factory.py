"""Resolve extension factories without modifying central registries."""

from importlib import import_module
from typing import Any


def resolve(target: str) -> Any:
    module, separator, name = target.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected 'module:attribute', got {target!r}")
    return getattr(import_module(module), name)


def instantiate(target: str, kwargs: dict | None = None, **runtime: Any) -> Any:
    values = dict(kwargs or {})
    duplicate = values.keys() & runtime.keys()
    if duplicate:
        raise ValueError(f"Runtime arguments cannot be overridden: {sorted(duplicate)}")
    return resolve(target)(**values, **runtime)

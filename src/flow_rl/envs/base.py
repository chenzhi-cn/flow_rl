"""The common contract for simulator and robot adapters."""

from typing import Any, Protocol, runtime_checkable

import gymnasium as gym

Observation = Any  # Vector, image, or a dictionary of sensor values.
Info = dict[str, Any]
StepResult = tuple[Observation, float, bool, bool, Info]


@runtime_checkable
class Environment(Protocol):
    """Gymnasium semantics: termination and time-limit truncation stay distinct."""

    observation_space: gym.Space
    action_space: gym.Space

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, Info]: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...

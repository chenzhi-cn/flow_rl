"""SDK-independent robot connection point.

Implement RobotTransport using the vendor SDK, and supply a task evaluator for
reward/success. No connection to real hardware is made by importing this module.
The driver is responsible for command deadlines, hardware limits and emergency
stop; this wrapper checks the declared action range before sending a command.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from flow_rl.envs.base import Info, Observation, StepResult


class RobotTransport(Protocol):
    observation_space: gym.Space
    action_space: gym.spaces.Box

    def reset(self, *, options: dict | None = None) -> tuple[Observation, Info]: ...

    def send_action(self, action: np.ndarray) -> None: ...

    def read_observation(self) -> tuple[Observation, Info]: ...

    def close(self) -> None: ...


TaskEvaluator = Callable[[Observation, np.ndarray, Observation], tuple[float, bool, Info]]


class RobotEnv:
    """Adapt a Franka, Ark or other vendor driver to the common environment API.

    Vendor implementations are supplied by the application. A task evaluator
    defines reward and termination; max_episode_steps defines truncation.
    """

    def __init__(
        self,
        transport: RobotTransport,
        evaluate: TaskEvaluator,
        max_episode_steps: int | None = None,
    ):
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.transport = transport
        self.evaluate = evaluate
        self.max_episode_steps = max_episode_steps
        self.observation_space = transport.observation_space
        self.action_space = transport.action_space
        self._observation = None
        self._steps = 0
        self._needs_reset = True

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, Info]:
        if seed is not None:
            self.action_space.seed(seed)
        observation, info = self.transport.reset(options=options)
        self._observation = deepcopy(observation)
        self._steps = 0
        self._needs_reset = False
        return observation, info

    def step(self, action: Any) -> StepResult:
        if self._needs_reset:
            raise RuntimeError("Call reset before stepping the robot")
        action = np.asarray(action, dtype=self.action_space.dtype)
        if not np.all(np.isfinite(action)) or not self.action_space.contains(action):
            raise ValueError("Robot action must be finite and within the declared action space")
        self.transport.send_action(action)
        observation, info = self.transport.read_observation()
        reward, terminated, task_info = self.evaluate(self._observation, action, observation)
        self._observation = deepcopy(observation)
        self._steps += 1
        truncated = self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        self._needs_reset = bool(terminated or truncated)
        return observation, float(reward), bool(terminated), bool(truncated), {**info, **task_info}

    def close(self) -> None:
        self._needs_reset = True
        self.transport.close()

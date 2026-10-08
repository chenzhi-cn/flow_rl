"""Environment adapters and a small registry for application extensions."""

from collections.abc import Callable

from flow_rl.envs.base import Environment
from flow_rl.envs.gym import make_gym_env
from flow_rl.envs.ogbench import DEFAULT_DATASET, make_ogbench_env
from flow_rl.envs.robot import RobotEnv, RobotTransport

_FACTORIES: dict[str, Callable[..., Environment]] = {
    "gym": make_gym_env,
    "ogbench": make_ogbench_env,
    "robot": RobotEnv,
}


def register_env(name: str, factory: Callable[..., Environment]) -> None:
    if name in _FACTORIES:
        raise ValueError(f"Environment backend already registered: {name}")
    _FACTORIES[name] = factory


def make_env(backend: str, **kwargs) -> Environment:
    try:
        factory = _FACTORIES[backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown environment backend {backend!r}; choose {sorted(_FACTORIES)}"
        ) from exc
    return factory(**kwargs)


__all__ = [
    "DEFAULT_DATASET",
    "Environment",
    "RobotEnv",
    "RobotTransport",
    "make_env",
    "make_gym_env",
    "make_ogbench_env",
    "register_env",
]

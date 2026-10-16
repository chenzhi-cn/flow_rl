"""Strict YAML configuration, inheritance and dotted command-line overrides."""

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 1
    output_dir: str = "runs/cube_awac"


@dataclass
class EnvironmentConfig:
    target: str = "flow_rl.envs:make_ogbench_env"
    kwargs: dict = field(
        default_factory=lambda: {"dataset_name": "cube-single-play-singletask-task2-v0"}
    )


@dataclass
class DataConfig:
    source: str = "ogbench"
    dataset_dir: str = "data/ogbench"
    path: str | None = None
    random_steps: int = 400
    max_transitions: int | None = None
    normalize_observations: bool = True


@dataclass
class ModelConfig:
    target: str = "flow_rl.models:GaussianChunkPolicy"
    chunk_size: int = 1
    parameter_budget: int | None = None
    kwargs: dict = field(
        default_factory=lambda: {"hidden_dims": [256, 256], "state_dependent_std": True}
    )


@dataclass
class ValueConfig:
    target: str = "flow_rl.models:EnsembleQ"
    parameter_budget: int | None = None
    kwargs: dict = field(
        default_factory=lambda: {"hidden_dims": [256, 256], "num_qs": 2, "layer_norm": False}
    )


@dataclass
class AlgorithmConfig:
    target: str = "flow_rl.algorithms:AWAC"
    kwargs: dict = field(default_factory=lambda: {"temperature": 1.0})


@dataclass
class TrainingConfig:
    mode: str = "offline_rl"
    steps: int = 10000
    batch_size: int = 256
    gamma: float = 0.99
    log_interval: int = 100
    checkpoint_interval: int = 5000
    eval_interval: int = 0
    warmup_steps: int = 1000
    replay_capacity: int = 100000
    updates_per_step: int = 1
    offline_ratio: float = 0.5


@dataclass
class InferenceConfig:
    asynchronous: bool = False
    realtime: bool = False
    control_dt: float = 0.05
    refresh_steps: int = 1
    stitching: str = "replace"
    smoothing: str = "none"
    fallback: str = "hold"
    ema_alpha: float = 0.5
    ensemble_decay: float = 0.5


@dataclass
class EvaluationConfig:
    episodes: int = 5
    max_steps: int = 200


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for section in ("model", "value"):
            budget = getattr(self, section).parameter_budget
            if budget is not None and (
                isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
            ):
                raise ValueError(f"{section}.parameter_budget must be a positive integer or null")
        if self.training.mode not in {"offline_rl", "online_rl", "supervised"}:
            raise ValueError("training.mode must be offline_rl, online_rl or supervised")
        if self.data.source not in {"ogbench", "npz", "random", "none"}:
            raise ValueError("data.source must be ogbench, npz, random or none")
        if self.data.source == "none" and self.training.mode != "online_rl":
            raise ValueError("Only online_rl can train without an offline dataset")
        if self.data.source == "npz" and not self.data.path:
            raise ValueError("data.path is required for an npz dataset")
        for name, value in {
            "chunk_size": self.model.chunk_size,
            "steps": self.training.steps,
            "batch_size": self.training.batch_size,
            "log_interval": self.training.log_interval,
            "torch_threads": self.experiment.torch_threads,
            "control_dt": self.inference.control_dt,
            "refresh_steps": self.inference.refresh_steps,
            "evaluation.episodes": self.evaluation.episodes,
            "evaluation.max_steps": self.evaluation.max_steps,
            "updates_per_step": self.training.updates_per_step,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.inference.refresh_steps > self.model.chunk_size:
            raise ValueError("refresh_steps must not exceed the action chunk size")
        if self.inference.realtime and self.model.chunk_size == 1:
            raise ValueError(
                "realtime latency compensation needs chunk_size >= 2; use the chunk configuration"
            )
        if not 0 < self.training.gamma <= 1 or not 0 <= self.training.offline_ratio <= 1:
            raise ValueError("gamma must be in (0,1]; offline_ratio must be in [0,1]")
        if self.training.replay_capacity < self.model.chunk_size:
            raise ValueError("replay_capacity must be at least chunk_size")
        for value in (
            self.training.checkpoint_interval,
            self.training.eval_interval,
            self.training.warmup_steps,
        ):
            if value < 0:
                raise ValueError("intervals and warmup_steps cannot be negative")
        if (
            self.data.max_transitions is not None
            and self.data.max_transitions < self.model.chunk_size
        ):
            raise ValueError("max_transitions must be at least chunk_size")


def merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    if "target" in update and update["target"] != base.get("target"):
        result["kwargs"] = {}
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in chain:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    with path.open() as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise TypeError("Configuration must be a YAML mapping")
    parent = values.pop("extends", None)
    if parent:
        values = merge(_read(path.parent / parent, (*chain, path)), values)
    return values


def from_dict(values: dict) -> Config:
    config = Config()
    unknown = set(values) - {f.name for f in fields(config)}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for section, incoming in values.items():
        old = getattr(config, section)
        if not isinstance(incoming, dict):
            raise TypeError(f"{section} must be a mapping")
        unknown = set(incoming) - {f.name for f in fields(old)}
        if unknown:
            raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
        # Component kwargs are a full parameter set when explicitly supplied.
        settings = {**asdict(old), **incoming}
        setattr(config, section, type(old)(**settings))
    config.validate()
    return config


def apply_overrides(values: dict, overrides: list[str] | None = None) -> dict:
    values = copy.deepcopy(values)
    for override in overrides or []:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"Expected dotted.key=value, got {override!r}")
        cursor = values
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"Unknown override path: {key}")
            cursor = cursor[part]
        if parts[-1] not in cursor and "kwargs" not in parts:
            raise ValueError(f"Unknown override option: {key}")
        value = yaml.safe_load(raw)
        if parts[-1] == "target" and cursor.get("target") != value:
            cursor["kwargs"] = {}
        cursor[parts[-1]] = value
    return values


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    values = merge(Config().to_dict(), _read(Path(path)))
    return from_dict(apply_overrides(values, overrides))


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

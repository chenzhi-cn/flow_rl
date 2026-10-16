"""Restore evaluation states before taking independent counterfactual actions.

The protocol is simulator-independent. The bundled adapter deliberately supports
OGBench's state-based single-cube environment only, whose saved qpos/qvel and
memoryless Cartesian controller are sufficient for a deterministic new rollout.
Historical contact forces and MuJoCo solver warm starts are absent from its NPZ
files, so reconstruction diagnostics are recorded instead of claiming exact
historical replay.
"""

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from flow_rl.data.ogbench import load_ogbench_dataset
from flow_rl.envs.ogbench import DEFAULT_DATASET, make_ogbench_env


@runtime_checkable
class RolloutAdapter(Protocol):
    """A source and simulator for independent one-step action interventions.

    Arrays in both dataset maps use noncompact transition indexing. At minimum
    they contain observations [N,O], actions [N,A], next_observations [N,O], and
    terminals [N]. Extra simulator state arrays remain aligned to those rows.
    Actions passed to ``rollout`` use the environment's action-space units.
    """

    action_low: np.ndarray
    action_high: np.ndarray
    metadata: dict[str, Any]

    def load_datasets(self) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]: ...

    def restore_observation(self, index: int) -> np.ndarray: ...

    def rollout(self, index: int, action: np.ndarray) -> np.ndarray: ...

    @property
    def restoration_statistics(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class OGBenchRolloutAdapter:
    """Official train/validation data and deterministic single-cube interventions.

    A reset followed by ``mj_resetData`` clears all solver, actuator, control,
    and contact history for *every* branch. qpos/qvel are then restored, and the
    fixed task goal is reinstated. This makes repeated branches identical even
    though original trajectory solver state is unavailable. The observation's
    contact channel (index 18) is recomputed, never copied into the simulator.
    """

    CONTACT_INDEX = 18

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        dataset_dir: str | Path = "data/ogbench",
        dataset_path: str | Path | None = None,
        seed: int = 0,
        env_kwargs: dict | None = None,
        restoration_tolerance: float = 0.05,
    ):
        if not dataset_name.startswith("cube-single-") or "singletask" not in dataset_name:
            raise ValueError("OGBenchRolloutAdapter supports cube-single singletask datasets only")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(restoration_tolerance) or restoration_tolerance <= 0:
            raise ValueError("restoration_tolerance must be finite and positive")
        env_kwargs = dict(env_kwargs or {})
        if "dataset_name" in env_kwargs:
            raise ValueError("Pass dataset_name directly, not through env_kwargs")
        if env_kwargs.get("ob_type", "states") != "states":
            raise ValueError("Manifold rollout requires state-vector observations")
        self._seed = int(seed)
        self._tolerance = float(restoration_tolerance)
        self._dataset_name = dataset_name
        self._dataset_dir = Path(dataset_dir).expanduser()
        self._dataset_path = None if dataset_path is None else Path(dataset_path).expanduser()
        self._datasets = None
        self._restore_errors: dict[int, np.ndarray] = {}
        self._restore_calls = 0
        self._closed = False
        self._env = make_ogbench_env(dataset_name, **env_kwargs)
        try:
            self._env.reset(seed=self._seed)
            self._base = self._env.unwrapped
            if self._base.model.na != 0:
                raise ValueError("Saved qpos/qvel cannot reconstruct actuator activation state")
            if self._base.compute_observation().shape != (28,):
                raise ValueError("Expected the standard 28-dimensional single-cube observation")
            self.action_low = np.asarray(self._env.action_space.low, dtype=np.float64).copy()
            self.action_high = np.asarray(self._env.action_space.high, dtype=np.float64).copy()
            self._goal_position = self._base.data.mocap_pos.copy()
            self._goal_quaternion = self._base.data.mocap_quat.copy()
            self.metadata = {
                "adapter": "flow_rl.manifold.rollout:OGBenchRolloutAdapter",
                "environment": {
                    "target": "flow_rl.envs:make_ogbench_env",
                    "kwargs": {"dataset_name": dataset_name, **env_kwargs},
                },
                "dataset_dir": str(self._dataset_dir.resolve()),
                "dataset_path": (
                    None if self._dataset_path is None else str(self._dataset_path.resolve())
                ),
                "reference_split": "train",
                "evaluation_split": "validation",
                "indexing": "ogbench_noncompact_transitions",
                "action_units": "environment_action_space",
                "control_timestep": float(self._base.control_timestep()),
                "packages": {name: version(name) for name in ("ogbench", "mujoco")},
                "restoration": {
                    "method": "fresh_mujoco_data_then_saved_qpos_qvel",
                    "seed": self._seed,
                    "tolerance_excluding_contact": self._tolerance,
                    "contact_observation_index": self.CONTACT_INDEX,
                    "historical_replay_exact": False,
                    "missing_historical_state": [
                        "contact_forces",
                        "solver_warm_start",
                        "cached_effector_kinematics_before_last_physics_integration",
                    ],
                },
            }
        except Exception:
            self.close()
            raise

    def load_datasets(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load once, preserving official terminal filtering for every state array."""
        self._check_open()
        if self._datasets is None:
            kwargs = {"cur_env": self._env, "add_info": True}
            if self._dataset_path is not None:
                kwargs["dataset_path"] = str(self._dataset_path)
            reference, evaluation = load_ogbench_dataset(
                self._dataset_name, self._dataset_dir, **kwargs
            )
            for name, dataset in (("reference", reference), ("evaluation", evaluation)):
                for key in ("observations", "actions", "next_observations", "terminals"):
                    if key not in dataset:
                        raise ValueError(f"{name} dataset is missing {key!r}")
                size = len(dataset["observations"])
                if size == 0:
                    raise ValueError(f"{name} dataset is empty")
                expected = {
                    "observations": (size, 28),
                    "actions": (size, len(self.action_low)),
                    "next_observations": (size, 28),
                    "terminals": (size,),
                    "qpos": (size, self._base.model.nq),
                    "qvel": (size, self._base.model.nv),
                }
                for key, shape in expected.items():
                    if key not in dataset or np.shape(dataset[key]) != shape:
                        raise ValueError(f"{name} {key} must have shape {shape}")
                    if not np.all(np.isfinite(dataset[key])):
                        raise ValueError(f"{name} {key} contains nonfinite values")
            self._datasets = reference, evaluation
        return self._datasets

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Rollout adapter is closed")

    def restore_observation(self, index: int) -> np.ndarray:
        """Return the reconstructed observation after restoring an evaluation row."""
        import mujoco

        _, evaluation = self.load_datasets()
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("Evaluation index must be an integer")
        if index < 0 or index >= len(evaluation["observations"]):
            raise IndexError("Evaluation index is outside the dataset")
        # Gym wrappers (including TimeLimit) must also begin a new branch. OGBench
        # reset takes random stabilization actions; mj_resetData below clears all
        # effects of those actions before the saved state is applied.
        self._env.reset(seed=self._seed)
        base = self._base
        mujoco.mj_resetData(base.model, base.data)
        base.data.mocap_pos[:] = self._goal_position
        base.data.mocap_quat[:] = self._goal_quaternion
        base._reset_next_step = False
        base.set_state(evaluation["qpos"][index], evaluation["qvel"][index])
        base.data.qacc_warmstart[:] = 0
        mujoco.mj_rnePostConstraint(base.model, base.data)
        base.pre_step()
        base.post_step()
        observation = np.asarray(base.compute_observation(), dtype=np.float32).copy()
        if observation.shape != (28,) or not np.all(np.isfinite(observation)):
            raise ValueError("Restored simulator observation is invalid")
        error = observation.astype(np.float64) - evaluation["observations"][index]
        noncontact_error = np.delete(error, self.CONTACT_INDEX)
        if np.max(np.abs(noncontact_error)) > self._tolerance:
            raise ValueError(
                "Saved simulator state does not reconstruct the dataset observation: "
                f"index={index}, max error excluding contact={np.max(np.abs(noncontact_error)):.6g}, "
                f"tolerance={self._tolerance:.6g}"
            )
        self._restore_errors[int(index)] = error
        self._restore_calls += 1
        return observation

    def rollout(self, index: int, action: np.ndarray) -> np.ndarray:
        """Restore afresh, apply one action, and copy the resulting observation."""
        self._check_open()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_low.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite vector matching the action space")
        if np.any(action < self.action_low) or np.any(action > self.action_high):
            raise ValueError("Action is outside environment bounds; clip explicitly before rollout")
        self.restore_observation(index)
        result = self._env.step(action)
        if len(result) != 5:
            raise RuntimeError("Simulator reset instead of executing the requested action")
        observation = np.asarray(result[0], dtype=np.float32).copy()
        if observation.shape != (28,) or not np.all(np.isfinite(observation)):
            raise ValueError("Simulator step produced an invalid observation")
        return observation

    @property
    def restoration_statistics(self) -> dict[str, Any]:
        """Errors measured on unique restored evaluation rows, in observation units."""
        result = {"restore_calls": self._restore_calls, "unique_states": len(self._restore_errors)}
        if self._restore_errors:
            error = np.stack(list(self._restore_errors.values()))
            result.update(
                observation_rmse=float(np.sqrt(np.mean(error**2))),
                observation_max_abs_error=float(np.max(np.abs(error))),
                observation_max_abs_error_excluding_contact=float(
                    np.max(np.abs(np.delete(error, self.CONTACT_INDEX, axis=1)))
                ),
                contact_max_abs_error=float(np.max(np.abs(error[:, self.CONTACT_INDEX]))),
            )
        return result

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

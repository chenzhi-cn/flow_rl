"""Official OGBench data loading and single-task reward relabeling."""

from pathlib import Path

from flow_rl.envs.ogbench import DEFAULT_DATASET, make_ogbench_env


def load_ogbench_dataset(
    dataset_name: str = DEFAULT_DATASET,
    dataset_dir: str | Path = "data/ogbench",
    **kwargs,
):
    """Return train/validation maps in OGBench's noncompact format.

    Missing files are downloaded by OGBench. Single-task rewards and bootstrap
    masks are relabeled by its official implementation. ``cur_env`` may supply
    an existing environment, otherwise a temporary one is created and closed.
    Additional kwargs (e.g. dataset_path) are forwarded to the official loader.
    """
    import ogbench

    forbidden = {"env_only", "dataset_only", "compact_dataset"} & kwargs.keys()
    if forbidden:
        raise ValueError(f"Dataset format/mode is controlled by this adapter: {sorted(forbidden)}")
    if "singletask" not in dataset_name.split("-"):
        raise ValueError("Reward-based training requires an OGBench singletask dataset name")
    env = kwargs.pop("cur_env", None)
    owns_env = env is None
    if owns_env:
        env_kwargs = {
            key: value for key, value in kwargs.items() if key not in {"dataset_path", "add_info"}
        }
        env = make_ogbench_env(dataset_name, **env_kwargs)
    try:
        return ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=str(Path(dataset_dir).expanduser()),
            compact_dataset=False,
            dataset_only=True,
            cur_env=env,
            **kwargs,
        )
    finally:
        if owns_env:
            env.close()

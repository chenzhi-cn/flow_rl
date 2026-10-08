"""OGBench environment creation without triggering a dataset download."""

DEFAULT_DATASET = "cube-single-play-singletask-task2-v0"


def make_ogbench_env(dataset_name: str = DEFAULT_DATASET, **kwargs):
    import ogbench

    if any(key in kwargs for key in ("env_only", "dataset_only", "cur_env")):
        raise ValueError("Environment mode is controlled by make_ogbench_env")
    return ogbench.make_env_and_datasets(dataset_name, env_only=True, **kwargs)

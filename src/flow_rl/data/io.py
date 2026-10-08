"""Portable transition files, distinct from OGBench's raw trajectory NPZs."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from flow_rl.data.dataset import REQUIRED_KEYS


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise ValueError(
                f"Missing transition fields: {sorted(missing)}. "
                "Load original OGBench archives with load_ogbench_dataset to construct transitions and rewards."
            )
        return {key: archive[key] for key in REQUIRED_KEYS}


def save_npz(path: str | Path, data: Mapping[str, np.ndarray]) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(REQUIRED_KEYS) - data.keys()
    if missing:
        raise ValueError(f"Missing transition fields: {sorted(missing)}")
    # Opening the file avoids numpy silently adding a second .npz suffix.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: data[key] for key in REQUIRED_KEYS})
    return path

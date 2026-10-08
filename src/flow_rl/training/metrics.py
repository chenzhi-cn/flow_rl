"""Append-only machine-readable experiment metrics."""

import json
from pathlib import Path


def write_metrics(path: Path, values: dict) -> None:
    line = json.dumps(values, allow_nan=False)
    with path.open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)

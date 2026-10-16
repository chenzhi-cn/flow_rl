"""Immutable, checksummed NPZ artifacts; loading never requires a simulator."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_arrays(arrays, metadata=None):
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(canonical_json(metadata).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"Object arrays are not supported: {name}")
        digest.update(canonical_json([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SampleBundle:
    arrays: dict[str, np.ndarray]
    metadata: dict

    @property
    def sample_id(self):
        return self.metadata["content_sha256"]


def save_bundle(path, arrays, metadata):
    """Publish atomically and exclusively: an existing sample set is never replaced."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Sample artifact already exists: {path}")
    if "metadata" in arrays:
        raise ValueError("metadata is a reserved array key")
    metadata = dict(metadata, format_version=FORMAT_VERSION)
    metadata.pop("content_sha256", None)
    metadata["content_sha256"] = hash_arrays(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays, metadata=np.array(canonical_json(metadata)))
            stream.flush()
            os.fsync(stream.fileno())
        # link() is atomic and refuses replacement even if another writer won the race.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return SampleBundle(arrays, metadata)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Missing sample metadata")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported manifold sample format")
    expected = metadata.get("content_sha256")
    unsigned = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if expected != hash_arrays(arrays, unsigned):
        raise ValueError("Sample artifact checksum mismatch")
    return SampleBundle(arrays, metadata)

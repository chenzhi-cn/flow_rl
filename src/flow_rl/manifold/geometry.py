"""Critical-state selection and local linear action-subspace measurements.

The action subspace is a linear span through the origin, as in Section G.2
of the reference paper. Actions are deliberately not mean-centered for SVD.
"""

from numbers import Real

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, name, ndim):
    array = np.asarray(value)
    if array.ndim not in ((ndim,) if isinstance(ndim, int) else ndim):
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be nonempty")
    return array.astype(np.float64, copy=False)


def _episode_bounds(terminals):
    terminals = np.asarray(terminals)
    if terminals.ndim != 1 or not terminals.size:
        raise ValueError("terminals must be a nonempty one-dimensional array")
    if terminals.dtype.kind not in "biuf" or not np.isin(terminals, [0, 1]).all():
        raise ValueError("terminals must contain only 0 and 1")
    ends = np.flatnonzero(terminals) + 1
    if not ends.size or ends[-1] != terminals.size:
        ends = np.append(ends, terminals.size)
    starts = np.concatenate(([0], ends[:-1]))
    lengths = ends - starts
    return starts, ends, np.repeat(starts, lengths), np.repeat(ends, lengths)


def select_critical_states(actions, terminals, n, window_before, window_after):
    """Choose the largest action-variance windows without crossing episodes.

    A state's window includes itself and up to ``window_before`` previous and
    ``window_after`` following actions. Its score is the sum of population
    variances across action coordinates, i.e. the trace of the covariance.
    Ties are broken by ascending transition index. Returned ``window_end`` is
    exclusive, so every window is ``actions[window_start[i]:window_end[i]]``.
    A final partial trajectory is treated as ending at the last array entry.
    """
    actions = _array(actions, "actions", 2)
    n = _integer(n, "n")
    before = _integer(window_before, "window_before", 0)
    after = _integer(window_after, "window_after", 0)
    starts, ends, episode_start, episode_end = _episode_bounds(terminals)
    size = len(actions)
    if len(episode_start) != size:
        raise ValueError("actions and terminals must have the same length")
    if n > size:
        raise ValueError("n cannot exceed the number of states")
    before, after = min(before, size), min(after, size)
    indices = np.arange(size, dtype=np.int64)
    window_start = np.maximum(indices - before, episode_start)
    window_end = np.minimum(indices + after + 1, episode_end)
    scores = np.empty(size, dtype=np.float64)
    # Prefix sums make selection linear in dataset size. Center each episode
    # first to reduce cancellation; this changes no action variance.
    for start, end in zip(starts, ends):
        episode = actions[start:end]
        centered = episode - episode.mean(axis=0)
        sums = np.vstack((np.zeros((1, actions.shape[1])), centered.cumsum(axis=0)))
        squares = np.concatenate(([0.0], np.square(centered).sum(axis=1).cumsum()))
        left = window_start[start:end] - start
        right = window_end[start:end] - start
        lengths = right - left
        means = (sums[right] - sums[left]) / lengths[:, None]
        variance = (squares[right] - squares[left]) / lengths - np.square(means).sum(axis=1)
        variance[lengths == 1] = 0.0
        scores[start:end] = np.maximum(variance, 0.0)
    if not np.isfinite(scores).all():
        raise ValueError("Action magnitudes overflowed variance calculations")
    selected = np.lexsort((indices, -scores))[:n]
    return {
        "indices": selected,
        "scores": scores[selected],
        "all_scores": scores,
        "window_start": window_start,
        "window_end": window_end,
    }


def full_chunk_starts(terminals, chunk_size):
    """Return starts whose entire action chunk remains in one trajectory."""
    chunk_size = _integer(chunk_size, "chunk_size")
    _, _, _, episode_end = _episode_bounds(terminals)
    indices = np.arange(len(episode_end), dtype=np.int64)
    if chunk_size > len(indices):
        return indices[:0]
    return indices[indices + chunk_size <= episode_end]


def _nearest_k(distances, indices, k):
    """Partial selection with exact, deterministic handling of cutoff ties."""
    count = min(k, distances.shape[1])
    selected_indices = np.empty((len(distances), count), dtype=np.int64)
    selected_distances = np.empty((len(distances), count), dtype=np.float64)
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        cutoff = np.partition(row_distances, count - 1)[count - 1]
        nearer = np.flatnonzero(row_distances < cutoff)
        tied = np.flatnonzero(row_distances == cutoff)
        tied = tied[np.argsort(row_indices[tied], kind="stable")[: count - len(nearer)]]
        chosen = np.concatenate((nearer, tied))
        order = np.lexsort((row_indices[chosen], row_distances[chosen]))
        chosen = chosen[order]
        selected_indices[row] = row_indices[chosen]
        selected_distances[row] = row_distances[chosen]
    return selected_distances, selected_indices


def nearest_action_subspaces(
    queries,
    reference_observations,
    reference_actions,
    k,
    rcond=1e-6,
    query_batch_size=128,
    reference_batch_size=8192,
    progress=None,
):
    """Construct local action spans using exact blockwise state-space KNN.

    Observations must already be in the desired distance coordinates (raw or
    consistently standardized by the caller). Reference actions may be [R,D]
    or [R,H,A]; chunks are flattened in time-major order. The returned basis
    has shape [Q,D,min(k,D)], with orthonormal active columns and zero padding.
    Numerical rank keeps singular values greater than ``rcond * s_max``.
    Distances are Euclidean, and equal distances prefer lower reference rows.
    """
    queries = _array(queries, "queries", 2)
    references = _array(reference_observations, "reference_observations", 2)
    actions = _array(reference_actions, "reference_actions", (2, 3))
    if queries.shape[1] != references.shape[1]:
        raise ValueError("Query and reference observation dimensions must match")
    if len(actions) != len(references):
        raise ValueError("Reference observations and actions must have the same length")
    k = _integer(k, "k")
    if k > len(references):
        raise ValueError("k cannot exceed the number of reference states")
    query_batch_size = _integer(query_batch_size, "query_batch_size")
    reference_batch_size = _integer(reference_batch_size, "reference_batch_size")
    if isinstance(rcond, (bool, np.bool_)) or not isinstance(rcond, Real):
        raise TypeError("rcond must be a finite number in [0, 1)")
    if not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("rcond must be a finite number in [0, 1)")
    actions = actions.reshape(len(actions), -1)
    neighbor_indices = np.empty((len(queries), k), dtype=np.int64)
    neighbor_distances = np.empty((len(queries), k), dtype=np.float64)
    for start in range(0, len(queries), query_batch_size):
        end = min(start + query_batch_size, len(queries))
        batch = queries[start:end]
        best_distances = np.empty((len(batch), 0), dtype=np.float64)
        best_indices = np.empty((len(batch), 0), dtype=np.int64)
        for ref_start in range(0, len(references), reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, len(references))
            ref_batch = references[ref_start:ref_end]
            # Accumulate coordinate differences directly: unlike ||q||² +
            # ||r||² - 2 q.r this retains precision for large nearby states.
            distances = np.zeros((len(batch), len(ref_batch)), dtype=np.float64)
            for coordinate in range(batch.shape[1]):
                delta = batch[:, coordinate, None] - ref_batch[None, :, coordinate]
                distances += delta * delta
            if not np.isfinite(distances).all():
                raise ValueError("Observation magnitudes overflowed distance calculations")
            indices = np.broadcast_to(np.arange(ref_start, ref_end), distances.shape)
            best_distances, best_indices = _nearest_k(
                np.concatenate((best_distances, distances), axis=1),
                np.concatenate((best_indices, indices), axis=1),
                k,
            )
        neighbor_indices[start:end] = best_indices
        neighbor_distances[start:end] = np.sqrt(best_distances)
        if progress is not None:
            progress(end, len(queries))
    # Vh spans rows of the action matrix. Keeping numerical null vectors would
    # spuriously enlarge the subspace, particularly for all-zero neighbors.
    _, singular_values, vh = np.linalg.svd(actions[neighbor_indices], full_matrices=False)
    if not np.isfinite(singular_values).all():
        raise ValueError("Action magnitudes overflowed singular value calculations")
    active = singular_values > rcond * singular_values[:, :1]
    basis = vh.transpose(0, 2, 1) * active[:, None, :]
    return {
        "basis": basis,
        "ranks": active.sum(axis=1),
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
        "singular_values": singular_values,
    }


def decompose_norms(predictions, basis):
    """Measure prediction components in and orthogonal to an action span.

    ``on_manifold_norm`` is ||P a||, not distance to a demonstration action;
    ``off_manifold_norm`` is ||a - P a||; ``total_norm`` is ||a||. A large
    on-manifold norm by itself therefore does not establish policy quality.
    """
    predictions = _array(predictions, "predictions", 2)
    basis = np.asarray(basis)
    if basis.ndim != 3 or basis.shape[:2] != predictions.shape:
        raise ValueError("basis must have shape [samples, action_dim, basis_dim]")
    if basis.dtype.kind not in "iuf" or not np.isfinite(basis).all():
        raise ValueError("basis must contain finite real numbers")
    basis = basis.astype(np.float64, copy=False)
    gram = np.einsum("qdk,qdl->qkl", basis, basis)
    expected = np.zeros_like(gram)
    column_norms = np.diagonal(gram, axis1=1, axis2=2)
    diagonal = np.arange(basis.shape[2])
    expected[:, diagonal, diagonal] = column_norms > 0.5
    if not np.allclose(gram, expected, atol=1e-7, rtol=1e-6):
        raise ValueError("basis columns must be orthonormal or zero padding")
    coordinates = np.einsum("qdk,qd->qk", basis, predictions)
    projection = np.einsum("qdk,qk->qd", basis, coordinates)
    result = {
        "off_manifold_norm": np.linalg.norm(predictions - projection, axis=1),
        "on_manifold_norm": np.linalg.norm(projection, axis=1),
        "total_norm": np.linalg.norm(predictions, axis=1),
    }
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Prediction magnitudes overflowed norm calculations")
    return result

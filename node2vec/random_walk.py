"""Numba-accelerated rejection-sampling random walks for node2vec.

The kernel implements the original node2vec rejection sampling for the second-
order biased walk, with a binary-search neighbour check (O(log d) per probe).

`rowptr` must be in CSR with an explicit sentinel: `len(rowptr) == num_nodes + 1`
and `rowptr[-1] == len(col)`. Use `prepare_rowptr_col` to convert arrays from
`graph_csr` (which omit the sentinel) into this layout.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:
    nb = None  # type: ignore
    _NUMBA_AVAILABLE = False


def _is_numba_compatible(arr: np.ndarray) -> bool:
    dtype = np.asarray(arr).dtype
    if dtype.kind not in {"i", "u", "f", "b"}:
        return False
    return dtype.isnative or dtype.byteorder in {"=", "|"}


def _maybe_make_native(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr)
    if _is_numba_compatible(arr_np):
        return arr_np
    return arr_np.astype(arr_np.dtype.newbyteorder("="), copy=True)


def prepare_rowptr_col(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return numba-friendly (rowptr, col) with an explicit sentinel at the end of rowptr.

    `indptr` from `graph_csr` has length `num_nodes` (no sentinel); we append
    `len(indices)` so the numba kernel can do `rowptr[v+1]` without a boundary
    check.
    """
    indptr_arr = _maybe_make_native(np.asarray(indptr))
    indices_arr = _maybe_make_native(np.asarray(indices))
    if indptr_arr.dtype != np.int64:
        indptr_arr = indptr_arr.astype(np.int64, copy=False)
    if indptr_arr.size == 0 or int(indptr_arr[-1]) != int(indices_arr.size):
        sentinel = np.asarray([indices_arr.size], dtype=np.int64)
        rowptr = np.concatenate([indptr_arr, sentinel])
    else:
        rowptr = indptr_arr
    return rowptr, indices_arr


if _NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _seed_numba_random(seed):  # pragma: no cover - numba
        np.random.seed(seed)

    @nb.njit(cache=True, inline="always")
    def _is_neighbor_log(rowptr, col, v, w):  # pragma: no cover - numba
        left = rowptr[v]
        right = rowptr[v + 1] - 1
        while left <= right:
            mid = (left + right) // 2
            val = col[mid]
            if val == w:
                return True
            elif val < w:
                left = mid + 1
            else:
                right = mid - 1
        return False

    @nb.njit(cache=True, inline="always")
    def _is_transition_neighbor(rowptr, col, t, x, is_directed):  # pragma: no cover - numba
        if is_directed:
            return _is_neighbor_log(rowptr, col, x, t)
        deg_t = rowptr[t + 1] - rowptr[t]
        deg_x = rowptr[x + 1] - rowptr[x]
        if deg_t <= deg_x:
            return _is_neighbor_log(rowptr, col, t, x)
        return _is_neighbor_log(rowptr, col, x, t)

    @nb.njit(cache=True, inline="always")
    def _sample_uniform_next(rowptr, col, v):  # pragma: no cover - numba
        row_start = rowptr[v]
        row_end = rowptr[v + 1]
        if row_end == row_start:
            return v
        return col[row_start + np.random.randint(row_end - row_start)]

    @nb.njit(cache=True, inline="always")
    def _sample_biased_next(
        rowptr,
        col,
        t,
        v,
        prob_0,
        prob_1,
        prob_2,
        min_prob_12,
        max_prob_12,
        is_directed,
    ):  # pragma: no cover - numba
        row_start = rowptr[v]
        row_end = rowptr[v + 1]
        if row_end == row_start:
            return v
        if row_end - row_start == 1:
            return col[row_start]

        while True:
            x = col[row_start + np.random.randint(row_end - row_start)]
            r = np.random.random()
            if x == t:
                if r < prob_0:
                    return x
                continue
            if r < min_prob_12:
                return x
            if r >= max_prob_12:
                continue
            if _is_transition_neighbor(rowptr, col, t, x, is_directed):
                if r < prob_1:
                    return x
            elif r < prob_2:
                return x

    @nb.njit(cache=True, inline="always")
    def _write_node_to_windows(
        n_out,
        total_walks,
        walk_idx,
        context_size,
        num_windows,
        position,
        node,
    ):  # pragma: no cover - numba
        first_window = position - context_size + 1
        if first_window < 0:
            first_window = 0
        last_window = position
        if last_window >= num_windows:
            last_window = num_windows - 1
        for window_idx in range(first_window, last_window + 1):
            out_pos = (window_idx * total_walks + walk_idx) * context_size + position - window_idx
            n_out[out_pos] = node

    @nb.njit(cache=True)
    def _random_walk_uniform_serial(rowptr, col, start, n_out, walk_length):  # pragma: no cover - numba
        numel = start.shape[0]
        stride = walk_length + 1

        for n in range(numel):
            v = start[n]
            n_out[n * stride] = v
            for l in range(walk_length):
                v = _sample_uniform_next(rowptr, col, v)
                n_out[n * stride + l + 1] = v

    @nb.njit(cache=True)
    def _random_walk_biased_serial(
        rowptr,
        col,
        start,
        n_out,
        walk_length,
        p,
        q,
        is_directed,
    ):  # pragma: no cover - numba
        inv_p = 1.0 / p
        inv_q = 1.0 / q
        max_prob = inv_p
        if max_prob < 1.0:
            max_prob = 1.0
        if max_prob < inv_q:
            max_prob = inv_q
        prob_0 = inv_p / max_prob
        prob_1 = 1.0 / max_prob
        prob_2 = inv_q / max_prob
        min_prob_12 = prob_1 if prob_1 < prob_2 else prob_2
        max_prob_12 = prob_1 if prob_1 > prob_2 else prob_2

        numel = start.shape[0]
        stride = walk_length + 1

        for n in range(numel):
            t = start[n]
            n_out[n * stride] = t

            row_start = rowptr[t]
            row_end = rowptr[t + 1]
            if row_end == row_start:
                v = t
            else:
                v = col[row_start + np.random.randint(row_end - row_start)]
            n_out[n * stride + 1] = v

            for l in range(1, walk_length):
                x = _sample_biased_next(
                    rowptr,
                    col,
                    t,
                    v,
                    prob_0,
                    prob_1,
                    prob_2,
                    min_prob_12,
                    max_prob_12,
                    is_directed,
                )
                n_out[n * stride + l + 1] = x
                t = v
                v = x

    @nb.njit(cache=True)
    def _random_walk_uniform_windows_serial(
        rowptr,
        col,
        start,
        n_out,
        walk_length,
        context_size,
        walks_per_start,
    ):  # pragma: no cover - numba
        start_count = start.shape[0]
        total_walks = start_count * walks_per_start
        num_windows = walk_length + 2 - context_size

        for n in range(total_walks):
            v = start[n % start_count]
            _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, 0, v)
            for l in range(walk_length):
                v = _sample_uniform_next(rowptr, col, v)
                _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, l + 1, v)

    @nb.njit(cache=True)
    def _random_walk_biased_windows_serial(
        rowptr,
        col,
        start,
        n_out,
        walk_length,
        context_size,
        walks_per_start,
        p,
        q,
        is_directed,
    ):  # pragma: no cover - numba
        inv_p = 1.0 / p
        inv_q = 1.0 / q
        max_prob = inv_p
        if max_prob < 1.0:
            max_prob = 1.0
        if max_prob < inv_q:
            max_prob = inv_q
        prob_0 = inv_p / max_prob
        prob_1 = 1.0 / max_prob
        prob_2 = inv_q / max_prob
        min_prob_12 = prob_1 if prob_1 < prob_2 else prob_2
        max_prob_12 = prob_1 if prob_1 > prob_2 else prob_2

        start_count = start.shape[0]
        total_walks = start_count * walks_per_start
        num_windows = walk_length + 2 - context_size

        for n in range(total_walks):
            t = start[n % start_count]
            _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, 0, t)

            v = _sample_uniform_next(rowptr, col, t)
            _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, 1, v)

            for l in range(1, walk_length):
                x = _sample_biased_next(
                    rowptr,
                    col,
                    t,
                    v,
                    prob_0,
                    prob_1,
                    prob_2,
                    min_prob_12,
                    max_prob_12,
                    is_directed,
                )
                _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, l + 1, x)
                t = v
                v = x

    @nb.njit(cache=True)
    def _negative_sample_windows_serial(
        start,
        n_out,
        walk_length,
        context_size,
        walks_per_start,
        num_negative_samples,
        num_nodes,
    ):  # pragma: no cover - numba
        start_count = start.shape[0]
        total_walks = start_count * walks_per_start * num_negative_samples
        num_windows = walk_length + 2 - context_size

        for n in range(total_walks):
            start_idx = (n // num_negative_samples) % start_count
            v = start[start_idx]
            _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, 0, v)
            for l in range(walk_length):
                v = np.random.randint(num_nodes)
                _write_node_to_windows(n_out, total_walks, n, context_size, num_windows, l + 1, v)

else:

    def _seed_numba_random(seed):  # type: ignore[no-redef]
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")

    def _random_walk_uniform_serial(rowptr, col, start, n_out, walk_length):  # type: ignore[no-redef]
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")

    def _random_walk_biased_serial(rowptr, col, start, n_out, walk_length, p, q, is_directed):  # type: ignore[no-redef]
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")

    def _random_walk_uniform_windows_serial(  # type: ignore[no-redef]
        rowptr, col, start, n_out, walk_length, context_size, walks_per_start
    ):
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")

    def _random_walk_biased_windows_serial(  # type: ignore[no-redef]
        rowptr, col, start, n_out, walk_length, context_size, walks_per_start, p, q, is_directed
    ):
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")

    def _negative_sample_windows_serial(  # type: ignore[no-redef]
        start, n_out, walk_length, context_size, walks_per_start, num_negative_samples, num_nodes
    ):
        raise RuntimeError("numba is not installed; install it to run node2vec random walks")


def seed_numba_random(seed: int) -> None:
    if not _NUMBA_AVAILABLE:
        return
    _seed_numba_random(int(seed) & 0xFFFFFFFF)


def random_walk(
    rowptr: np.ndarray,
    col: np.ndarray,
    start: np.ndarray,
    walk_length: int,
    p: float,
    q: float,
    *,
    is_directed: bool = False,
) -> np.ndarray:
    """Run node2vec random walks. Returns walks of shape `[len(start), walk_length + 1]`.

    `rowptr` must have an explicit sentinel — use `prepare_rowptr_col` first.
    """
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")
    start_arr = np.ascontiguousarray(start, dtype=np.int64)
    n_out = np.empty(start_arr.size * (walk_length + 1), dtype=np.int64)
    if float(p) == 1.0 and float(q) == 1.0:
        _random_walk_uniform_serial(rowptr, col, start_arr, n_out, int(walk_length))
    else:
        _random_walk_biased_serial(rowptr, col, start_arr, n_out, int(walk_length), float(p), float(q), bool(is_directed))
    return n_out.reshape(start_arr.size, walk_length + 1)


def random_walk_windows(
    rowptr: np.ndarray,
    col: np.ndarray,
    start: np.ndarray,
    walk_length: int,
    context_size: int,
    p: float,
    q: float,
    *,
    walks_per_start: int = 1,
    is_directed: bool = False,
) -> np.ndarray:
    """Run random walks and return context windows without materializing full batch walks."""
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")
    if context_size < 2:
        raise ValueError("context_size must be >= 2")
    num_windows = int(walk_length) + 2 - int(context_size)
    if num_windows < 1:
        raise ValueError("context_size is incompatible with walk_length")
    if walks_per_start < 1:
        raise ValueError("walks_per_start must be >= 1")

    start_arr = np.ascontiguousarray(start, dtype=np.int64)
    total_walks = int(start_arr.size) * int(walks_per_start)
    n_out = np.empty(total_walks * num_windows * int(context_size), dtype=np.int64)
    if float(p) == 1.0 and float(q) == 1.0:
        _random_walk_uniform_windows_serial(
            rowptr,
            col,
            start_arr,
            n_out,
            int(walk_length),
            int(context_size),
            int(walks_per_start),
        )
    else:
        _random_walk_biased_windows_serial(
            rowptr,
            col,
            start_arr,
            n_out,
            int(walk_length),
            int(context_size),
            int(walks_per_start),
            float(p),
            float(q),
            bool(is_directed),
        )
    return n_out.reshape(total_walks * num_windows, int(context_size))


def negative_sample_windows(
    start: np.ndarray,
    *,
    walk_length: int,
    context_size: int,
    walks_per_start: int,
    num_negative_samples: int,
    num_nodes: int,
) -> np.ndarray:
    """Generate negative random-walk windows without materializing full negative walks."""
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")
    if context_size < 2:
        raise ValueError("context_size must be >= 2")
    num_windows = int(walk_length) + 2 - int(context_size)
    if num_windows < 1:
        raise ValueError("context_size is incompatible with walk_length")
    if walks_per_start < 1:
        raise ValueError("walks_per_start must be >= 1")
    if num_negative_samples < 1:
        raise ValueError("num_negative_samples must be >= 1")
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")

    start_arr = np.ascontiguousarray(start, dtype=np.int64)
    total_walks = int(start_arr.size) * int(walks_per_start) * int(num_negative_samples)
    n_out = np.empty(total_walks * num_windows * int(context_size), dtype=np.int64)
    _negative_sample_windows_serial(
        start_arr,
        n_out,
        int(walk_length),
        int(context_size),
        int(walks_per_start),
        int(num_negative_samples),
        int(num_nodes),
    )
    return n_out.reshape(total_walks * num_windows, int(context_size))

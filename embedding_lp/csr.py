from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_csr.graph import GraphCSR


@dataclass(slots=True)
class CSRGraphView:
    indptr: np.ndarray
    indices: np.ndarray
    num_nodes: int
    num_edges: int
    is_directed: bool = False


def validate_csr(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indptr_arr = np.asarray(indptr)
    indices_arr = np.asarray(indices)
    if indptr_arr.ndim != 1:
        raise ValueError("indptr must be 1-D")
    if indices_arr.ndim != 1:
        raise ValueError("indices must be 1-D")
    if indptr_arr.dtype.kind not in {"i", "u"}:
        raise ValueError("indptr must be an integer array")
    if indices_arr.dtype.kind not in {"i", "u"}:
        raise ValueError("indices must be an integer array")
    if indptr_arr.dtype.itemsize < 8:
        raise ValueError("indptr must use at least 64-bit offsets")
    if indices_arr.dtype.itemsize < 4:
        raise ValueError("indices must use at least 32-bit integers")
    if indptr_arr.size == 0:
        raise ValueError("indptr must not be empty")
    if indptr_arr[0] != 0:
        raise ValueError("indptr must start from 0")
    if np.any(indptr_arr[1:] < indptr_arr[:-1]):
        raise ValueError("indptr must be non-decreasing")
    if indptr_arr[-1] > indices_arr.size:
        raise ValueError("last indptr cannot exceed len(indices)")
    return indptr_arr, indices_arr


def graph_to_csr_view(graph: GraphCSR, *, is_directed: bool = False) -> CSRGraphView:
    if graph.edge_starts is None or graph.edge_ends is None:
        raise ValueError("GraphCSR must contain edge_starts and edge_ends")
    indptr = graph.edge_starts.numpy()
    indices = graph.edge_ends.numpy()
    validate_csr(indptr, indices)
    return CSRGraphView(
        indptr=indptr,
        indices=indices,
        num_nodes=int(indptr.size),
        num_edges=int(indices.size),
        is_directed=is_directed,
    )


def raw_arrays_to_csr_view(
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    is_directed: bool = False,
) -> CSRGraphView:
    indptr_arr, indices_arr = validate_csr(indptr, indices)
    return CSRGraphView(
        indptr=indptr_arr,
        indices=indices_arr,
        num_nodes=int(indptr_arr.size),
        num_edges=int(indices_arr.size),
        is_directed=is_directed,
    )

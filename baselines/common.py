from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from graph_csr.serializer import GraphCSRSerializer


LOGGER = logging.getLogger(__name__)

GraphKind = Literal["undirected-symmetric-csr", "directed", "undirected-single-edge-list"]


@dataclass(frozen=True)
class GraphInfo:
    num_nodes: int
    raw_edges: int


@dataclass(frozen=True)
class EdgeChunk:
    src: np.ndarray
    dst: np.ndarray
    node_from: int
    node_to: int


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def format_int(value: int) -> str:
    return f"{int(value):,}"


def edge_range(starts: np.ndarray, edge_ends_size: int, node_from: int, node_to: int) -> tuple[int, int]:
    lo = int(starts[node_from])
    if node_to < starts.size:
        hi = int(starts[node_to])
    else:
        hi = int(edge_ends_size)
    return lo, hi


def make_chunk_src(
    starts: np.ndarray,
    edge_ends_size: int,
    node_from: int,
    node_to: int,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    local_starts = np.asarray(starts[node_from:node_to], dtype=np.int64)
    if local_starts.size == 0:
        return np.empty((0,), dtype=dtype)

    if node_to < starts.size:
        next_values = np.asarray(starts[node_from + 1:node_to + 1], dtype=np.int64)
    else:
        next_values = np.empty(node_to - node_from, dtype=np.int64)
        if node_to - node_from > 1:
            next_values[:-1] = np.asarray(starts[node_from + 1:node_to], dtype=np.int64)
        next_values[-1] = int(edge_ends_size)

    degrees = next_values - local_starts
    if np.any(degrees < 0):
        bad_local = int(np.flatnonzero(degrees < 0)[0])
        raise ValueError(f"edge_starts is not monotonic near node_id={node_from + bad_local}")

    return np.repeat(
        np.arange(node_from, node_to, dtype=dtype),
        degrees.astype(np.int64, copy=False),
    )


def validate_csr_arrays(starts: np.ndarray, raw_edges: int) -> None:
    if starts.size == 0:
        raise ValueError("Empty GraphCSR is not supported by baseline exporters")
    if int(starts[0]) != 0:
        raise ValueError(f"Invalid CSR: edge_starts[0] must be 0, got {int(starts[0])}")
    if int(starts[-1]) > int(raw_edges):
        raise ValueError(
            f"Invalid CSR: last start {int(starts[-1])} > edge_ends.size {int(raw_edges)}"
        )


def iter_csr_edge_chunks(
    graph,
    *,
    chunk_nodes: int,
    node_dtype: np.dtype,
) -> Iterator[EdgeChunk]:
    if int(chunk_nodes) <= 0:
        raise ValueError(f"chunk_nodes must be positive, got {chunk_nodes}")
    starts = graph.edge_starts.numpy()
    ends = graph.edge_ends.numpy()
    num_nodes = int(starts.size)
    raw_edges = int(graph.edge_ends.size)
    validate_csr_arrays(starts, raw_edges)

    for node_from in range(0, num_nodes, int(chunk_nodes)):
        node_to = min(num_nodes, node_from + int(chunk_nodes))
        lo, hi = edge_range(starts, raw_edges, node_from, node_to)
        src = make_chunk_src(starts, raw_edges, node_from, node_to, dtype=node_dtype)
        dst = np.asarray(ends[lo:hi], dtype=node_dtype)
        if src.size != dst.size:
            raise RuntimeError(
                f"CSR decode mismatch for rows [{node_from}, {node_to}): "
                f"src.size={src.size}, dst.size={dst.size}"
            )
        yield EdgeChunk(src=src, dst=dst, node_from=node_from, node_to=node_to)


def edge_mask(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    graph_kind: GraphKind,
    drop_selfloops: bool,
) -> np.ndarray:
    mask = np.ones(dst.size, dtype=bool)
    if drop_selfloops:
        mask &= src != dst
    if graph_kind == "undirected-symmetric-csr":
        mask &= src < dst
    elif graph_kind in ("directed", "undirected-single-edge-list"):
        pass
    else:
        raise ValueError(f"Unknown graph_kind: {graph_kind}")
    return mask


def graph_info_from_dir(
    graph_dir: str | Path,
    *,
    use_mmap: bool,
    file_endian: str,
    allow_non_native: bool,
    chunk_bytes: int,
) -> GraphInfo:
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        starts = graph.edge_starts.numpy()
        raw_edges = int(graph.edge_ends.size)
        validate_csr_arrays(starts, raw_edges)
        return GraphInfo(num_nodes=int(starts.size), raw_edges=raw_edges)


def count_filtered_edges(
    graph_dir: str | Path,
    *,
    use_mmap: bool,
    file_endian: str,
    allow_non_native: bool,
    chunk_bytes: int,
    chunk_nodes: int,
    node_dtype: np.dtype,
    graph_kind: GraphKind,
    drop_selfloops: bool,
    log_every_chunks: int = 10,
) -> tuple[GraphInfo, int, int]:
    kept_edges = 0
    dropped_selfloops = 0
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        info = GraphInfo(
            num_nodes=int(graph.edge_starts.numpy().size),
            raw_edges=int(graph.edge_ends.size),
        )
        for chunk_id, chunk in enumerate(
            iter_csr_edge_chunks(graph, chunk_nodes=chunk_nodes, node_dtype=node_dtype)
        ):
            mask = edge_mask(
                chunk.src,
                chunk.dst,
                graph_kind=graph_kind,
                drop_selfloops=drop_selfloops,
            )
            if drop_selfloops:
                dropped_selfloops += int((chunk.src == chunk.dst).sum())
            kept_edges += int(mask.sum())
            if log_every_chunks > 0 and chunk_id % log_every_chunks == 0:
                LOGGER.info(
                    "Counted rows [%s, %s) / %s; kept_edges=%s",
                    format_int(chunk.node_from),
                    format_int(chunk.node_to),
                    format_int(info.num_nodes),
                    format_int(kept_edges),
                )
    return info, kept_edges, dropped_selfloops


def write_edge_index_npy_from_graphcsr(
    graph_dir: str | Path,
    out_path: str | Path,
    *,
    use_mmap: bool,
    file_endian: str,
    allow_non_native: bool,
    chunk_bytes: int,
    chunk_nodes: int,
    node_dtype: np.dtype,
    graph_kind: GraphKind,
    drop_selfloops: bool,
    log_every_chunks: int = 10,
) -> tuple[GraphInfo, int, int]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info, kept_edges, dropped_selfloops = count_filtered_edges(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        chunk_nodes=chunk_nodes,
        node_dtype=node_dtype,
        graph_kind=graph_kind,
        drop_selfloops=drop_selfloops,
        log_every_chunks=log_every_chunks,
    )
    edge_index = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int64,
        shape=(2, int(kept_edges)),
    )
    write_pos = 0
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        for chunk_id, chunk in enumerate(
            iter_csr_edge_chunks(graph, chunk_nodes=chunk_nodes, node_dtype=node_dtype)
        ):
            mask = edge_mask(
                chunk.src,
                chunk.dst,
                graph_kind=graph_kind,
                drop_selfloops=drop_selfloops,
            )
            n_kept = int(mask.sum())
            if n_kept:
                edge_index[0, write_pos:write_pos + n_kept] = chunk.src[mask]
                edge_index[1, write_pos:write_pos + n_kept] = chunk.dst[mask]
                write_pos += n_kept
            if log_every_chunks > 0 and chunk_id % log_every_chunks == 0:
                LOGGER.info(
                    "Wrote edge_index rows [%s, %s) / %s; edges=%s/%s",
                    format_int(chunk.node_from),
                    format_int(chunk.node_to),
                    format_int(info.num_nodes),
                    format_int(write_pos),
                    format_int(kept_edges),
                )
    if write_pos != kept_edges:
        raise RuntimeError(f"Expected to write {kept_edges} edges, wrote {write_pos}")
    edge_index.flush()
    return info, kept_edges, dropped_selfloops


def write_edge_pairs_npy_from_graphcsr(
    graph_dir: str | Path,
    out_path: str | Path,
    *,
    use_mmap: bool,
    file_endian: str,
    allow_non_native: bool,
    chunk_bytes: int,
    chunk_nodes: int,
    node_dtype: np.dtype,
    graph_kind: GraphKind,
    drop_selfloops: bool,
    log_every_chunks: int = 10,
) -> tuple[GraphInfo, int, int]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info, kept_edges, dropped_selfloops = count_filtered_edges(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        chunk_nodes=chunk_nodes,
        node_dtype=node_dtype,
        graph_kind=graph_kind,
        drop_selfloops=drop_selfloops,
        log_every_chunks=log_every_chunks,
    )
    pairs = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int64,
        shape=(int(kept_edges), 2),
    )
    write_pos = 0
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        for chunk_id, chunk in enumerate(
            iter_csr_edge_chunks(graph, chunk_nodes=chunk_nodes, node_dtype=node_dtype)
        ):
            mask = edge_mask(
                chunk.src,
                chunk.dst,
                graph_kind=graph_kind,
                drop_selfloops=drop_selfloops,
            )
            n_kept = int(mask.sum())
            if n_kept:
                pairs[write_pos:write_pos + n_kept, 0] = chunk.src[mask]
                pairs[write_pos:write_pos + n_kept, 1] = chunk.dst[mask]
                write_pos += n_kept
            if log_every_chunks > 0 and chunk_id % log_every_chunks == 0:
                LOGGER.info(
                    "Wrote pairs rows [%s, %s) / %s; edges=%s/%s",
                    format_int(chunk.node_from),
                    format_int(chunk.node_to),
                    format_int(info.num_nodes),
                    format_int(write_pos),
                    format_int(kept_edges),
                )
    if write_pos != kept_edges:
        raise RuntimeError(f"Expected to write {kept_edges} pairs, wrote {write_pos}")
    pairs.flush()
    return info, kept_edges, dropped_selfloops


def validate_node_id_bounds(min_dst: int | None, max_dst: int | None, num_nodes: int) -> None:
    if min_dst is not None and min_dst < 0:
        raise ValueError(f"Invalid dst node id: min dst = {min_dst}")
    if max_dst is not None and max_dst >= num_nodes:
        raise ValueError(f"Invalid dst node id: max dst = {max_dst}, num_nodes = {num_nodes}")

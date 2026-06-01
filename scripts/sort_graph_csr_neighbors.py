from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from graph_csr.io_utils import Endian
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.serializer import GraphCSRSerializer

from numba import njit
from scripts._common import chunk_counts, configure_logging, format_int, format_seconds, raw_int64_memmap


LOGGER = logging.getLogger("sort_graph_csr_neighbors")


@njit(cache=True)
def _sort_rows_no_ts_numba(counts: np.ndarray, block_dst: np.ndarray) -> np.ndarray:
    out_dst = np.empty(block_dst.shape[0], dtype=np.int32)
    pos = 0
    for i in range(counts.shape[0]):
        cnt = int(counts[i])
        row = block_dst[pos:pos + cnt].copy()
        row.sort()
        for j in range(cnt):
            out_dst[pos + j] = row[j]
        pos += cnt
    return out_dst


@njit(cache=True)
def _sort_rows_with_ts_numba(
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_dst = np.empty(block_dst.shape[0], dtype=np.int32)
    out_ts = np.empty(block_ts.shape[0], dtype=np.int32)
    pos = 0
    for i in range(counts.shape[0]):
        cnt = int(counts[i])
        order = np.argsort(block_dst[pos:pos + cnt])
        for j in range(cnt):
            src_pos = pos + int(order[j])
            out_dst[pos + j] = block_dst[src_pos]
            out_ts[pos + j] = block_ts[src_pos]
        pos += cnt
    return out_dst, out_ts


def _sort_rows_numpy(
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    out_dst = np.empty(block_dst.shape[0], dtype=np.int32)
    out_ts = None if block_ts is None else np.empty(block_ts.shape[0], dtype=np.int32)
    pos = 0
    for cnt_raw in counts:
        cnt = int(cnt_raw)
        if cnt <= 1:
            out_dst[pos:pos + cnt] = block_dst[pos:pos + cnt]
            if out_ts is not None and block_ts is not None:
                out_ts[pos:pos + cnt] = block_ts[pos:pos + cnt]
        else:
            order = np.argsort(block_dst[pos:pos + cnt], kind="stable")
            out_dst[pos:pos + cnt] = np.asarray(block_dst[pos:pos + cnt][order], dtype=np.int32)
            if out_ts is not None and block_ts is not None:
                out_ts[pos:pos + cnt] = np.asarray(block_ts[pos:pos + cnt][order], dtype=np.int32)
        pos += cnt
    return out_dst, out_ts


def _sort_rows(
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: Optional[np.ndarray],
    use_numba: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if use_numba:
        dst_native = np.asarray(block_dst, dtype=np.int32)
        if block_ts is None:
            return _sort_rows_no_ts_numba(counts, dst_native), None
        ts_native = np.asarray(block_ts, dtype=np.int32)
        return _sort_rows_with_ts_numba(counts, dst_native, ts_native)
    return _sort_rows_numpy(counts, block_dst, block_ts)


def sort_graph_csr_neighbors(
    *,
    graph_dir: Path,
    out_dir: Path,
    use_mmap: bool,
    file_endian: Endian,
    writable: bool,
    allow_non_native: bool,
    chunk_bytes: int,
    node_chunk_size: int,
    use_numba: bool,
    log_every_chunks: int,
    show_progress: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_starts_path = out_dir / GraphCSRSerializer.EDGE_STARTS
    edge_ends_path = out_dir / GraphCSRSerializer.EDGE_ENDS
    timestamps_path = out_dir / GraphCSRSerializer.TIMESTAMPS

    t0 = time.time()
    LOGGER.info("Opening input GraphCSR from %s", graph_dir)
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=writable,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        starts = graph.edge_starts.numpy()
        edge_ends = graph.edge_ends.numpy()
        timestamps = graph.timestamps.numpy() if graph.timestamps is not None else None

        num_nodes = int(starts.size)
        total_edges = int(graph.edge_ends.size)
        LOGGER.info(
            "Graph opened: num_nodes=%s edges=%s timestamps=%s backend=%s",
            format_int(num_nodes),
            format_int(total_edges),
            timestamps is not None,
            "numba" if use_numba else "numpy",
        )

        LOGGER.info("Writing edge_starts unchanged")
        out_starts = raw_int64_memmap(edge_starts_path, num_nodes, file_endian)
        for lo in range(0, num_nodes, node_chunk_size):
            hi = min(lo + node_chunk_size, num_nodes)
            out_starts[lo:hi] = np.asarray(starts[lo:hi], dtype=np.int64)
        out_starts.flush()
        del out_starts

        out_edge_ends = LargeInt32Array.create(
            total_edges,
            use_mmap=True,
            path=edge_ends_path,
            file_endian=file_endian,
            writable=True,
        )
        out_edge_ends_arr = out_edge_ends.numpy()

        if timestamps is not None:
            out_timestamps = LargeInt32Array.create(
                total_edges,
                use_mmap=True,
                path=timestamps_path,
                file_endian=file_endian,
                writable=True,
            )
            out_timestamps_arr = out_timestamps.numpy()
        else:
            timestamps_path.write_bytes(b"")
            out_timestamps = None
            out_timestamps_arr = None

        chunk_idx = 0
        chunk_iter = range(0, num_nodes, node_chunk_size)
        bar = tqdm(
            chunk_iter,
            total=(num_nodes + node_chunk_size - 1) // node_chunk_size,
            desc="sort csr rows",
            unit="chunk",
            disable=not show_progress,
        )
        for lo in bar:
            hi = min(lo + node_chunk_size, num_nodes)
            chunk_idx += 1

            chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
            if chunk_starts.size == 0:
                continue

            start_edge = int(chunk_starts[0])
            end_edge = int(starts[hi]) if hi < num_nodes else total_edges
            counts = chunk_counts(chunk_starts, end_edge)
            block_dst = edge_ends[start_edge:end_edge]
            block_ts = timestamps[start_edge:end_edge] if timestamps is not None else None

            sorted_dst, sorted_ts = _sort_rows(counts, block_dst, block_ts, use_numba)
            if sorted_dst.size:
                out_edge_ends_arr[start_edge:end_edge] = sorted_dst
                if out_timestamps_arr is not None and sorted_ts is not None:
                    out_timestamps_arr[start_edge:end_edge] = sorted_ts

            if show_progress:
                bar.set_postfix(nodes=format_int(hi), edges=format_int(end_edge))

            if log_every_chunks > 0 and chunk_idx % log_every_chunks == 0:
                elapsed = time.time() - t0
                LOGGER.info(
                    "Chunks=%s nodes=%s/%s (%.2f%%) edges=%s/%s elapsed=%s",
                    format_int(chunk_idx),
                    format_int(hi),
                    format_int(num_nodes),
                    100.0 * hi / max(num_nodes, 1),
                    format_int(end_edge),
                    format_int(total_edges),
                    format_seconds(elapsed),
                )

        out_edge_ends.flush()
        out_edge_ends.close()
        if out_timestamps is not None:
            out_timestamps.flush()
            out_timestamps.close()
        if show_progress:
            bar.close()

    elapsed = time.time() - t0
    LOGGER.info(
        "Finished sorting GraphCSR rows: num_nodes=%s edges=%s elapsed=%s output=%s",
        format_int(num_nodes),
        format_int(total_edges),
        format_seconds(elapsed),
        out_dir,
    )
    return {
        "num_nodes": num_nodes,
        "edges": total_edges,
        "timestamps": timestamps is not None,
        "used_numba": bool(use_numba),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GraphCSR copy where every adjacency list is sorted. "
            "edge_starts are preserved; edge_ends are sorted row-by-row; timestamps, "
            "when present, are reordered together with edge_ends."
        )
    )
    parser.add_argument("--graph-dir", type=str, required=True, help="Input GraphCSR folder")
    parser.add_argument("--out-dir", type=str, required=True, help="Output sorted GraphCSR folder")
    parser.add_argument("--use-mmap", action="store_true", help="Use mmap for reading input graph")
    parser.add_argument("--file-endian", type=str, default="big", choices=["big", "little"])
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--node-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--allow-non-native", action="store_true")
    parser.add_argument("--writable", action="store_true")
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable numba kernels even if numba is installed",
    )
    parser.add_argument("--log-every-chunks", type=int, default=10)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    sort_graph_csr_neighbors(
        graph_dir=Path(args.graph_dir),
        out_dir=Path(args.out_dir),
        use_mmap=args.use_mmap,
        file_endian=args.file_endian,
        writable=args.writable,
        allow_non_native=args.allow_non_native,
        chunk_bytes=args.chunk_bytes,
        node_chunk_size=args.node_chunk_size,
        use_numba=not args.no_numba,
        log_every_chunks=args.log_every_chunks,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()

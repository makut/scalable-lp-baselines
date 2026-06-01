from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_csr.io_utils import Endian
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.serializer import GraphCSRSerializer
from scripts._common import chunk_counts, configure_logging, format_int, format_seconds, raw_int64_memmap

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


LOGGER = logging.getLogger("graph_csr_to_upper")


@njit(cache=True)
def _count_upper_numba(lo: int, counts: np.ndarray, block_dst: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.shape[0], dtype=np.int64)
    pos = 0
    for i in range(counts.shape[0]):
        src = lo + i
        cnt = counts[i]
        keep = 0
        for _ in range(cnt):
            if block_dst[pos] > src:
                keep += 1
            pos += 1
        out[i] = keep
    return out


@njit(cache=True)
def _filter_upper_no_ts_numba(
    lo: int,
    counts: np.ndarray,
    block_dst: np.ndarray,
) -> np.ndarray:
    kept_counts = _count_upper_numba(lo, counts, block_dst)
    total_kept = int(kept_counts.sum())

    out_dst = np.empty(total_kept, dtype=np.int32)

    read_pos = 0
    write_pos = 0
    for i in range(counts.shape[0]):
        src = lo + i
        cnt = counts[i]
        for _ in range(cnt):
            dst = block_dst[read_pos]
            if dst > src:
                out_dst[write_pos] = dst
                write_pos += 1
            read_pos += 1

    return out_dst


@njit(cache=True)
def _filter_upper_with_ts_numba(
    lo: int,
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kept_counts = _count_upper_numba(lo, counts, block_dst)
    total_kept = int(kept_counts.sum())

    out_dst = np.empty(total_kept, dtype=np.int32)
    out_ts = np.empty(total_kept, dtype=np.int32)

    read_pos = 0
    write_pos = 0
    for i in range(counts.shape[0]):
        src = lo + i
        cnt = counts[i]
        for _ in range(cnt):
            dst = block_dst[read_pos]
            if dst > src:
                out_dst[write_pos] = dst
                out_ts[write_pos] = block_ts[read_pos]
                write_pos += 1
            read_pos += 1

    return out_dst, out_ts


def _count_upper_numpy(lo: int, counts: np.ndarray, block_dst: np.ndarray) -> np.ndarray:
    if counts.size == 0:
        return np.empty((0,), dtype=np.int64)
    src = np.repeat(np.arange(lo, lo + counts.shape[0], dtype=np.int64), counts)
    mask = block_dst.astype(np.int64, copy=False) > src
    offsets = np.empty(counts.shape[0], dtype=np.int64)
    offsets[0] = 0
    if counts.shape[0] > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    prefix = np.empty(mask.shape[0] + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(mask.astype(np.int64, copy=False), out=prefix[1:])
    return prefix[offsets + counts] - prefix[offsets]


def _filter_upper_numpy(
    lo: int,
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if counts.size == 0:
        return np.empty((0,), dtype=np.int32), None if block_ts is None else np.empty((0,), dtype=np.int32)
    src = np.repeat(np.arange(lo, lo + counts.shape[0], dtype=np.int64), counts)
    mask = block_dst.astype(np.int64, copy=False) > src
    kept_dst = np.asarray(block_dst[mask], dtype=np.int32)
    kept_ts = np.asarray(block_ts[mask], dtype=np.int32) if block_ts is not None else None
    return kept_dst, kept_ts


def _count_upper(
    lo: int,
    counts: np.ndarray,
    block_dst: np.ndarray,
    use_numba: bool,
) -> np.ndarray:
    if use_numba:
        return _count_upper_numba(lo, counts, np.asarray(block_dst, dtype=np.int32))
    return _count_upper_numpy(lo, counts, block_dst)


def _filter_upper(
    lo: int,
    counts: np.ndarray,
    block_dst: np.ndarray,
    block_ts: Optional[np.ndarray],
    use_numba: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if use_numba:
        dst_native = np.asarray(block_dst, dtype=np.int32)
        if block_ts is None:
            return _filter_upper_no_ts_numba(lo, counts, dst_native), None
        ts_native = np.asarray(block_ts, dtype=np.int32)
        return _filter_upper_with_ts_numba(lo, counts, dst_native, ts_native)
    return _filter_upper_numpy(lo, counts, block_dst, block_ts)


def convert_graph_to_upper(
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

    if use_numba and not NUMBA_AVAILABLE:
        LOGGER.warning("numba is not installed; falling back to numpy implementation")
        use_numba = False

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
        total_directed_edges = int(graph.edge_ends.size)
        LOGGER.info(
            "Graph opened: num_nodes=%s total_directed_edges=%s timestamps=%s backend=%s",
            format_int(num_nodes),
            format_int(total_directed_edges),
            timestamps is not None,
            "numba" if use_numba else "numpy",
        )

        out_starts = raw_int64_memmap(edge_starts_path, num_nodes, file_endian)

        kept_total = 0
        chunk_idx = 0
        pass1_iter = range(0, num_nodes, node_chunk_size)
        pass1_bar = tqdm(
            pass1_iter,
            total=(num_nodes + node_chunk_size - 1) // node_chunk_size,
            desc="pass1 count upper",
            unit="chunk",
            disable=not show_progress,
        )
        for lo in pass1_bar:
            hi = min(lo + node_chunk_size, num_nodes)
            chunk_idx += 1

            chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
            if chunk_starts.size == 0:
                continue

            end_edge = int(starts[hi]) if hi < num_nodes else total_directed_edges
            counts = chunk_counts(chunk_starts, end_edge)
            start_edge = int(chunk_starts[0])
            block_dst = edge_ends[start_edge:end_edge]

            kept_counts = _count_upper(lo, counts, block_dst, use_numba)
            local_offsets = np.empty(kept_counts.shape[0], dtype=np.int64)
            local_offsets[0] = 0
            if kept_counts.shape[0] > 1:
                local_offsets[1:] = np.cumsum(kept_counts[:-1], dtype=np.int64)
            out_starts[lo:hi] = kept_total + local_offsets
            kept_total += int(kept_counts.sum())
            if show_progress:
                pass1_bar.set_postfix(nodes=format_int(hi), kept=format_int(kept_total))

            if log_every_chunks > 0 and chunk_idx % log_every_chunks == 0:
                elapsed = time.time() - t0
                LOGGER.info(
                    "Pass 1 chunks=%s nodes=%s/%s (%.2f%%) kept_edges=%s elapsed=%s",
                    format_int(chunk_idx),
                    format_int(hi),
                    format_int(num_nodes),
                    100.0 * hi / max(num_nodes, 1),
                    format_int(kept_total),
                    format_seconds(elapsed),
                )

        out_starts.flush()
        del out_starts
        if show_progress:
            pass1_bar.close()

        LOGGER.info(
            "Pass 1 done: output_upper_edges=%s elapsed=%s",
            format_int(kept_total),
            format_seconds(time.time() - t0),
        )

        out_edge_ends = LargeInt32Array.create(
            kept_total,
            use_mmap=True,
            path=edge_ends_path,
            file_endian=file_endian,
            writable=True,
        )
        out_edge_ends_arr = out_edge_ends.numpy()

        if timestamps is not None:
            out_timestamps = LargeInt32Array.create(
                kept_total,
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

        written = 0
        chunk_idx = 0
        pass2_iter = range(0, num_nodes, node_chunk_size)
        pass2_bar = tqdm(
            pass2_iter,
            total=(num_nodes + node_chunk_size - 1) // node_chunk_size,
            desc="pass2 write upper",
            unit="chunk",
            disable=not show_progress,
        )
        for lo in pass2_bar:
            hi = min(lo + node_chunk_size, num_nodes)
            chunk_idx += 1

            chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
            if chunk_starts.size == 0:
                continue

            end_edge = int(starts[hi]) if hi < num_nodes else total_directed_edges
            counts = chunk_counts(chunk_starts, end_edge)
            start_edge = int(chunk_starts[0])
            block_dst = edge_ends[start_edge:end_edge]
            block_ts = timestamps[start_edge:end_edge] if timestamps is not None else None

            kept_dst, kept_ts = _filter_upper(lo, counts, block_dst, block_ts, use_numba)
            n_kept = int(kept_dst.shape[0])
            if n_kept:
                out_edge_ends_arr[written:written + n_kept] = kept_dst
                if out_timestamps_arr is not None and kept_ts is not None:
                    out_timestamps_arr[written:written + n_kept] = kept_ts
                written += n_kept
            if show_progress:
                pass2_bar.set_postfix(nodes=format_int(hi), written=format_int(written))

            if log_every_chunks > 0 and chunk_idx % log_every_chunks == 0:
                elapsed = time.time() - t0
                LOGGER.info(
                    "Pass 2 chunks=%s nodes=%s/%s (%.2f%%) written_edges=%s/%s elapsed=%s",
                    format_int(chunk_idx),
                    format_int(hi),
                    format_int(num_nodes),
                    100.0 * hi / max(num_nodes, 1),
                    format_int(written),
                    format_int(kept_total),
                    format_seconds(elapsed),
                )

        out_edge_ends.flush()
        out_edge_ends.close()
        if out_timestamps is not None:
            out_timestamps.flush()
            out_timestamps.close()
        if show_progress:
            pass2_bar.close()

    elapsed = time.time() - t0
    LOGGER.info(
        "Finished conversion: num_nodes=%s upper_edges=%s elapsed=%s output=%s",
        format_int(num_nodes),
        format_int(kept_total),
        format_seconds(elapsed),
        out_dir,
    )
    return {
        "num_nodes": num_nodes,
        "input_directed_edges": total_directed_edges,
        "output_upper_edges": kept_total,
        "timestamps": timestamps is not None,
        "used_numba": bool(use_numba),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an undirected bidirectional GraphCSR into a GraphCSR that stores "
            "only canonical edges with u < v while preserving the same on-disk format."
        )
    )
    parser.add_argument("--graph-dir", type=str, required=True, help="Input GraphCSR folder")
    parser.add_argument("--out-dir", type=str, required=True, help="Output GraphCSR folder")
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
    convert_graph_to_upper(
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

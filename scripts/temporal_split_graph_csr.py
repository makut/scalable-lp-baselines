"""
Temporal split of a GraphCSR by edge timestamp.

The K most-recent edges (largest timestamp values) go to the validation
positive pair list; the rest stay in the train GraphCSR (written in the
same format, with stable node ids; per-row neighbour lists are sorted
ascending so that `_has_edge_sorted` works on the result). The validation
positives are saved as `[N, 2] int64` via `np.save` — the format consumed
by `dataset_utils.eval_data._load_edges`.

Timestamps are treated as opaque int32 values whose only meaningful
property is order: smaller value ⇒ strictly earlier (with ~1s granularity
for equal values). No base, unit, or epoch is assumed.

Selection of the threshold timestamp `T*` (the K-th largest value) is
done via a two-pass radix histogram select: O(E) time, ~512 KB memory,
sequential I/O over the mmap'd `timestamps` array.

Ties at `T*` (edges with `ts == T*`) are handled by an explicit policy:
  * "exclude" (default): val = {ts > T*}; ties go to train.
    val size ≤ K, never splits a simultaneous batch.
  * "include": val = {ts >= T*}; ties go to val. val size ≥ K.

For each unique source vertex in the validation positives, the script
also generates negative samples via length-2 random walks over the train
graph and saves them as a sibling `[M, 2] int64` array (consumable by
`LabeledEdgeDataset` as `valid_edge_neg`).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from tqdm import tqdm

from dataset_utils.negative_sampling import sample_two_hop_unique_dsts
from graph_csr.io_utils import Endian
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.serializer import GraphCSRSerializer
from scripts.sort_graph_csr_neighbors import _sort_rows_with_ts_numba
from scripts._common import (
    chunk_counts,
    configure_logging,
    delete_dir_safely,
    format_int,
    format_seconds,
    raw_int64_memmap,
)

LOGGER = logging.getLogger("temporal_split_graph_csr")

TiesPolicy = Literal["exclude", "include"]

_SIGN_FLIP = np.uint32(0x80000000)
_RADIX_CHUNK_DEFAULT = 16 * 1024 * 1024  # int32 elements per radix scan chunk


def _hash_undirected_edge_pair(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Symmetric int32 hash for an undirected edge.

    Returns the same value for (u, v) and (v, u) — required so both
    directions of an edge get the same synthetic timestamp, which keeps
    the temporal split coherent on a symmetric CSR. Output covers the
    full signed int32 range so `_radix_select_kth_largest` sees a roughly
    uniform distribution (no bias toward positive or negative buckets).

    Mixing is a 64-bit SplitMix finalizer on `(min(u,v) << 32) | max(u,v)`.
    """
    u = src.astype(np.uint64, copy=False)
    v = dst.astype(np.uint64, copy=False)
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    x = (a << np.uint64(32)) | b
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    # Truncate to low 32 bits, reinterpret as signed int32 (full range).
    return x.astype(np.uint32, copy=False).view(np.int32)


def _build_hash_timestamps(
    *,
    starts: np.ndarray,
    edge_ends: np.ndarray,
    num_nodes: int,
    total_edges: int,
    node_chunk_size: int,
    show_progress: bool,
) -> np.ndarray:
    """Materialize synthetic ts = hash(min(u,v), max(u,v)) for every edge in CSR.

    One sequential pass over the graph, node-chunked: `src` for the chunk
    is built via `np.repeat(np.arange(lo, hi), counts)` (cheap vector op),
    `dst` is the corresponding `edge_ends` slice. Memory is O(N_edges)
    int32 = 4 B/edge — same as a regular `timestamps.bin`, just kept in
    RAM instead of written to disk.

    The output is suitable for `_radix_select_kth_largest` and the
    existing pass1/pass2 emitters: they only ever read `timestamps`
    sequentially or by slice.
    """
    out = np.empty(total_edges, dtype=np.int32)
    bar = tqdm(
        list(_node_chunks(num_nodes, node_chunk_size)),
        desc="hash timestamps",
        unit="chunk",
        disable=not show_progress,
    )
    for lo, hi in bar:
        chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
        if chunk_starts.size == 0:
            continue
        start_edge = int(chunk_starts[0])
        end_edge = int(starts[hi]) if hi < num_nodes else int(total_edges)
        counts = chunk_counts(chunk_starts, end_edge)
        src_per_edge = np.repeat(np.arange(lo, hi, dtype=np.int32), counts)
        dst_per_edge = np.asarray(edge_ends[start_edge:end_edge], dtype=np.int32)
        out[start_edge:end_edge] = _hash_undirected_edge_pair(src_per_edge, dst_per_edge)
    return out


def _radix_select_kth_largest(
    timestamps: np.ndarray,
    *,
    k: int,
    chunk_size: int = _RADIX_CHUNK_DEFAULT,
    show_progress: bool,
) -> Tuple[int, int, int]:
    """Find the K-th largest int32 in `timestamps` via two radix-histogram passes.

    Returns `(threshold, strictly_above, at_threshold)` where:
      * `count(ts > threshold) == strictly_above < k`
      * `count(ts == threshold) == at_threshold`
      * `strictly_above + at_threshold >= k`

    Memory: 2 × 65536 × 8 B histograms (one at a time). I/O: two sequential
    passes over `timestamps` (mmap-friendly). Time: O(E).
    """
    total = int(timestamps.size)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > total:
        raise ValueError(f"k={k} exceeds total edges {total}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    def _iter_chunks(desc: str):
        bar = tqdm(
            range(0, total, chunk_size),
            desc=desc,
            unit="chunk",
            disable=not show_progress,
        )
        for start in bar:
            end = min(start + chunk_size, total)
            block = np.asarray(timestamps[start:end], dtype=np.int32)
            yield block

    # Pass A: high-16 histogram over (u = t ^ 0x80000000) so signed order ↔ unsigned order.
    hi_hist = np.zeros(1 << 16, dtype=np.int64)
    for block in _iter_chunks("radix pass A (hi-16)"):
        u = block.view(np.uint32) ^ _SIGN_FLIP
        hi = (u >> np.uint32(16)).astype(np.uint16, copy=False)
        hi_hist += np.bincount(hi, minlength=1 << 16).astype(np.int64, copy=False)

    if int(hi_hist.sum()) != total:
        raise RuntimeError("radix pass A histogram total mismatch")

    above_hi = 0
    selected_hi: int | None = None
    for b in range(65535, -1, -1):
        cnt = int(hi_hist[b])
        if cnt == 0:
            continue
        if above_hi + cnt >= k:
            selected_hi = b
            break
        above_hi += cnt
    if selected_hi is None:
        raise RuntimeError("radix-select pass A failed to locate bucket")
    need_in_bucket = k - above_hi
    LOGGER.info(
        "radix pass A: selected_hi=%s above_hi=%s need_in_bucket=%s",
        selected_hi,
        format_int(above_hi),
        format_int(need_in_bucket),
    )

    # Pass B: low-16 histogram restricted to hi == selected_hi.
    selected_hi_u32 = np.uint32(selected_hi)
    lo_hist = np.zeros(1 << 16, dtype=np.int64)
    for block in _iter_chunks("radix pass B (lo-16)"):
        u = block.view(np.uint32) ^ _SIGN_FLIP
        hi = u >> np.uint32(16)
        mask = hi == selected_hi_u32
        if not mask.any():
            continue
        lo = (u[mask] & np.uint32(0xFFFF)).astype(np.uint16, copy=False)
        lo_hist += np.bincount(lo, minlength=1 << 16).astype(np.int64, copy=False)

    above_in_bucket = 0
    selected_lo: int | None = None
    for b in range(65535, -1, -1):
        cnt = int(lo_hist[b])
        if cnt == 0:
            continue
        if above_in_bucket + cnt >= need_in_bucket:
            selected_lo = b
            break
        above_in_bucket += cnt
    if selected_lo is None:
        raise RuntimeError("radix-select pass B failed to locate bucket")

    u_star = np.array(
        [(np.uint32(selected_hi) << np.uint32(16)) | np.uint32(selected_lo)],
        dtype=np.uint32,
    )
    threshold = int((u_star ^ _SIGN_FLIP).view(np.int32)[0])

    strictly_above = above_hi + above_in_bucket
    at_threshold = int(lo_hist[selected_lo])

    if not (strictly_above < k <= strictly_above + at_threshold):
        raise RuntimeError(
            f"radix-select invariant violated: k={k} strictly_above={strictly_above} "
            f"at_threshold={at_threshold}"
        )

    return threshold, strictly_above, at_threshold


def _node_chunks(num_nodes: int, chunk_size: int):
    for lo in range(0, num_nodes, chunk_size):
        yield lo, min(lo + chunk_size, num_nodes)


def _row_counts_in_chunk(
    flag_block: np.ndarray,
    chunk_starts: np.ndarray,
    start_edge: int,
    end_edge: int,
) -> np.ndarray:
    """Return per-node count of True values in `flag_block` for nodes [lo, hi).

    `flag_block` covers edges [start_edge, end_edge). `chunk_starts` are absolute
    CSR row offsets for nodes [lo, hi).
    """
    if chunk_starts.size == 0:
        return np.empty((0,), dtype=np.int64)
    local_starts = (np.asarray(chunk_starts, dtype=np.int64) - int(start_edge)).astype(np.int64)
    local_ends = np.empty_like(local_starts)
    local_ends[:-1] = local_starts[1:]
    local_ends[-1] = int(end_edge) - int(start_edge)
    cum = np.concatenate(([0], np.cumsum(flag_block.astype(np.int64))))
    return cum[local_ends] - cum[local_starts]


def _train_mask(ts_block: np.ndarray, threshold: int, ties: TiesPolicy) -> np.ndarray:
    """Boolean mask of edges that go to the train split.

    `ties="exclude"` ⇒ val = {ts > T*}, so train = {ts <= T*} (ties → train).
    `ties="include"` ⇒ val = {ts >= T*}, so train = {ts < T*} (ties → val).
    """
    t32 = np.int32(threshold)
    if ties == "exclude":
        return ts_block <= t32
    if ties == "include":
        return ts_block < t32
    raise ValueError(f"Unknown ties policy: {ties!r}")


def _pass1_count_train(
    *,
    starts: np.ndarray,
    timestamps: np.ndarray,
    threshold: int,
    ties: TiesPolicy,
    num_nodes: int,
    total_edges: int,
    node_chunk_size: int,
    show_progress: bool,
) -> np.ndarray:
    train_count_per_node = np.empty(num_nodes, dtype=np.int64)
    bar = tqdm(
        list(_node_chunks(num_nodes, node_chunk_size)),
        desc="pass 1 / count",
        unit="chunk",
        disable=not show_progress,
    )
    for lo, hi in bar:
        chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
        if chunk_starts.size == 0:
            continue
        start_edge = int(chunk_starts[0])
        end_edge = int(starts[hi]) if hi < num_nodes else int(total_edges)
        block_ts = np.asarray(timestamps[start_edge:end_edge], dtype=np.int32)
        mask = _train_mask(block_ts, threshold, ties)
        train_count_per_node[lo:hi] = _row_counts_in_chunk(mask, chunk_starts, start_edge, end_edge)
    return train_count_per_node


def _pass2_emit(
    *,
    starts: np.ndarray,
    edge_ends: np.ndarray,
    timestamps: np.ndarray,
    threshold: int,
    ties: TiesPolicy,
    num_nodes: int,
    total_edges: int,
    node_chunk_size: int,
    train_count_per_node: np.ndarray,
    out_edge_ends_arr: np.ndarray,
    out_timestamps_arr: np.ndarray,
    val_pairs: np.ndarray,
    show_progress: bool,
    val_pairs_ts: np.ndarray | None = None,
) -> Tuple[int, int]:
    train_offset = 0
    val_offset = 0
    bar = tqdm(
        list(_node_chunks(num_nodes, node_chunk_size)),
        desc="pass 2 / emit",
        unit="chunk",
        disable=not show_progress,
    )
    for lo, hi in bar:
        chunk_starts = np.asarray(starts[lo:hi], dtype=np.int64)
        if chunk_starts.size == 0:
            continue
        start_edge = int(chunk_starts[0])
        end_edge = int(starts[hi]) if hi < num_nodes else int(total_edges)
        block_dst = np.asarray(edge_ends[start_edge:end_edge], dtype=np.int32)
        block_ts = np.asarray(timestamps[start_edge:end_edge], dtype=np.int32)

        is_train = _train_mask(block_ts, threshold, ties)
        n_train = int(is_train.sum())
        n_val = int(block_dst.size) - n_train

        if n_train:
            chunk_train_counts = np.asarray(train_count_per_node[lo:hi], dtype=np.int64)
            filtered_dst = np.ascontiguousarray(block_dst[is_train], dtype=np.int32)
            filtered_ts = np.ascontiguousarray(block_ts[is_train], dtype=np.int32)
            sorted_dst, sorted_ts = _sort_rows_with_ts_numba(
                chunk_train_counts, filtered_dst, filtered_ts
            )
            out_edge_ends_arr[train_offset:train_offset + n_train] = sorted_dst
            out_timestamps_arr[train_offset:train_offset + n_train] = sorted_ts
            train_offset += n_train

        if n_val:
            is_val = ~is_train
            val_per_node = _row_counts_in_chunk(is_val, chunk_starts, start_edge, end_edge)
            src = np.repeat(np.arange(lo, hi, dtype=np.int64), val_per_node)
            val_pairs[val_offset:val_offset + n_val, 0] = src
            val_pairs[val_offset:val_offset + n_val, 1] = block_dst[is_val].astype(np.int64)
            if val_pairs_ts is not None:
                val_pairs_ts[val_offset:val_offset + n_val] = block_ts[is_val]
            val_offset += n_val
    return train_offset, val_offset


def _generate_val_negatives(
    *,
    train_graph_dir: Path,
    val_pairs: np.ndarray,
    n_walks_per_src: int,
    seed: int,
    use_mmap: bool,
    file_endian: Endian,
    allow_non_native: bool,
    chunk_bytes: int,
    show_progress: bool,
    desc: str = "val negatives",
) -> np.ndarray:
    """Generate validation negatives via 2-hop random walks over the train graph.

    For each unique src in `val_pairs`, run `n_walks_per_src` length-2 walks
    over the (sorted) train CSR; collect unique final vertices that are not
    src and not direct neighbours of src. Output is `[total, 2] int64`.

    `desc` is used for log/progress strings — pass "test negatives" when
    sampling for the test split so logs are unambiguous.
    """
    if val_pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    unique_srcs = np.unique(np.asarray(val_pairs[:, 0], dtype=np.int64))
    LOGGER.info(
        "Generating %s: unique_srcs=%s walks_per_src=%s",
        desc,
        format_int(unique_srcs.size),
        format_int(n_walks_per_src),
    )

    with GraphCSRSerializer.deserialize(
        train_graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        edge_starts = graph.edge_starts.numpy()
        edge_ends = graph.edge_ends.numpy()
        rng = np.random.default_rng(int(seed))

        chunks: list[np.ndarray] = []
        bar = tqdm(
            unique_srcs,
            desc=desc,
            unit="src",
            disable=not show_progress,
        )
        empty_srcs = 0
        for src in bar:
            dsts = sample_two_hop_unique_dsts(
                edge_starts,
                edge_ends,
                src=int(src),
                n_walks=int(n_walks_per_src),
                rng=rng,
            )
            if dsts.size == 0:
                empty_srcs += 1
                continue
            block = np.empty((dsts.size, 2), dtype=np.int64)
            block[:, 0] = int(src)
            block[:, 1] = dsts
            chunks.append(block)

    if empty_srcs:
        LOGGER.warning(
            "%s/%s %s source vertices produced 0 negatives (isolated in train or fully-connected)",
            format_int(empty_srcs),
            format_int(unique_srcs.size),
            desc,
        )

    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


def _default_val_neg_path(val_out_pairs: Path) -> Path:
    """Derive `<stem>_neg.npy` next to the val positives path."""
    stem = val_out_pairs.stem
    suffix = val_out_pairs.suffix or ".npy"
    return val_out_pairs.with_name(f"{stem}_neg{suffix}")


def temporal_split_graph_csr(
    *,
    graph_dir: Path,
    train_out_dir: Path,
    val_out_pairs: Path,
    val_edges: int,
    ties: TiesPolicy,
    use_mmap: bool,
    file_endian: Endian,
    out_file_endian: Endian,
    allow_non_native: bool,
    chunk_bytes: int,
    node_chunk_size: int,
    show_progress: bool,
    val_out_neg: Path | None,
    val_neg_walks_per_src: int,
    val_neg_seed: int,
    radix_chunk_size: int = _RADIX_CHUNK_DEFAULT,
    delete_input_after_success: bool = False,
    use_edge_hash_as_timestamps: bool = False,
) -> dict:
    if val_edges <= 0:
        raise ValueError(f"val_edges must be positive, got {val_edges}")
    if ties not in ("exclude", "include"):
        raise ValueError(f"ties must be 'exclude' or 'include', got {ties!r}")

    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_pairs.parent.mkdir(parents=True, exist_ok=True)
    edge_starts_path = train_out_dir / GraphCSRSerializer.EDGE_STARTS
    edge_ends_path = train_out_dir / GraphCSRSerializer.EDGE_ENDS
    timestamps_path = train_out_dir / GraphCSRSerializer.TIMESTAMPS

    t0 = time.time()
    LOGGER.info("Opening input GraphCSR from %s", graph_dir)
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        starts = graph.edge_starts.numpy()
        edge_ends = graph.edge_ends.numpy()
        num_nodes = int(starts.size)
        total_edges = int(graph.edge_ends.size)

        if use_edge_hash_as_timestamps:
            LOGGER.info(
                "Synthetic-ts mode: ignoring on-disk timestamps (present=%s); "
                "computing hash(min(u,v), max(u,v)) for %s edges",
                graph.timestamps is not None,
                format_int(total_edges),
            )
            timestamps = _build_hash_timestamps(
                starts=starts,
                edge_ends=edge_ends,
                num_nodes=num_nodes,
                total_edges=total_edges,
                node_chunk_size=node_chunk_size,
                show_progress=show_progress,
            )
        else:
            if graph.timestamps is None:
                raise ValueError(
                    "Input graph has no timestamps; temporal split is not possible. "
                    "Pass use_edge_hash_as_timestamps=True to synthesize them "
                    "from edge identities."
                )
            timestamps = graph.timestamps.numpy()

        LOGGER.info(
            "Graph opened: nodes=%s edges=%s val_edges=%s ties=%s synthetic_ts=%s",
            format_int(num_nodes),
            format_int(total_edges),
            format_int(val_edges),
            ties,
            use_edge_hash_as_timestamps,
        )

        if val_edges > total_edges:
            raise ValueError(
                f"val_edges={val_edges} exceeds total edges {total_edges}"
            )

        threshold, strictly_above, at_threshold = _radix_select_kth_largest(
            timestamps,
            k=val_edges,
            chunk_size=radix_chunk_size,
            show_progress=show_progress,
        )
        if ties == "exclude":
            total_val = strictly_above
        else:
            total_val = strictly_above + at_threshold
        total_train = total_edges - total_val
        LOGGER.info(
            "Threshold T*=%s: strictly_above=%s at_threshold=%s → val=%s (requested %s) train=%s",
            threshold,
            format_int(strictly_above),
            format_int(at_threshold),
            format_int(total_val),
            format_int(val_edges),
            format_int(total_train),
        )

        train_count_per_node = _pass1_count_train(
            starts=starts,
            timestamps=timestamps,
            threshold=threshold,
            ties=ties,
            num_nodes=num_nodes,
            total_edges=total_edges,
            node_chunk_size=node_chunk_size,
            show_progress=show_progress,
        )
        counted_train = int(train_count_per_node.sum())
        if counted_train != total_train:
            raise RuntimeError(
                f"Train count mismatch between radix-select and pass 1: "
                f"select={total_train} pass1={counted_train}"
            )

        new_edge_starts = np.zeros(num_nodes, dtype=np.int64)
        if num_nodes > 1:
            np.cumsum(train_count_per_node[:-1], out=new_edge_starts[1:])

        LOGGER.info("Writing train edge_starts to %s", edge_starts_path)
        out_starts = raw_int64_memmap(edge_starts_path, num_nodes, out_file_endian)
        out_starts[:] = new_edge_starts.astype(out_starts.dtype, copy=False)
        out_starts.flush()
        del out_starts

        if total_train == 0:
            timestamps_path.write_bytes(b"")
            edge_ends_path.write_bytes(b"")
            LOGGER.warning("No edges left in train split; writing empty edge_ends/timestamps")
            out_edge_ends = None
            out_timestamps = None
            out_edge_ends_arr = np.empty((0,), dtype=np.int32)
            out_timestamps_arr = np.empty((0,), dtype=np.int32)
        else:
            out_edge_ends = LargeInt32Array.create(
                total_train,
                use_mmap=True,
                path=edge_ends_path,
                file_endian=out_file_endian,
                writable=True,
            )
            out_timestamps = LargeInt32Array.create(
                total_train,
                use_mmap=True,
                path=timestamps_path,
                file_endian=out_file_endian,
                writable=True,
            )
            out_edge_ends_arr = out_edge_ends.numpy()
            out_timestamps_arr = out_timestamps.numpy()

        val_pairs = np.empty((total_val, 2), dtype=np.int64)

        train_written, val_written = _pass2_emit(
            starts=starts,
            edge_ends=edge_ends,
            timestamps=timestamps,
            threshold=threshold,
            ties=ties,
            num_nodes=num_nodes,
            total_edges=total_edges,
            node_chunk_size=node_chunk_size,
            train_count_per_node=train_count_per_node,
            out_edge_ends_arr=out_edge_ends_arr,
            out_timestamps_arr=out_timestamps_arr,
            val_pairs=val_pairs,
            show_progress=show_progress,
        )

        if train_written != total_train:
            raise RuntimeError(f"Train edge count mismatch: emitted={train_written} expected={total_train}")
        if val_written != total_val:
            raise RuntimeError(f"Val edge count mismatch: emitted={val_written} expected={total_val}")

        if out_edge_ends is not None:
            out_edge_ends.flush()
            out_edge_ends.close()
        if out_timestamps is not None:
            out_timestamps.flush()
            out_timestamps.close()

    if val_pairs.shape[0] > 0:
        before_dedup = val_pairs.shape[0]
        val_pairs = np.unique(val_pairs, axis=0)
        deduped = before_dedup - val_pairs.shape[0]
        if deduped > 0:
            LOGGER.info(
                "Deduplicated val pairs: %s → %s (%s duplicates removed)",
                format_int(before_dedup),
                format_int(val_pairs.shape[0]),
                format_int(deduped),
            )
    LOGGER.info("Saving val pairs (%s rows) to %s", format_int(val_pairs.shape[0]), val_out_pairs)
    np.save(val_out_pairs, val_pairs)

    val_neg_count = 0
    if val_out_neg is not None and total_train > 0:
        val_out_neg.parent.mkdir(parents=True, exist_ok=True)
        val_neg = _generate_val_negatives(
            train_graph_dir=train_out_dir,
            val_pairs=val_pairs,
            n_walks_per_src=val_neg_walks_per_src,
            seed=val_neg_seed,
            use_mmap=use_mmap,
            file_endian=out_file_endian,
            allow_non_native=allow_non_native,
            chunk_bytes=chunk_bytes,
            show_progress=show_progress,
        )
        val_neg_count = int(val_neg.shape[0])
        LOGGER.info("Saving val negatives (%s rows) to %s", format_int(val_neg_count), val_out_neg)
        np.save(val_out_neg, val_neg)
    elif val_out_neg is not None:
        LOGGER.warning("Skipping val negatives because train graph is empty")

    input_deleted = False
    if delete_input_after_success:
        preserve = [train_out_dir, val_out_pairs.parent]
        if val_out_neg is not None:
            preserve.append(val_out_neg.parent)
        delete_dir_safely(graph_dir, preserve_paths=preserve)
        input_deleted = True

    elapsed = time.time() - t0
    LOGGER.info(
        "Done. nodes=%s train_edges=%s val_pos=%s val_neg=%s elapsed=%s",
        format_int(num_nodes),
        format_int(total_train),
        format_int(total_val),
        format_int(val_neg_count),
        format_seconds(elapsed),
    )
    return {
        "num_nodes": num_nodes,
        "total_edges": total_edges,
        "train_edges": total_train,
        "val_edges": total_val,
        "val_edges_requested": int(val_edges),
        "ties": ties,
        "threshold": int(threshold),
        "strictly_above_threshold": int(strictly_above),
        "at_threshold": int(at_threshold),
        "val_neg_edges": val_neg_count,
        "input_deleted": input_deleted,
        "elapsed_seconds": elapsed,
    }


def _partition_eval_into_val_and_test(
    *,
    eval_pairs: np.ndarray,
    eval_ts: np.ndarray,
    test_edges: int,
    ties: TiesPolicy,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split an in-memory (eval_pairs, eval_ts) buffer into val and test.

    The eval buffer is assumed to already be the (val+test) recency bucket,
    selected by `_radix_select_kth_largest` with k = val_edges + test_edges
    and the same `ties` policy. Here we apply the same ties policy a second
    time, but at the *inner* boundary T_test that separates the K_test most
    recent eval entries from the rest.

    Returns (val_pairs, test_pairs, T_test_or_minint).
    """
    total_eval = int(eval_pairs.shape[0])
    if test_edges <= 0 or total_eval == 0:
        return eval_pairs, np.empty((0, 2), dtype=np.int64), np.iinfo(np.int32).min
    if test_edges >= total_eval:
        return np.empty((0, 2), dtype=np.int64), eval_pairs, np.iinfo(np.int32).min

    # T_test* is the K_test-th largest timestamp inside the eval bucket.
    # np.partition is O(N); we only need the value at one index, no full sort.
    kth_smallest_index = total_eval - int(test_edges)  # ascending position of the K_test-th largest
    partitioned = np.partition(eval_ts, kth_smallest_index)
    t_test_star = np.int32(partitioned[kth_smallest_index])

    if ties == "exclude":
        test_mask = eval_ts > t_test_star
    else:  # "include"
        test_mask = eval_ts >= t_test_star

    test_pairs = eval_pairs[test_mask]
    val_pairs = eval_pairs[~test_mask]
    return val_pairs, test_pairs, int(t_test_star)


def temporal_split_3way_graph_csr(
    *,
    graph_dir: Path,
    train_out_dir: Path,
    val_out_pairs: Path,
    test_out_pairs: Path,
    val_edges: int,
    test_edges: int,
    ties: TiesPolicy,
    use_mmap: bool,
    file_endian: Endian,
    out_file_endian: Endian,
    allow_non_native: bool,
    chunk_bytes: int,
    node_chunk_size: int,
    show_progress: bool,
    val_out_neg: Path | None,
    test_out_neg: Path | None,
    val_neg_walks_per_src: int,
    val_neg_seed: int,
    test_neg_walks_per_src: int,
    test_neg_seed: int,
    radix_chunk_size: int = _RADIX_CHUNK_DEFAULT,
    delete_input_after_success: bool = False,
    use_edge_hash_as_timestamps: bool = False,
) -> dict:
    """Single-pass 3-way temporal split: train + val + test in one go.

    Compared to running `temporal_split_graph_csr` twice (carve test, then
    carve val from the remainder), this version writes the train CSR only
    once and never produces an intermediate file. The (val ∪ test) bucket
    is collected in memory during pass 2 along with per-edge timestamps,
    then partitioned by `np.partition` on the inner threshold T_test (the
    K_test-th largest timestamp inside the bucket).

    Semantics: K_test most-recent edges → test, the next K_val by recency →
    val, the rest → train. The same `ties` policy is applied at both
    boundaries (T_val between train and eval; T_test inside eval).
    "exclude" pushes ties down: eval = train's complement ≤ K_test + K_val,
    test ≤ K_test, val = (eval - test) — i.e. val absorbs the "exclude
    slack" from the inner T_test boundary and may exceed K_val when many
    edges sit exactly on T_test. "include" mirrors: eval ≥ K_test + K_val,
    test ≥ K_test, val = (eval - test). val's count is therefore best-
    effort relative to K_val; only K_test and the total eval target are
    bounded by the policy.

    Both val and test negatives are 2-hop random walks over the FINAL train
    CSR (i.e. neither val nor test edges count as direct neighbours), with
    independent RNG seeds.
    """
    if val_edges <= 0:
        raise ValueError(f"val_edges must be positive, got {val_edges}")
    if test_edges <= 0:
        raise ValueError(f"test_edges must be positive, got {test_edges}")
    if ties not in ("exclude", "include"):
        raise ValueError(f"ties must be 'exclude' or 'include', got {ties!r}")

    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_pairs.parent.mkdir(parents=True, exist_ok=True)
    test_out_pairs.parent.mkdir(parents=True, exist_ok=True)
    edge_starts_path = train_out_dir / GraphCSRSerializer.EDGE_STARTS
    edge_ends_path = train_out_dir / GraphCSRSerializer.EDGE_ENDS
    timestamps_path = train_out_dir / GraphCSRSerializer.TIMESTAMPS

    eval_target = int(val_edges) + int(test_edges)

    t0 = time.time()
    LOGGER.info("Opening input GraphCSR from %s", graph_dir)
    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        starts = graph.edge_starts.numpy()
        edge_ends = graph.edge_ends.numpy()
        num_nodes = int(starts.size)
        total_edges = int(graph.edge_ends.size)

        if use_edge_hash_as_timestamps:
            LOGGER.info(
                "Synthetic-ts mode: ignoring on-disk timestamps (present=%s); "
                "computing hash(min(u,v), max(u,v)) for %s edges",
                graph.timestamps is not None,
                format_int(total_edges),
            )
            timestamps = _build_hash_timestamps(
                starts=starts,
                edge_ends=edge_ends,
                num_nodes=num_nodes,
                total_edges=total_edges,
                node_chunk_size=node_chunk_size,
                show_progress=show_progress,
            )
        else:
            if graph.timestamps is None:
                raise ValueError(
                    "Input graph has no timestamps; temporal split is not possible. "
                    "Pass use_edge_hash_as_timestamps=True to synthesize them "
                    "from edge identities."
                )
            timestamps = graph.timestamps.numpy()

        LOGGER.info(
            "Graph opened: nodes=%s edges=%s val_edges=%s test_edges=%s ties=%s synthetic_ts=%s",
            format_int(num_nodes),
            format_int(total_edges),
            format_int(val_edges),
            format_int(test_edges),
            ties,
            use_edge_hash_as_timestamps,
        )
        if eval_target > total_edges:
            raise ValueError(
                f"val_edges+test_edges={eval_target} exceeds total edges {total_edges}"
            )

        # Outer threshold T_val: separates train from (val ∪ test).
        threshold_val, strictly_above_val, at_threshold_val = _radix_select_kth_largest(
            timestamps,
            k=eval_target,
            chunk_size=radix_chunk_size,
            show_progress=show_progress,
        )
        if ties == "exclude":
            total_eval = strictly_above_val
        else:
            total_eval = strictly_above_val + at_threshold_val
        total_train = total_edges - total_eval
        LOGGER.info(
            "Threshold T_val=%s: strictly_above=%s at_threshold=%s → eval=%s (requested %s) train=%s",
            threshold_val,
            format_int(strictly_above_val),
            format_int(at_threshold_val),
            format_int(total_eval),
            format_int(eval_target),
            format_int(total_train),
        )

        train_count_per_node = _pass1_count_train(
            starts=starts,
            timestamps=timestamps,
            threshold=threshold_val,
            ties=ties,
            num_nodes=num_nodes,
            total_edges=total_edges,
            node_chunk_size=node_chunk_size,
            show_progress=show_progress,
        )
        counted_train = int(train_count_per_node.sum())
        if counted_train != total_train:
            raise RuntimeError(
                f"Train count mismatch between radix-select and pass 1: "
                f"select={total_train} pass1={counted_train}"
            )

        new_edge_starts = np.zeros(num_nodes, dtype=np.int64)
        if num_nodes > 1:
            np.cumsum(train_count_per_node[:-1], out=new_edge_starts[1:])

        LOGGER.info("Writing train edge_starts to %s", edge_starts_path)
        out_starts = raw_int64_memmap(edge_starts_path, num_nodes, out_file_endian)
        out_starts[:] = new_edge_starts.astype(out_starts.dtype, copy=False)
        out_starts.flush()
        del out_starts

        if total_train == 0:
            timestamps_path.write_bytes(b"")
            edge_ends_path.write_bytes(b"")
            LOGGER.warning("No edges left in train split; writing empty edge_ends/timestamps")
            out_edge_ends = None
            out_timestamps = None
            out_edge_ends_arr = np.empty((0,), dtype=np.int32)
            out_timestamps_arr = np.empty((0,), dtype=np.int32)
        else:
            out_edge_ends = LargeInt32Array.create(
                total_train,
                use_mmap=True,
                path=edge_ends_path,
                file_endian=out_file_endian,
                writable=True,
            )
            out_timestamps = LargeInt32Array.create(
                total_train,
                use_mmap=True,
                path=timestamps_path,
                file_endian=out_file_endian,
                writable=True,
            )
            out_edge_ends_arr = out_edge_ends.numpy()
            out_timestamps_arr = out_timestamps.numpy()

        # The eval bucket is collected in memory: ~ (val_edges+test_edges)*12 B
        # for OGB sizes this is well under 1 GB.
        eval_pairs = np.empty((total_eval, 2), dtype=np.int64)
        eval_ts = np.empty((total_eval,), dtype=np.int32)

        train_written, eval_written = _pass2_emit(
            starts=starts,
            edge_ends=edge_ends,
            timestamps=timestamps,
            threshold=threshold_val,
            ties=ties,
            num_nodes=num_nodes,
            total_edges=total_edges,
            node_chunk_size=node_chunk_size,
            train_count_per_node=train_count_per_node,
            out_edge_ends_arr=out_edge_ends_arr,
            out_timestamps_arr=out_timestamps_arr,
            val_pairs=eval_pairs,
            val_pairs_ts=eval_ts,
            show_progress=show_progress,
        )
        if train_written != total_train:
            raise RuntimeError(f"Train edge count mismatch: emitted={train_written} expected={total_train}")
        if eval_written != total_eval:
            raise RuntimeError(f"Eval edge count mismatch: emitted={eval_written} expected={total_eval}")

        if out_edge_ends is not None:
            out_edge_ends.flush()
            out_edge_ends.close()
        if out_timestamps is not None:
            out_timestamps.flush()
            out_timestamps.close()

    # Inner partition: split eval into val and test by T_test (the K_test-th
    # largest ts inside the eval bucket). Same ties policy applies.
    val_pairs, test_pairs, t_test_star = _partition_eval_into_val_and_test(
        eval_pairs=eval_pairs,
        eval_ts=eval_ts,
        test_edges=test_edges,
        ties=ties,
    )
    LOGGER.info(
        "Inner threshold T_test=%s: val=%s (requested %s) test=%s (requested %s)",
        t_test_star,
        format_int(val_pairs.shape[0]),
        format_int(val_edges),
        format_int(test_pairs.shape[0]),
        format_int(test_edges),
    )
    if val_pairs.shape[0] == 0 and val_edges > 0:
        LOGGER.warning(
            "Val bucket is empty after partition (likely all eval edges fell on the same ts under 'exclude' ties)."
        )
    if test_pairs.shape[0] == 0 and test_edges > 0:
        LOGGER.warning("Test bucket is empty after partition.")
    del eval_pairs, eval_ts

    def _dedup_and_save(pairs: np.ndarray, out_path: Path, name: str) -> np.ndarray:
        if pairs.shape[0] > 0:
            before = pairs.shape[0]
            pairs = np.unique(pairs, axis=0)
            if before - pairs.shape[0] > 0:
                LOGGER.info(
                    "Deduplicated %s pairs: %s → %s (%s duplicates removed)",
                    name,
                    format_int(before),
                    format_int(pairs.shape[0]),
                    format_int(before - pairs.shape[0]),
                )
        LOGGER.info("Saving %s pairs (%s rows) to %s", name, format_int(pairs.shape[0]), out_path)
        np.save(out_path, pairs)
        return pairs

    val_pairs = _dedup_and_save(val_pairs, val_out_pairs, "val")
    test_pairs = _dedup_and_save(test_pairs, test_out_pairs, "test")

    val_neg_count = 0
    test_neg_count = 0
    if total_train > 0:
        if val_out_neg is not None:
            val_out_neg.parent.mkdir(parents=True, exist_ok=True)
            val_neg = _generate_val_negatives(
                train_graph_dir=train_out_dir,
                val_pairs=val_pairs,
                n_walks_per_src=val_neg_walks_per_src,
                seed=val_neg_seed,
                use_mmap=use_mmap,
                file_endian=out_file_endian,
                allow_non_native=allow_non_native,
                chunk_bytes=chunk_bytes,
                show_progress=show_progress,
                desc="val negatives",
            )
            val_neg_count = int(val_neg.shape[0])
            LOGGER.info("Saving val negatives (%s rows) to %s", format_int(val_neg_count), val_out_neg)
            np.save(val_out_neg, val_neg)
        if test_out_neg is not None:
            test_out_neg.parent.mkdir(parents=True, exist_ok=True)
            test_neg = _generate_val_negatives(
                train_graph_dir=train_out_dir,
                val_pairs=test_pairs,
                n_walks_per_src=test_neg_walks_per_src,
                seed=test_neg_seed,
                use_mmap=use_mmap,
                file_endian=out_file_endian,
                allow_non_native=allow_non_native,
                chunk_bytes=chunk_bytes,
                show_progress=show_progress,
                desc="test negatives",
            )
            test_neg_count = int(test_neg.shape[0])
            LOGGER.info("Saving test negatives (%s rows) to %s", format_int(test_neg_count), test_out_neg)
            np.save(test_out_neg, test_neg)
    else:
        if val_out_neg is not None or test_out_neg is not None:
            LOGGER.warning("Skipping eval negatives because train graph is empty")

    input_deleted = False
    if delete_input_after_success:
        preserve = [train_out_dir, val_out_pairs.parent, test_out_pairs.parent]
        if val_out_neg is not None:
            preserve.append(val_out_neg.parent)
        if test_out_neg is not None:
            preserve.append(test_out_neg.parent)
        delete_dir_safely(graph_dir, preserve_paths=preserve)
        input_deleted = True

    elapsed = time.time() - t0
    LOGGER.info(
        "Done. nodes=%s train_edges=%s val_pos=%s val_neg=%s test_pos=%s test_neg=%s elapsed=%s",
        format_int(num_nodes),
        format_int(total_train),
        format_int(val_pairs.shape[0]),
        format_int(val_neg_count),
        format_int(test_pairs.shape[0]),
        format_int(test_neg_count),
        format_seconds(elapsed),
    )
    return {
        "num_nodes": num_nodes,
        "total_edges": total_edges,
        "train_edges": total_train,
        "val_edges": int(val_pairs.shape[0]),
        "val_edges_requested": int(val_edges),
        "test_edges": int(test_pairs.shape[0]),
        "test_edges_requested": int(test_edges),
        "ties": ties,
        "threshold_val": int(threshold_val),
        "threshold_test": int(t_test_star),
        "val_neg_edges": val_neg_count,
        "test_neg_edges": test_neg_count,
        "input_deleted": input_deleted,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Temporal split of a GraphCSR by edge timestamp. "
            "The K most-recent edges (by int32 timestamp value) are dumped "
            "as a validation pair list ([N, 2] int64 .npy); the rest stay "
            "in the train GraphCSR."
        )
    )
    parser.add_argument("--graph-dir", type=str, required=True, help="Input GraphCSR folder")
    parser.add_argument("--train-out-dir", type=str, required=True, help="Output train GraphCSR folder")
    parser.add_argument("--val-out-pairs", type=str, required=True, help="Output val pairs .npy path")
    parser.add_argument(
        "--val-edges",
        type=int,
        required=True,
        help="Number of most-recent edges (largest int32 timestamp values) to put in val.",
    )
    parser.add_argument(
        "--ties",
        choices=["exclude", "include"],
        default="exclude",
        help=(
            "Policy for edges with ts == threshold. 'exclude' (default): such "
            "edges go to train, so val size ≤ --val-edges. 'include': they "
            "go to val, so val size ≥ --val-edges."
        ),
    )
    parser.add_argument("--use-mmap", action="store_true", help="Use mmap when reading input graph")
    parser.add_argument("--file-endian", choices=["big", "little"], default="big", help="Input endianness")
    parser.add_argument(
        "--out-file-endian",
        choices=["big", "little"],
        default=None,
        help="Output endianness (default: same as input)",
    )
    parser.add_argument("--allow-non-native", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--node-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--radix-chunk-size",
        type=int,
        default=_RADIX_CHUNK_DEFAULT,
        help="Number of int32 timestamps per radix-pass chunk (sequential scan).",
    )
    parser.add_argument(
        "--val-out-neg",
        type=str,
        default=None,
        help="Output path for val negatives .npy (default: <val-out-pairs stem>_neg.npy).",
    )
    parser.add_argument(
        "--skip-val-neg",
        action="store_true",
        help="Do not generate validation negatives.",
    )
    parser.add_argument(
        "--val-neg-walks-per-src",
        type=int,
        default=100,
        help="Number of length-2 random walks to attempt per unique val src.",
    )
    parser.add_argument("--val-neg-seed", type=int, default=12345)
    parser.add_argument(
        "--use-edge-hash-as-timestamps",
        action="store_true",
        help=(
            "Treat the graph as untimestamped: synthesize per-edge ts as "
            "hash(min(u,v), max(u,v)) (same for both directions). Required "
            "when the input GraphCSR has an empty timestamps.bin; off by "
            "default so a missing ts file fails loudly instead of silently "
            "producing a pseudo-random split."
        ),
    )
    parser.add_argument(
        "--delete-input-after-success",
        action="store_true",
        help=(
            "DESTRUCTIVE: after a successful split, recursively remove --graph-dir "
            "to reclaim disk. Refuses if --graph-dir overlaps with any output path."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)

    out_file_endian = args.out_file_endian or args.file_endian

    val_out_pairs = Path(args.val_out_pairs)
    if args.skip_val_neg:
        val_out_neg: Path | None = None
    elif args.val_out_neg is not None:
        val_out_neg = Path(args.val_out_neg)
    else:
        val_out_neg = _default_val_neg_path(val_out_pairs)

    temporal_split_graph_csr(
        graph_dir=Path(args.graph_dir),
        train_out_dir=Path(args.train_out_dir),
        val_out_pairs=val_out_pairs,
        val_edges=args.val_edges,
        ties=args.ties,
        use_mmap=args.use_mmap,
        file_endian=args.file_endian,
        out_file_endian=out_file_endian,
        allow_non_native=args.allow_non_native,
        chunk_bytes=args.chunk_bytes,
        node_chunk_size=args.node_chunk_size,
        show_progress=not args.no_progress,
        val_out_neg=val_out_neg,
        val_neg_walks_per_src=args.val_neg_walks_per_src,
        val_neg_seed=args.val_neg_seed,
        radix_chunk_size=args.radix_chunk_size,
        delete_input_after_success=args.delete_input_after_success,
        use_edge_hash_as_timestamps=args.use_edge_hash_as_timestamps,
    )


if __name__ == "__main__":
    main()

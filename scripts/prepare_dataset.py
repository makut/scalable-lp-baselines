"""End-to-end graph preprocessing for link prediction.

Given a full GraphCSR with int32 timestamps, this script produces the unified
dataset layout consumed by all four training methods (MF / GraphSAGE / SEAL /
VERSE):

    <out_root>/
      train_csr/             # train GraphCSR (both directions, sorted, with timestamps)
      train_pairs_csr/       # canonical pairs CSR (only u < v) for iterating positives
      valid_edge.npy         # [N, 2] int64 — val positives
      valid_edge_neg.npy     # [M, 2] int64 — val negatives (2-hop walks over train graph)
      test_edge.npy          # [N, 2] int64 — test positives  (only with --test-edges)
      test_edge_neg.npy      # [M, 2] int64 — test negatives  (only with --test-edges)

Timestamps are treated as opaque int32 values: only their relative order is
meaningful. No base, unit, or epoch is assumed.

Splitting modes:
  * `--val-edges K` only (default): 2-way split. K most-recent edges → val;
    the rest → train.
  * `--val-edges K_val --test-edges K_test`: 3-way temporal split, done in
    a single pass — K_test most-recent edges → test; the next K_val by
    recency → val; the rest → train. Both eval positives go through one
    `_pass2_emit` (the train CSR is written exactly once); the val/test
    partition is then a cheap `np.partition` on the in-memory eval bucket.
    Val and test negatives are sampled via 2-hop random walks over the
    final train graph (the one the model actually sees), with independent
    RNG seeds.

Test pos/neg without `--test-edges` are not produced by this script — the
user supplies them externally.

Untimestamped graphs:
  By default a missing `timestamps.bin` fails — temporal split is not
  meaningful without an edge ordering. With `--use-edge-hash-as-timestamps`
  the script synthesizes ts = hash(min(u,v), max(u,v)) per edge (same for
  both directions of an undirected edge). The split is then deterministic
  in (u,v) but carries no real temporal information; the flag is required
  so missing timestamps fail loudly by default.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from graph_csr.io_utils import Endian
from scripts._common import configure_logging, delete_dir_safely
from scripts.convert_graph_csr_to_upper import convert_graph_to_upper
from scripts.temporal_split_graph_csr import (
    _RADIX_CHUNK_DEFAULT,
    _default_val_neg_path,
    temporal_split_3way_graph_csr,
    temporal_split_graph_csr,
)

LOGGER = logging.getLogger("prepare_dataset")


def prepare_dataset(
    *,
    graph_dir: Path,
    out_root: Path,
    val_edges: int,
    ties: str,
    use_mmap: bool,
    file_endian: Endian,
    out_file_endian: Endian,
    allow_non_native: bool,
    chunk_bytes: int,
    node_chunk_size: int,
    radix_chunk_size: int,
    val_neg_walks_per_src: int,
    val_neg_seed: int,
    skip_val_neg: bool,
    use_numba_for_pairs: bool,
    show_progress: bool,
    test_edges: Optional[int] = None,
    test_neg_walks_per_src: int = 100,
    test_neg_seed: int = 23456,
    skip_test_neg: bool = False,
    delete_input_after_success: bool = False,
    use_edge_hash_as_timestamps: bool = False,
) -> dict:
    if test_edges is not None and test_edges <= 0:
        raise ValueError(f"test_edges must be positive when provided, got {test_edges}")

    out_root.mkdir(parents=True, exist_ok=True)
    train_dir = out_root / "train_csr"
    pairs_dir = out_root / "train_pairs_csr"
    val_pos_path = out_root / "valid_edge.npy"
    val_neg_path: Optional[Path] = None if skip_val_neg else _default_val_neg_path(val_pos_path)
    test_pos_path = out_root / "test_edge.npy"
    test_neg_path: Optional[Path] = None if (test_edges is None or skip_test_neg) else _default_val_neg_path(test_pos_path)

    if test_edges is None:
        LOGGER.info("Step 1/2: temporal val split → %s", train_dir)
        split_stats = temporal_split_graph_csr(
            graph_dir=graph_dir,
            train_out_dir=train_dir,
            val_out_pairs=val_pos_path,
            val_edges=val_edges,
            ties=ties,  # type: ignore[arg-type]
            use_mmap=use_mmap,
            file_endian=file_endian,
            out_file_endian=out_file_endian,
            allow_non_native=allow_non_native,
            chunk_bytes=chunk_bytes,
            node_chunk_size=node_chunk_size,
            show_progress=show_progress,
            val_out_neg=val_neg_path,
            val_neg_walks_per_src=val_neg_walks_per_src,
            val_neg_seed=val_neg_seed,
            radix_chunk_size=radix_chunk_size,
            use_edge_hash_as_timestamps=use_edge_hash_as_timestamps,
        )
    else:
        LOGGER.info(
            "Step 1/2: single-pass 3-way temporal split (val=%d, test=%d) → %s",
            val_edges, test_edges, train_dir,
        )
        split_stats = temporal_split_3way_graph_csr(
            graph_dir=graph_dir,
            train_out_dir=train_dir,
            val_out_pairs=val_pos_path,
            test_out_pairs=test_pos_path,
            val_edges=val_edges,
            test_edges=test_edges,
            ties=ties,  # type: ignore[arg-type]
            use_mmap=use_mmap,
            file_endian=file_endian,
            out_file_endian=out_file_endian,
            allow_non_native=allow_non_native,
            chunk_bytes=chunk_bytes,
            node_chunk_size=node_chunk_size,
            show_progress=show_progress,
            val_out_neg=val_neg_path,
            test_out_neg=test_neg_path,
            val_neg_walks_per_src=val_neg_walks_per_src,
            val_neg_seed=val_neg_seed,
            test_neg_walks_per_src=test_neg_walks_per_src,
            test_neg_seed=test_neg_seed,
            radix_chunk_size=radix_chunk_size,
            use_edge_hash_as_timestamps=use_edge_hash_as_timestamps,
        )

    LOGGER.info("Step 2/2: building canonical pairs CSR (u < v) → %s", pairs_dir)
    pairs_stats = convert_graph_to_upper(
        graph_dir=train_dir,
        out_dir=pairs_dir,
        use_mmap=use_mmap,
        file_endian=out_file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        node_chunk_size=node_chunk_size,
        use_numba=use_numba_for_pairs,
        log_every_chunks=10,
        show_progress=show_progress,
    )

    input_deleted = False
    if delete_input_after_success:
        delete_dir_safely(graph_dir, preserve_paths=[out_root])
        input_deleted = True

    LOGGER.info(
        "Done. train_csr=%s train_pairs_csr=%s val_pos=%s val_neg=%s test_pos=%s test_neg=%s input_deleted=%s",
        train_dir,
        pairs_dir,
        val_pos_path,
        val_neg_path,
        test_pos_path if test_edges is not None else None,
        test_neg_path,
        input_deleted,
    )
    return {
        "train_csr_dir": str(train_dir),
        "train_pairs_csr_dir": str(pairs_dir),
        "val_pos_path": str(val_pos_path),
        "val_neg_path": None if val_neg_path is None else str(val_neg_path),
        "test_pos_path": None if test_edges is None else str(test_pos_path),
        "test_neg_path": None if test_neg_path is None else str(test_neg_path),
        "input_deleted": input_deleted,
        "split": split_stats,
        "pairs": pairs_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end preprocessing: temporal train/val(/test) split (K "
            "most-recent edges → val; an extra K_test on top → test) + "
            "canonical pairs CSR (u < v). Produces the unified dataset layout "
            "consumed by all link-prediction methods in this repo."
        )
    )
    parser.add_argument("--graph-dir", type=str, required=True, help="Input GraphCSR folder")
    parser.add_argument("--out-root", type=str, required=True, help="Output root directory")
    parser.add_argument(
        "--val-edges",
        type=int,
        required=True,
        help="Number of most-recent edges (largest int32 timestamp values) to put in val.",
    )
    parser.add_argument(
        "--test-edges",
        type=int,
        default=None,
        help=(
            "If set, do a 3-way temporal split: this many MOST-recent edges "
            "go to test, the NEXT --val-edges go to val, the rest stay in "
            "train. If omitted, only val is carved out (test files are not "
            "produced)."
        ),
    )
    parser.add_argument(
        "--ties",
        choices=["exclude", "include"],
        default="exclude",
        help=(
            "Policy for edges with ts == threshold. 'exclude' (default): such "
            "edges go to the lower-priority bucket (train in the 2-way mode; "
            "train for the val cut, val for the test cut in 3-way mode), so "
            "the eval bucket size ≤ requested. 'include': they go up, so size "
            "≥ requested."
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
        "--val-neg-walks-per-src",
        type=int,
        default=100,
        help="Number of length-2 random walks to attempt per unique val src.",
    )
    parser.add_argument("--val-neg-seed", type=int, default=12345)
    parser.add_argument("--skip-val-neg", action="store_true", help="Do not generate validation negatives.")
    parser.add_argument(
        "--test-neg-walks-per-src",
        type=int,
        default=100,
        help="Number of length-2 random walks to attempt per unique test src (only with --test-edges).",
    )
    parser.add_argument("--test-neg-seed", type=int, default=23456)
    parser.add_argument("--skip-test-neg", action="store_true", help="Do not generate test negatives (only with --test-edges).")
    parser.add_argument(
        "--use-edge-hash-as-timestamps",
        action="store_true",
        help=(
            "Use this when the input GraphCSR has no real timestamps. "
            "Instead of the on-disk ts, the split synthesizes one per edge "
            "as hash(min(u,v), max(u,v)) — same value for both directions, "
            "covers the full int32 range so radix-select sees a uniform "
            "distribution. The resulting split is deterministic in (u,v) "
            "but has no real temporal meaning; this flag must be set "
            "explicitly so a missing timestamps.bin fails loudly by default."
        ),
    )
    parser.add_argument(
        "--no-numba-for-pairs",
        action="store_true",
        help="Disable numba kernels in upper-triangle conversion (slower fallback).",
    )
    parser.add_argument(
        "--delete-input-after-success",
        action="store_true",
        help=(
            "DESTRUCTIVE: after the full pipeline (split + pairs) succeeds, "
            "recursively remove --graph-dir to reclaim disk. Refuses if "
            "--graph-dir overlaps with --out-root."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)

    out_file_endian: Endian = args.out_file_endian or args.file_endian

    prepare_dataset(
        graph_dir=Path(args.graph_dir),
        out_root=Path(args.out_root),
        val_edges=args.val_edges,
        test_edges=args.test_edges,
        ties=args.ties,
        use_mmap=args.use_mmap,
        file_endian=args.file_endian,
        out_file_endian=out_file_endian,
        allow_non_native=args.allow_non_native,
        chunk_bytes=args.chunk_bytes,
        node_chunk_size=args.node_chunk_size,
        radix_chunk_size=args.radix_chunk_size,
        val_neg_walks_per_src=args.val_neg_walks_per_src,
        val_neg_seed=args.val_neg_seed,
        skip_val_neg=args.skip_val_neg,
        test_neg_walks_per_src=args.test_neg_walks_per_src,
        test_neg_seed=args.test_neg_seed,
        skip_test_neg=args.skip_test_neg,
        use_numba_for_pairs=not args.no_numba_for_pairs,
        show_progress=not args.no_progress,
        delete_input_after_success=args.delete_input_after_success,
        use_edge_hash_as_timestamps=args.use_edge_hash_as_timestamps,
    )


if __name__ == "__main__":
    main()

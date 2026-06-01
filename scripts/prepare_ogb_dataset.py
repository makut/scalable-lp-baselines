"""Convert OGB datasets to the LPP GraphCSR format.

Supported:
  * ogbl-citation2     — link prediction (provides train/valid/test edge splits)
  * ogbn-papers100M    — node classification (no edge splits; user must split)

Output layout (inside `--out-dir`):
  graph_csr/                  GraphCSR — symmetrized full graph with per-edge
                              int32 timestamps (derived from node_year).
                              `edge_starts.bin`, `edge_ends.bin`, `timestamps.bin`.
  num_nodes.txt               Plain text: total number of nodes.

  For ogbl-citation2 the OGB-provided splits are also dumped:
    valid_edge.npy            [Nv, 2]   int64 — positive (src, dst) pairs.
    valid_edge_neg.npy        [Nv*K, 2] int64 — 1000 negatives per src, flattened.
    test_edge.npy             [Nt, 2]   int64
    test_edge_neg.npy         [Nt*K, 2] int64

GraphCSR conventions:
  * `edge_starts` size == num_nodes (NOT num_nodes+1; the last row implicitly
    ends at `edge_ends.size`).
  * The graph is symmetric: each undirected edge (u,v) appears twice — once
    as (u,v) and once as (v,u) — so message-passing / sampling code reads
    both directions.
  * Per-row neighbours are sorted ascending (so the downstream
    `_has_edge_sorted` works).
  * `timestamps[i]` is the int32 timestamp of the i-th edge in `edge_ends`.
    For both copies of an undirected edge the value is the same:
    `max(node_year[u], node_year[v])` — the year by which the citation
    necessarily existed. Years missing in OGB (rare) are replaced by 0.
    Only the relative order is meaningful.

Test-split status by dataset:
  * ogbl-citation2 has built-in train/valid/test edge splits provided as
    `(source_node, target_node, target_node_neg[1000])` — these are dumped
    verbatim (negatives are flattened to `[N*1000, 2]`). No extra work needed.
  * ogbn-papers100M does NOT ship edge splits. After this script you have
    only the full GraphCSR. To produce val/test, either:
       (a) run `scripts/prepare_dataset.py --val-edges K_val
           --test-edges K_test` to carve out a three-way temporal split;
       (b) sample your own held-out test edges before running a two-way split
           with `scripts/prepare_dataset.py --val-edges K_val`.
    For test negatives, `dataset_utils/negative_sampling.sample_two_hop_unique_dsts`
    (same routine used by `prepare_dataset.py` for val negatives) is the
    natural choice.

Usage:
  pip install ogb

  python -m scripts.prepare_ogb_dataset \\
      --dataset ogbl-citation2 \\
      --out-dir /data/lpp/citation2 \\
      --ogb-root /data/ogb

  python -m scripts.prepare_ogb_dataset \\
      --dataset ogbn-papers100M \\
      --out-dir /data/lpp/papers100M \\
      --ogb-root /data/ogb

Memory:
  ogbl-citation2 (~30M directed edges) is comfortable on any workstation.
  ogbn-papers100M (~1.6B directed → ~3.2B symmetric) peaks at ~80 GB RAM
  during the in-memory sort step (int64 keys + ts + reorder buffer). Run
  on a machine with ≥128 GB RAM, or adapt `_build_csr_with_timestamps`
  to spill to disk (np.memmap + chunked radix sort).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_csr.io_utils import Endian
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.large_int64_raw import LargeInt64RawArray
from graph_csr.graph import GraphCSR
from graph_csr.serializer import GraphCSRSerializer
from scripts._common import configure_logging, format_int, format_seconds

LOGGER = logging.getLogger("prepare_ogb_dataset")

_INT32_MAX = np.iinfo(np.int32).max


def _load_ogbl_citation2(ogb_root: Path) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    """Return (edge_index[2, E] int32, node_year[N] int32, num_nodes, edge_split).

    `edge_index` covers only train edges (OGB removes val/test from the main
    graph). `edge_split` is the dict returned by `dataset.get_edge_split()`.
    """
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-citation2", root=str(ogb_root))
    graph, _ = dataset[0], None  # link-pred dataset has no label
    edge_index = np.asarray(graph["edge_index"])  # [2, E] int64
    num_nodes = int(graph["num_nodes"])
    node_year_raw = np.asarray(graph["node_year"]).reshape(-1)  # [N]
    edge_split = dataset.get_edge_split()

    edge_index = edge_index.astype(np.int32, copy=False)
    node_year = _normalize_years(node_year_raw)
    return edge_index, node_year, num_nodes, edge_split


def _load_ogbn_papers100m(ogb_root: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (edge_index[2, E] int32, node_year[N] int32, num_nodes)."""
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(ogb_root))
    graph, _ = dataset[0]
    edge_index = np.asarray(graph["edge_index"])  # [2, E] int64 (citation u→v)
    num_nodes = int(graph["num_nodes"])
    node_year_raw = np.asarray(graph["node_year"]).reshape(-1)

    if num_nodes > _INT32_MAX:
        raise ValueError(
            f"papers100M has {num_nodes} nodes which exceeds int32 range "
            f"({_INT32_MAX}); LPP CSR uses int32 node ids."
        )
    edge_index = edge_index.astype(np.int32, copy=False)
    node_year = _normalize_years(node_year_raw)
    return edge_index, node_year, num_nodes


def _normalize_years(raw: np.ndarray) -> np.ndarray:
    """Coerce node_year to int32 with safe fallback for missing values."""
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.floating):
        nan_mask = np.isnan(arr)
        arr = np.where(nan_mask, 0, arr)
    arr = arr.astype(np.int64, copy=False)
    arr = np.clip(arr, 0, _INT32_MAX)
    return arr.astype(np.int32, copy=False)


def _symmetrize(edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) arrays for the symmetric directed view.

    Drops self-loops and exact duplicates while remaining streaming-friendly:
    we sort once on the encoded edge key, then unique-by-diff. Memory peak is
    ~3× the symmetric edge count in int64.
    """
    src_dir = edge_index[0]
    dst_dir = edge_index[1]

    # Build the symmetric edge list: original + reversed.
    src = np.concatenate([src_dir, dst_dir])
    dst = np.concatenate([dst_dir, src_dir])

    # Drop self-loops.
    keep = src != dst
    if not keep.all():
        src = src[keep]
        dst = dst[keep]

    return src, dst


def _build_csr_with_timestamps(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    node_year: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort the symmetric edge list and emit CSR arrays.

    Returns:
      edge_starts[num_nodes] int64,
      edge_ends[num_edges]   int32 (sorted within each row by dst ascending),
      timestamps[num_edges]  int32 (aligned with edge_ends).

    Per-edge timestamps use `max(node_year[u], node_year[v])` so that both
    directions of an undirected edge get the same value (needed for any
    downstream temporal split to be coherent).
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    # Pre-compute per-edge timestamp = max(year[u], year[v]).
    ts = np.maximum(node_year[src], node_year[dst]).astype(np.int32, copy=False)

    # Drop exact duplicates by encoding (src, dst) into a single int64 key
    # then unique-sorting. Each undirected edge survives once per direction.
    LOGGER.info("Encoding (src, dst) into int64 keys ...")
    key = (src.astype(np.int64) << 32) | dst.astype(np.int64) & 0xFFFFFFFF
    LOGGER.info("Sorting %s directed entries (with dups) ...", format_int(key.size))
    order = np.argsort(key, kind="stable")
    key = key[order]
    ts = ts[order]
    del order

    unique_mask = np.empty(key.size, dtype=bool)
    unique_mask[0] = True
    if key.size > 1:
        np.not_equal(key[1:], key[:-1], out=unique_mask[1:])
    n_unique = int(unique_mask.sum())
    LOGGER.info(
        "Deduplicated: %s → %s directed edges",
        format_int(key.size),
        format_int(n_unique),
    )

    key = key[unique_mask]
    ts = ts[unique_mask]
    del unique_mask

    src_sorted = (key >> 32).astype(np.int64, copy=False)
    dst_sorted = (key & np.int64(0xFFFFFFFF)).astype(np.int32, copy=False)
    del key

    edge_starts = np.zeros(num_nodes, dtype=np.int64)
    per_node = np.bincount(src_sorted, minlength=num_nodes).astype(np.int64, copy=False)
    if num_nodes > 1:
        np.cumsum(per_node[:-1], out=edge_starts[1:])
    return edge_starts, dst_sorted, ts


def _write_graph_csr(
    *,
    out_dir: Path,
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    timestamps: np.ndarray,
    file_endian: Endian,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_ends_arr = LargeInt32Array(
        size=int(edge_ends.size),
        arr=np.ascontiguousarray(edge_ends, dtype=np.int32),
    )
    timestamps_arr = LargeInt32Array(
        size=int(timestamps.size),
        arr=np.ascontiguousarray(timestamps, dtype=np.int32),
    )
    edge_starts_arr = LargeInt64RawArray(
        arr=np.ascontiguousarray(edge_starts, dtype=np.int64),
    )

    graph = GraphCSR(
        edge_ends=edge_ends_arr,
        edge_starts=edge_starts_arr,
        timestamps=timestamps_arr,
    )
    GraphCSRSerializer.serialize(graph, out_dir, file_endian=file_endian)


def _flatten_neg_edges(source_node: np.ndarray, target_node_neg: np.ndarray) -> np.ndarray:
    """Expand `[N, K]` negatives to `[N*K, 2]` int64 pairs."""
    k = int(target_node_neg.shape[1])
    src_repeat = np.repeat(source_node.astype(np.int64), k)
    dst_flat = np.asarray(target_node_neg, dtype=np.int64).reshape(-1)
    return np.stack([src_repeat, dst_flat], axis=1)


def _dump_edge_split(out_dir: Path, edge_split: dict) -> None:
    """Dump OGB-provided valid/test edge splits as LPP-compatible npy files."""
    for split_name, prefix in [("valid", "valid"), ("test", "test")]:
        split = edge_split[split_name]
        src = np.asarray(split["source_node"], dtype=np.int64)
        tgt = np.asarray(split["target_node"], dtype=np.int64)
        neg = np.asarray(split["target_node_neg"])  # [N, K]

        pos = np.stack([src, tgt], axis=1)
        neg_flat = _flatten_neg_edges(src, neg)

        pos_path = out_dir / f"{prefix}_edge.npy"
        neg_path = out_dir / f"{prefix}_edge_neg.npy"
        np.save(pos_path, pos)
        np.save(neg_path, neg_flat)
        LOGGER.info(
            "%s: positives=%s @ %s, negatives=%s @ %s",
            split_name,
            format_int(pos.shape[0]),
            pos_path,
            format_int(neg_flat.shape[0]),
            neg_path,
        )


def prepare_ogb_dataset(
    *,
    dataset: str,
    out_dir: Path,
    ogb_root: Path,
    file_endian: Endian,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "graph_csr"

    t0 = time.time()
    LOGGER.info("Loading %s from %s ...", dataset, ogb_root)
    if dataset == "ogbl-citation2":
        edge_index, node_year, num_nodes, edge_split = _load_ogbl_citation2(ogb_root)
    elif dataset == "ogbn-papers100M":
        edge_index, node_year, num_nodes = _load_ogbn_papers100m(ogb_root)
        edge_split = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    LOGGER.info(
        "Loaded: num_nodes=%s directed_edges=%s years_min=%s years_max=%s",
        format_int(num_nodes),
        format_int(edge_index.shape[1]),
        int(node_year.min()),
        int(node_year.max()),
    )

    LOGGER.info("Symmetrizing edge list ...")
    src, dst = _symmetrize(edge_index)
    del edge_index
    LOGGER.info("Symmetric directed entries (pre-dedup): %s", format_int(src.size))

    LOGGER.info("Building CSR with per-edge timestamps ...")
    edge_starts, edge_ends, timestamps = _build_csr_with_timestamps(
        src=src, dst=dst, node_year=node_year, num_nodes=num_nodes,
    )
    del src, dst

    LOGGER.info(
        "CSR ready: nodes=%s edges=%s. Writing to %s (endian=%s) ...",
        format_int(num_nodes),
        format_int(edge_ends.size),
        graph_dir,
        file_endian,
    )
    _write_graph_csr(
        out_dir=graph_dir,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        timestamps=timestamps,
        file_endian=file_endian,
    )

    (out_dir / "num_nodes.txt").write_text(f"{num_nodes}\n")

    if edge_split is not None:
        LOGGER.info("Dumping OGB-provided valid/test edge splits ...")
        _dump_edge_split(out_dir, edge_split)
    else:
        LOGGER.info(
            "%s has no built-in edge splits — run scripts/prepare_dataset.py "
            "to carve val edges by timestamp, and supply test_edge.npy / "
            "test_edge_neg.npy yourself.",
            dataset,
        )

    LOGGER.info("Done in %s. Output: %s", format_seconds(time.time() - t0), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an OGB dataset and convert it to LPP GraphCSR format.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ogbl-citation2", "ogbn-papers100M"],
        required=True,
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for LPP layout")
    parser.add_argument("--ogb-root", type=str, required=True, help="OGB cache root (where datasets get downloaded/extracted)")
    parser.add_argument(
        "--file-endian",
        choices=["big", "little"],
        default="big",
        help="Endianness for written CSR files (default: big, matching the rest of the project).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    prepare_ogb_dataset(
        dataset=args.dataset,
        out_dir=Path(args.out_dir),
        ogb_root=Path(args.ogb_root),
        file_endian=args.file_endian,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .config import PositiveEdgesConfig


Endian = Literal["little", "big"]


@dataclass(slots=True)
class DatasetConfig:
    """Unified dataset config shared across all link-prediction algorithms.

    Three groups of fields:
      * prepared dataset root — `root` expands to the standard layout produced
        by `scripts/prepare_dataset.py`: `train_csr/`, `train_pairs_csr/`, and
        eval split files stored directly under the root.
      * graph paths — `graph_csr_root` (full train graph, both directions, used
        for negative-sampling rejection / message-passing / k-hop sampling) and
        `pairs_graph_csr_root` (canonical `u < v` pairs, used to iterate train
        positives without duplicates). Explicit paths override `root`.
      * eval paths — `valid_edge_path`, `valid_edge_neg_path`, `test_edge_path`,
        `test_edge_neg_path` as `[N, 2] int64` npy files. May be left None and
        resolved against `split_root`.
      * misc — `num_nodes`, `has_self_loops`, `mmap`, `sample_seed`, plus CSR
        I/O parameters reused for both graphs.
    """

    num_nodes: int = 0

    # Prepared dataset layout from scripts.prepare_dataset
    root: str | None = None

    # Graph paths
    graph_csr_root: str | None = None
    pairs_graph_csr_root: str | None = None

    # CSR I/O (shared for both graph_csr_root and pairs_graph_csr_root)
    graph_csr_use_mmap: bool = True
    graph_csr_file_endian: Endian = "big"
    graph_csr_allow_non_native: bool = True
    graph_csr_chunk_bytes: int = 256 * 1024 * 1024

    # Eval splits
    split_root: str | None = None
    valid_edge_path: str | None = None
    valid_edge_neg_path: str | None = None
    test_edge_path: str | None = None
    test_edge_neg_path: str | None = None
    mmap: bool = True

    # Misc
    has_self_loops: bool = False
    sample_seed: int = 12345

    def resolve(self) -> "DatasetConfig":
        dataset_root = None if self.root is None else Path(self.root)
        split_root = Path(self.split_root) if self.split_root is not None else dataset_root

        def _resolve(maybe_path: str | None, default_name: str) -> str | None:
            if maybe_path is not None:
                return str(Path(maybe_path))
            if split_root is None:
                return None
            return str(split_root / default_name)

        def _resolve_graph(maybe_path: str | None, default_name: str) -> str | None:
            if maybe_path is not None:
                return str(Path(maybe_path))
            if dataset_root is None:
                return None
            return str(dataset_root / default_name)

        return DatasetConfig(
            num_nodes=int(self.num_nodes),
            root=None if dataset_root is None else str(dataset_root),
            graph_csr_root=_resolve_graph(self.graph_csr_root, "train_csr"),
            pairs_graph_csr_root=_resolve_graph(self.pairs_graph_csr_root, "train_pairs_csr"),
            graph_csr_use_mmap=bool(self.graph_csr_use_mmap),
            graph_csr_file_endian=self.graph_csr_file_endian,
            graph_csr_allow_non_native=bool(self.graph_csr_allow_non_native),
            graph_csr_chunk_bytes=int(self.graph_csr_chunk_bytes),
            split_root=None if split_root is None else str(split_root),
            valid_edge_path=_resolve(self.valid_edge_path, "valid_edge.npy"),
            valid_edge_neg_path=_resolve(self.valid_edge_neg_path, "valid_edge_neg.npy"),
            test_edge_path=_resolve(self.test_edge_path, "test_edge.npy"),
            test_edge_neg_path=_resolve(self.test_edge_neg_path, "test_edge_neg.npy"),
            mmap=bool(self.mmap),
            has_self_loops=bool(self.has_self_loops),
            sample_seed=int(self.sample_seed),
        )

    def to_positive_edges_config(self) -> PositiveEdgesConfig:
        return PositiveEdgesConfig(
            num_nodes=int(self.num_nodes),
            has_self_loops=bool(self.has_self_loops),
            graph_csr_root=self.graph_csr_root,
            pairs_graph_csr_root=self.pairs_graph_csr_root,
            graph_csr_use_mmap=bool(self.graph_csr_use_mmap),
            graph_csr_file_endian=str(self.graph_csr_file_endian),
            graph_csr_allow_non_native=bool(self.graph_csr_allow_non_native),
            graph_csr_chunk_bytes=int(self.graph_csr_chunk_bytes),
        ).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

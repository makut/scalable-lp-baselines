from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class NegativeSamplingConfig:
    name: str = "uniform"
    neg_per_pos: int = 1
    seed: int = 42
    reject_self_loops: bool = True
    reject_existing_edges: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchTransformConfig:
    name: str = "edge_label"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositiveEdgesConfig:
    num_nodes: int = 0
    has_self_loops: bool = False
    # Full train graph (both directions). Used for negative-sampling rejection
    # via _has_edge_sorted, GraphSAGE neighbour sampling, SEAL k-hop extraction.
    graph_csr_root: str | None = None
    # Canonical pairs CSR (u < v). Used to iterate train positives without
    # duplicates. If None, falls back to `graph_csr_root` (each undirected edge
    # iterated twice).
    pairs_graph_csr_root: str | None = None
    graph_csr_use_mmap: bool = True
    graph_csr_file_endian: str = "big"
    graph_csr_allow_non_native: bool = True
    graph_csr_chunk_bytes: int = 256 * 1024 * 1024

    def resolve(self) -> "PositiveEdgesConfig":
        return PositiveEdgesConfig(
            num_nodes=int(self.num_nodes),
            has_self_loops=bool(self.has_self_loops),
            graph_csr_root=None if self.graph_csr_root is None else str(Path(self.graph_csr_root)),
            pairs_graph_csr_root=None if self.pairs_graph_csr_root is None else str(Path(self.pairs_graph_csr_root)),
            graph_csr_use_mmap=bool(self.graph_csr_use_mmap),
            graph_csr_file_endian=str(self.graph_csr_file_endian),
            graph_csr_allow_non_native=bool(self.graph_csr_allow_non_native),
            graph_csr_chunk_bytes=int(self.graph_csr_chunk_bytes),
        )


@dataclass(slots=True)
class TrainLoaderConfig:
    batch_size: int = 65_536
    seed: int = 42
    num_workers: int = 0
    positive_edges: PositiveEdgesConfig = field(default_factory=PositiveEdgesConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainLoaderConfig":
        data = dict(raw)
        positive_edges = data.pop("positive_edges", {})
        negative_sampling = data.pop("negative_sampling", {})
        if "seed" not in negative_sampling and "seed" in data:
            negative_sampling = dict(negative_sampling)
            negative_sampling["seed"] = data["seed"]
        return cls(
            **data,
            positive_edges=PositiveEdgesConfig(**positive_edges).resolve(),
            negative_sampling=NegativeSamplingConfig(**negative_sampling),
        )

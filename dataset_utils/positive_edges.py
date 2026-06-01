from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from graph_csr import GraphCSRSerializer


class GraphCSRPositiveEdgeDataset(Dataset[torch.Tensor]):
    """Iterates over positive edges from a GraphCSR.

    Reads from `pairs_graph_csr_root` if set (canonical pairs `u < v`, no
    duplicates), otherwise falls back to `graph_csr_root` (full graph, may
    yield each undirected edge twice). The full graph is independently
    consulted for negative-sampling rejection elsewhere.
    """

    def __init__(self, config: Any) -> None:
        pairs_root = getattr(config, "pairs_graph_csr_root", None)
        graph_root = getattr(config, "graph_csr_root", None)
        positives_root = pairs_root if pairs_root is not None else graph_root
        if positives_root is None:
            raise ValueError(
                "either pairs_graph_csr_root or graph_csr_root must be set "
                "for GraphCSRPositiveEdgeDataset"
            )
        self.config = config
        self.positives_root = str(positives_root)
        self.uses_canonical_pairs = pairs_root is not None
        self.graph = None

        with GraphCSRSerializer.deserialize(
            self.positives_root,
            use_mmap=bool(getattr(config, "graph_csr_use_mmap", True)),
            file_endian=getattr(config, "graph_csr_file_endian", "big"),
            writable=False,
            allow_non_native=bool(getattr(config, "graph_csr_allow_non_native", True)),
            chunk_bytes=int(getattr(config, "graph_csr_chunk_bytes", 256 * 1024 * 1024)),
        ) as graph:
            self.num_nodes = int(graph.edge_starts.numpy().size)
            self.num_positive_edges = int(graph.edge_ends.size)

        if self.uses_canonical_pairs and graph_root is not None:
            with GraphCSRSerializer.deserialize(
                str(graph_root),
                use_mmap=bool(getattr(config, "graph_csr_use_mmap", True)),
                file_endian=getattr(config, "graph_csr_file_endian", "big"),
                writable=False,
                allow_non_native=bool(getattr(config, "graph_csr_allow_non_native", True)),
                chunk_bytes=int(getattr(config, "graph_csr_chunk_bytes", 256 * 1024 * 1024)),
            ) as full_graph:
                full_num_nodes = int(full_graph.edge_starts.numpy().size)
            if full_num_nodes != self.num_nodes:
                raise ValueError(
                    "graph_csr_root and pairs_graph_csr_root must have the same num_nodes, "
                    f"got {full_num_nodes} and {self.num_nodes}"
                )

        config_num_nodes = int(getattr(config, "num_nodes", 0))
        if config_num_nodes > 0 and config_num_nodes != self.num_nodes:
            raise ValueError(
                f"dataset.num_nodes={config_num_nodes} does not match graph num_nodes={self.num_nodes}"
            )

    def _ensure_graph(self) -> None:
        if self.graph is None:
            self.graph = GraphCSRSerializer.deserialize(
                self.positives_root,
                use_mmap=bool(getattr(self.config, "graph_csr_use_mmap", True)),
                file_endian=getattr(self.config, "graph_csr_file_endian", "big"),
                writable=False,
                allow_non_native=bool(getattr(self.config, "graph_csr_allow_non_native", True)),
                chunk_bytes=int(getattr(self.config, "graph_csr_chunk_bytes", 256 * 1024 * 1024)),
            )

    def _positive_edge_at(self, edge_pos: int) -> tuple[int, int]:
        self._ensure_graph()
        starts = self.graph.edge_starts.numpy()
        edge_index = int(edge_pos)
        src = int(np.searchsorted(starts, edge_index, side="right") - 1)
        if src < 0:
            raise ValueError(f"Bad positive edge position: {edge_pos}")
        offset = edge_index - int(starts[src])
        dst = int(self.graph.neighbor_by_index(src, offset))
        if self.uses_canonical_pairs and src >= dst:
            raise ValueError(
                f"pairs_graph_csr_root must store only canonical edges u < v, got {(src, dst)}"
            )
        return src, dst

    def __len__(self) -> int:
        return int(self.num_positive_edges)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.num_positive_edges <= 0:
            raise ValueError("Train graph is empty")
        local_idx = int(idx)
        if local_idx < 0 or local_idx >= self.num_positive_edges:
            raise IndexError(f"Index out of range: {local_idx}")
        src, dst = self._positive_edge_at(local_idx)
        return torch.tensor([src, dst], dtype=torch.int64)

# Based on facebookresearch/SEAL_OGB, licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

from dataset_utils import RawLinkBatch
from graph_csr.serializer import GraphCSRSerializer

try:
    from .graph_csr_extraction import _k_hop_subgraph_graph_csr
    from .utils import construct_pyg_graph
except ImportError:
    from graph_csr_extraction import _k_hop_subgraph_graph_csr
    from utils import construct_pyg_graph


class _GraphCSRSEALGraphBackend:
    def __init__(self, data, *, directed: bool) -> None:
        if directed:
            raise ValueError("GraphCSR mode currently supports only undirected graphs")
        self.data = data
        self.directed = bool(directed)
        self.graph = None

    def _ensure_graph(self) -> None:
        if self.graph is not None:
            return
        self.graph = GraphCSRSerializer.deserialize(
            self.data.graph_root,
            use_mmap=self.data.use_mmap,
            file_endian=self.data.file_endian,
            writable=False,
            allow_non_native=self.data.allow_non_native,
            chunk_bytes=self.data.chunk_bytes,
        )

    def build_graph(
        self,
        src,
        dst,
        label,
        *,
        num_hops,
        node_label,
        ratio_per_hop,
        max_nodes_per_hop,
        per_vertex_oversample,
        graph_csr_use_per_vertex_sampling,
        graph_csr_use_pairwise_subgraph,
    ):
        self._ensure_graph()
        tmp = _k_hop_subgraph_graph_csr(
            int(src),
            int(dst),
            int(num_hops),
            self.graph,
            ratio_per_hop,
            max_nodes_per_hop,
            per_vertex_oversample=per_vertex_oversample,
            graph_csr_use_per_vertex_sampling=graph_csr_use_per_vertex_sampling,
            graph_csr_use_pairwise_subgraph=graph_csr_use_pairwise_subgraph,
            node_features=self.data.x,
            y=label,
        )
        return construct_pyg_graph(*tmp, node_label)


class SEALSubgraphExtractor:
    def __init__(
        self,
        *,
        graph_data,
        num_hops: int,
        node_label: str,
        ratio_per_hop: float,
        max_nodes_per_hop: int | None,
        directed: bool,
        per_vertex_oversample: float = 1.5,
        graph_csr_use_per_vertex_sampling: bool = True,
        graph_csr_use_pairwise_subgraph: bool = True,
    ) -> None:
        self.graph_data = graph_data
        self.num_hops = int(num_hops)
        self.node_label = str(node_label)
        self.ratio_per_hop = float(ratio_per_hop)
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = bool(directed)
        self.per_vertex_oversample = float(per_vertex_oversample)
        self.graph_csr_use_per_vertex_sampling = bool(graph_csr_use_per_vertex_sampling)
        self.graph_csr_use_pairwise_subgraph = bool(graph_csr_use_pairwise_subgraph)
        self._backend = None

    def _ensure_backend(self):
        if self._backend is not None:
            return self._backend
        self._backend = _GraphCSRSEALGraphBackend(self.graph_data, directed=self.directed)
        return self._backend

    def build_graph(self, src: int, dst: int, label: float):
        return self._ensure_backend().build_graph(
            src,
            dst,
            label,
            num_hops=self.num_hops,
            node_label=self.node_label,
            ratio_per_hop=self.ratio_per_hop,
            max_nodes_per_hop=self.max_nodes_per_hop,
            per_vertex_oversample=self.per_vertex_oversample,
            graph_csr_use_per_vertex_sampling=self.graph_csr_use_per_vertex_sampling,
            graph_csr_use_pairwise_subgraph=self.graph_csr_use_pairwise_subgraph,
        )


def _batch_graphs(graphs: list[Any]) -> Batch:
    return Batch.from_data_list(graphs)


class SEALBatchTransform:
    def __init__(self, extractor: SEALSubgraphExtractor) -> None:
        self.extractor = extractor

    def __call__(self, batch: RawLinkBatch) -> Batch:
        edges = torch.cat([batch.pos_edges, batch.neg_edges], dim=0)
        labels = torch.cat([batch.pos_labels, batch.neg_labels], dim=0)
        graphs = [
            self.extractor.build_graph(int(edge[0]), int(edge[1]), float(label))
            for edge, label in zip(edges.tolist(), labels.tolist())
        ]
        return _batch_graphs(graphs)


class SEALEvalCollator:
    def __init__(self, extractor: SEALSubgraphExtractor) -> None:
        self.extractor = extractor

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
        graphs = []
        for edge, label in batch:
            edge_list = edge.tolist()
            src = int(edge_list[0])
            dst = int(edge_list[1])
            g = self.extractor.build_graph(src, dst, float(label.item()))
            # Stash original edge endpoints so per-source ranking metrics can
            # group predictions by src after batching. PyG concatenates
            # graph-level tensors of shape [1, …] across the batch axis.
            g.edge_src = torch.tensor([src], dtype=torch.int64)
            g.edge_dst = torch.tensor([dst], dtype=torch.int64)
            graphs.append(g)
        return _batch_graphs(graphs)


def sample_train_graphs_for_sortpool(
    *,
    positive_dataset,
    negative_sampler,
    extractor: SEALSubgraphExtractor,
    sample_size: int,
    seed: int = 0,
) -> list[Any]:
    total = len(positive_dataset)
    count = max(0, min(int(sample_size), total))
    if count <= 0:
        return []
    # train_pairs_csr is sorted by src, so range(count) gives all (0, *) pairs
    # when src=0 is a hub — biased toward one source. Sample uniformly instead.
    if count >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(total, size=count, replace=False).tolist()
    pos_edges = torch.stack([positive_dataset[int(idx)] for idx in indices], dim=0).to(dtype=torch.int64)
    neg_edges = negative_sampler.sample(pos_edges, meta={"worker_id": 0, "rank": 0}).to(dtype=torch.int64)

    graphs = [
        extractor.build_graph(int(edge[0]), int(edge[1]), 1.0)
        for edge in pos_edges.tolist()
    ]
    graphs.extend(
        extractor.build_graph(int(edge[0]), int(edge[1]), 0.0)
        for edge in neg_edges.tolist()
    )
    return graphs


@dataclass
class SEALTrainDatasetFacade:
    positive_dataset: Any
    negative_sampler: Any
    extractor: SEALSubgraphExtractor
    train_loader: Any
    num_features: int = 0

    def __len__(self) -> int:
        examples_per_positive = getattr(self.negative_sampler, "examples_per_positive", lambda: 1.0)
        return int(len(self.positive_dataset) * max(1.0, float(examples_per_positive())))

    def set_epoch(self, epoch: int) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def sample_graphs_for_sortpool(self, sample_size: int):
        return sample_train_graphs_for_sortpool(
            positive_dataset=self.positive_dataset,
            negative_sampler=self.negative_sampler,
            extractor=self.extractor,
            sample_size=sample_size,
        )

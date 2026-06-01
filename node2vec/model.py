from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from embedding_table_utils import BaseEmbeddingTable, EmbeddingTableConfig, create_embedding_table

from .config import Node2VecConfig


class Node2VecEmbeddingModule(nn.Module):
    def __init__(self, embedding_table: BaseEmbeddingTable) -> None:
        super().__init__()
        self.embedding_table = embedding_table

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table.lookup(node_ids)


def build_embedding_table_config(config: Node2VecConfig, num_nodes: int) -> EmbeddingTableConfig:
    return EmbeddingTableConfig.from_dict(
        config.embedding_table_config,
        num_embeddings=num_nodes,
        device=config.device,
    )


def create_node2vec_embedding_table(
    *,
    num_nodes: int,
    config: Node2VecConfig,
    device: torch.device,
    process_group: "dist.ProcessGroup | None" = None,
) -> BaseEmbeddingTable:
    return create_embedding_table(
        build_embedding_table_config(config, num_nodes),
        device=device,
        process_group=process_group,
    )

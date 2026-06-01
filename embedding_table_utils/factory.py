from __future__ import annotations

import torch
import torch.distributed as dist

from .api import BaseEmbeddingTable
from .config import EmbeddingTableConfig
from .torchrec_backend import TorchRecShardedEmbeddingTable
from .vanilla_backend import VanillaEmbeddingTable


def create_embedding_table(
    config: EmbeddingTableConfig,
    *,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> BaseEmbeddingTable:
    if config.backend == "torchrec":
        return TorchRecShardedEmbeddingTable(
            config=config,
            device=device,
            process_group=process_group,
        )
    if config.backend == "vanilla":
        return VanillaEmbeddingTable(
            config=config,
            device=device,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


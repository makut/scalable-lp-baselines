from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .api import BaseEmbeddingTable
from .config import EmbeddingTableConfig
from .init import apply_init, resolve_torch_dtype
from .optimizer_adapters import build_vanilla_optimizer_adapter


class VanillaEmbeddingTable(BaseEmbeddingTable):
    def __init__(self, config: EmbeddingTableConfig, *, device: torch.device) -> None:
        super().__init__(config=config, device=device)
        self.embedding = nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            device=device,
            dtype=resolve_torch_dtype(config.dtype),
        )
        apply_init(self.embedding.weight, config)
        self._optimizer_adapter = build_vanilla_optimizer_adapter(self.embedding.parameters(), config)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(device=self.embedding.weight.device, dtype=torch.int64)
        return self.embedding(ids)

    def local_model_state_dict(self) -> dict[str, Any]:
        return self.state_dict()

    def load_local_model_state_dict(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state)

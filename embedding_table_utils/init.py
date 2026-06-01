from __future__ import annotations

import math

import torch
from torch import nn

from .config import EmbeddingTableConfig


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def apply_init(weight: torch.Tensor, config: EmbeddingTableConfig) -> None:
    if getattr(weight, "is_meta", False):
        return

    init_kwargs = dict(config.init_kwargs)
    if config.init_type == "normal":
        mean = float(init_kwargs.get("mean", 0.0))
        std = float(init_kwargs.get("std", 1.0 / max(1, config.embedding_dim)))
        nn.init.normal_(weight, mean=mean, std=std)
        return
    if config.init_type == "uniform":
        if "low" in init_kwargs or "high" in init_kwargs:
            low = float(init_kwargs["low"])
            high = float(init_kwargs["high"])
        else:
            bound = float(init_kwargs.get("bound", 1.0 / math.sqrt(max(1, config.embedding_dim))))
            low = -bound
            high = bound
        nn.init.uniform_(weight, a=low, b=high)
        return
    if config.init_type == "zeros":
        nn.init.zeros_(weight)
        return
    raise ValueError(f"Unsupported init_type: {config.init_type}")


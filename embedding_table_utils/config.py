from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import torch


EmbeddingBackend = Literal["torchrec", "vanilla"]
EmbeddingDType = Literal["fp16", "fp32"]
EmbeddingInitType = Literal["normal", "uniform", "zeros"]
ComputeKernelPolicy = Literal["auto", "prefer_hbm", "allow_uvm"]

_SUPPORTED_BACKENDS = {"torchrec", "vanilla"}
_SUPPORTED_DTYPES = {"fp16", "fp32"}
_SUPPORTED_INIT_TYPES = {"normal", "uniform", "zeros"}
_SUPPORTED_OPTIMIZERS = {None, "sgd", "adam", "adamw", "adagrad", "rowwise_adagrad"}
_SUPPORTED_COMPUTE_KERNEL_POLICIES = {"auto", "prefer_hbm", "allow_uvm"}


def _validate_init_kwargs(init_type: str, init_kwargs: dict[str, Any]) -> None:
    if init_type == "normal":
        allowed = {"mean", "std"}
    elif init_type == "uniform":
        allowed = {"bound", "low", "high"}
        if "bound" in init_kwargs and ("low" in init_kwargs or "high" in init_kwargs):
            raise ValueError("uniform init accepts either 'bound' or ('low', 'high'), not both")
    elif init_type == "zeros":
        allowed = set()
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")

    unknown = set(init_kwargs) - allowed
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported init_kwargs for init_type={init_type!r}: {unknown_str}")

    if init_type == "normal" and "std" in init_kwargs and float(init_kwargs["std"]) < 0.0:
        raise ValueError("normal init requires std >= 0")
    if init_type == "uniform" and "bound" in init_kwargs and float(init_kwargs["bound"]) < 0.0:
        raise ValueError("uniform init requires bound >= 0")
    if init_type == "uniform" and (
        ("low" in init_kwargs and "high" not in init_kwargs)
        or ("high" in init_kwargs and "low" not in init_kwargs)
    ):
        raise ValueError("uniform init requires both 'low' and 'high' when explicit bounds are used")
    if init_type == "uniform" and "low" in init_kwargs and "high" in init_kwargs:
        if float(init_kwargs["low"]) > float(init_kwargs["high"]):
            raise ValueError("uniform init requires low <= high")


@dataclass(slots=True)
class EmbeddingTableConfig:
    backend: EmbeddingBackend
    num_embeddings: int
    embedding_dim: int

    dtype: EmbeddingDType = "fp32"
    table_name: str = "node_table"
    feature_name: str = "node"

    init_type: EmbeddingInitType = "normal"
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    optimizer_type: str | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    sharding_type: str = "row_wise"
    compute_kernel_policy: ComputeKernelPolicy = "auto"

    device: str | None = None

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        num_embeddings: int | None = None,
        device: str | torch.device | None = None,
    ) -> "EmbeddingTableConfig":
        data = dict(raw)
        valid_keys = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown embedding_table_config keys: {unknown}")

        if num_embeddings is not None:
            existing_num_embeddings = data.get("num_embeddings")
            if existing_num_embeddings is not None and int(existing_num_embeddings) != int(num_embeddings):
                raise ValueError(
                    "embedding_table_config.num_embeddings does not match resolved num_embeddings: "
                    f"{existing_num_embeddings} != {num_embeddings}"
                )
            data["num_embeddings"] = int(num_embeddings)
        if device is not None and data.get("device") is None:
            data["device"] = str(device)
        return cls(**data)

    def __post_init__(self) -> None:
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if not self.table_name:
            raise ValueError("table_name must be non-empty")
        if not self.feature_name:
            raise ValueError("feature_name must be non-empty")
        if self.init_type not in _SUPPORTED_INIT_TYPES:
            raise ValueError(f"Unsupported init_type: {self.init_type}")
        _validate_init_kwargs(self.init_type, self.init_kwargs)
        if self.optimizer_type not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")
        if self.compute_kernel_policy not in _SUPPORTED_COMPUTE_KERNEL_POLICIES:
            raise ValueError(f"Unsupported compute_kernel_policy: {self.compute_kernel_policy}")
        if self.backend == "torchrec" and self.sharding_type not in {"row_wise", "column_wise"}:
            raise ValueError(
                "TorchRec backend currently supports only "
                "sharding_type in {'row_wise', 'column_wise'}"
            )
        if self.backend == "torchrec" and self.optimizer_type == "adamw":
            raise ValueError("TorchRec backend does not expose AdamW semantics for sharded embedding tables")
        if self.device is not None:
            torch.device(self.device)

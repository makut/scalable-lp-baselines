from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .api import BaseEmbeddingTable
from .config import EmbeddingTableConfig
from .init import apply_init
from .optimizer_adapters import build_torchrec_optimizer_adapter


logger = logging.getLogger(__name__)
_LOG_STATS_SAMPLE_SIZE = 1_000_000


def _lazy_import_torchrec() -> dict[str, Any]:
    try:
        from fbgemm_gpu.split_embedding_configs import EmbOptimType
        from torchrec import KeyedJaggedTensor
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
        from torchrec.distributed.model_parallel import DistributedModelParallel
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.planner.types import ParameterConstraints
        from torchrec.distributed.types import ShardingEnv, ShardingType
        from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
        from torchrec.modules.embedding_modules import EmbeddingBagCollection
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise RuntimeError(
            "TorchRec backend is unavailable. Ensure torchrec and fbgemm_gpu are installed "
            "for the current platform/runtime before using backend='torchrec'."
        ) from exc

    return {
        "DataType": DataType,
        "DistributedModelParallel": DistributedModelParallel,
        "EmbeddingBagCollection": EmbeddingBagCollection,
        "EmbeddingBagCollectionSharder": EmbeddingBagCollectionSharder,
        "EmbeddingBagConfig": EmbeddingBagConfig,
        "EmbeddingComputeKernel": EmbeddingComputeKernel,
        "EmbeddingShardingPlanner": EmbeddingShardingPlanner,
        "EmbOptimType": EmbOptimType,
        "KeyedJaggedTensor": KeyedJaggedTensor,
        "ParameterConstraints": ParameterConstraints,
        "PoolingType": PoolingType,
        "ShardingEnv": ShardingEnv,
        "ShardingType": ShardingType,
        "Topology": Topology,
    }


def _torchrec_data_type(config: EmbeddingTableConfig) -> Any:
    torchrec = _lazy_import_torchrec()
    data_type = torchrec["DataType"]
    if config.dtype == "fp32":
        return data_type.FP32
    if config.dtype == "fp16":
        return data_type.FP16
    raise ValueError(f"Unsupported TorchRec dtype: {config.dtype}")


def _torchrec_fused_params(config: EmbeddingTableConfig) -> dict[str, Any] | None:
    if config.optimizer_type is None:
        return None

    torchrec = _lazy_import_torchrec()
    emb_optim_type = torchrec["EmbOptimType"]
    optimizer_type = config.optimizer_type
    if optimizer_type == "sgd":
        optimizer = emb_optim_type.EXACT_SGD
    elif optimizer_type == "adam":
        optimizer = emb_optim_type.ADAM
    elif optimizer_type in {"adagrad", "rowwise_adagrad"}:
        optimizer = emb_optim_type.EXACT_ROWWISE_ADAGRAD
    else:
        raise ValueError(
            f"Unsupported optimizer_type={optimizer_type!r} for TorchRec fused embedding path"
        )

    fused_params = dict(config.optimizer_kwargs)
    fused_params["optimizer"] = optimizer
    return fused_params


def _append_compute_kernel(
    out: list[str],
    kernel: Any,
    *candidate_names: str,
) -> None:
    for name in candidate_names:
        member = getattr(kernel, name, None)
        if member is None:
            continue
        value = member.value
        if value not in out:
            out.append(value)
        return
    raise ValueError(
        "Installed TorchRec does not expose any of the expected EmbeddingComputeKernel "
        f"members: {candidate_names}"
    )


def _compute_kernel_constraints(config: EmbeddingTableConfig, device: torch.device) -> list[str] | None:
    if config.compute_kernel_policy == "auto":
        return None

    torchrec = _lazy_import_torchrec()
    kernel = torchrec["EmbeddingComputeKernel"]
    if device.type != "cuda":
        return [kernel.DENSE.value]
    if config.compute_kernel_policy == "prefer_hbm":
        kernels: list[str] = []
        _append_compute_kernel(kernels, kernel, "BATCHED_FUSED", "FUSED")
        _append_compute_kernel(kernels, kernel, "BATCHED_DENSE", "DENSE")
        _append_compute_kernel(kernels, kernel, "DENSE")
        return kernels
    if config.compute_kernel_policy == "allow_uvm":
        kernels = []
        _append_compute_kernel(kernels, kernel, "BATCHED_FUSED_UVM_CACHING", "FUSED_UVM_CACHING")
        _append_compute_kernel(kernels, kernel, "BATCHED_FUSED_UVM", "FUSED_UVM")
        _append_compute_kernel(kernels, kernel, "BATCHED_FUSED", "FUSED")
        _append_compute_kernel(kernels, kernel, "BATCHED_DENSE", "DENSE")
        _append_compute_kernel(kernels, kernel, "DENSE")
        return kernels
    raise ValueError(f"Unsupported compute_kernel_policy: {config.compute_kernel_policy}")


def _torchrec_sharding_type(config: EmbeddingTableConfig) -> str:
    torchrec = _lazy_import_torchrec()
    sharding_type = torchrec["ShardingType"]
    if config.sharding_type == "row_wise":
        return sharding_type.ROW_WISE.value
    if config.sharding_type == "column_wise":
        return sharding_type.COLUMN_WISE.value
    raise ValueError(f"Unsupported TorchRec sharding_type: {config.sharding_type}")


def _column_wise_min_partition(config: EmbeddingTableConfig, world_size: int) -> int | None:
    if config.sharding_type != "column_wise":
        return None
    if world_size <= 1:
        return int(config.embedding_dim)

    embedding_dim = int(config.embedding_dim)
    if embedding_dim % world_size != 0:
        raise ValueError(
            "Equal column-wise sharding requires embedding_dim to be divisible by world_size: "
            f"embedding_dim={embedding_dim}, world_size={world_size}"
        )

    shard_dim = embedding_dim // world_size
    if shard_dim % 4 != 0:
        raise ValueError(
            "Equal column-wise sharding requires per-rank shard dim to be divisible by 4: "
            f"embedding_dim={embedding_dim}, world_size={world_size}, shard_dim={shard_dim}"
        )
    return int(shard_dim)


class _TorchRecLookupModule(nn.Module):
    def __init__(self, config: EmbeddingTableConfig, *, device: torch.device) -> None:
        super().__init__()
        torchrec = _lazy_import_torchrec()
        self._feature_name = config.feature_name
        self._table_name = config.table_name
        self._embedding_dim = int(config.embedding_dim)
        self.ebc = torchrec["EmbeddingBagCollection"](
            tables=[
                torchrec["EmbeddingBagConfig"](
                    name=config.table_name,
                    embedding_dim=config.embedding_dim,
                    num_embeddings=config.num_embeddings,
                    feature_names=[config.feature_name],
                    pooling=torchrec["PoolingType"].SUM,
                    data_type=_torchrec_data_type(config),
                )
            ],
            device=device,
        )
        self._kjt_cls = torchrec["KeyedJaggedTensor"]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        input_shape = tuple(ids.shape)
        values = ids.reshape(-1).to(dtype=torch.int64)
        unique_values, inverse = torch.unique(values, return_inverse=True)
        lengths = torch.ones(unique_values.numel(), device=unique_values.device, dtype=torch.int32)
        kjt = self._kjt_cls(
            keys=[self._feature_name],
            values=unique_values,
            lengths=lengths,
            stride=int(unique_values.numel()),
        )
        outputs = self.ebc(kjt)
        if hasattr(outputs, "wait"):
            outputs = outputs.wait()
        unique_emb = outputs.to_dict()[self._feature_name]
        result = unique_emb.index_select(0, inverse)
        if input_shape:
            return result.reshape(*input_shape, self._embedding_dim)
        return result.reshape(self._embedding_dim)


class TorchRecShardedEmbeddingTable(BaseEmbeddingTable):
    def __init__(
        self,
        config: EmbeddingTableConfig,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(config=config, device=device)
        if dist.is_available() and dist.is_initialized():
            self._process_group = process_group if process_group is not None else dist.group.WORLD
        else:
            self._process_group = None
        self._use_dmp = self.device.type == "cuda" or (
            self._process_group is not None and int(dist.get_world_size(self._process_group)) > 1
        )
        self._lookup_module = _TorchRecLookupModule(
            config,
            device=torch.device("meta") if self._use_dmp else self.device,
        )
        self._module = self._build_sharded_module() if self._use_dmp else self._lookup_module
        self._initialize_local_shards()
        self._optimizer_adapter = build_torchrec_optimizer_adapter(module=self._module, config=config)
        self._log_local_weight_stats()

    @property
    def process_group(self) -> dist.ProcessGroup | None:
        return self._process_group

    def _build_sharded_module(self) -> nn.Module:
        torchrec = _lazy_import_torchrec()
        sharder = torchrec["EmbeddingBagCollectionSharder"](
            fused_params=_torchrec_fused_params(self.config)
        )
        if self._process_group is None:
            world_size = 1
            rank = 0
            local_world_size = 1
            sharding_env = torchrec["ShardingEnv"].from_local(world_size=world_size, rank=rank)
        else:
            world_size = int(dist.get_world_size(self._process_group))
            rank = int(dist.get_rank(self._process_group))
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
            sharding_env = torchrec["ShardingEnv"].from_process_group(self._process_group)
        constraints = {
            self.config.table_name: torchrec["ParameterConstraints"](
                sharding_types=[_torchrec_sharding_type(self.config)],
                min_partition=_column_wise_min_partition(self.config, world_size),
                compute_kernels=_compute_kernel_constraints(self.config, self.device),
            )
        }
        planner = torchrec["EmbeddingShardingPlanner"](
            topology=torchrec["Topology"](
                world_size=world_size,
                compute_device=self.device.type,
                local_world_size=local_world_size,
            ),
            constraints=constraints,
        )
        if self._process_group is None:
            plan = planner.plan(self._lookup_module, [sharder])
        else:
            plan = planner.collective_plan(self._lookup_module, [sharder], self._process_group)
        # Log the sharding plan once on rank 0 so users get the standard TorchRec
        # summary (per-rank shards, sharding type, compute kernel) at startup.
        if rank == 0:
            logger.info(
                "TorchRec sharding plan for table=%s sharding_type=%s "
                "compute_kernel_policy=%s world_size=%d:\n%s",
                self.config.table_name,
                self.config.sharding_type,
                self.config.compute_kernel_policy,
                world_size,
                plan,
            )
        return torchrec["DistributedModelParallel"](
            module=self._lookup_module,
            env=sharding_env,
            device=self.device,
            plan=plan,
            sharders=[sharder],
        )

    @torch.no_grad()
    def _initialize_local_shards(self) -> None:
        for param in self._module.parameters():
            if getattr(param, "is_meta", False):
                continue
            apply_init(param, self.config)

    @torch.no_grad()
    def _log_local_weight_stats(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        sampled_values: list[torch.Tensor] = []
        total_values = 0
        sample_shapes: list[tuple[int, ...]] = []
        for param in self._module.parameters():
            if getattr(param, "is_meta", False):
                continue
            try:
                flat_param = param.detach().view(-1)
            except RuntimeError:
                flat_param = param.detach().reshape(-1)

            numel = int(flat_param.numel())
            if numel == 0:
                continue
            total_values += numel
            sample_shapes.append(tuple(param.shape))

            sample_size = min(numel, _LOG_STATS_SAMPLE_SIZE)
            if sample_size < numel:
                stride = max(1, numel // sample_size)
                flat_param = flat_param[::stride][:sample_size]
            sampled_values.append(flat_param.float().cpu())

        if not sampled_values:
            logger.debug("No materialized local TorchRec weights were found for stats logging")
            return
        flat = torch.cat(sampled_values, dim=0)
        logger.debug(
            "Local TorchRec shard sampled stats for table=%s: shapes=%s "
            "sampled_values=%d total_values=%d mean=%.6f std=%.6f min=%.6f max=%.6f",
            self.config.table_name,
            sample_shapes,
            int(flat.numel()),
            total_values,
            float(flat.mean().item()),
            float(flat.std(unbiased=False).item()),
            float(flat.min().item()),
            float(flat.max().item()),
        )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(device=self.device, dtype=torch.int64)
        return self._module(ids)

    def local_model_state_dict(self) -> dict[str, Any]:
        return self._module.state_dict()

    def load_local_model_state_dict(self, state: dict[str, Any]) -> None:
        self._module.load_state_dict(state)

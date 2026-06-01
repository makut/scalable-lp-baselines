from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from embedding_table_utils import EmbeddingTableConfig, create_embedding_table
from embedding_table_utils.checkpointing import distributed_rank, distributed_world_size


CHECKPOINT_METADATA_FORMAT = "baseline_embedding_table_conversion_v1"
DEFAULT_CHUNK_ROWS = 1_000_000
PREFERRED_TORCH_TENSOR_KEYS = (
    "node_embedding.weight",
    "node_embedding.embedding.weight",
    "embedding.weight",
    "embeddings.weight",
    "weight",
)


@dataclass(slots=True)
class ConversionResult:
    out_dir: str
    backend: str
    dtype: str
    num_embeddings: int
    embedding_dim: int
    table_name: str
    feature_name: str
    step: int | None
    source: dict[str, Any]
    rank: int
    world_size: int


class EmbeddingMatrixSource:
    source_type: str
    path: Path | None

    @property
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def dtype(self) -> str:
        raise NotImplementedError

    def copy_slice_to(
        self,
        target: torch.Tensor,
        *,
        row_start: int,
        col_start: int,
        chunk_rows: int,
    ) -> None:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.source_type,
            "path": None if self.path is None else str(self.path),
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


class NumpyEmbeddingMatrixSource(EmbeddingMatrixSource):
    def __init__(self, array: np.ndarray, *, path: Path | None = None, source_type: str = "npy") -> None:
        if array.ndim != 2:
            raise ValueError(f"Embedding matrix must be 2D, got shape={array.shape}")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"Embedding matrix dtype must be floating, got {array.dtype}")
        self.array = array
        self.path = path
        self.source_type = source_type

    @classmethod
    def from_npy(cls, path: str | Path) -> "NumpyEmbeddingMatrixSource":
        p = Path(path)
        return cls(np.load(p, mmap_mode="r"), path=p, source_type="npy")

    @classmethod
    def from_raw_binary(
        cls,
        path: str | Path,
        *,
        num_nodes: int,
        dim: int,
        dtype: str,
    ) -> "NumpyEmbeddingMatrixSource":
        p = Path(path)
        np_dtype = np.dtype(dtype)
        expected_bytes = int(num_nodes) * int(dim) * np_dtype.itemsize
        actual_bytes = p.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Unexpected raw embedding size for {p}: got {actual_bytes} bytes, "
                f"expected {expected_bytes} for num_nodes={num_nodes}, dim={dim}, dtype={np_dtype}"
            )
        matrix = np.memmap(p, mode="r", dtype=np_dtype, shape=(int(num_nodes), int(dim)))
        return cls(matrix, path=p, source_type="raw_binary")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.array.shape[0]), int(self.array.shape[1])

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)

    def copy_slice_to(
        self,
        target: torch.Tensor,
        *,
        row_start: int,
        col_start: int,
        chunk_rows: int,
    ) -> None:
        rows, cols = _validate_target_slice(self.shape, target, row_start=row_start, col_start=col_start)
        for local_start in range(0, rows, max(1, int(chunk_rows))):
            local_end = min(rows, local_start + max(1, int(chunk_rows)))
            source_rows = slice(row_start + local_start, row_start + local_end)
            source_cols = slice(col_start, col_start + cols)
            chunk = np.asarray(self.array[source_rows, source_cols])
            if not chunk.flags.writeable:
                chunk = np.array(chunk, copy=True)
            target[local_start:local_end].copy_(
                torch.as_tensor(chunk).to(device=target.device, dtype=target.dtype)
            )


class TorchTensorEmbeddingMatrixSource(EmbeddingMatrixSource):
    def __init__(self, tensor: torch.Tensor, *, path: Path | None, tensor_key: str | None) -> None:
        if tensor.ndim != 2:
            raise ValueError(f"Embedding tensor must be 2D, got shape={tuple(tensor.shape)}")
        if not torch.is_floating_point(tensor):
            raise ValueError(f"Embedding tensor dtype must be floating, got {tensor.dtype}")
        self.tensor = tensor.detach().cpu().contiguous()
        self.path = path
        self.tensor_key = tensor_key
        self.source_type = "torch_checkpoint"

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        tensor_key: str | None = None,
    ) -> "TorchTensorEmbeddingMatrixSource":
        p = Path(path)
        payload = _torch_load_compatible(p)
        selected_key, tensor = _extract_embedding_tensor(payload, tensor_key=tensor_key)
        return cls(tensor, path=p, tensor_key=selected_key)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.tensor.shape[0]), int(self.tensor.shape[1])

    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta["tensor_key"] = self.tensor_key
        return meta

    def copy_slice_to(
        self,
        target: torch.Tensor,
        *,
        row_start: int,
        col_start: int,
        chunk_rows: int,
    ) -> None:
        rows, cols = _validate_target_slice(self.shape, target, row_start=row_start, col_start=col_start)
        for local_start in range(0, rows, max(1, int(chunk_rows))):
            local_end = min(rows, local_start + max(1, int(chunk_rows)))
            chunk = self.tensor[
                row_start + local_start: row_start + local_end,
                col_start: col_start + cols,
            ]
            target[local_start:local_end].copy_(chunk.to(device=target.device, dtype=target.dtype))


def _torch_load_compatible(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _lookup_key(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping) and key in payload:
        return payload[key]
    current = payload
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise KeyError(key)
    return current


def _iter_floating_2d_tensors(payload: Any, prefix: str = ""):
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 2 and torch.is_floating_point(payload):
            yield prefix.removeprefix("."), payload
        return
    if isinstance(payload, torch.nn.Parameter):
        tensor = payload.detach()
        if tensor.ndim == 2 and torch.is_floating_point(tensor):
            yield prefix.removeprefix("."), tensor
        return
    if hasattr(payload, "state_dict"):
        try:
            yield from _iter_floating_2d_tensors(payload.state_dict(), prefix)
        except Exception:
            return
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_floating_2d_tensors(value, next_prefix)
        return
    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_floating_2d_tensors(value, next_prefix)


def _extract_embedding_tensor(payload: Any, *, tensor_key: str | None) -> tuple[str | None, torch.Tensor]:
    if tensor_key is not None:
        value = _lookup_key(payload, tensor_key)
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Tensor key {tensor_key!r} resolved to {type(value).__name__}, not torch.Tensor")
        return tensor_key, value

    if isinstance(payload, torch.nn.Embedding):
        return "weight", payload.weight.detach()
    if isinstance(payload, torch.Tensor):
        return None, payload

    candidates = list(_iter_floating_2d_tensors(payload))
    if not candidates:
        raise ValueError(
            "Could not find a 2D floating tensor in the torch checkpoint. "
            "Pass --tensor-key, for example --tensor-key node_embedding.weight."
        )
    by_key = {key: tensor for key, tensor in candidates}
    for preferred in PREFERRED_TORCH_TENSOR_KEYS:
        if preferred in by_key:
            return preferred, by_key[preferred]
        suffix = f".{preferred}"
        suffix_matches = [(key, tensor) for key, tensor in candidates if key.endswith(suffix)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    if len(candidates) == 1:
        return candidates[0]
    keys = ", ".join(key or "<root>" for key, _ in candidates[:20])
    raise ValueError(
        "Found multiple 2D floating tensors in the torch checkpoint; pass --tensor-key. "
        f"Candidates: {keys}"
    )


def _validate_target_slice(
    source_shape: tuple[int, int],
    target: torch.Tensor,
    *,
    row_start: int,
    col_start: int,
) -> tuple[int, int]:
    if target.ndim != 2:
        raise ValueError(f"Target embedding shard must be 2D, got shape={tuple(target.shape)}")
    rows = int(target.shape[0])
    cols = int(target.shape[1])
    if row_start < 0 or col_start < 0:
        raise ValueError(f"Negative embedding slice offset: row_start={row_start}, col_start={col_start}")
    if row_start + rows > source_shape[0] or col_start + cols > source_shape[1]:
        raise ValueError(
            "Target embedding shard is outside source matrix: "
            f"source_shape={source_shape}, row_start={row_start}, col_start={col_start}, "
            f"target_shape={tuple(target.shape)}"
        )
    return rows, cols


def _local_shards(value: Any) -> list[Any] | None:
    if not hasattr(value, "local_shards"):
        return None
    try:
        return list(value.local_shards())
    except Exception:
        return None


def _shard_offsets_and_sizes(shard: Any, tensor: torch.Tensor) -> tuple[list[int], list[int]]:
    metadata = getattr(shard, "metadata", None)
    offsets = getattr(metadata, "shard_offsets", None) if metadata is not None else None
    sizes = getattr(metadata, "shard_sizes", None) if metadata is not None else None
    if offsets is None:
        offsets = [0, 0]
    if sizes is None:
        sizes = list(tensor.shape)
    return [int(v) for v in offsets], [int(v) for v in sizes]


def _copy_source_into_model_state(
    state: Mapping[str, Any],
    source: EmbeddingMatrixSource,
    *,
    chunk_rows: int,
) -> int:
    copied = 0
    for value in state.values():
        shards = _local_shards(value)
        if shards is not None:
            for shard in shards:
                tensor = shard.tensor
                offsets, sizes = _shard_offsets_and_sizes(shard, tensor)
                if len(offsets) < 2 or len(sizes) < 2:
                    raise ValueError(f"Unsupported embedding shard metadata offsets={offsets}, sizes={sizes}")
                expected_shape = tuple(sizes[:2])
                if tuple(tensor.shape[:2]) != expected_shape:
                    raise ValueError(
                        f"Shard tensor shape {tuple(tensor.shape)} does not match metadata sizes={sizes}"
                    )
                source.copy_slice_to(
                    tensor,
                    row_start=offsets[0],
                    col_start=offsets[1],
                    chunk_rows=chunk_rows,
                )
                copied += 1
            continue

        if isinstance(value, torch.Tensor) and tuple(value.shape) == source.shape:
            source.copy_slice_to(value, row_start=0, col_start=0, chunk_rows=chunk_rows)
            copied += 1
    if copied == 0:
        raise ValueError(
            "Could not find an embedding weight tensor in the backend model state "
            f"for source shape={source.shape}"
        )
    return copied


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        resolved = device if isinstance(device, torch.device) else torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            resolved = torch.device("cuda", local_rank)
        return resolved
    return torch.device("cpu")


def maybe_init_distributed_for_conversion(backend: str, *, device: torch.device, dist_backend: str = "auto") -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    if backend != "torchrec":
        raise ValueError("Distributed conversion is supported only for backend='torchrec'")
    if dist_backend == "auto":
        dist_backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(dist_backend)


def save_embedding_matrix_as_checkpoint(
    source: EmbeddingMatrixSource,
    *,
    out_dir: str | Path,
    backend: str,
    dtype: str = "fp32",
    table_name: str = "node_table",
    feature_name: str = "node",
    sharding_type: str = "row_wise",
    compute_kernel_policy: str = "auto",
    device: str | torch.device | None = None,
    step: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ConversionResult:
    if backend not in {"torchrec", "vanilla"}:
        raise ValueError(f"Unsupported embedding checkpoint backend: {backend}")
    num_embeddings, embedding_dim = source.shape
    resolved_device = _resolve_device(device)
    process_group = dist.group.WORLD if backend == "torchrec" and dist.is_available() and dist.is_initialized() else None
    config = EmbeddingTableConfig(
        backend=backend,  # type: ignore[arg-type]
        num_embeddings=int(num_embeddings),
        embedding_dim=int(embedding_dim),
        dtype=dtype,  # type: ignore[arg-type]
        table_name=table_name,
        feature_name=feature_name,
        init_type="zeros",
        init_kwargs={},
        optimizer_type=None,
        optimizer_kwargs={},
        sharding_type=sharding_type,
        compute_kernel_policy=compute_kernel_policy,  # type: ignore[arg-type]
        device=None,
    )
    table = create_embedding_table(config, device=resolved_device, process_group=process_group)
    state = table.local_model_state_dict()
    _copy_source_into_model_state(state, source, chunk_rows=chunk_rows)
    table.load_local_model_state_dict(state)
    table.save_local(str(out_dir), step=step)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    result = ConversionResult(
        out_dir=str(out_dir),
        backend=backend,
        dtype=dtype,
        num_embeddings=int(num_embeddings),
        embedding_dim=int(embedding_dim),
        table_name=table_name,
        feature_name=feature_name,
        step=step,
        source=source.metadata(),
        rank=distributed_rank(process_group),
        world_size=distributed_world_size(process_group),
    )
    if result.rank == 0:
        metadata = {
            "format": CHECKPOINT_METADATA_FORMAT,
            "embedding_table_config": asdict(config),
            "conversion": asdict(result),
            "extra": dict(extra_metadata or {}),
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return result

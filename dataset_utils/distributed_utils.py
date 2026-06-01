"""Shared distributed-training primitives reused by all link-prediction trainers.

These small helpers are shared by the embedding trainers. Consolidating them
here keeps the distributed bootstrap consistent and means future fixes (NCCL
handling, env overrides) only need to land in one place.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def distributed_is_initialized() -> bool:
    """True iff `torch.distributed` is available AND has been initialised."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def maybe_init_distributed(backend: str) -> tuple[int, int]:
    """Initialise `torch.distributed` if `WORLD_SIZE > 1` and not already set up.

    Returns `(rank, world_size)`. In a single-process run returns `(0, 1)`.
    """
    if distributed_is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1
    torch.distributed.init_process_group(backend=backend)
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def infer_device(device_name: str | None) -> torch.device:
    """Resolve a torch.device from an explicit string or, failing that, from
    `LOCAL_RANK` (CUDA) or fall back to CPU.
    """
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def local_batch_limit(global_limit: int | None, *, rank: int, world_size: int) -> int | None:
    """Translate a *global* maximum number of batches into a per-rank quota.

    Returns None when no limit is set (caller iterates the loader to its end).
    """
    if global_limit is None:
        return None
    limit = int(global_limit)
    if limit <= 0:
        return 0
    return max(0, (limit + int(world_size) - 1 - int(rank)) // max(1, int(world_size)))


def gather_eval_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Gather a tuple of per-rank ndarrays to rank 0; non-zero ranks get empty
    arrays. Use during evaluation when metrics need the full prediction set.
    """
    if not distributed_is_initialized():
        return tuple(arrays)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(tuple(arrays), gathered, dst=0)
    if rank != 0 or gathered is None:
        return tuple(np.empty(0, dtype=arr.dtype) for arr in arrays)
    out: list[np.ndarray] = []
    for idx in range(len(arrays)):
        chunks = [np.asarray(item[idx]) for item in gathered if item is not None]
        out.append(np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=arrays[idx].dtype))
    return tuple(out)


def all_reduce_float_array(values: np.ndarray) -> np.ndarray:
    """Sum a small float array across all ranks and return the reduced copy."""
    values_arr = np.asarray(values, dtype=np.float64)
    if not distributed_is_initialized():
        return values_arr
    backend = str(torch.distributed.get_backend()).lower()
    device = torch.device("cpu")
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.as_tensor(values_arr, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.cpu().numpy()


def broadcast_metrics(metrics: dict[str, float] | None) -> dict[str, float] | None:
    """Broadcast a metrics dict computed only on rank 0 back to every rank."""
    if not distributed_is_initialized():
        return metrics
    container = [metrics]
    torch.distributed.broadcast_object_list(container, src=0)
    return container[0]

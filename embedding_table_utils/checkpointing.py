from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch.distributed as dist

from .chunked_state_dict import (
    chunked_restore,
    chunked_save_state_dict,
    load_chunked_manifest,
)
from .config import EmbeddingTableConfig


FORMAT_VERSION = 2


def distributed_rank(process_group: dist.ProcessGroup | None = None) -> int:
    if dist.is_available() and dist.is_initialized():
        group = process_group if process_group is not None else dist.group.WORLD
        return int(dist.get_rank(group))
    return 0


def distributed_world_size(process_group: dist.ProcessGroup | None = None) -> int:
    if dist.is_available() and dist.is_initialized():
        group = process_group if process_group is not None else dist.group.WORLD
        return int(dist.get_world_size(group))
    return 1


def rank_checkpoint_dir(ckpt_dir: str | Path, rank: int) -> Path:
    return Path(ckpt_dir) / f"rank{rank}"


def save_local_checkpoint(
    *,
    ckpt_dir: str | Path,
    config: EmbeddingTableConfig,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    step: int | None,
    process_group: dist.ProcessGroup | None,
) -> Path:
    ckpt_root = Path(ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    rank = distributed_rank(process_group)
    rank_dir = rank_checkpoint_dir(ckpt_root, rank)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": asdict(config),
        "model": model_state,
        "optimizer": optimizer_state,
        "step": step,
        "rank": rank,
        "world_size": distributed_world_size(process_group),
    }
    chunked_save_state_dict(payload, rank_dir)
    return rank_dir


def load_local_checkpoint(
    *,
    ckpt_dir: str | Path,
    process_group: dist.ProcessGroup | None,
    model_template: dict[str, Any] | None = None,
    optimizer_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the per-rank checkpoint, streaming any chunked tensors directly
    into ``model_template`` / ``optimizer_template`` in-place where possible.

    The returned dict has the same shape as the saved payload: ``config``,
    ``model``, ``optimizer``, ``step``, ``rank``, ``world_size``. Tensors in
    ``model`` and ``optimizer`` are the same objects as in the templates when
    streaming was possible, otherwise freshly-allocated CPU tensors.
    """
    rank = distributed_rank(process_group)
    rank_dir = rank_checkpoint_dir(ckpt_dir, rank)
    if not rank_dir.is_dir():
        raise FileNotFoundError(f"Local checkpoint shard not found for rank {rank}: {rank_dir}")

    manifest = load_chunked_manifest(rank_dir)
    stored_format = int(manifest.get("format_version", -1))
    if stored_format != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported local checkpoint format_version={stored_format}, expected {FORMAT_VERSION}"
        )

    expected_world_size = distributed_world_size(process_group)
    stored_world_size = int(manifest.get("world_size", expected_world_size))
    if stored_world_size != expected_world_size:
        raise ValueError(
            f"Checkpoint world_size={stored_world_size} does not match current world_size={expected_world_size}. "
            "Local per-rank checkpoints are intended only for resume with the same number of ranks."
        )

    restored_model = chunked_restore(manifest["model"], model_template, rank_dir)
    saved_optimizer = manifest.get("optimizer")
    if saved_optimizer is None:
        restored_optimizer: dict[str, Any] | None = None
    else:
        restored_optimizer = chunked_restore(saved_optimizer, optimizer_template, rank_dir)

    return {
        "format_version": stored_format,
        "config": manifest["config"],
        "model": restored_model,
        "optimizer": restored_optimizer,
        "step": manifest.get("step"),
        "rank": manifest.get("rank"),
        "world_size": stored_world_size,
    }

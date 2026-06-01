"""Method-agnostic frozen embedding table.

Embedding trainers and downstream LP methods need to load a trained embedding
table for read-only use during downstream classifier training.
This module centralises that path so each method doesn't ship its own copy.

Key idea: `save_local_checkpoint` already serialises `EmbeddingTableConfig`
inside each rank's manifest. That config has everything we need to
reconstruct the table — `num_embeddings`, `embedding_dim`, `dtype`,
`sharding_type`, `backend` — without ever touching the method-specific
training config.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from .api import BaseEmbeddingTable
from .checkpointing import (
    FORMAT_VERSION,
    distributed_rank,
    distributed_world_size,
    rank_checkpoint_dir,
)
from .chunked_state_dict import (
    MANIFEST_FILE_NAME,
    chunked_restore,
    load_chunked_manifest,
)
from .config import EmbeddingTableConfig
from .factory import create_embedding_table


def _has_rank0_shard(p: Path) -> bool:
    return (p / "rank0" / MANIFEST_FILE_NAME).exists()


def _resolve_inner_ckpt_dir(ckpt_dir: str | Path) -> Path:
    """Resolve a path that may point at the embedding-table dir directly,
    its containing step dir, or a parent dir with a `latest_checkpoint.txt`.

    Recognised layouts (from outer to inner):
      - <parent>/latest_checkpoint.txt -> step subdir
      - <step>/embedding_table/rank0/manifest.pt
      - <inner>/rank0/manifest.pt              (already inner)
    """
    p = Path(ckpt_dir)
    if _has_rank0_shard(p):
        return p
    if _has_rank0_shard(p / "embedding_table"):
        return p / "embedding_table"
    latest = p / "latest_checkpoint.txt"
    if latest.exists():
        name = latest.read_text(encoding="utf-8").strip()
        candidate = p / name / "embedding_table"
        if _has_rank0_shard(candidate):
            return candidate
        candidate = p / name
        if _has_rank0_shard(candidate):
            return candidate
    raise FileNotFoundError(
        f"Failed to resolve embedding-table checkpoint directory from {p}. "
        "Expected either an inner dir with rank0/manifest.pt, an outer step dir with "
        "embedding_table/rank0/manifest.pt, or a parent dir with latest_checkpoint.txt."
    )


def _load_inner_manifest(
    ckpt_dir: Path,
    *,
    process_group: dist.ProcessGroup | None,
) -> tuple[dict, Path]:
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
            f"Checkpoint world_size={stored_world_size} does not match current world_size={expected_world_size}"
        )
    return manifest, rank_dir


class ReadOnlyEmbeddingStore(nn.Module):
    """Frozen embedding table loaded from a checkpoint (or initialised randomly).

    Construct via the two classmethod factories:

      * `from_checkpoint(ckpt_dir, device, process_group=None, ...)` —
        load weights from disk. The `EmbeddingTableConfig` is recovered
        directly from the saved shards, so this works for any LP method
        that uses `save_local_checkpoint`.

      * `from_config(config, device, process_group=None, seed=None)` —
        build a fresh randomly-initialised table for callers that explicitly
        need one.

    The returned module exposes `lookup(ids)` under `@torch.no_grad`, plus
    `num_nodes` / `embedding_dim` properties for sizing downstream layers.
    """

    embedding_table: BaseEmbeddingTable

    def __init__(self, embedding_table: BaseEmbeddingTable) -> None:
        super().__init__()
        self.embedding_table = embedding_table
        self.embedding_table.eval()
        for param in self.embedding_table.parameters():
            param.requires_grad_(False)

    @property
    def num_nodes(self) -> int:
        return int(self.embedding_table.config.num_embeddings)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding_table.config.embedding_dim)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
        expected_num_nodes: int | None = None,
        expected_embedding_dim: int | None = None,
    ) -> "ReadOnlyEmbeddingStore":
        inner_dir = _resolve_inner_ckpt_dir(ckpt_dir)
        manifest, rank_dir = _load_inner_manifest(inner_dir, process_group=process_group)
        config = EmbeddingTableConfig(**manifest["config"])
        if expected_num_nodes is not None and int(expected_num_nodes) != int(config.num_embeddings):
            raise ValueError(
                f"Checkpoint num_embeddings={config.num_embeddings} does not match "
                f"expected={expected_num_nodes}"
            )
        if expected_embedding_dim is not None and int(expected_embedding_dim) != int(config.embedding_dim):
            raise ValueError(
                f"Checkpoint embedding_dim={config.embedding_dim} does not match "
                f"expected={expected_embedding_dim}"
            )
        embedding_table = create_embedding_table(
            config,
            device=device,
            process_group=process_group,
        )
        # Stream weights directly into the freshly-built table's storage.
        model_template = embedding_table.local_model_state_dict()
        restored_model = chunked_restore(manifest["model"], model_template, rank_dir)
        embedding_table.load_local_model_state_dict(restored_model)
        # Optimizer state is irrelevant for a frozen table; skip restoring it.
        return cls(embedding_table=embedding_table)

    @classmethod
    def from_config(
        cls,
        config: EmbeddingTableConfig,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
        seed: int | None = None,
    ) -> "ReadOnlyEmbeddingStore":
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        embedding_table = create_embedding_table(
            config,
            device=device,
            process_group=process_group,
        )
        return cls(embedding_table=embedding_table)

    @torch.no_grad()
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table.lookup(ids)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lookup(ids)

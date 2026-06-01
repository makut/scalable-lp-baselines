from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .positive_edges import GraphCSRPositiveEdgeDataset


@dataclass(slots=True)
class EdgeSplits:
    val_pos: np.ndarray
    val_neg: np.ndarray
    test_pos: np.ndarray
    test_neg: np.ndarray


def _load_edges(path: str | None, *, mmap: bool) -> np.ndarray | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    array = np.load(path, mmap_mode="r" if mmap else None)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Edge array at {path} must have shape [N, 2], got {array.shape}")
    if np.asarray(array).dtype == np.int64:
        return array
    return np.asarray(array, dtype=np.int64)


def _splitmix64_vec(values: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 — deterministic per-element hash for sort keys."""
    v = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
    v = (v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    v = (v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    v = v ^ (v >> np.uint64(31))
    return v


class LabeledEdgeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pos_edges: np.ndarray,
        neg_edges: np.ndarray,
    ) -> None:
        pos = np.asarray(pos_edges, dtype=np.int64)
        neg = np.asarray(neg_edges, dtype=np.int64)
        all_edges = np.concatenate([pos, neg], axis=0)
        all_labels = np.concatenate(
            [np.ones(pos.shape[0], dtype=np.float32), np.zeros(neg.shape[0], dtype=np.float32)],
            axis=0,
        )
        # Sort by a randomized source key, then by src. This keeps every source
        # contiguous for streaming per-source eval while avoiding low-id bias
        # when max_val_batches / max_test_batches truncates the pass.
        order = np.lexsort((all_edges[:, 0], _splitmix64_vec(all_edges[:, 0])))
        self.edges = all_edges[order]
        self.labels = all_labels[order]

    def __len__(self) -> int:
        return int(self.edges.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(idx)
        return torch.from_numpy(self.edges[index]), torch.tensor(self.labels[index], dtype=torch.float32)


class ContiguousShardSampler(Sampler[int]):
    def __init__(self, dataset_len: int, *, rank: int, world_size: int) -> None:
        self.dataset_len = int(dataset_len)
        shard_bounds = np.linspace(0, self.dataset_len, int(world_size) + 1, dtype=np.int64)
        self.start = int(shard_bounds[int(rank)])
        self.end = int(shard_bounds[int(rank) + 1])

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return max(0, self.end - self.start)


def build_eval_loader(
    *,
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    device_type: str,
    collate_fn=None,
) -> DataLoader:
    dataset_len = len(dataset)  # type: ignore[arg-type]
    sampler = ContiguousShardSampler(dataset_len, rank=rank, world_size=world_size) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_eval_dataset(
    *,
    dataset_config: Any,
    split_name: str,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
) -> Dataset[Any]:
    del dataset_config, split_name
    return LabeledEdgeDataset(pos_edges, neg_edges)


def load_edge_splits(config: Any) -> EdgeSplits:
    """Load val/test positive and negative edges from npy files.

    Required config attributes: `valid_edge_path`, `valid_edge_neg_path`,
    `test_edge_path`, `test_edge_neg_path`. Optional `mmap` (default True).
    """
    use_mmap = bool(getattr(config, "mmap", True))
    paths = {
        "valid_edge_path": getattr(config, "valid_edge_path", None),
        "valid_edge_neg_path": getattr(config, "valid_edge_neg_path", None),
        "test_edge_path": getattr(config, "test_edge_path", None),
        "test_edge_neg_path": getattr(config, "test_edge_neg_path", None),
    }
    loaded = {name: _load_edges(path, mmap=use_mmap) for name, path in paths.items()}
    missing = [name for name, value in loaded.items() if value is None]
    if missing:
        raise ValueError(f"Missing required dataset paths: {missing}")
    return EdgeSplits(
        val_pos=np.asarray(loaded["valid_edge_path"], dtype=np.int64),
        val_neg=np.asarray(loaded["valid_edge_neg_path"], dtype=np.int64),
        test_pos=np.asarray(loaded["test_edge_path"], dtype=np.int64),
        test_neg=np.asarray(loaded["test_edge_neg_path"], dtype=np.int64),
    )


def build_train_positive_dataset(*, dataset_config: Any) -> Dataset[torch.Tensor]:
    return GraphCSRPositiveEdgeDataset(dataset_config)

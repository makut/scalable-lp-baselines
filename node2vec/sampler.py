"""Dataset and collator for node2vec training batches.

Each item is a node id; the collator runs a random walk for each start id in
the batch, replicates by `walks_per_node`, and slices each walk into
`context_size`-length windows. The positive sample is the window itself; the
negative sample has the same anchor with `walk_length` uniformly random nodes
substituted for the walk body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Node2VecConfig
from .random_walk import (
    negative_sample_windows,
    prepare_rowptr_col,
    random_walk_windows,
    seed_numba_random,
)


@dataclass(slots=True)
class Node2VecBatch:
    pos_rw: torch.Tensor
    # Exactly one of `neg_rw` / `shared_neg_ids` is populated, depending on
    # `enable_shared_negatives` in the collator.
    neg_rw: torch.Tensor | None = None
    shared_neg_ids: torch.Tensor | None = None


class NodeRangeDataset(Dataset[int]):
    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = int(num_nodes)

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> int:
        return int(idx)


class NodeDistributedSampler(Sampler[int]):
    """Shards node ids across ranks. Mirrors `torch.utils.data.distributed.DistributedSampler`
    but lives inside the module to avoid importing the full distributed package at module
    import time when distributed is not initialised.
    """

    def __init__(
        self,
        dataset: NodeRangeDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        n = len(dataset)
        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            if len(indices) < self.total_size:
                indices = indices + indices[: (self.total_size - len(indices))]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class Node2VecCollator:
    """Turns a list of start-node ids into (pos_rw, neg_rw) tensors via random walk.

    Picklable for `num_workers > 0`: only numpy arrays + plain config fields are held.
    Worker-process RNG is seeded once per worker process on first use.
    """

    def __init__(
        self,
        rowptr: np.ndarray,
        col: np.ndarray,
        *,
        walk_length: int,
        context_size: int,
        walks_per_node: int,
        p: float,
        q: float,
        num_negative_samples: int,
        num_nodes: int,
        base_seed: int = 0,
        is_directed: bool = False,
        enable_shared_negatives: bool = False,
    ) -> None:
        self.rowptr = rowptr
        self.col = col
        self.walk_length = int(walk_length)
        self.context_size = int(context_size)
        self.walks_per_node = int(walks_per_node)
        self.p = float(p)
        self.q = float(q)
        self.num_negative_samples = int(num_negative_samples)
        self.num_nodes = int(num_nodes)
        self.base_seed = int(base_seed)
        self.is_directed = bool(is_directed)
        self.enable_shared_negatives = bool(enable_shared_negatives)
        # `internal_walk_length` matches the reference: number of additional
        # sampled positions after the initial start node.
        self._internal_walk_length = self.walk_length - 1
        self._num_windows = self._internal_walk_length + 1 + 1 - self.context_size
        if self._num_windows < 1:
            raise ValueError(
                "context_size is incompatible with walk_length: "
                f"got walk_length={self.walk_length}, context_size={self.context_size}"
            )

    def _seed_worker(self) -> None:
        # Called inside worker / per-call. Combine base seed with worker id so each
        # worker draws independent walks.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        # The seed itself bumps every call via os.getpid + per-call counter? No —
        # numpy.random uses a global state; we just need different workers to be
        # different. Reseed only once per worker process to avoid expensive reseeds
        # in the hot path.
        if getattr(self, "_seeded_worker_id", None) == worker_id:
            return
        seed = (self.base_seed + worker_id) & 0xFFFFFFFF
        np.random.seed(seed)
        seed_numba_random(seed)
        self._seeded_worker_id = worker_id

    def _pos_walks(self, starts: np.ndarray) -> np.ndarray:
        return random_walk_windows(
            self.rowptr,
            self.col,
            starts,
            self._internal_walk_length,
            self.context_size,
            self.p,
            self.q,
            walks_per_start=self.walks_per_node,
            is_directed=self.is_directed,
        )

    def _neg_walks(self, starts: np.ndarray) -> np.ndarray:
        return negative_sample_windows(
            starts,
            walk_length=self._internal_walk_length,
            context_size=self.context_size,
            walks_per_start=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            num_nodes=self.num_nodes,
        )

    def _shared_neg_ids(self) -> np.ndarray:
        # Pool size matches num_negative_samples: every positive anchor in the
        # batch will be scored against this single shared pool.
        return np.random.randint(
            0, self.num_nodes, size=self.num_negative_samples, dtype=np.int64
        )

    def __call__(self, batch: list[int]) -> Node2VecBatch:
        self._seed_worker()
        starts = np.asarray(batch, dtype=np.int64)
        pos_rw = self._pos_walks(starts)
        if self.enable_shared_negatives:
            shared_neg_ids = self._shared_neg_ids()
            return Node2VecBatch(
                pos_rw=torch.from_numpy(pos_rw),
                shared_neg_ids=torch.from_numpy(shared_neg_ids),
            )
        neg_rw = self._neg_walks(starts)
        return Node2VecBatch(
            pos_rw=torch.from_numpy(pos_rw),
            neg_rw=torch.from_numpy(neg_rw),
        )


def build_train_loader(
    *,
    rowptr: np.ndarray,
    col: np.ndarray,
    num_nodes: int,
    config: Node2VecConfig,
    rank: int,
    world_size: int,
    device_type: str,
) -> torch.utils.data.DataLoader:
    num_sampler_workers = int(config.num_sampler_workers)
    dataset = NodeRangeDataset(num_nodes)
    sampler = NodeDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=config.drop_last,
    )
    collator = Node2VecCollator(
        rowptr=rowptr,
        col=col,
        walk_length=config.walk_length,
        context_size=config.context_size,
        walks_per_node=config.walks_per_node,
        p=config.p,
        q=config.q,
        num_negative_samples=config.num_negative_samples,
        num_nodes=num_nodes,
        base_seed=int(config.seed) + 1_000 * int(rank),
        is_directed=bool(config.is_directed),
        enable_shared_negatives=bool(config.enable_shared_negatives),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        shuffle=False,
        drop_last=config.drop_last,
        collate_fn=collator,
        num_workers=num_sampler_workers,
        pin_memory=bool(config.pin_memory) and device_type == "cuda",
        persistent_workers=int(config.num_sampler_workers) > 0,
    )

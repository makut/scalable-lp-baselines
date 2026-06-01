from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import get_worker_info

from .batch_types import RawLinkBatch
from .negative_sampling import NegativeSampler


class LinkPredictionCollator:
    def __init__(
        self,
        *,
        negative_sampler: NegativeSampler,
        batch_transform: Any | None = None,
        static_meta: dict[str, Any] | None = None,
    ) -> None:
        self.negative_sampler = negative_sampler
        self.batch_transform = batch_transform
        self.static_meta = {} if static_meta is None else dict(static_meta)

    def __call__(self, batch: list[torch.Tensor]):
        pos_edges = torch.stack(batch, dim=0).to(dtype=torch.int64)
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        meta = dict(self.static_meta)
        meta.update(
            worker_id=worker_id,
            num_pos=int(pos_edges.shape[0]),
        )
        neg_edges = self.negative_sampler.sample(pos_edges, meta=meta).to(dtype=torch.int64)
        meta["num_neg"] = int(neg_edges.shape[0])

        raw_batch = RawLinkBatch(
            pos_edges=pos_edges,
            neg_edges=neg_edges,
            pos_labels=torch.ones(pos_edges.shape[0], dtype=torch.float32),
            neg_labels=torch.zeros(neg_edges.shape[0], dtype=torch.float32),
            meta=meta,
        )

        if self.batch_transform is None:
            return raw_batch
        return self.batch_transform(raw_batch)

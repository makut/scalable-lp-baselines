from __future__ import annotations

from typing import Any, Protocol

import torch

from .batch_types import EdgeLabelBatch, RawLinkBatch
from .config import BatchTransformConfig


class BatchTransform(Protocol):
    def __call__(self, batch: RawLinkBatch) -> Any:
        ...


class EdgeLabelBatchTransform:
    def __call__(self, batch: RawLinkBatch) -> EdgeLabelBatch:
        edges = torch.cat([batch.pos_edges, batch.neg_edges], dim=0)
        labels = torch.cat([batch.pos_labels, batch.neg_labels], dim=0)
        return EdgeLabelBatch(edges=edges, labels=labels, meta=dict(batch.meta))


class IdentityBatchTransform:
    def __call__(self, batch: RawLinkBatch) -> RawLinkBatch:
        return batch


BATCH_TRANSFORM_REGISTRY = {
    "edge_label": EdgeLabelBatchTransform,
    "identity": IdentityBatchTransform,
}


def build_batch_transform(cfg: BatchTransformConfig) -> BatchTransform:
    name = str(cfg.name)
    if name not in BATCH_TRANSFORM_REGISTRY:
        raise ValueError(f"Unsupported batch transform: {name}")
    transform_cls = BATCH_TRANSFORM_REGISTRY[name]
    return transform_cls(**dict(cfg.extra_kwargs))

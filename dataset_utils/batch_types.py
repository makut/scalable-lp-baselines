from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class RawLinkBatch:
    pos_edges: torch.Tensor
    neg_edges: torch.Tensor
    pos_labels: torch.Tensor
    neg_labels: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeLabelBatch:
    edges: torch.Tensor
    labels: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)

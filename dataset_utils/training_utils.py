"""Small training-loop helpers shared across LP trainers.

These helpers are shared by the training entry points. Centralising them makes
the contract explicit and removes incidental drift between methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from .batch_types import EdgeLabelBatch


MetricMode = Literal["min", "max"]


def metric_improved(value: float, *, best: float | None, mode: MetricMode) -> bool:
    """Return True iff `value` is a strict improvement over `best` for `mode`.

    `best` may be None (no prior best, so any value is an improvement).
    `mode` is `"min"` (lower-is-better, e.g. logloss) or `"max"`
    (higher-is-better, e.g. roc_auc).
    """
    if best is None:
        return True
    if mode == "min":
        return float(value) < float(best)
    return float(value) > float(best)


@dataclass
class EarlyStopping:
    """Patience-based early stopping over an arbitrary validation metric.

    Counts consecutive evaluations without `min_delta`-improvement on the
    monitored metric. Once the counter reaches `patience`, `update()`
    returns `True` and the trainer should break out of its loop.

    Built independently of `metric_improved` because patience semantics use
    `min_delta` (a metric drift smaller than `min_delta` does NOT reset the
    counter), while best-checkpoint tracking typically uses strict
    improvement. The two coexist on the same metric, but their definitions
    of "improvement" differ.

    Disabled when `patience is None` (acts as a no-op) — convenient for
    threading through configs.
    """

    patience: int | None
    mode: MetricMode = "max"
    min_delta: float = 0.0
    best: float | None = None
    num_bad: int = 0
    stopped: bool = False
    last_value: float | None = None

    @property
    def enabled(self) -> bool:
        return self.patience is not None and int(self.patience) >= 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        delta = float(self.min_delta)
        if self.mode == "min":
            return float(value) < float(self.best) - delta
        return float(value) > float(self.best) + delta

    def update(self, value: float | None) -> bool:
        """Record a new monitored value. Returns True iff training should stop.

        `None` is treated as "no observation this round" and is a no-op so
        trainers can pass through missing metrics without special-casing.
        """
        if not self.enabled or self.stopped or value is None:
            return self.stopped
        value_f = float(value)
        self.last_value = value_f
        if self._is_improvement(value_f):
            self.best = value_f
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= int(self.patience):  # type: ignore[arg-type]
                self.stopped = True
        return self.stopped

    def reason(self) -> str:
        if not self.stopped:
            return ""
        return (
            f"early stopping: no improvement for {self.num_bad} eval(s), "
            f"best={self.best} last={self.last_value} mode={self.mode} "
            f"min_delta={self.min_delta}"
        )


def unpack_edge_label_batch(
    batch: EdgeLabelBatch | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(edges, labels)` regardless of whether the batch is the typed
    `EdgeLabelBatch` (used by train collator) or a plain `(edges, labels)`
    tuple (used by `LabeledEdgeDataset` for eval)."""
    if isinstance(batch, EdgeLabelBatch):
        return batch.edges, batch.labels
    return batch


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and assert that the top-level value is a mapping."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML object at {p} must be a mapping")
    return data

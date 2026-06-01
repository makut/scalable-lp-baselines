"""Unified evaluation metrics for link prediction.

The current distributed eval path is source-local: candidates are streamed in
contiguous source groups, each completed group is reduced to per-source metric
sums, and distributed workers only all-reduce those small sums.

Supported source-local metrics:
  * `hits@K`: whether the source has at least one positive in top-K.
  * `recall@K`: #positives in top-K / total positives for the source.
  * `ndcg@K`: binary nDCG at K.
  * `mrr` / `mrr@K`: reciprocal rank of the first positive.
  * `roc_auc@K`: binary AUC inside the top-K prefix, averaged per source.

Sources without any positive label are skipped for averaged ranking metrics.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalise_metric_name(metric: str) -> str:
    return str(metric).replace("-", "_")


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _per_source_metric_keys(metrics: Iterable[str], k_values: tuple[int, ...]) -> list[str]:
    keys: list[str] = []
    for raw_metric in metrics:
        metric = _normalise_metric_name(raw_metric)
        if metric == "roc_auc":
            # Compatibility: source-local eval replaces global ROC-AUC with
            # the top-K source-local version.
            metric = "roc_auc@k"
        if metric in {"hits@k", "recall@k", "ndcg@k", "mrr@k", "roc_auc@k"}:
            prefix = metric.removesuffix("@k")
            for k in k_values:
                _append_unique(keys, f"{prefix}@{int(k)}")
        elif metric == "mrr":
            _append_unique(keys, "mrr")
    return keys


def _requested_count_metrics(metrics: Iterable[str]) -> tuple[bool, bool]:
    names = {_normalise_metric_name(metric) for metric in metrics}
    return "num_sources" in names, "num_sources_with_positives" in names


def _auc_from_ranked_prefix(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0:
        return 0.0
    if n_neg == 0:
        return 1.0
    neg_below = 0
    wins = 0.0
    end = int(labels.size)
    while end > 0:
        start = end - 1
        while start > 0 and float(scores[start - 1]) == float(scores[end - 1]):
            start -= 1
        group = labels[start:end]
        group_pos = int(group.sum())
        group_neg = int(group.size - group_pos)
        wins += float(group_pos * neg_below) + 0.5 * float(group_pos * group_neg)
        neg_below += group_neg
        end = start
    return float(wins / max(1, n_pos * n_neg))


class PerSourceMetricAccumulator:
    """Streaming accumulator for source-local ranking metrics.

    Inputs must be ordered so all rows for a source are contiguous. For
    distributed row shards, set `drop_first_src` / `drop_last_src` to discard
    boundary groups that may have been split across ranks.
    """

    def __init__(
        self,
        *,
        metrics: Iterable[str] = ("roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr", "mrr@k"),
        metrics_at_k: Iterable[int] = (10, 50, 100),
        apply_sigmoid: bool = True,
        drop_first_src: bool = False,
        drop_last_src: bool = False,
    ) -> None:
        self.metrics = tuple(metrics)
        self.k_values = tuple(int(k) for k in metrics_at_k)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.drop_first_src = bool(drop_first_src)
        self.drop_last_src = bool(drop_last_src)
        self.metric_keys = _per_source_metric_keys(self.metrics, self.k_values)
        self.wants_num_sources, self.wants_num_sources_with_positives = _requested_count_metrics(self.metrics)

        self._sums = {key: 0.0 for key in self.metric_keys}
        self._num_sources = 0.0
        self._num_sources_with_positives = 0.0
        self._current_src: int | None = None
        self._current_labels: list[np.ndarray] = []
        self._current_scores: list[np.ndarray] = []
        self._skip_current_on_finalize = self.drop_first_src
        self._closed = False

    def update(self, *, srcs: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot update a finalized PerSourceMetricAccumulator")
        srcs_arr = np.asarray(srcs, dtype=np.int64).reshape(-1)
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        raw_scores = np.asarray(scores).reshape(-1)
        score_arr = sigmoid(raw_scores) if self.apply_sigmoid else np.asarray(raw_scores, dtype=np.float64)
        if not (srcs_arr.shape[0] == labels_arr.shape[0] == score_arr.shape[0]):
            raise ValueError("srcs, labels and scores must have the same length")

        start = 0
        n = int(srcs_arr.shape[0])
        while start < n:
            src = int(srcs_arr[start])
            end = start + 1
            while end < n and int(srcs_arr[end]) == src:
                end += 1
            self._append_group_slice(src, labels_arr[start:end], score_arr[start:end])
            start = end

    def _append_group_slice(self, src: int, labels: np.ndarray, scores: np.ndarray) -> None:
        if self._current_src is None:
            self._current_src = int(src)
        elif int(src) != self._current_src:
            self._finalize_current_group()
            self._current_src = int(src)
        self._current_labels.append(np.asarray(labels, dtype=np.int64))
        self._current_scores.append(np.asarray(scores, dtype=np.float64))

    def _reset_current_group(self) -> None:
        self._current_src = None
        self._current_labels = []
        self._current_scores = []

    def _finalize_current_group(self) -> None:
        if self._current_src is None:
            return
        if self._skip_current_on_finalize:
            self._skip_current_on_finalize = False
            self._reset_current_group()
            return

        labels = np.concatenate(self._current_labels, axis=0) if self._current_labels else np.empty(0, dtype=np.int64)
        scores = np.concatenate(self._current_scores, axis=0) if self._current_scores else np.empty(0, dtype=np.float64)
        self._num_sources += 1.0
        n_pos = int(labels.sum())
        if n_pos <= 0:
            self._reset_current_group()
            return
        self._num_sources_with_positives += 1.0

        order = np.argsort(-scores, kind="mergesort")
        ranked_labels = labels[order].astype(np.int64, copy=False)
        ranked_scores = scores[order].astype(np.float64, copy=False)
        pos_positions = np.flatnonzero(ranked_labels == 1)
        first_pos_rank = int(pos_positions[0]) + 1 if pos_positions.size else None
        if "mrr" in self._sums:
            self._sums["mrr"] += 0.0 if first_pos_rank is None else 1.0 / float(first_pos_rank)

        for k in self.k_values:
            top = ranked_labels[: int(k)]
            top_scores = ranked_scores[: int(k)]
            n_top = int(top.size)
            n_pos_top = int(top.sum())
            key = f"hits@{int(k)}"
            if key in self._sums:
                self._sums[key] += 1.0 if n_pos_top > 0 else 0.0
            key = f"recall@{int(k)}"
            if key in self._sums:
                self._sums[key] += float(n_pos_top / n_pos)
            key = f"ndcg@{int(k)}"
            if key in self._sums:
                log2_idx = np.log2(np.arange(2, n_top + 2, dtype=np.float64))
                dcg = float((top.astype(np.float64) / log2_idx).sum()) if n_top else 0.0
                ideal_n = min(n_pos, int(k))
                ideal_log2 = np.log2(np.arange(2, ideal_n + 2, dtype=np.float64))
                idcg = float((1.0 / ideal_log2).sum()) if ideal_n > 0 else 0.0
                self._sums[key] += dcg / idcg if idcg > 0 else 0.0
            key = f"mrr@{int(k)}"
            if key in self._sums:
                self._sums[key] += (
                    1.0 / float(first_pos_rank)
                    if first_pos_rank is not None and first_pos_rank <= int(k)
                    else 0.0
                )
            key = f"roc_auc@{int(k)}"
            if key in self._sums:
                self._sums[key] += _auc_from_ranked_prefix(top, top_scores)

        self._reset_current_group()

    def finalize(self) -> None:
        if self._closed:
            return
        if not self.drop_last_src:
            self._finalize_current_group()
        self._reset_current_group()
        self._closed = True

    def to_reduction_array(self) -> np.ndarray:
        self.finalize()
        return np.asarray(
            [
                self._num_sources,
                self._num_sources_with_positives,
                *[self._sums[key] for key in self.metric_keys],
            ],
            dtype=np.float64,
        )

    def compute(self, reduced: np.ndarray | None = None) -> dict[str, float]:
        values = self.to_reduction_array() if reduced is None else np.asarray(reduced, dtype=np.float64)
        num_sources = float(values[0]) if values.size > 0 else 0.0
        num_sources_with_positives = float(values[1]) if values.size > 1 else 0.0
        metric_sums = values[2:]
        denom = num_sources_with_positives
        out: dict[str, float] = {}
        for key, value in zip(self.metric_keys, metric_sums):
            out[key] = float(value / denom) if denom > 0 else 0.0
        if self.wants_num_sources:
            out["num_sources"] = num_sources
        if self.wants_num_sources_with_positives:
            out["num_sources_with_positives"] = num_sources_with_positives
        return out


def per_source_ranking_metrics(
    srcs: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Compute per-source ranking metrics at a single K, averaged over sources.

    Returns dict with keys `hits`, `recall`, `ndcg`. Sources without any
    positive label are skipped.
    """
    srcs_arr = np.asarray(srcs, dtype=np.int64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if srcs_arr.size == 0:
        return {"hits": 0.0, "recall": 0.0, "ndcg": 0.0}

    order = np.argsort(srcs_arr, kind="mergesort")
    acc = PerSourceMetricAccumulator(
        metrics=("hits@k", "recall@k", "ndcg@k"),
        metrics_at_k=(int(k),),
        apply_sigmoid=False,
    )
    acc.update(srcs=srcs_arr[order], labels=labels_arr[order], scores=scores_arr[order])
    result = acc.compute()
    return {
        "hits": result.get(f"hits@{int(k)}", 0.0),
        "recall": result.get(f"recall@{int(k)}", 0.0),
        "ndcg": result.get(f"ndcg@{int(k)}", 0.0),
    }


def compute_metrics(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    srcs: np.ndarray | None = None,
    metrics: Iterable[str] = ("roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr", "mrr@k"),
    metrics_at_k: Iterable[int] = (10, 50, 100),
    threshold: float = 0.5,
    apply_sigmoid: bool = True,
) -> dict[str, float]:
    """Compute global classification + per-source ranking metrics.

    Args:
        y_true: `[N]` int labels {0, 1}.
        scores: `[N]` float scores. By default treated as logits and passed
            through sigmoid; set `apply_sigmoid=False` if already in [0, 1].
        srcs: `[N]` source vertex per (src, dst) candidate, required when any
            source-local metric is requested.
        metrics: list of metric names. Supported:
            global classification: `roc_auc`, `pr_auc`, `accuracy`, `logloss`,
                `precision`, `recall`, `f1`.
            per-source ranking: `roc_auc@k`, `hits@k`, `recall@k`,
                `ndcg@k`, `mrr`, `mrr@k`.
        metrics_at_k: K values for ranking metrics.
        threshold: cutoff for binarized predictions (used by accuracy /
            precision / recall / f1).
        apply_sigmoid: whether to map scores → probabilities via sigmoid.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    raw_scores = np.asarray(scores).reshape(-1)
    score_arr = sigmoid(raw_scores) if apply_sigmoid else np.asarray(raw_scores, dtype=np.float64)
    y_pred = (score_arr >= float(threshold)).astype(np.int64)
    k_values = tuple(int(k) for k in metrics_at_k)
    metric_names = tuple(metrics)
    normalised_metrics = tuple(_normalise_metric_name(metric) for metric in metric_names)

    per_source_result: dict[str, float] | None = None

    def _per_source_result() -> dict[str, float]:
        nonlocal per_source_result
        if per_source_result is None:
            if srcs is None:
                raise ValueError("source-local ranking metrics require `srcs` argument")
            srcs_arr = np.asarray(srcs, dtype=np.int64).reshape(-1)
            order = np.argsort(srcs_arr, kind="mergesort")
            acc = PerSourceMetricAccumulator(
                metrics=normalised_metrics,
                metrics_at_k=k_values,
                apply_sigmoid=False,
            )
            acc.update(srcs=srcs_arr[order], labels=y_true_arr[order], scores=score_arr[order])
            per_source_result = acc.compute()
        return per_source_result

    out: dict[str, float] = {}
    for raw_metric, metric in zip(metric_names, normalised_metrics):
        if metric == "accuracy":
            out["accuracy"] = float(accuracy_score(y_true_arr, y_pred))
        elif metric == "roc_auc":
            if np.unique(y_true_arr).size >= 2:
                out["roc_auc"] = float(roc_auc_score(y_true_arr, score_arr))
        elif metric == "pr_auc":
            out["pr_auc"] = float(average_precision_score(y_true_arr, score_arr))
        elif metric == "logloss":
            out["logloss"] = float(log_loss(y_true_arr, score_arr, labels=[0, 1]))
        elif metric == "precision":
            out["precision"] = float(precision_score(y_true_arr, y_pred, zero_division=0))
        elif metric == "recall":
            out["recall"] = float(recall_score(y_true_arr, y_pred, zero_division=0))
        elif metric == "f1":
            out["f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))
        elif metric in {"roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr@k"}:
            prefix = metric.removesuffix("@k")
            for k in k_values:
                key = f"{prefix}@{k}"
                out[key] = _per_source_result().get(key, 0.0)
        elif metric == "mrr":
            out["mrr"] = _per_source_result().get("mrr", 0.0)
        elif metric == "num_sources":
            if srcs is None:
                raise ValueError("`num_sources` requires `srcs` argument")
            out["num_sources"] = float(np.unique(np.asarray(srcs, dtype=np.int64)).size)
        elif metric == "num_sources_with_positives":
            if srcs is None:
                raise ValueError("`num_sources_with_positives` requires `srcs` argument")
            srcs_arr = np.asarray(srcs, dtype=np.int64).reshape(-1)
            pos_mask = y_true_arr == 1
            out["num_sources_with_positives"] = float(
                np.unique(srcs_arr[pos_mask]).size if pos_mask.any() else 0
            )
        else:
            raise ValueError(f"Unsupported metric: {raw_metric}")
    return out

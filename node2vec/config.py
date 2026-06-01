from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from embedding_lp.config import (
    LinkPredictionConfig,
    LinkPredictionTrainConfig,
    MetricMode,
    map_lp_negative_strategy,
    to_dataset_utils_train_loader_config,
)

LossReduction = Literal["mean", "sum"]


def default_embedding_table_config() -> dict[str, Any]:
    return {
        "backend": "torchrec",
        "embedding_dim": 128,
        "dtype": "fp32",
        "init_type": "normal",
        "init_kwargs": {},
        "table_name": "node_table",
        "feature_name": "node",
        "sharding_type": "row_wise",
        "compute_kernel_policy": "auto",
        "optimizer_type": "adam",
        "optimizer_kwargs": {"learning_rate": 0.01},
    }


@dataclass(slots=True)
class Node2VecConfig:
    """Configuration for node2vec embedding training.

    Random walk parameters follow the original node2vec paper: `p` is the
    return parameter (lower => more BFS-like, more revisits), `q` is the in-out
    parameter (lower => more DFS-like, walks venture further).
    """

    embedding_table_config: dict[str, Any] = field(default_factory=default_embedding_table_config)

    batch_size: int = 2048
    walk_length: int = 10
    context_size: int = 5
    walks_per_node: int = 5
    p: float = 1.0
    q: float = 1.0
    num_negative_samples: int = 1

    # When True, negatives are sampled once per batch (uniformly over node ids)
    # and every positive anchor in the batch is scored against the same pool.
    # Drastically reduces embedding-lookup traffic vs. the per-anchor scheme.
    enable_shared_negatives: bool = False

    use_nce_bias: bool = False
    loss_reduction: LossReduction = "mean"

    backend: str = "nccl"

    num_epochs: int = 1
    seed: int = 42
    device: str | None = None
    log_every: int = 100
    show_progress: bool = True

    checkpoint_metric: str = "train_loss"
    checkpoint_metric_mode: MetricMode = "min"
    save_best_checkpoint: bool = False
    checkpoint_every_steps: int | None = None
    checkpoint_every_epoch: bool = True
    checkpoint_dir: str | None = None
    resume_checkpoint_dir: str | None = None
    tensorboard_log_dir: str | None = None

    num_sampler_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True

    is_directed: bool = False

    # Parameter-free LP validation on val edges (dot-product scoring).
    # Enabled when val_eval_every is set and val_pos_edges/val_neg_edges are
    # passed to the trainer (CLI loads them from `dataset.split_root`).
    val_eval_every: int | None = None  # global steps between val evals; null disables
    val_batch_size: int = 65_536
    val_num_workers: int = 0
    val_metrics: tuple[str, ...] = ("roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr", "mrr@k")
    val_metrics_at_k: tuple[int, ...] = (10, 50, 100)
    val_treat_as_undirected: bool = True
    val_max_batches: int | None = None

    def __post_init__(self) -> None:
        if self.context_size < 2:
            raise ValueError("context_size must be >= 2 (anchor + at least one neighbor)")
        if self.walk_length + 1 < self.context_size:
            raise ValueError("walk_length + 1 must be >= context_size")
        if self.walks_per_node < 1:
            raise ValueError("walks_per_node must be >= 1")
        if self.num_negative_samples < 1:
            raise ValueError("num_negative_samples must be >= 1")
        if self.p <= 0.0 or self.q <= 0.0:
            raise ValueError("p and q must be positive")
        if not self.embedding_table_config:
            raise ValueError("embedding_table_config must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

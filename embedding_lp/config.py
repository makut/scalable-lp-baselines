from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dataset_utils import NegativeSamplingConfig, PositiveEdgesConfig, TrainLoaderConfig


MetricMode = Literal["min", "max"]
LPOperatorName = Literal["average", "concat", "hadamard", "weighted_l1", "weighted_l2"]
LPOptimizerName = Literal["adam", "sgd", "lbfgs"]
LPDeviceDType = Literal["fp32", "bf16", "fp16"]


@dataclass(slots=True)
class LinkPredictionTrainConfig:
    embedding_dim: int | None = None
    lp_operator: LPOperatorName = "hadamard"
    batch_size_edges: int = 262_144
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    optimizer: LPOptimizerName = "adam"
    gradient_clip_norm: float | None = None
    neg_per_pos_train: int = 1
    neg_per_pos_val: int = 1
    neg_per_pos_test: int = 1
    num_workers: int = 0
    val_num_workers: int | None = None
    device_dtype: LPDeviceDType = "fp32"
    backend: str = "nccl"
    is_directed: bool = False
    treat_as_undirected_for_lp: bool = True
    has_self_loops: bool = False
    graph_csr_root: str | None = None
    pairs_graph_csr_root: str | None = None
    graph_csr_use_mmap: bool = True
    graph_csr_file_endian: str = "little"
    graph_csr_allow_non_native: bool = True
    graph_csr_chunk_bytes: int = 256 * 1024 * 1024
    negative_edge_strategy: str = "uniform_nonedge"
    checkpoint_metric: str = "roc_auc@100"
    checkpoint_metric_mode: MetricMode = "max"
    save_best_checkpoint: bool = True
    checkpoint_every_epoch: bool = True
    checkpoint_dir: str | None = None
    resume_checkpoint_path: str | None = None
    tensorboard_log_dir: str | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    evaluate_train_split: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    log_every: int = 100
    show_progress: bool = True
    device: str | None = None
    seed: int = 42
    metrics: tuple[str, ...] = ("roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr", "mrr@k")
    metrics_at_k: tuple[int, ...] = field(default_factory=lambda: (10, 50, 100))
    eval_threshold: float = 0.5
    save_predictions: bool = False
    eval_every: int | None = 1000  # global steps between val evals; test runs only after training

    def positive_batch_size(self, neg_per_pos: int) -> int:
        if neg_per_pos < 0:
            raise ValueError("neg_per_pos must be non-negative")
        return max(1, int(self.batch_size_edges) // max(1, neg_per_pos + 1))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LinkPredictionConfig = LinkPredictionTrainConfig


def resolve_embedding_checkpoint_dir(raw_embeddings: dict[str, Any]) -> str:
    mode = str(raw_embeddings.get("mode", "checkpoint"))
    if mode != "checkpoint":
        raise ValueError("Only embeddings.mode='checkpoint' is supported")
    checkpoint_dir = raw_embeddings.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("embeddings.checkpoint_dir must be set")
    return str(checkpoint_dir)


_LP_NEGATIVE_STRATEGIES = frozenset({"uniform_nonedge", "two_hop"})


def map_lp_negative_strategy(name: str) -> str:
    if name in _LP_NEGATIVE_STRATEGIES:
        return name
    raise ValueError(
        f"Unsupported LP negative_edge_strategy for dataset_utils: {name}. "
        f"Supported: {sorted(_LP_NEGATIVE_STRATEGIES)}"
    )


def to_dataset_utils_train_loader_config(
    config: LinkPredictionTrainConfig,
    *,
    num_nodes: int,
    indptr: Any | None = None,
    indices: Any | None = None,
    neg_per_pos: int | None = None,
    seed: int | None = None,
) -> TrainLoaderConfig:
    extra_kwargs: dict[str, Any] = {
        "is_directed": bool(config.is_directed),
        "treat_as_undirected_for_lp": bool(config.treat_as_undirected_for_lp),
    }
    if config.graph_csr_root is None and indptr is not None and indices is not None:
        extra_kwargs.update({"indptr": indptr, "indices": indices})
    return TrainLoaderConfig(
        batch_size=int(config.batch_size_edges),
        seed=int(config.seed if seed is None else seed),
        num_workers=int(config.num_workers),
        positive_edges=PositiveEdgesConfig(
            num_nodes=int(num_nodes),
            has_self_loops=bool(config.has_self_loops),
            graph_csr_root=config.graph_csr_root,
            pairs_graph_csr_root=config.pairs_graph_csr_root,
            graph_csr_use_mmap=bool(config.graph_csr_use_mmap),
            graph_csr_file_endian=str(config.graph_csr_file_endian),
            graph_csr_allow_non_native=bool(config.graph_csr_allow_non_native),
            graph_csr_chunk_bytes=int(config.graph_csr_chunk_bytes),
        ),
        negative_sampling=NegativeSamplingConfig(
            name=map_lp_negative_strategy(config.negative_edge_strategy),
            neg_per_pos=int(config.neg_per_pos_train if neg_per_pos is None else neg_per_pos),
            seed=int(config.seed if seed is None else seed),
            reject_self_loops=not bool(config.has_self_loops),
            extra_kwargs=extra_kwargs,
        ),
    )

"""Method-agnostic downstream link prediction over frozen node embeddings."""

from .config import LinkPredictionConfig, LinkPredictionTrainConfig
from .trainer import (
    EdgeSplit,
    LinkPredictionArtifacts,
    LinkPredictionTrainer,
    LogisticRegression,
    compute_lp_metrics,
    edge_exists,
    make_csr_graph_view,
    make_edge_features,
    train_link_prediction_classifier,
)

__all__ = [
    "EdgeSplit",
    "LinkPredictionArtifacts",
    "LinkPredictionConfig",
    "LinkPredictionTrainConfig",
    "LinkPredictionTrainer",
    "LogisticRegression",
    "compute_lp_metrics",
    "edge_exists",
    "make_csr_graph_view",
    "make_edge_features",
    "train_link_prediction_classifier",
]

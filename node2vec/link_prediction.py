"""Compatibility facade for the shared frozen-embedding LP trainer."""

from embedding_lp.trainer import (
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
    "LinkPredictionTrainer",
    "LogisticRegression",
    "compute_lp_metrics",
    "edge_exists",
    "make_csr_graph_view",
    "make_edge_features",
    "train_link_prediction_classifier",
]

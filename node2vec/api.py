from __future__ import annotations

import numpy as np

from graph_csr.graph import GraphCSR

from .config import LinkPredictionTrainConfig, Node2VecConfig
from .csr import CSRGraphView, graph_to_csr_view, raw_arrays_to_csr_view
from .link_prediction import LinkPredictionArtifacts, train_link_prediction_classifier
from .trainer import Node2VecTrainer, Node2VecTrainingArtifacts


def make_csr_graph_view(
    *,
    graph: GraphCSR | None = None,
    indptr: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    is_directed: bool = False,
) -> CSRGraphView:
    if graph is not None:
        return graph_to_csr_view(graph, is_directed=is_directed)
    if indptr is None or indices is None:
        raise ValueError("Either graph or both indptr and indices must be provided")
    return raw_arrays_to_csr_view(indptr, indices, is_directed=is_directed)


def train_node2vec_embeddings(
    *,
    graph: GraphCSR | None = None,
    indptr: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    config: Node2VecConfig | None = None,
    is_directed: bool = False,
    val_pos_edges: np.ndarray | None = None,
    val_neg_edges: np.ndarray | None = None,
) -> Node2VecTrainingArtifacts:
    cfg = config or Node2VecConfig()
    csr = make_csr_graph_view(graph=graph, indptr=indptr, indices=indices, is_directed=is_directed)
    trainer = Node2VecTrainer(
        csr.indptr,
        csr.indices,
        cfg,
        val_pos_edges=val_pos_edges,
        val_neg_edges=val_neg_edges,
    )
    return trainer.fit()


def train_link_prediction(
    *,
    embedding_checkpoint_dir: str,
    graph: GraphCSR | None = None,
    indptr: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    lp_config: LinkPredictionTrainConfig | None = None,
    val_pos_edges: np.ndarray | None = None,
    val_neg_edges: np.ndarray | None = None,
    test_pos_edges: np.ndarray | None = None,
    test_neg_edges: np.ndarray | None = None,
    is_directed: bool | None = None,
) -> LinkPredictionArtifacts:
    cfg = lp_config or LinkPredictionTrainConfig()
    resolved_is_directed = cfg.is_directed if is_directed is None else bool(is_directed)
    cfg.is_directed = resolved_is_directed
    return train_link_prediction_classifier(
        embedding_checkpoint_dir=embedding_checkpoint_dir,
        graph=graph,
        indptr=indptr,
        indices=indices,
        config=cfg,
        val_pos_edges=val_pos_edges,
        val_neg_edges=val_neg_edges,
        test_pos_edges=test_pos_edges,
        test_neg_edges=test_neg_edges,
    )

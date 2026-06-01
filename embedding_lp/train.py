from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from dataset_utils import DatasetConfig, load_edge_splits, load_yaml
from graph_csr import GraphCSRSerializer

from .config import LinkPredictionTrainConfig, resolve_embedding_checkpoint_dir
from .trainer import _log_memory_snapshot, train_link_prediction_classifier


def _build_lp_config(raw_training: dict[str, Any], raw_eval: dict[str, Any]) -> LinkPredictionTrainConfig:
    merged = dict(raw_training)
    merged.update(raw_eval)
    valid_keys = {field.name for field in fields(LinkPredictionTrainConfig)}
    unknown = sorted(set(merged) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown link prediction config keys: {unknown}")
    if "metrics" in merged:
        merged["metrics"] = tuple(merged["metrics"])
    if "metrics_at_k" in merged:
        merged["metrics_at_k"] = tuple(int(x) for x in merged["metrics_at_k"])
    return LinkPredictionTrainConfig(**merged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train downstream logistic regression on top of frozen node embeddings"
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    raw_config = load_yaml(args.config)

    dataset_cfg = raw_config.get("dataset", {})
    embeddings_cfg = raw_config.get("embeddings", {})
    output_cfg = raw_config.get("output", {})
    training_cfg = raw_config.get("training", {})
    eval_cfg = raw_config.get("evaluation", {})

    if not all(isinstance(x, dict) for x in (dataset_cfg, embeddings_cfg, output_cfg, training_cfg, eval_cfg)):
        raise ValueError("All config sections must be mappings")

    dataset = DatasetConfig(**dataset_cfg).resolve()
    if dataset.num_nodes <= 0:
        raise ValueError("dataset.num_nodes must be positive")
    if dataset.graph_csr_root is None:
        raise ValueError("dataset.graph_csr_root must be set")

    embedding_checkpoint_dir = resolve_embedding_checkpoint_dir(embeddings_cfg)

    out_dir = Path(output_cfg["dir"])
    history_name = str(output_cfg.get("history_file", "metrics_history.json"))
    final_metrics_name = str(output_cfg.get("final_metrics_file", "final_metrics.json"))
    config_name = str(output_cfg.get("resolved_config_file", "resolved_config.json"))
    out_dir.mkdir(parents=True, exist_ok=True)

    lp_config = _build_lp_config(training_cfg, eval_cfg)
    lp_config.graph_csr_root = dataset.graph_csr_root
    lp_config.pairs_graph_csr_root = dataset.pairs_graph_csr_root
    lp_config.graph_csr_use_mmap = bool(dataset.graph_csr_use_mmap)
    lp_config.graph_csr_file_endian = str(dataset.graph_csr_file_endian)
    lp_config.graph_csr_allow_non_native = bool(dataset.graph_csr_allow_non_native)
    lp_config.graph_csr_chunk_bytes = int(dataset.graph_csr_chunk_bytes)
    lp_config.has_self_loops = bool(dataset.has_self_loops)
    if lp_config.checkpoint_dir is None:
        lp_config.checkpoint_dir = str(out_dir / "checkpoints")
    if lp_config.tensorboard_log_dir is None:
        lp_config.tensorboard_log_dir = str(out_dir / "tensorboard")

    logging.getLogger(__name__).info(
        "Loading val/test edge splits via DatasetConfig (train positives are iterated from the CSR)"
    )
    splits = load_edge_splits(dataset)
    rank = int(os.environ.get("RANK", "0"))
    _log_memory_snapshot("train_lp:after_load_eval_splits", rank=rank)

    with GraphCSRSerializer.deserialize(
        dataset.graph_csr_root,
        use_mmap=dataset.graph_csr_use_mmap,
        file_endian=dataset.graph_csr_file_endian,
    ) as graph:
        _log_memory_snapshot("train_lp:after_graph_deserialize", rank=rank)
        artifacts = train_link_prediction_classifier(
            embedding_checkpoint_dir=embedding_checkpoint_dir,
            graph=graph,
            config=lp_config,
            val_pos_edges=splits.val_pos,
            val_neg_edges=splits.val_neg,
            test_pos_edges=splits.test_pos,
            test_neg_edges=splits.test_neg,
        )

    if rank == 0:
        (out_dir / history_name).write_text(json.dumps(artifacts.history, indent=2), encoding="utf-8")
        (out_dir / final_metrics_name).write_text(json.dumps(artifacts.final_metrics, indent=2), encoding="utf-8")
        resolved_config = {
            "dataset": dataset.to_dict(),
            "embeddings": {
                "checkpoint_dir": embedding_checkpoint_dir,
            },
            "output": {
                "dir": str(out_dir),
                "history_file": history_name,
                "final_metrics_file": final_metrics_name,
                "resolved_config_file": config_name,
            },
            "training": artifacts.config,
        }
        (out_dir / config_name).write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
        print(f"Finished downstream link prediction training. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

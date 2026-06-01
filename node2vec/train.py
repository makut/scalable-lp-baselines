from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from dataset_utils import load_yaml
from graph_csr import GraphCSRSerializer
from embedding_table_utils import EmbeddingTableConfig

from .api import train_node2vec_embeddings
from .config import Node2VecConfig


def _load_val_edges(dataset_cfg: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve val_pos / val_neg edge files from YAML's `dataset.split_root`
    (and/or explicit `valid_edge_path`, `valid_edge_neg_path`). Returns
    (None, None) when nothing is configured.
    """
    split_root = dataset_cfg.get("split_root")
    val_pos_path = dataset_cfg.get("valid_edge_path")
    val_neg_path = dataset_cfg.get("valid_edge_neg_path")
    if val_pos_path is None and split_root is not None:
        val_pos_path = str(Path(split_root) / "valid_edge.npy")
    if val_neg_path is None and split_root is not None:
        val_neg_path = str(Path(split_root) / "valid_edge_neg.npy")
    if val_pos_path is None or val_neg_path is None:
        return None, None
    if not Path(val_pos_path).exists() or not Path(val_neg_path).exists():
        return None, None
    mmap = bool(dataset_cfg.get("mmap", True))
    pos = np.load(val_pos_path, mmap_mode="r" if mmap else None)
    neg = np.load(val_neg_path, mmap_mode="r" if mmap else None)
    return np.asarray(pos, dtype=np.int64), np.asarray(neg, dtype=np.int64)


_EMBEDDING_TABLE_CONFIG_KEYS = {f.name for f in fields(EmbeddingTableConfig)}


def _normalize_embedding_table_config(raw_embedding_table_config: dict[str, Any]) -> dict[str, Any]:
    table_config = dict(raw_embedding_table_config)
    unknown = sorted(set(table_config) - _EMBEDDING_TABLE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown embedding_table_config keys: {unknown}")
    if table_config.get("num_embeddings") is None:
        table_config.pop("num_embeddings", None)
    return table_config


def _build_config(
    raw_training: dict[str, Any],
    raw_embedding_table_config: dict[str, Any] | None = None,
) -> Node2VecConfig:
    valid_keys = {f.name for f in fields(Node2VecConfig)}
    config_data = dict(raw_training)

    if raw_embedding_table_config is not None:
        config_data["embedding_table_config"] = _normalize_embedding_table_config(raw_embedding_table_config)

    unknown = sorted(set(config_data) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown training config keys: {unknown}")
    return Node2VecConfig(**config_data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train node2vec embeddings from GraphCSR dataset")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args()
    raw_config = load_yaml(args.config)

    dataset_cfg = raw_config.get("dataset", {})
    output_cfg = raw_config.get("output", {})
    embedding_table_cfg = raw_config.get("embedding_table_config")
    training_cfg = raw_config.get("training", {})

    if not isinstance(dataset_cfg, dict) or not isinstance(output_cfg, dict) or not isinstance(training_cfg, dict):
        raise ValueError("dataset/output/training sections must be mappings")
    if embedding_table_cfg is None:
        raise ValueError("embedding_table_config section is required")
    if embedding_table_cfg is not None and not isinstance(embedding_table_cfg, dict):
        raise ValueError("embedding_table_config section must be a mapping")

    dataset_path = Path(dataset_cfg["path"])
    use_mmap = bool(dataset_cfg.get("use_mmap", True))
    file_endian = str(dataset_cfg.get("file_endian", "little"))
    is_directed = bool(dataset_cfg.get("is_directed", False))

    out_dir = Path(output_cfg["dir"])
    history_name = str(output_cfg.get("history_file", "train_history.json"))
    config_name = str(output_cfg.get("resolved_config_file", "resolved_config.json"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(training_cfg, embedding_table_cfg)
    if cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = str(out_dir / "checkpoints")
    if cfg.tensorboard_log_dir is None:
        cfg.tensorboard_log_dir = str(out_dir / "tensorboard")

    val_pos, val_neg = _load_val_edges(dataset_cfg)
    if cfg.val_eval_every is not None and (val_pos is None or val_neg is None):
        raise ValueError(
            "training.val_eval_every is set but val edges were not found "
            "(provide dataset.split_root or dataset.valid_edge_path / dataset.valid_edge_neg_path)"
        )

    with GraphCSRSerializer.deserialize(dataset_path, use_mmap=use_mmap, file_endian=file_endian) as graph:
        artifacts = train_node2vec_embeddings(
            graph=graph,
            config=cfg,
            is_directed=is_directed,
            val_pos_edges=val_pos,
            val_neg_edges=val_neg,
        )

    (out_dir / history_name).write_text(json.dumps(artifacts.history, indent=2), encoding="utf-8")
    resolved_config = {
        "dataset": {
            "path": str(dataset_path),
            "use_mmap": use_mmap,
            "file_endian": file_endian,
            "is_directed": is_directed,
        },
        "output": {
            "dir": str(out_dir),
            "history_file": history_name,
            "resolved_config_file": config_name,
            "checkpoint_dir": artifacts.checkpoint_dir,
            "embedding_shards_dir": artifacts.embedding_shards_dir,
        },
        "training": artifacts.config,
    }
    (out_dir / config_name).write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    print(f"Finished node2vec training. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from embedding_table_utils import ReadOnlyEmbeddingStore

from node2vec.config import LinkPredictionTrainConfig, Node2VecConfig
from node2vec.link_prediction import LinkPredictionTrainer
from node2vec.loss import node2vec_loss
from node2vec.model import build_embedding_table_config, create_node2vec_embedding_table
from node2vec.random_walk import prepare_rowptr_col, random_walk, random_walk_windows, seed_numba_random
from node2vec.sampler import Node2VecCollator, NodeRangeDataset, build_train_loader
from node2vec.train import _build_config
from node2vec.trainer import Node2VecTrainer


def _toy_graph() -> tuple[np.ndarray, np.ndarray]:
    """4-node graph with sorted neighbours per node, in lpp graph_csr layout
    (indptr.size == num_nodes, no sentinel)."""
    indptr = np.asarray([0, 2, 4, 6], dtype=np.int64)
    indices = np.asarray([1, 2, 0, 2, 0, 3, 1, 2], dtype=np.int64)
    return indptr, indices


def _embedding_table_config(
    *,
    embedding_dim: int = 8,
    backend: str = "vanilla",
    optimizer_type: str | None = "adam",
    lr: float = 0.01,
    init_type: str = "normal",
    init_kwargs: dict | None = None,
) -> dict:
    optimizer_kwargs = None
    if optimizer_type is not None:
        lr_key = "learning_rate" if backend == "torchrec" else "lr"
        optimizer_kwargs = {lr_key: lr}
    return {
        "backend": backend,
        "embedding_dim": embedding_dim,
        "dtype": "fp32",
        "init_type": init_type,
        "init_kwargs": {} if init_kwargs is None else init_kwargs,
        "optimizer_type": optimizer_type,
        "optimizer_kwargs": {} if optimizer_kwargs is None else optimizer_kwargs,
        "sharding_type": "row_wise",
        "compute_kernel_policy": "auto",
    }


class RandomWalkTests(unittest.TestCase):
    def test_prepare_rowptr_col_adds_sentinel(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        self.assertEqual(rowptr.size, indptr.size + 1)
        self.assertEqual(int(rowptr[-1]), int(indices.size))
        self.assertEqual(col.shape, indices.shape)

    def test_random_walk_shape_and_starts(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        starts = np.array([0, 1, 2, 3], dtype=np.int64)
        walks = random_walk(rowptr, col, starts, walk_length=5, p=1.0, q=1.0)
        self.assertEqual(walks.shape, (4, 6))
        self.assertTrue(np.array_equal(walks[:, 0], starts))
        self.assertTrue(((walks >= 0) & (walks < 4)).all())

    def test_random_walk_pq_extremes(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        starts = np.array([0, 1, 2, 3], dtype=np.int64)
        # Just sanity-check the algorithm runs and produces valid node ids.
        walks = random_walk(rowptr, col, starts, walk_length=8, p=4.0, q=0.25)
        self.assertEqual(walks.shape, (4, 9))
        self.assertTrue(((walks >= 0) & (walks < 4)).all())

    def test_random_walk_windows_matches_sliced_walks(self) -> None:
        # Deterministic one-neighbour graph: window order can be checked without
        # fighting random streams.
        indptr = np.array([0, 1, 2], dtype=np.int64)
        indices = np.array([1, 2, 0], dtype=np.int64)
        rowptr, col = prepare_rowptr_col(indptr, indices)
        starts = np.array([0, 1, 2], dtype=np.int64)
        repeated = np.tile(starts, 2)

        walks = random_walk(rowptr, col, repeated, walk_length=4, p=1.0, q=1.0)
        expected = np.concatenate([walks[:, j : j + 3] for j in range(3)], axis=0)
        actual = random_walk_windows(
            rowptr,
            col,
            starts,
            walk_length=4,
            context_size=3,
            p=1.0,
            q=1.0,
            walks_per_start=2,
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_numba_random_seed_is_reproducible(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        starts = np.array([0, 1, 2, 3], dtype=np.int64)

        seed_numba_random(123)
        first = random_walk(rowptr, col, starts, walk_length=5, p=1.0, q=1.0)
        seed_numba_random(123)
        second = random_walk(rowptr, col, starts, walk_length=5, p=1.0, q=1.0)
        self.assertTrue(np.array_equal(first, second))


class SamplerTests(unittest.TestCase):
    def test_collator_shapes(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        collator = Node2VecCollator(
            rowptr=rowptr,
            col=col,
            walk_length=6,
            context_size=3,
            walks_per_node=2,
            p=1.0,
            q=1.0,
            num_negative_samples=2,
            num_nodes=4,
        )
        batch = collator([0, 1, 2, 3])
        # walk_length=6, context_size=3 => internal_walk_length=5, num_windows=5+1+1-3=4
        # pos rows: batch=4 * walks_per_node=2 * num_windows=4 = 32
        # neg rows: 4 * 2 * num_neg=2 * 4 = 64
        self.assertEqual(batch.pos_rw.shape, (32, 3))
        self.assertEqual(batch.neg_rw.shape, (64, 3))
        self.assertEqual(batch.pos_rw.dtype, torch.int64)
        self.assertEqual(batch.neg_rw.dtype, torch.int64)

    def test_loader_iterates_each_node_once_per_epoch(self) -> None:
        indptr, indices = _toy_graph()
        rowptr, col = prepare_rowptr_col(indptr, indices)
        cfg = Node2VecConfig(
            embedding_table_config=_embedding_table_config(),
            walk_length=4,
            context_size=2,
            walks_per_node=1,
            batch_size=2,
            num_negative_samples=1,
            num_sampler_workers=0,
            pin_memory=False,
            drop_last=False,
            device="cpu",
        )
        loader = build_train_loader(
            rowptr=rowptr,
            col=col,
            num_nodes=4,
            config=cfg,
            rank=0,
            world_size=1,
            device_type="cpu",
        )
        loader.sampler.set_epoch(0)  # type: ignore[attr-defined]
        seen = []
        for batch in loader:
            # First column of pos_rw with walks_per_node=1, num_windows = walk_length-2+2 = 3
            self.assertEqual(batch.pos_rw.shape[1], cfg.context_size)
            seen.append(batch.pos_rw[:, 0].cpu().numpy())
        all_anchors = np.concatenate(seen)
        # Each node should appear at least once as a starting anchor across windows.
        self.assertEqual(set(all_anchors.tolist()), {0, 1, 2, 3})


class TrainerSmokeTests(unittest.TestCase):
    def test_trainer_runs_and_saves_checkpoint(self) -> None:
        indptr, indices = _toy_graph()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_root = Path(tmp_dir) / "checkpoints"
            cfg = Node2VecConfig(
                embedding_table_config=_embedding_table_config(embedding_dim=8, lr=0.05),
                walk_length=4,
                context_size=2,
                walks_per_node=1,
                batch_size=2,
                num_negative_samples=1,
                num_epochs=1,
                device="cpu",
                num_sampler_workers=0,
                pin_memory=False,
                drop_last=False,
                checkpoint_dir=str(checkpoint_root),
                tensorboard_log_dir=None,
                show_progress=False,
            )
            trainer = Node2VecTrainer(indptr, indices, cfg)
            artifacts = trainer.fit()

            self.assertEqual(artifacts.checkpoint_dir, str(checkpoint_root))
            self.assertEqual(artifacts.embedding_shards_dir, str(checkpoint_root / "final_embedding_shards"))
            latest_name = (checkpoint_root / "latest_checkpoint.txt").read_text(encoding="utf-8").strip()
            metadata = json.loads(
                (checkpoint_root / latest_name / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_nodes"], 4)
            self.assertEqual(metadata["embedding_dim"], 8)
            self.assertIn("config", metadata)
            self.assertTrue((checkpoint_root / latest_name / "embedding_table" / "rank0").is_dir())
            self.assertTrue((checkpoint_root / "final_embedding_shards" / "rank0").is_dir())

    def test_trainer_runs_dot_product_val_eval(self) -> None:
        indptr, indices = _toy_graph()
        val_pos = np.array([[0, 1], [2, 3]], dtype=np.int64)
        val_neg = np.array([[0, 3], [1, 3]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_root = Path(tmp_dir) / "checkpoints"
            cfg = Node2VecConfig(
                embedding_table_config=_embedding_table_config(embedding_dim=4),
                walk_length=3,
                context_size=2,
                walks_per_node=1,
                batch_size=2,
                num_negative_samples=1,
                num_epochs=2,
                device="cpu",
                num_sampler_workers=0,
                pin_memory=False,
                drop_last=False,
                checkpoint_dir=str(checkpoint_root),
                tensorboard_log_dir=None,
                show_progress=False,
                val_eval_every=1,  # eval every step
                val_batch_size=2,
                val_num_workers=0,
                val_metrics=("roc_auc@k", "mrr@k"),
                val_metrics_at_k=(2,),
                checkpoint_metric="roc_auc@2",
                checkpoint_metric_mode="max",
                save_best_checkpoint=True,
                checkpoint_every_epoch=False,
            )
            trainer = Node2VecTrainer(
                indptr,
                indices,
                cfg,
                val_pos_edges=val_pos,
                val_neg_edges=val_neg,
            )
            artifacts = trainer.fit()
            self.assertIsNotNone(artifacts.final_val_metrics)
            self.assertIn("roc_auc@2", artifacts.final_val_metrics)
            self.assertIn("mrr@2", artifacts.final_val_metrics)
            # `best.pt` should exist because save_best_checkpoint is on and we eval every step.
            self.assertTrue((checkpoint_root / "best.pt" / "embedding_table" / "rank0").is_dir())

    def test_read_only_embedding_store_loads_checkpoint(self) -> None:
        indptr, indices = _toy_graph()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_root = Path(tmp_dir) / "checkpoints"
            cfg = Node2VecConfig(
                embedding_table_config=_embedding_table_config(embedding_dim=8),
                walk_length=4,
                context_size=2,
                walks_per_node=1,
                batch_size=2,
                num_negative_samples=1,
                num_epochs=1,
                device="cpu",
                num_sampler_workers=0,
                pin_memory=False,
                drop_last=False,
                checkpoint_dir=str(checkpoint_root),
                tensorboard_log_dir=None,
                show_progress=False,
            )
            Node2VecTrainer(indptr, indices, cfg).fit()

            store = ReadOnlyEmbeddingStore.from_checkpoint(
                str(checkpoint_root),
                device=torch.device("cpu"),
                expected_num_nodes=4,
                expected_embedding_dim=cfg.embedding_table_config["embedding_dim"],
            )
            lookup = store.lookup(torch.tensor([0, 1, 3], dtype=torch.int64))
            self.assertEqual(tuple(lookup.shape), (3, cfg.embedding_table_config["embedding_dim"]))
            self.assertTrue(all(not p.requires_grad for p in store.embedding_table.parameters()))


class LinkPredictionEvalTests(unittest.TestCase):
    def test_test_split_is_evaluated_only_when_explicitly_requested(self) -> None:
        trainer = LinkPredictionTrainer.__new__(LinkPredictionTrainer)
        trainer.val_loader = object()
        trainer.test_loader = object()
        trainer._last_val_metrics = None
        trainer._last_test_metrics = None
        trainer.rank = 0
        trainer.global_step = 10
        trainer.best_metric_value = None
        trainer.config = LinkPredictionTrainConfig(checkpoint_dir=None)
        trainer.early_stopping = Mock()
        trainer.early_stopping.update.return_value = False
        trainer._record_metrics = Mock()
        trainer._evaluate_loader = Mock(
            side_effect=lambda loader, *, split_name: ({"roc_auc@100": 0.75}, None)
        )

        trainer._run_eval_and_track(epoch=0)

        self.assertEqual(
            [call.kwargs["split_name"] for call in trainer._evaluate_loader.call_args_list],
            ["val"],
        )
        self.assertIsNone(trainer._last_test_metrics)
        trainer._record_metrics.assert_called_once_with(0, {"val/roc_auc@100": 0.75})

        trainer._evaluate_loader.reset_mock()
        trainer._record_metrics.reset_mock()
        trainer.test_loader = None
        trainer.test_split = object()
        trainer._build_eval_loader_for_split = Mock(return_value=object())
        trainer._run_eval_and_track(epoch=0, include_test=True)

        trainer._build_eval_loader_for_split.assert_called_once_with(trainer.test_split, split_name="test")
        self.assertEqual(
            [call.kwargs["split_name"] for call in trainer._evaluate_loader.call_args_list],
            ["val", "test"],
        )
        self.assertEqual(trainer._last_test_metrics, {"roc_auc@100": 0.75})
        trainer._record_metrics.assert_called_once_with(
            0,
            {"val/roc_auc@100": 0.75, "test/roc_auc@100": 0.75},
        )


class LossTests(unittest.TestCase):
    def test_loss_is_finite(self) -> None:
        cfg = Node2VecConfig(
            embedding_table_config=_embedding_table_config(embedding_dim=4),
            walk_length=4,
            context_size=2,
            walks_per_node=1,
            num_negative_samples=2,
            device="cpu",
        )
        table = create_node2vec_embedding_table(num_nodes=8, config=cfg, device=torch.device("cpu"))

        pos_rw = torch.randint(0, 8, (5, 3), dtype=torch.int64)
        neg_rw = torch.randint(0, 8, (10, 3), dtype=torch.int64)
        loss = node2vec_loss(table, pos_rw, neg_rw, num_nodes=8, use_nce_bias=False)
        self.assertTrue(torch.isfinite(loss).item())


class ConfigTests(unittest.TestCase):
    def test_build_embedding_table_config_vanilla(self) -> None:
        cfg = Node2VecConfig(
            embedding_table_config={
                **_embedding_table_config(
                    embedding_dim=16,
                    backend="vanilla",
                    lr=0.125,
                    init_type="uniform",
                    init_kwargs={"bound": 0.5},
                ),
                "optimizer_kwargs": {"lr": 0.125, "weight_decay": 0.01},
            },
        )
        table_cfg = build_embedding_table_config(cfg, num_nodes=20)
        self.assertEqual(table_cfg.backend, "vanilla")
        self.assertEqual(table_cfg.optimizer_type, "adam")
        self.assertEqual(table_cfg.optimizer_kwargs["lr"], 0.125)
        self.assertEqual(table_cfg.optimizer_kwargs["weight_decay"], 0.01)
        self.assertEqual(table_cfg.init_type, "uniform")

    def test_build_embedding_table_config_torchrec_uses_learning_rate(self) -> None:
        cfg = Node2VecConfig(
            embedding_table_config=_embedding_table_config(
                embedding_dim=8,
                backend="torchrec",
                optimizer_type="sgd",
                lr=0.5,
            ),
        )
        table_cfg = build_embedding_table_config(cfg, num_nodes=10)
        self.assertEqual(table_cfg.backend, "torchrec")
        self.assertEqual(table_cfg.optimizer_kwargs["learning_rate"], 0.5)
        self.assertEqual(table_cfg.init_type, "normal")
        self.assertEqual(table_cfg.init_kwargs, {})

    def test_build_embedding_table_config_from_common_block(self) -> None:
        cfg = _build_config(
            {"walk_length": 4, "context_size": 2, "num_epochs": 1},
            raw_embedding_table_config={
                "backend": "torchrec",
                "num_embeddings": None,
                "embedding_dim": 16,
                "dtype": "fp32",
                "init_type": "normal",
                "init_kwargs": {},
                "optimizer_type": "adam",
                "optimizer_kwargs": {"learning_rate": 0.125},
                "sharding_type": "row_wise",
                "compute_kernel_policy": "prefer_hbm",
            },
        )
        table_cfg = build_embedding_table_config(cfg, num_nodes=20)
        self.assertEqual(table_cfg.backend, "torchrec")
        self.assertEqual(table_cfg.num_embeddings, 20)
        self.assertEqual(table_cfg.embedding_dim, 16)
        self.assertEqual(table_cfg.optimizer_type, "adam")
        self.assertEqual(table_cfg.optimizer_kwargs["learning_rate"], 0.125)
        self.assertEqual(table_cfg.compute_kernel_policy, "prefer_hbm")

    def test_invalid_context_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_size"):
            Node2VecConfig(context_size=1)

    def test_invalid_walk_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "walk_length"):
            Node2VecConfig(walk_length=2, context_size=5)


if __name__ == "__main__":
    unittest.main()

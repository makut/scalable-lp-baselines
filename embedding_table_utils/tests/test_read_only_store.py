from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from embedding_table_utils import (
    EmbeddingTableConfig,
    ReadOnlyEmbeddingStore,
    create_embedding_table,
)


def _train_and_save(ckpt_dir: Path) -> EmbeddingTableConfig:
    config = EmbeddingTableConfig(
        backend="vanilla",
        num_embeddings=8,
        embedding_dim=4,
        init_type="normal",
        init_kwargs={"mean": 0.0, "std": 0.25},
        optimizer_type=None,
    )
    table = create_embedding_table(config, device=torch.device("cpu"))
    table.save_local(str(ckpt_dir), step=42)
    return config


class ReadOnlyEmbeddingStoreTests(unittest.TestCase):
    def test_from_checkpoint_inner_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "inner"
            _train_and_save(ckpt)

            store = ReadOnlyEmbeddingStore.from_checkpoint(
                str(ckpt),
                device=torch.device("cpu"),
                expected_num_nodes=8,
                expected_embedding_dim=4,
            )
            self.assertEqual(store.num_nodes, 8)
            self.assertEqual(store.embedding_dim, 4)
            out = store.lookup(torch.tensor([0, 3, 7], dtype=torch.int64))
            self.assertEqual(tuple(out.shape), (3, 4))
            for param in store.embedding_table.parameters():
                self.assertFalse(param.requires_grad)

    def test_from_checkpoint_via_step_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            step_dir = root / "step_001"
            step_dir.mkdir()
            _train_and_save(step_dir / "embedding_table")
            store = ReadOnlyEmbeddingStore.from_checkpoint(
                str(step_dir),
                device=torch.device("cpu"),
            )
            self.assertEqual(store.num_nodes, 8)

    def test_from_checkpoint_via_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            step_dir = root / "step_002"
            step_dir.mkdir()
            _train_and_save(step_dir / "embedding_table")
            (root / "latest_checkpoint.txt").write_text("step_002\n", encoding="utf-8")
            store = ReadOnlyEmbeddingStore.from_checkpoint(
                str(root),
                device=torch.device("cpu"),
            )
            self.assertEqual(store.num_nodes, 8)

    def test_expected_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "inner"
            _train_and_save(ckpt)
            with self.assertRaisesRegex(ValueError, "num_embeddings"):
                ReadOnlyEmbeddingStore.from_checkpoint(
                    str(ckpt), device=torch.device("cpu"), expected_num_nodes=99
                )
            with self.assertRaisesRegex(ValueError, "embedding_dim"):
                ReadOnlyEmbeddingStore.from_checkpoint(
                    str(ckpt), device=torch.device("cpu"), expected_embedding_dim=99
                )

    def test_from_config(self) -> None:
        config = EmbeddingTableConfig(
            backend="vanilla",
            num_embeddings=12,
            embedding_dim=6,
            init_type="normal",
            init_kwargs={"mean": 0.0, "std": 0.1},
            optimizer_type=None,
        )
        store = ReadOnlyEmbeddingStore.from_config(
            config, device=torch.device("cpu"), seed=123
        )
        self.assertEqual(store.num_nodes, 12)
        self.assertEqual(store.embedding_dim, 6)
        out = store.lookup(torch.tensor([0, 1, 2], dtype=torch.int64))
        self.assertEqual(tuple(out.shape), (3, 6))
        for param in store.embedding_table.parameters():
            self.assertFalse(param.requires_grad)

    def test_no_grad_flows_through_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "inner"
            _train_and_save(ckpt)
            store = ReadOnlyEmbeddingStore.from_checkpoint(
                str(ckpt), device=torch.device("cpu")
            )
            out = store.lookup(torch.tensor([0, 1], dtype=torch.int64))
            self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()

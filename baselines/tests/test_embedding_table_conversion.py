from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.embedding_table_conversion import (
    NumpyEmbeddingMatrixSource,
    TorchTensorEmbeddingMatrixSource,
    save_embedding_matrix_as_checkpoint,
)
from embedding_table_utils import ReadOnlyEmbeddingStore


class BaselineEmbeddingTableConversionTests(unittest.TestCase):
    def test_npy_to_vanilla_checkpoint_loads_with_read_only_store(self) -> None:
        matrix = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            npy_path = root / "emb.npy"
            out_dir = root / "embedding_table"
            np.save(npy_path, matrix)

            save_embedding_matrix_as_checkpoint(
                NumpyEmbeddingMatrixSource.from_npy(npy_path),
                out_dir=out_dir,
                backend="vanilla",
            )
            store = ReadOnlyEmbeddingStore.from_checkpoint(
                out_dir,
                device=torch.device("cpu"),
                expected_num_nodes=5,
                expected_embedding_dim=4,
            )

            ids = torch.tensor([0, 3, 4], dtype=torch.int64)
            actual = store.lookup(ids)
            expected = torch.from_numpy(matrix[[0, 3, 4]])
            self.assertTrue(torch.allclose(actual, expected))

    def test_raw_binary_source_validates_size_and_converts(self) -> None:
        matrix = (np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "embeddings.bin"
            out_dir = root / "embedding_table"
            matrix.tofile(raw_path)

            source = NumpyEmbeddingMatrixSource.from_raw_binary(
                raw_path,
                num_nodes=3,
                dim=4,
                dtype="<f4",
            )
            save_embedding_matrix_as_checkpoint(source, out_dir=out_dir, backend="vanilla")
            store = ReadOnlyEmbeddingStore.from_checkpoint(out_dir, device=torch.device("cpu"))

            actual = store.lookup(torch.tensor([2], dtype=torch.int64))
            self.assertTrue(torch.allclose(actual, torch.from_numpy(matrix[[2]])))

    def test_torch_checkpoint_auto_detects_node_embedding_weight(self) -> None:
        matrix = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ckpt_path = root / "best_model.pth"
            out_dir = root / "embedding_table"
            torch.save(
                {
                    "format": "seal_checkpoint_v2",
                    "model": {"lin.weight": torch.ones(2, 3)},
                    "node_embedding": {"weight": matrix},
                },
                ckpt_path,
            )

            source = TorchTensorEmbeddingMatrixSource.from_checkpoint(ckpt_path)
            save_embedding_matrix_as_checkpoint(source, out_dir=out_dir, backend="vanilla")
            store = ReadOnlyEmbeddingStore.from_checkpoint(out_dir, device=torch.device("cpu"))

            actual = store.lookup(torch.tensor([1, 5], dtype=torch.int64))
            self.assertTrue(torch.equal(actual, matrix[[1, 5]]))


if __name__ == "__main__":
    unittest.main()

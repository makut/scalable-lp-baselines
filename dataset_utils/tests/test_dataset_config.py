from __future__ import annotations

import unittest

from dataset_utils import DatasetConfig


class DatasetConfigTests(unittest.TestCase):
    def test_root_expands_prepare_dataset_layout(self) -> None:
        config = DatasetConfig(root="/data/prepared").resolve()

        self.assertEqual(config.root, "/data/prepared")
        self.assertEqual(config.graph_csr_root, "/data/prepared/train_csr")
        self.assertEqual(config.pairs_graph_csr_root, "/data/prepared/train_pairs_csr")
        self.assertEqual(config.split_root, "/data/prepared")
        self.assertEqual(config.valid_edge_path, "/data/prepared/valid_edge.npy")
        self.assertEqual(config.valid_edge_neg_path, "/data/prepared/valid_edge_neg.npy")
        self.assertEqual(config.test_edge_path, "/data/prepared/test_edge.npy")
        self.assertEqual(config.test_edge_neg_path, "/data/prepared/test_edge_neg.npy")

    def test_explicit_paths_override_root_defaults(self) -> None:
        config = DatasetConfig(
            root="/data/prepared",
            graph_csr_root="/custom/train",
            split_root="/custom/splits",
            test_edge_path="/custom/test.npy",
        ).resolve()

        self.assertEqual(config.graph_csr_root, "/custom/train")
        self.assertEqual(config.pairs_graph_csr_root, "/data/prepared/train_pairs_csr")
        self.assertEqual(config.split_root, "/custom/splits")
        self.assertEqual(config.valid_edge_path, "/custom/splits/valid_edge.npy")
        self.assertEqual(config.test_edge_path, "/custom/test.npy")


if __name__ == "__main__":
    unittest.main()

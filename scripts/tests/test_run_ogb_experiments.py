from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from embedding_lp.train import _build_lp_config
from node2vec.train import _build_config as build_node2vec_config
from seal.config import load_config as load_seal_config
from scripts.run_ogb_experiments import PROJECT_ROOT, make_dataset_paths, write_run_configs


class RunOgbExperimentsTests(unittest.TestCase):
    def test_write_run_configs_points_every_method_at_prepared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = make_dataset_paths(
                dataset="ogbl-citation2",
                data_root=root / "data",
                runs_root=root / "runs",
            )
            configs = write_run_configs(
                paths=paths,
                num_nodes=123,
                file_endian="big",
                seal_template_path=PROJECT_ROOT / "seal" / "configs" / "link_prediction_default.yaml",
                node2vec_template_path=PROJECT_ROOT / "node2vec" / "configs" / "node2vec_default.yaml",
                lp_template_path=PROJECT_ROOT / "embedding_lp" / "configs" / "link_prediction_default.yaml",
            )

            seal = yaml.safe_load(configs["seal"].read_text(encoding="utf-8"))
            self.assertEqual(seal["dataset"]["root"], str(paths.prepared_dir))
            self.assertEqual(seal["dataset"]["num_nodes"], 123)
            load_seal_config(configs["seal"])

            node2vec = yaml.safe_load(configs["node2vec"].read_text(encoding="utf-8"))
            self.assertEqual(node2vec["dataset"]["split_root"], str(paths.prepared_dir))
            self.assertIsNone(node2vec["training"]["resume_checkpoint_dir"])
            build_node2vec_config(node2vec["training"], node2vec["embedding_table_config"])

            for method in ("node2vec", "grape"):
                lp = yaml.safe_load(configs[f"{method}_lp"].read_text(encoding="utf-8"))
                self.assertEqual(lp["dataset"]["root"], str(paths.prepared_dir))
                self.assertEqual(lp["dataset"]["num_nodes"], 123)
                self.assertEqual(lp["embeddings"]["checkpoint_dir"], str(paths.embedding_checkpoint(method)))
                self.assertIsNone(lp["training"]["resume_checkpoint_path"])
                _build_lp_config(lp["training"], lp["evaluation"])


if __name__ == "__main__":
    unittest.main()

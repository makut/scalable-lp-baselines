from __future__ import annotations

import inspect
import unittest

from embedding_lp.config import LinkPredictionTrainConfig, resolve_embedding_checkpoint_dir
from embedding_lp.train import _build_lp_config
from embedding_lp.trainer import (
    LinkPredictionTrainer,
    train_link_prediction_classifier,
)
from node2vec.config import LinkPredictionTrainConfig as Node2VecLPConfig
from node2vec.link_prediction import LinkPredictionTrainer as Node2VecLPTrainer
from node2vec.train_lp import _build_lp_config as build_node2vec_lp_config


class SharedFacadeTests(unittest.TestCase):
    def test_method_packages_reexport_shared_lp_types(self) -> None:
        self.assertIs(Node2VecLPConfig, LinkPredictionTrainConfig)
        self.assertIs(Node2VecLPTrainer, LinkPredictionTrainer)

    def test_method_cli_modules_reexport_shared_config_builder(self) -> None:
        self.assertIs(build_node2vec_lp_config, _build_lp_config)


class CheckpointOnlyTests(unittest.TestCase):
    def test_classifier_api_has_no_embedding_source_mode(self) -> None:
        signature = inspect.signature(train_link_prediction_classifier)
        self.assertIn("embedding_checkpoint_dir", signature.parameters)
        self.assertNotIn("embedding_source_mode", signature.parameters)

    def test_checkpoint_dir_resolver_accepts_checkpoint_only(self) -> None:
        self.assertEqual(
            resolve_embedding_checkpoint_dir({"checkpoint_dir": "/tmp/checkpoint"}),
            "/tmp/checkpoint",
        )
        self.assertEqual(
            resolve_embedding_checkpoint_dir({"mode": "checkpoint", "checkpoint_dir": "/tmp/checkpoint"}),
            "/tmp/checkpoint",
        )

    def test_checkpoint_dir_resolver_rejects_random_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only embeddings.mode='checkpoint'"):
            resolve_embedding_checkpoint_dir({"mode": "random_init"})

    def test_checkpoint_dir_resolver_requires_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "embeddings.checkpoint_dir"):
            resolve_embedding_checkpoint_dir({})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import logging
import tempfile
import unittest
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from embedding_table_utils import EmbeddingTableConfig, create_embedding_table
from embedding_table_utils.optimizer_adapters import build_torchrec_optimizer_adapter
from embedding_table_utils import torchrec_backend


def _torchrec_runtime_is_available() -> bool:
    try:
        from embedding_table_utils.torchrec_backend import _lazy_import_torchrec

        _lazy_import_torchrec()
    except Exception:
        return False
    return True


def _torchrec_roundtrip_worker(rank: int, world_size: int, init_method: str, ckpt_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        config = EmbeddingTableConfig(
            backend="torchrec",
            num_embeddings=16,
            embedding_dim=4,
            optimizer_type=None,
        )
        table = create_embedding_table(
            config,
            device=torch.device("cpu"),
            process_group=dist.group.WORLD,
        )
        ids = torch.tensor([0, 2, 5], dtype=torch.int32)
        result = table.lookup(ids)
        assert tuple(result.shape) == (3, 4)
        table.save_local(ckpt_dir, step=7)
        dist.barrier()
        loaded_step = table.load_local(ckpt_dir)
        assert loaded_step == 7
    finally:
        dist.destroy_process_group()


class VanillaEmbeddingTableTests(unittest.TestCase):
    def test_create_vanilla_backend_and_lookup(self) -> None:
        config = EmbeddingTableConfig(
            backend="vanilla",
            num_embeddings=10,
            embedding_dim=6,
            init_type="uniform",
            init_kwargs={"bound": 0.1},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 0.05},
        )
        table = create_embedding_table(config, device=torch.device("cpu"))
        output = table.lookup(torch.tensor([1, 3, 5], dtype=torch.int32))
        self.assertEqual(tuple(output.shape), (3, 6))
        self.assertEqual(output.dtype, torch.float32)

    def test_checkpoint_roundtrip_preserves_lookup(self) -> None:
        config = EmbeddingTableConfig(
            backend="vanilla",
            num_embeddings=12,
            embedding_dim=5,
            optimizer_type="sgd",
            optimizer_kwargs={"lr": 0.1},
        )
        table = create_embedding_table(config, device=torch.device("cpu"))
        ids = torch.tensor([1, 4, 7], dtype=torch.int64)

        before = table.lookup(ids).detach().clone()
        loss = table.lookup(ids).sum()
        table.zero_grad()
        loss.backward()
        table.step()
        after_step = table.lookup(ids).detach().clone()
        self.assertFalse(torch.allclose(before, after_step))

        with tempfile.TemporaryDirectory() as tmp_dir:
            table.save_local(tmp_dir, step=11)
            restored = create_embedding_table(config, device=torch.device("cpu"))
            loaded_step = restored.load_local(tmp_dir)
            after_load = restored.lookup(ids).detach().clone()

        self.assertEqual(loaded_step, 11)
        self.assertTrue(torch.allclose(after_step, after_load))

    def test_inference_mode_zero_grad_and_step_are_noop(self) -> None:
        config = EmbeddingTableConfig(
            backend="vanilla",
            num_embeddings=8,
            embedding_dim=4,
            optimizer_type=None,
        )
        table = create_embedding_table(config, device=torch.device("cpu"))
        ids = torch.tensor([0, 1], dtype=torch.int64)
        before = table.lookup(ids).detach().clone()
        table.zero_grad()
        table.step()
        after = table.lookup(ids).detach().clone()
        self.assertTrue(torch.allclose(before, after))
        self.assertIsNone(table.local_optimizer_state_dict())

    def test_torchrec_requires_row_wise(self) -> None:
        with self.assertRaisesRegex(ValueError, "row_wise"):
            EmbeddingTableConfig(
                backend="torchrec",
                num_embeddings=10,
                embedding_dim=4,
                sharding_type="table_wise",
            )

    def test_torchrec_adapter_prefers_fused_optimizer(self) -> None:
        class FakeFusedOptimizer:
            def __init__(self) -> None:
                self.zero_grad_called = False
                self.step_called = False

            def zero_grad(self) -> None:
                self.zero_grad_called = True

            def step(self) -> None:
                self.step_called = True

            def state_dict(self) -> dict[str, object]:
                return {"fused": True}

            def load_state_dict(self, state: dict[str, object]) -> None:
                self.loaded_state = state

        class FakeTorchRecModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, 3))
                self.fused_optimizer = FakeFusedOptimizer()

        config = EmbeddingTableConfig(
            backend="torchrec",
            num_embeddings=4,
            embedding_dim=3,
            optimizer_type="rowwise_adagrad",
            optimizer_kwargs={"learning_rate": 0.1},
        )
        module = FakeTorchRecModule()
        adapter = build_torchrec_optimizer_adapter(module=module, config=config)

        adapter.zero_grad()
        adapter.step()

        self.assertTrue(module.fused_optimizer.zero_grad_called)
        self.assertTrue(module.fused_optimizer.step_called)
        self.assertEqual(adapter.state_dict(), {"fused": True})

    def test_torchrec_rowwise_adagrad_requires_fused_optimizer(self) -> None:
        config = EmbeddingTableConfig(
            backend="torchrec",
            num_embeddings=4,
            embedding_dim=3,
            optimizer_type="rowwise_adagrad",
            optimizer_kwargs={"learning_rate": 0.1},
        )
        module = nn.Embedding(4, 3)

        with self.assertRaisesRegex(RuntimeError, "requires a fused TorchRec optimizer"):
            build_torchrec_optimizer_adapter(module=module, config=config)

    def test_compute_kernel_constraints_support_current_torchrec_names(self) -> None:
        class FakeKernel(Enum):
            DENSE = "dense"
            FUSED = "fused"
            FUSED_UVM = "fused_uvm"
            FUSED_UVM_CACHING = "fused_uvm_caching"

        config = EmbeddingTableConfig(
            backend="torchrec",
            num_embeddings=4,
            embedding_dim=3,
            compute_kernel_policy="allow_uvm",
        )
        original = torchrec_backend._lazy_import_torchrec
        torchrec_backend._lazy_import_torchrec = lambda: {"EmbeddingComputeKernel": FakeKernel}
        try:
            kernels = torchrec_backend._compute_kernel_constraints(config, torch.device("cuda"))
        finally:
            torchrec_backend._lazy_import_torchrec = original

        self.assertEqual(kernels, ["fused_uvm_caching", "fused_uvm", "fused", "dense"])

    def test_compute_kernel_constraints_support_batched_torchrec_names(self) -> None:
        class FakeKernel(Enum):
            DENSE = "dense"
            BATCHED_DENSE = "batched_dense"
            BATCHED_FUSED = "batched_fused"

        config = EmbeddingTableConfig(
            backend="torchrec",
            num_embeddings=4,
            embedding_dim=3,
            compute_kernel_policy="prefer_hbm",
        )
        original = torchrec_backend._lazy_import_torchrec
        torchrec_backend._lazy_import_torchrec = lambda: {"EmbeddingComputeKernel": FakeKernel}
        try:
            kernels = torchrec_backend._compute_kernel_constraints(config, torch.device("cuda"))
        finally:
            torchrec_backend._lazy_import_torchrec = original

        self.assertEqual(kernels, ["batched_fused", "batched_dense", "dense"])

    def test_torchrec_weight_stats_do_not_scan_params_when_debug_is_disabled(self) -> None:
        class BadModule:
            def parameters(self):
                raise AssertionError("parameters should not be scanned when DEBUG logging is disabled")

        table = object.__new__(torchrec_backend.TorchRecShardedEmbeddingTable)
        table._module = BadModule()
        original_level = torchrec_backend.logger.level
        torchrec_backend.logger.setLevel(logging.INFO)
        try:
            torchrec_backend.TorchRecShardedEmbeddingTable._log_local_weight_stats(table)
        finally:
            torchrec_backend.logger.setLevel(original_level)


@unittest.skipUnless(_torchrec_runtime_is_available(), "TorchRec runtime is unavailable in this environment")
class TorchRecEmbeddingTableTests(unittest.TestCase):
    def test_two_rank_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_file = Path(tmp_dir) / "dist_init"
            init_file.touch()
            init_method = f"file://{init_file}"
            mp.spawn(
                _torchrec_roundtrip_worker,
                args=(2, init_method, tmp_dir),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()

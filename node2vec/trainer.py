from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from dataset_utils import (
    EarlyStopping,
    LabeledEdgeDataset,
    PerSourceMetricAccumulator,
    all_reduce_float_array,
    build_eval_loader,
    distributed_is_initialized,
    infer_device,
    local_batch_limit,
    maybe_init_distributed,
    metric_improved,
)

from .config import Node2VecConfig
from .loss import node2vec_loss
from .model import Node2VecEmbeddingModule, create_node2vec_embedding_table
from .random_walk import prepare_rowptr_col
from .sampler import Node2VecBatch, build_train_loader

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(slots=True)
class Node2VecTrainingArtifacts:
    history: list[dict[str, float]]
    config: dict[str, Any]
    checkpoint_dir: str | None
    embedding_shards_dir: str | None
    final_val_metrics: dict[str, float] | None = None


def _all_reduce_mean(value: torch.Tensor) -> float:
    if not distributed_is_initialized():
        return float(value.item())
    reduced = value.detach().clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    reduced /= float(torch.distributed.get_world_size())
    return float(reduced.item())


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        return checkpoint_path
    latest_path = checkpoint_path / "latest_checkpoint.txt"
    if latest_path.exists():
        checkpoint_name = latest_path.read_text(encoding="utf-8").strip()
        resolved = checkpoint_path / checkpoint_name
        if (resolved / "metadata.json").exists():
            return resolved
    raise FileNotFoundError(f"Failed to resolve checkpoint directory from {checkpoint_path}")


class Node2VecTrainer:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        config: Node2VecConfig,
        *,
        val_pos_edges: np.ndarray | None = None,
        val_neg_edges: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.rowptr, self.col = prepare_rowptr_col(indptr, indices)
        self.num_nodes = int(self.rowptr.size - 1)
        self.rank, self.world_size = maybe_init_distributed(config.backend)
        self.device = infer_device(config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._val_pos_edges = val_pos_edges
        self._val_neg_edges = val_neg_edges

        process_group = torch.distributed.group.WORLD if distributed_is_initialized() else None
        embedding_backend = str(config.embedding_table_config.get("backend"))
        use_torchrec_backend = embedding_backend == "torchrec"
        self.embedding_table = create_node2vec_embedding_table(
            num_nodes=self.num_nodes,
            config=config,
            device=self.device,
            process_group=process_group if use_torchrec_backend else None,
        )
        self.model = Node2VecEmbeddingModule(self.embedding_table)
        if distributed_is_initialized() and not use_torchrec_backend:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )

        self.train_loader = build_train_loader(
            rowptr=self.rowptr,
            col=self.col,
            num_nodes=self.num_nodes,
            config=config,
            rank=self.rank,
            world_size=self.world_size,
            device_type=self.device.type,
        )

        self.history: list[dict[str, float]] = []
        self.best_metric_value: float | None = None
        self.early_stopping = EarlyStopping(patience=None)
        self.writer = self._build_writer()
        self.start_epoch = 0
        self.global_step = 0
        self._last_val_metrics: dict[str, float] | None = None
        self.val_loader = self._build_val_loader()
        self._load_checkpoint_if_needed()

    def _build_val_loader(self):
        if (
            self.config.val_eval_every is None
            or self._val_pos_edges is None
            or self._val_neg_edges is None
        ):
            return None
        dataset = LabeledEdgeDataset(self._val_pos_edges, self._val_neg_edges)
        return build_eval_loader(
            dataset=dataset,
            batch_size=int(self.config.val_batch_size),
            num_workers=int(self.config.val_num_workers),
            rank=self.rank,
            world_size=self.world_size,
            device_type=self.device.type,
        )

    def _build_writer(self) -> Any | None:
        if self.rank != 0 or SummaryWriter is None or self.config.tensorboard_log_dir is None:
            return None
        log_dir = Path(self.config.tensorboard_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _checkpoint_root(self) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir)

    def _save_checkpoint(self, *, epoch: int, step: int, metrics: dict[str, float], name: str | None = None) -> None:
        checkpoint_root = self._checkpoint_root()
        if checkpoint_root is None:
            return
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_name = name or f"epoch_{epoch:03d}_step_{step:09d}"
        checkpoint_path = checkpoint_root / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.embedding_table.save_local(str(checkpoint_path / "embedding_table"), step=step)
        if distributed_is_initialized():
            torch.distributed.barrier()
        if self.rank == 0:
            metadata = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "global_step": int(step),
                "epoch": int(epoch),
                "metrics": metrics,
                "best_metric": self.best_metric_value,
                "world_size": self.world_size,
                "num_nodes": self.num_nodes,
                "embedding_dim": int(self.embedding_table.config.embedding_dim),
                "config": self.config.to_dict(),
                "checkpoint_name": checkpoint_name,
            }
            (checkpoint_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if name is None:
                (checkpoint_root / "latest_checkpoint.txt").write_text(checkpoint_name + "\n", encoding="utf-8")
        if distributed_is_initialized():
            torch.distributed.barrier()

    def _load_checkpoint_if_needed(self) -> None:
        if self.config.resume_checkpoint_dir is None:
            return
        checkpoint_path = _resolve_checkpoint_dir(self.config.resume_checkpoint_dir)
        metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
        expected_world_size = int(metadata["world_size"])
        if expected_world_size != self.world_size:
            raise ValueError(
                f"Checkpoint world_size={expected_world_size} does not match current world_size={self.world_size}"
            )
        stored_num_nodes = int(metadata["num_nodes"])
        if stored_num_nodes != self.num_nodes:
            raise ValueError(f"Checkpoint num_nodes={stored_num_nodes} does not match current num_nodes={self.num_nodes}")
        stored_embedding_dim = int(metadata["embedding_dim"])
        current_embedding_dim = int(self.embedding_table.config.embedding_dim)
        if stored_embedding_dim != current_embedding_dim:
            raise ValueError(
                f"Checkpoint embedding_dim={stored_embedding_dim} does not match current embedding_dim={current_embedding_dim}"
            )
        loaded_step = self.embedding_table.load_local(str(checkpoint_path / "embedding_table"))
        metadata_step = int(metadata["global_step"])
        if loaded_step is not None and loaded_step != metadata_step:
            raise ValueError(
                f"Embedding-table checkpoint step={loaded_step} does not match metadata step={metadata_step}"
            )
        self.global_step = metadata_step
        self.start_epoch = int(metadata["epoch"]) + 1
        best_metric = metadata.get("best_metric")
        if best_metric is not None and np.isfinite(best_metric):
            self.best_metric_value = float(best_metric)

    @torch.no_grad()
    def _run_val_eval(self, epoch: int) -> dict[str, float] | None:
        """Parameter-free LP eval over val edges using dot-product scoring."""
        if self.val_loader is None:
            return None
        from dataset_utils import unpack_edge_label_batch

        self.model.eval()
        local_max_batches = local_batch_limit(
            self.config.val_max_batches, rank=self.rank, world_size=self.world_size
        )
        accumulator = PerSourceMetricAccumulator(
            metrics=self.config.val_metrics,
            metrics_at_k=self.config.val_metrics_at_k,
            apply_sigmoid=True,
            drop_first_src=self.world_size > 1 and self.rank > 0,
            drop_last_src=self.world_size > 1 and self.rank < self.world_size - 1,
        )
        truncated = False
        for batch_idx, batch in enumerate(self.val_loader):
            if local_max_batches is not None and batch_idx >= local_max_batches:
                truncated = True
                break
            edges, labels = unpack_edge_label_batch(batch)
            edges = edges.to(device=self.device, dtype=torch.int64, non_blocking=True)
            src_ids = edges[:, 0]
            dst_ids = edges[:, 1]
            if self.config.val_treat_as_undirected:
                lo = torch.minimum(src_ids, dst_ids)
                hi = torch.maximum(src_ids, dst_ids)
                src_ids, dst_ids = lo, hi
            src_emb = self.embedding_table.lookup(src_ids)
            dst_emb = self.embedding_table.lookup(dst_ids)
            scores = (src_emb * dst_emb).sum(dim=-1).float()
            accumulator.update(
                srcs=edges[:, 0].detach().cpu().numpy().astype(np.int64, copy=False),
                labels=labels.numpy().astype(np.int64, copy=False),
                scores=scores.detach().cpu().numpy().astype(np.float64, copy=False),
            )
        self.model.train()

        if truncated:
            accumulator.drop_last_src = True
        reduced = all_reduce_float_array(accumulator.to_reduction_array())
        return accumulator.compute(reduced)

    def _maybe_run_val_eval(self, epoch: int) -> None:
        if self.val_loader is None or self.config.val_eval_every is None:
            return
        if self.global_step == 0 or self.global_step % int(self.config.val_eval_every) != 0:
            return
        val_metrics = self._run_val_eval(epoch)
        if not val_metrics:
            return
        self._last_val_metrics = val_metrics
        self._record_metrics(
            epoch, self.global_step, {f"val/{k}": float(v) for k, v in val_metrics.items()}
        )
        if self.config.save_best_checkpoint and self.config.checkpoint_metric in val_metrics:
            value = float(val_metrics[self.config.checkpoint_metric])
            if metric_improved(value, best=self.best_metric_value, mode=self.config.checkpoint_metric_mode):
                self.best_metric_value = value
                self._save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    metrics={**val_metrics, "best_metric_value": value},
                    name="best.pt",
                )

    def _record_metrics(self, epoch: int, step: int, metrics: dict[str, float]) -> None:
        if self.rank != 0:
            return
        entry = {"epoch": float(epoch), "step": float(step)}
        entry.update({k: float(v) for k, v in metrics.items()})
        self.history.append(entry)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def _forward_loss(self, batch: Node2VecBatch) -> torch.Tensor:
        pos_rw = batch.pos_rw.to(self.device, dtype=torch.int64, non_blocking=True)
        neg_rw = (
            batch.neg_rw.to(self.device, dtype=torch.int64, non_blocking=True)
            if batch.neg_rw is not None
            else None
        )
        shared_neg_ids = (
            batch.shared_neg_ids.to(self.device, dtype=torch.int64, non_blocking=True)
            if batch.shared_neg_ids is not None
            else None
        )
        return node2vec_loss(
            self.model,
            pos_rw,
            neg_rw,
            num_nodes=self.num_nodes,
            use_nce_bias=self.config.use_nce_bias,
            reduction=self.config.loss_reduction,
            shared_neg_ids=shared_neg_ids,
        )

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        sampler = self.train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        try:
            total_batches = len(self.train_loader)
        except TypeError:
            total_batches = None

        progress = None
        if self.rank == 0 and self.config.show_progress and tqdm is not None:
            progress = tqdm(total=total_batches, desc=f"node2vec epoch {epoch}", unit="batch", leave=False)

        epoch_loss_sum = 0.0
        epoch_examples = 0
        start_time = time.perf_counter()
        last_log_time = start_time
        last_log_step = self.global_step

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._forward_loss(batch)
            self.embedding_table.zero_grad()
            loss.backward()
            self.embedding_table.step()
            self.global_step += 1

            mean_loss = _all_reduce_mean(loss)
            batch_examples = int(batch.pos_rw.shape[0])
            epoch_loss_sum += mean_loss * batch_examples
            epoch_examples += batch_examples

            if progress is not None:
                progress.update(1)
                progress.set_postfix({"loss": f"{mean_loss:.6f}"})

            if self.config.log_every > 0 and self.global_step % self.config.log_every == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-9)
                steps_done = max(self.global_step - last_log_step, 1)
                self._record_metrics(
                    epoch,
                    self.global_step,
                    {
                        "train/loss": mean_loss,
                        "train/steps_per_sec": steps_done / elapsed,
                        "train/elapsed_sec": now - start_time,
                    },
                )
                last_log_time = now
                last_log_step = self.global_step

            if (
                self.config.checkpoint_every_steps is not None
                and self.global_step > 0
                and self.global_step % int(self.config.checkpoint_every_steps) == 0
            ):
                self._save_checkpoint(epoch=epoch, step=self.global_step, metrics={"train_loss": mean_loss})

            self._maybe_run_val_eval(epoch)

        if progress is not None:
            progress.close()

        avg_loss = float(epoch_loss_sum / epoch_examples) if epoch_examples > 0 else float("nan")
        return {"train_loss": avg_loss}

    def _save_final_embedding_shards(self) -> str | None:
        checkpoint_root = self._checkpoint_root()
        if checkpoint_root is None:
            return None
        output_dir = checkpoint_root / "final_embedding_shards"
        self.embedding_table.save_local(str(output_dir), step=self.global_step)
        if distributed_is_initialized():
            torch.distributed.barrier()
        return str(output_dir)

    def fit(self) -> Node2VecTrainingArtifacts:
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_metrics = self._train_epoch(epoch)
            self._record_metrics(epoch, self.global_step, {"epoch/train_loss": epoch_metrics["train_loss"]})

            # If val eval is enabled, `_maybe_run_val_eval` already drives the
            # best-checkpoint decision intra-step; skip the per-epoch fallback
            # so we don't compare train_loss against a val metric.
            if self.config.save_best_checkpoint and self.val_loader is None:
                metric_value = float(
                    epoch_metrics.get(self.config.checkpoint_metric, epoch_metrics["train_loss"])
                )
                if metric_improved(
                    metric_value,
                    best=self.best_metric_value,
                    mode=self.config.checkpoint_metric_mode,
                ):
                    self.best_metric_value = metric_value
                    self._save_checkpoint(
                        epoch=epoch,
                        step=self.global_step,
                        metrics={**epoch_metrics, "best_metric_value": metric_value},
                        name="best.pt",
                    )
            if self.config.checkpoint_every_epoch:
                self._save_checkpoint(epoch=epoch, step=self.global_step, metrics=epoch_metrics)

        # Always finalize with a val eval so artifacts reflect end-of-training state.
        if self.val_loader is not None and self._last_val_metrics is None:
            val_metrics = self._run_val_eval(self.config.num_epochs - 1)
            if val_metrics:
                self._last_val_metrics = val_metrics
                self._record_metrics(
                    self.config.num_epochs - 1,
                    self.global_step,
                    {f"val/{k}": float(v) for k, v in val_metrics.items()},
                )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        embedding_shards_dir = self._save_final_embedding_shards()
        return Node2VecTrainingArtifacts(
            history=self.history,
            config=self.config.to_dict(),
            checkpoint_dir=self.config.checkpoint_dir,
            embedding_shards_dir=embedding_shards_dir,
            final_val_metrics=self._last_val_metrics,
        )

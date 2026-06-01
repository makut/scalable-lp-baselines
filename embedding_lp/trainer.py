from __future__ import annotations

import os
import time
import logging
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dataset_utils import (
    EdgeLabelBatchTransform,
    EarlyStopping,
    PerSourceMetricAccumulator,
    all_reduce_float_array,
    build_eval_dataset,
    build_eval_loader,
    build_negative_sampler,
    build_train_loader,
    distributed_is_initialized,
    infer_device,
    local_batch_limit,
    maybe_init_distributed,
    metric_improved,
    unpack_edge_label_batch,
)

from graph_csr.graph import GraphCSR

from embedding_table_utils import ReadOnlyEmbeddingStore

from .config import LinkPredictionTrainConfig, to_dataset_utils_train_loader_config
from .csr import CSRGraphView, graph_to_csr_view, raw_arrays_to_csr_view


logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass(slots=True)
class EdgeSplit:
    pos_edges: np.ndarray
    neg_edges: np.ndarray | None = None


@dataclass(slots=True)
class LinkPredictionArtifacts:
    classifier_checkpoint_path: str | None
    history: list[dict[str, float]]
    final_metrics: dict[str, dict[str, float] | None]
    config: dict[str, Any]




def _process_rss_gb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return float(rss_kb) / (1024**3)
    return float(rss_kb) * 1024.0 / (1024**3)


def _log_memory_snapshot(tag: str, *, rank: int, device: torch.device | None = None) -> None:
    parts = [f"rss={_process_rss_gb():.2f}GB"]
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        alloc = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        parts.append(f"cuda_alloc={alloc:.2f}GB")
        parts.append(f"cuda_reserved={reserved:.2f}GB")
    logger.info("Memory snapshot [%s] rank=%d %s", tag, rank, " ".join(parts))


def _array_storage_kind(arr: np.ndarray | None) -> str:
    if arr is None:
        return "none"
    if isinstance(arr, np.memmap):
        return "memmap"
    base = getattr(arr, "base", None)
    if isinstance(base, np.memmap):
        return "memmap-view"
    return "ndarray"


def _log_edge_array_details(name: str, arr: np.ndarray | None, *, rank: int) -> None:
    if arr is None:
        logger.info("Edge array [%s] rank=%d missing", name, rank)
        return
    edge_arr = np.asarray(arr)
    logical_gb = edge_arr.size * edge_arr.dtype.itemsize / (1024**3)
    logger.info(
        "Edge array [%s] rank=%d shape=%s dtype=%s storage=%s logical_size=%.2fGB",
        name,
        rank,
        tuple(edge_arr.shape),
        edge_arr.dtype,
        _array_storage_kind(arr),
        logical_gb,
    )


def _canonicalize_edges(
    edges: np.ndarray,
    *,
    is_directed: bool,
    treat_as_undirected_for_lp: bool,
) -> np.ndarray:
    edge_arr = np.asarray(edges, dtype=np.int64)
    if edge_arr.ndim != 2 or edge_arr.shape[1] != 2:
        raise ValueError("Edges must have shape [N, 2]")
    if not treat_as_undirected_for_lp:
        return edge_arr
    mask = edge_arr[:, 0] > edge_arr[:, 1]
    if not np.any(mask):
        return edge_arr
    out = edge_arr.copy()
    swapped = out[mask, 0].copy()
    out[mask, 0] = out[mask, 1]
    out[mask, 1] = swapped
    return out


def _edge_key(
    u: int,
    v: int,
    *,
    is_directed: bool,
    treat_as_undirected_for_lp: bool,
) -> tuple[int, int]:
    if treat_as_undirected_for_lp and u > v:
        return v, u
    return u, v


def _row_bounds(indptr: np.ndarray, num_edges: int, node_id: int) -> tuple[int, int]:
    start = int(indptr[node_id])
    end = int(indptr[node_id + 1]) if node_id + 1 < indptr.size else int(num_edges)
    return start, end


def _edge_exists_one_way(indptr: np.ndarray, indices: np.ndarray, num_edges: int, u: int, v: int) -> bool:
    start, end = _row_bounds(indptr, num_edges, u)
    neigh = indices[start:end]
    pos = int(np.searchsorted(neigh, v))
    return pos < neigh.size and int(neigh[pos]) == int(v)


def edge_exists(
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    u: int,
    v: int,
    is_directed: bool,
    treat_as_undirected_for_lp: bool,
) -> bool:
    num_edges = int(indices.size)
    if not treat_as_undirected_for_lp:
        return _edge_exists_one_way(indptr, indices, num_edges, u, v)
    if _edge_exists_one_way(indptr, indices, num_edges, u, v):
        return True
    if u != v:
        return _edge_exists_one_way(indptr, indices, num_edges, v, u)
    return False


def _enumerate_graph_positive_edges(
    csr: CSRGraphView,
    *,
    treat_as_undirected_for_lp: bool,
    has_self_loops: bool,
    show_progress: bool = False,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] | None = set() if treat_as_undirected_for_lp else None
    num_edges = int(csr.indices.size)
    node_iter: Iterable[int]
    if show_progress and tqdm is not None:
        node_iter = tqdm(range(csr.num_nodes), desc="LP split edges", unit="node", leave=False)
    else:
        node_iter = range(csr.num_nodes)
    for u in node_iter:
        start, end = _row_bounds(csr.indptr, num_edges, u)
        for pos in range(start, end):
            v = int(csr.indices[pos])
            if not has_self_loops and u == v:
                continue
            a, b = _edge_key(
                u,
                v,
                is_directed=csr.is_directed,
                treat_as_undirected_for_lp=treat_as_undirected_for_lp,
            )
            if seen is not None:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
            rows.append((a, b))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


class LogisticRegression(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def make_edge_features(src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor, operator: str) -> torch.Tensor:
    if operator == "hadamard":
        return src_embeddings * dst_embeddings
    if operator == "concat":
        return torch.cat([src_embeddings, dst_embeddings], dim=-1)
    if operator == "average":
        return (src_embeddings + dst_embeddings) / 2.0
    if operator == "weighted_l1":
        return torch.abs(src_embeddings - dst_embeddings)
    if operator == "weighted_l2":
        diff = src_embeddings - dst_embeddings
        return diff * diff
    raise ValueError(f"Unsupported lp_operator: {operator}")


def compute_lp_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    srcs: np.ndarray | None = None,
    metrics: tuple[str, ...],
    threshold: float,
    metrics_at_k: tuple[int, ...],
) -> dict[str, float]:
    """Thin wrapper around the shared `dataset_utils.compute_metrics`.

    The LP classifier produces sigmoided probabilities, so `apply_sigmoid=False`.
    """
    from dataset_utils import compute_metrics as _shared_compute

    return _shared_compute(
        y_true=y_true,
        scores=y_score,
        srcs=srcs,
        metrics=metrics,
        metrics_at_k=metrics_at_k,
        threshold=threshold,
        apply_sigmoid=False,
    )


def _autocast_dtype(name: str) -> torch.dtype | None:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return None


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _count_params_with_grad(module: nn.Module) -> tuple[int, int]:
    total = 0
    with_grad = 0
    for param in module.parameters():
        total += 1
        if param.grad is not None:
            with_grad += 1
    return total, with_grad


def _build_optimizer(model: nn.Module, config: LinkPredictionTrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "lbfgs":
        return torch.optim.LBFGS(params, lr=config.learning_rate)
    raise ValueError(f"Unsupported classifier optimizer: {config.optimizer}")


class LinkPredictionTrainer:
    def __init__(
        self,
        *,
        csr: CSRGraphView,
        embedding_checkpoint_dir: str,
        val_split: EdgeSplit | None,
        test_split: EdgeSplit | None,
        config: LinkPredictionTrainConfig,
    ) -> None:
        if not config.graph_csr_root:
            raise ValueError("config.graph_csr_root must be set; train positives are iterated from the CSR")
        self.csr = csr
        self.embedding_checkpoint_dir = embedding_checkpoint_dir
        self.val_split = val_split
        self.test_split = test_split
        self.config = config
        self.rank, self.world_size = maybe_init_distributed(config.backend)
        self.device = infer_device(config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        _log_memory_snapshot("trainer:init:start", rank=self.rank, device=self.device)

        process_group = torch.distributed.group.WORLD if distributed_is_initialized() else None
        self.embedding_store = ReadOnlyEmbeddingStore.from_checkpoint(
            embedding_checkpoint_dir,
            device=self.device,
            process_group=process_group,
            expected_num_nodes=self.csr.num_nodes,
            expected_embedding_dim=config.embedding_dim,
        )
        _log_memory_snapshot("trainer:after_embedding_store", rank=self.rank, device=self.device)
        in_dim = self.embedding_store.embedding_dim * (2 if config.lp_operator == "concat" else 1)
        classifier = LogisticRegression(in_dim).to(self.device)
        if distributed_is_initialized():
            classifier = DistributedDataParallel(
                classifier,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        self.classifier = classifier
        self.optimizer = _build_optimizer(self.classifier, config)
        self.criterion = nn.BCEWithLogitsLoss()
        self.writer = self._build_writer()
        self.history: list[dict[str, float]] = []
        self.best_metric_value: float | None = None
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.checkpoint_metric_mode,
            min_delta=float(config.early_stopping_min_delta),
        )
        self.start_epoch = 0
        self.global_step = 0
        self._checked_gradient_routing = False
        self._should_stop_training = False
        self._last_val_metrics: dict[str, float] | None = None
        self._last_test_metrics: dict[str, float] | None = None
        self._last_val_preds: tuple[np.ndarray, np.ndarray] | None = None
        self._last_test_preds: tuple[np.ndarray, np.ndarray] | None = None
        logger.info(
            "Building LP loaders rank=%d val_pos=%d test_pos=%d batch_size_edges=%d neg_per_pos_train=%d",
            self.rank,
            0 if val_split is None else int(val_split.pos_edges.shape[0]),
            0 if test_split is None else int(test_split.pos_edges.shape[0]),
            int(config.batch_size_edges),
            int(config.neg_per_pos_train),
        )
        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_eval_loader_for_split(val_split, split_name="val") if val_split is not None else None
        self.test_loader = None
        _log_memory_snapshot("trainer:after_build_loaders", rank=self.rank, device=self.device)
        self._load_checkpoint_if_needed()
        _log_memory_snapshot("trainer:init:done", rank=self.rank, device=self.device)

    def _build_writer(self) -> Any | None:
        if self.rank != 0 or SummaryWriter is None or self.config.tensorboard_log_dir is None:
            return None
        log_dir = Path(self.config.tensorboard_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _build_dataset_utils_config(self, *, neg_per_pos: int, seed_offset: int = 0):
        cfg = to_dataset_utils_train_loader_config(
            self.config,
            num_nodes=self.csr.num_nodes,
            neg_per_pos=neg_per_pos,
            seed=int(self.config.seed) + int(seed_offset),
        )
        if cfg.positive_edges.graph_csr_root is None:
            cfg.negative_sampling.extra_kwargs.update(
                {
                    "indptr": self.csr.indptr,
                    "indices": self.csr.indices,
                }
            )
        return cfg

    def _materialize_negative_edges(self, pos_edges: np.ndarray, *, split_name: str, neg_per_pos: int) -> np.ndarray:
        cfg = self._build_dataset_utils_config(
            neg_per_pos=neg_per_pos,
            seed_offset=10_000 if split_name == "val" else 20_000 if split_name == "test" else 0,
        )
        sampler = build_negative_sampler(cfg.negative_sampling, cfg.positive_edges)
        pos_tensor = torch.as_tensor(np.asarray(pos_edges, dtype=np.int64), dtype=torch.int64)
        neg_edges = sampler.sample(pos_tensor, meta={"rank": 0, "world_size": 1, "worker_id": 0})
        return neg_edges.cpu().numpy().astype(np.int64, copy=False)

    def _build_train_loader(self) -> DataLoader:
        neg_per_pos = self.config.neg_per_pos_train
        cfg = self._build_dataset_utils_config(neg_per_pos=neg_per_pos)
        logger.info(
            "Building dataset_utils LP train loader rank=%d batch_size_edges=%d neg_per_pos=%d "
            "graph_csr_root=%s",
            self.rank,
            int(self.config.batch_size_edges),
            int(neg_per_pos),
            cfg.positive_edges.graph_csr_root,
        )
        return build_train_loader(
            train_loader_config=cfg,
            batch_transform=EdgeLabelBatchTransform(),
            rank=self.rank,
            world_size=self.world_size,
            device_type=self.device.type,
        )

    def _build_eval_loader_for_split(self, split: EdgeSplit, *, split_name: str) -> DataLoader:
        neg_per_pos = {
            "val": self.config.neg_per_pos_val,
            "test": self.config.neg_per_pos_test,
        }[split_name]
        neg_edges = split.neg_edges
        if neg_edges is None:
            logger.info(
                "Materializing dataset_utils LP eval negatives rank=%d split=%s pos=%d neg_per_pos=%d",
                self.rank,
                split_name,
                int(split.pos_edges.shape[0]),
                int(neg_per_pos),
            )
            neg_edges = self._materialize_negative_edges(split.pos_edges, split_name=split_name, neg_per_pos=neg_per_pos)
        dataset = build_eval_dataset(
            dataset_config=self._build_dataset_utils_config(neg_per_pos=neg_per_pos).positive_edges,
            split_name=split_name,
            pos_edges=split.pos_edges,
            neg_edges=neg_edges,
        )
        logger.info(
            "Building dataset_utils LP eval loader rank=%d split=%s dataset_len=%d batch_size_edges=%d",
            self.rank,
            split_name,
            len(dataset),
            int(self.config.batch_size_edges),
        )
        eval_num_workers = (
            int(self.config.val_num_workers)
            if self.config.val_num_workers is not None
            else int(self.config.num_workers)
        )
        return build_eval_loader(
            dataset=dataset,
            batch_size=self.config.batch_size_edges,
            num_workers=eval_num_workers,
            rank=self.rank,
            world_size=self.world_size,
            device_type=self.device.type,
        )

    def _checkpoint_dir(self) -> Path:
        if self.config.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not set")
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_checkpoint(self, *, epoch: int, metrics: dict[str, float], best: bool = False) -> None:
        if self.rank != 0 or self.config.checkpoint_dir is None:
            return
        logger.info(
            "Starting LP checkpoint save rank=%d epoch=%d best=%s checkpoint_dir=%s",
            self.rank,
            epoch,
            best,
            self.config.checkpoint_dir,
        )
        _log_memory_snapshot("checkpoint:before_payload", rank=self.rank, device=self.device)
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric_value": self.best_metric_value,
            "metrics": metrics,
            "model": _unwrap_module(self.classifier).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "embedding_checkpoint_dir": self.embedding_checkpoint_dir,
        }
        checkpoint_dir = self._checkpoint_dir()
        latest_path = checkpoint_dir / "latest.pt"
        logger.info("Saving LP checkpoint latest path=%s", latest_path)
        torch.save(payload, latest_path)
        logger.info("Saved LP checkpoint latest path=%s", latest_path)
        if self.config.checkpoint_every_epoch:
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            logger.info("Saving LP checkpoint epoch path=%s", epoch_path)
            torch.save(payload, epoch_path)
            logger.info("Saved LP checkpoint epoch path=%s", epoch_path)
        if best:
            best_path = checkpoint_dir / "best.pt"
            logger.info("Saving LP checkpoint best path=%s", best_path)
            torch.save(payload, best_path)
            logger.info("Saved LP checkpoint best path=%s", best_path)
        _log_memory_snapshot("checkpoint:after_save", rank=self.rank, device=self.device)

    def _resolve_resume_path(self) -> Path | None:
        if self.config.resume_checkpoint_path is None:
            return None
        path = Path(self.config.resume_checkpoint_path)
        if path.is_dir():
            latest = path / "latest.pt"
            if latest.exists():
                return latest
        return path

    def _load_checkpoint_if_needed(self) -> None:
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return
        payload = torch.load(resume_path, map_location=self.device)
        _unwrap_module(self.classifier).load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.best_metric_value = payload.get("best_metric_value")
        self.start_epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload.get("global_step", 0))

    def _record_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.rank != 0:
            return
        entry = {"epoch": float(epoch), "global_step": float(self.global_step)}
        entry.update({k: float(v) for k, v in metrics.items()})
        self.history.append(entry)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

    def _lookup_features(self, edges: torch.Tensor) -> torch.Tensor:
        src_ids = edges[:, 0].to(device=self.device, dtype=torch.int64, non_blocking=True)
        dst_ids = edges[:, 1].to(device=self.device, dtype=torch.int64, non_blocking=True)
        if self.config.treat_as_undirected_for_lp:
            lo = torch.minimum(src_ids, dst_ids)
            hi = torch.maximum(src_ids, dst_ids)
            src_ids, dst_ids = lo, hi
        with torch.no_grad():
            src_emb = self.embedding_store.lookup(src_ids)
            dst_emb = self.embedding_store.lookup(dst_ids)
            features = make_edge_features(src_emb, dst_emb, self.config.lp_operator)
        return features

    def _assert_gradient_routing(self) -> None:
        emb_total, emb_with_grad = _count_params_with_grad(self.embedding_store)
        clf_total, clf_with_grad = _count_params_with_grad(_unwrap_module(self.classifier))
        logger.info(
            "Gradient routing check rank=%d embedding_params_with_grad=%d/%d classifier_params_with_grad=%d/%d",
            self.rank,
            emb_with_grad,
            emb_total,
            clf_with_grad,
            clf_total,
        )
        if emb_with_grad != 0:
            raise RuntimeError(
                f"Embedding store unexpectedly received gradients on rank {self.rank}: "
                f"{emb_with_grad}/{emb_total} parameters have non-None grad"
            )
        if clf_total > 0 and clf_with_grad == 0:
            raise RuntimeError(
                f"Classifier parameters did not receive gradients on rank {self.rank}: "
                f"{clf_with_grad}/{clf_total} parameters have non-None grad"
            )

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.classifier.train()
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)
        max_batches = local_batch_limit(
            self.config.max_train_batches,
            rank=self.rank,
            world_size=self.world_size,
        )
        total_batches = len(self.train_loader) if max_batches is None else min(len(self.train_loader), int(max_batches))
        logger.info(
            "Starting LP train epoch rank=%d epoch=%d num_batches=%d global_step=%d",
            self.rank,
            epoch,
            total_batches,
            self.global_step,
        )
        autocast_dtype = _autocast_dtype(self.config.device_dtype)
        progress = None
        if self.rank == 0 and self.config.show_progress and tqdm is not None:
            progress = tqdm(total=total_batches, desc=f"LP train {epoch}", unit="batch", leave=False)

        losses: list[float] = []
        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(
                    "Stopping LP train epoch early rank=%d epoch=%d at batch_idx=%d due to max_train_batches=%d",
                    self.rank,
                    epoch,
                    batch_idx,
                    int(max_batches),
                )
                break
            if batch_idx == 0:
                logger.info("LP train epoch=%d rank=%d received first batch", epoch, self.rank)
                _log_memory_snapshot("train_epoch:first_batch_received", rank=self.rank, device=self.device)
            edges, labels = unpack_edge_label_batch(batch)
            edges = edges.to(device=self.device, dtype=torch.int64, non_blocking=True)
            labels = labels.to(device=self.device, dtype=torch.float32, non_blocking=True)
            features = self._lookup_features(edges)
            if batch_idx == 0:
                logger.info(
                    "LP train epoch=%d rank=%d built features for first batch shape=%s",
                    epoch,
                    self.rank,
                    tuple(features.shape),
                )
                _log_memory_snapshot("train_epoch:first_batch_features_ready", rank=self.rank, device=self.device)

            self.optimizer.zero_grad()
            if self.config.optimizer == "lbfgs":
                def closure() -> torch.Tensor:
                    self.optimizer.zero_grad()
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=autocast_dtype is not None and self.device.type == "cuda",
                        dtype=autocast_dtype,
                    ):
                        logits_local = self.classifier(features)
                        loss_local = self.criterion(logits_local.float(), labels)
                    loss_local.backward()
                    return loss_local

                loss = self.optimizer.step(closure)
                logits = self.classifier(features)
                loss = self.criterion(logits.float(), labels)
            else:
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=autocast_dtype is not None and self.device.type == "cuda",
                    dtype=autocast_dtype,
                ):
                    logits = self.classifier(features)
                    loss = self.criterion(logits.float(), labels)
                loss.backward()
                if not self._checked_gradient_routing:
                    self._assert_gradient_routing()
                    self._checked_gradient_routing = True
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            if batch_idx == 0:
                logger.info(
                    "LP train epoch=%d rank=%d finished optimization for first batch loss=%.6f",
                    epoch,
                    self.rank,
                    float(loss.detach().item()),
                )
                _log_memory_snapshot("train_epoch:first_batch_step_done", rank=self.rank, device=self.device)

            self.global_step += 1
            losses.append(float(loss.detach().item()))
            if progress is not None:
                progress.update(1)
                progress.set_postfix({"loss": f"{losses[-1]:.6f}"})
            if self.rank == 0 and batch_idx % self.config.log_every == 0 and self.writer is not None:
                self.writer.add_scalar("train/batch_loss", losses[-1], self.global_step)

            if (
                self.config.eval_every is not None
                and self.global_step > 0
                and self.global_step % int(self.config.eval_every) == 0
            ):
                if self._run_eval_and_track(epoch):
                    self._should_stop_training = True
                    break

        if progress is not None:
            progress.close()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        logger.info(
            "Finished LP train epoch rank=%d epoch=%d mean_loss=%.6f",
            self.rank,
            epoch,
            mean_loss,
        )
        return {"train_loss": mean_loss}

    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader: DataLoader | None,
        *,
        split_name: str,
    ) -> tuple[dict[str, float] | None, tuple[np.ndarray, np.ndarray] | None]:
        if loader is None:
            return None, None
        self.classifier.eval()
        max_batches_by_split = {
            "train": self.config.max_train_batches,
            "val": self.config.max_val_batches,
            "test": self.config.max_test_batches,
        }
        max_batches = local_batch_limit(
            max_batches_by_split[split_name],
            rank=self.rank,
            world_size=self.world_size,
        )
        total_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
        logger.info(
            "Starting LP evaluation rank=%d split=%s num_batches=%d",
            self.rank,
            split_name,
            total_batches,
        )
        autocast_dtype = _autocast_dtype(self.config.device_dtype)
        accumulator = PerSourceMetricAccumulator(
            metrics=self.config.metrics,
            metrics_at_k=self.config.metrics_at_k,
            apply_sigmoid=False,
            drop_first_src=self.world_size > 1 and self.rank > 0,
            drop_last_src=self.world_size > 1 and self.rank < self.world_size - 1,
        )
        truncated = False
        progress = None
        if self.rank == 0 and self.config.show_progress and tqdm is not None:
            progress = tqdm(total=total_batches, desc=f"LP {split_name}", unit="batch", leave=False)
        local_examples = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                truncated = True
                logger.info(
                    "Stopping LP evaluation early rank=%d split=%s at batch_idx=%d due to limit=%d",
                    self.rank,
                    split_name,
                    batch_idx,
                    int(max_batches),
                )
                break
            if batch_idx == 0:
                logger.info("LP evaluation split=%s rank=%d received first batch", split_name, self.rank)
                _log_memory_snapshot(f"eval:{split_name}:first_batch_received", rank=self.rank, device=self.device)
            edges, labels = unpack_edge_label_batch(batch)
            edges = edges.to(device=self.device, dtype=torch.int64, non_blocking=True)
            labels = labels.to(device=self.device, dtype=torch.float32, non_blocking=True)
            features = self._lookup_features(edges)
            if batch_idx == 0:
                logger.info(
                    "LP evaluation split=%s rank=%d built first batch features shape=%s",
                    split_name,
                    self.rank,
                    tuple(features.shape),
                )
                _log_memory_snapshot(f"eval:{split_name}:first_batch_features_ready", rank=self.rank, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=autocast_dtype is not None and self.device.type == "cuda",
                dtype=autocast_dtype,
            ):
                logits = self.classifier(features)
            logits_fp32 = logits.float()
            probs_tensor = torch.sigmoid(logits_fp32)
            accumulator.update(
                srcs=edges[:, 0].detach().cpu().numpy().astype(np.int64, copy=False),
                labels=labels.detach().cpu().numpy().astype(np.int64, copy=False),
                scores=probs_tensor.detach().cpu().numpy().astype(np.float64, copy=False),
            )
            local_examples += int(labels.numel())
            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()
        logger.info(
            "Finished LP evaluation local pass rank=%d split=%s local_examples=%d",
            self.rank,
            split_name,
            int(local_examples),
        )
        if truncated:
            accumulator.drop_last_src = True
        reduced_metrics = all_reduce_float_array(accumulator.to_reduction_array())
        return accumulator.compute(reduced_metrics), None

    def _run_eval_and_track(self, epoch: int, *, include_test: bool = False) -> bool:
        """Run validation at the current global step and optionally final test eval.

        Updates the best metric, saves checkpoints, and advances early-stopping.
        Returns True if early stop fired.
        """
        val_metrics, val_preds = self._evaluate_loader(self.val_loader, split_name="val")
        self._last_val_metrics = val_metrics
        self._last_val_preds = val_preds
        test_metrics = None
        if include_test:
            if self.test_loader is None and self.test_split is not None:
                self.test_loader = self._build_eval_loader_for_split(self.test_split, split_name="test")
            test_metrics, test_preds = self._evaluate_loader(self.test_loader, split_name="test")
            self._last_test_metrics = test_metrics
            self._last_test_preds = test_preds

        metrics_to_record: dict[str, float] = {}
        for name, metrics in (("val", val_metrics), ("test", test_metrics)):
            if metrics is None:
                continue
            for key, value in metrics.items():
                metrics_to_record[f"{name}/{key}"] = float(value)
        self._record_metrics(epoch, metrics_to_record)

        monitored = None if val_metrics is None else val_metrics.get(self.config.checkpoint_metric)
        if self.rank == 0:
            is_best = monitored is not None and metric_improved(
                float(monitored),
                best=self.best_metric_value,
                mode=self.config.checkpoint_metric_mode,
            )
            if is_best:
                self.best_metric_value = float(monitored)
            merged = {
                **({f"val_{k}": v for k, v in (val_metrics or {}).items()}),
                **({f"test_{k}": v for k, v in (test_metrics or {}).items()}),
            }
            if self.config.checkpoint_dir is not None:
                self._save_checkpoint(
                    epoch=epoch,
                    metrics=merged,
                    best=bool(is_best and self.config.save_best_checkpoint),
                )
        if distributed_is_initialized():
            torch.distributed.barrier()

        if monitored is not None and self.early_stopping.update(float(monitored)):
            logger.info(
                "LP training stopped early at epoch=%d step=%d rank=%d: %s",
                epoch,
                self.global_step,
                self.rank,
                self.early_stopping.reason(),
            )
            return True
        return False

    def fit(self) -> LinkPredictionArtifacts:
        final_metrics: dict[str, dict[str, float] | None] = {"train": None, "val": None, "test": None}
        logger.info(
            "Starting LP fit rank=%d start_epoch=%d num_epochs=%d eval_every=%s",
            self.rank,
            self.start_epoch,
            self.config.num_epochs,
            self.config.eval_every,
        )
        last_epoch = max(self.start_epoch, self.config.num_epochs - 1)
        for epoch in range(self.start_epoch, self.config.num_epochs):
            last_epoch = epoch
            epoch_train = self._train_epoch(epoch)
            self._record_metrics(epoch, {"train/loss": epoch_train["train_loss"]})
            if self._should_stop_training:
                break

        # Always finalize with a fresh val/test pass. Test is intentionally
        # evaluated only here, after classifier training has finished.
        if self.val_loader is not None or self.test_split is not None:
            self._run_eval_and_track(last_epoch, include_test=True)

        train_metrics = None
        if self.config.evaluate_train_split:
            logger.warning(
                "Skipping train-split evaluation: source-local streaming metrics require grouped eval loaders"
            )
        final_metrics["train"] = train_metrics
        final_metrics["val"] = self._last_val_metrics
        final_metrics["test"] = self._last_test_metrics

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        classifier_checkpoint_path = None
        if self.rank == 0 and self.config.checkpoint_dir is not None:
            classifier_checkpoint_path = str(self._checkpoint_dir() / "latest.pt")

        if self.rank == 0 and self.config.save_predictions and self.config.checkpoint_dir is not None:
            checkpoint_dir = self._checkpoint_dir()
            for split_name, preds in (("val", self._last_val_preds), ("test", self._last_test_preds)):
                if preds is None:
                    continue
                labels, scores = preds
                np.savez(
                    checkpoint_dir / f"predictions_{split_name}.npz",
                    labels=labels,
                    scores=scores,
                )

        return LinkPredictionArtifacts(
            classifier_checkpoint_path=classifier_checkpoint_path,
            history=self.history,
            final_metrics=final_metrics,
            config=self.config.to_dict(),
        )


def make_csr_graph_view(
    *,
    graph: GraphCSR | None = None,
    indptr: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    is_directed: bool = False,
) -> CSRGraphView:
    if graph is not None:
        return graph_to_csr_view(graph, is_directed=is_directed)
    if indptr is None or indices is None:
        raise ValueError("Either graph or both indptr and indices must be provided")
    return raw_arrays_to_csr_view(indptr, indices, is_directed=is_directed)


def _prepare_splits(
    *,
    val_pos_edges: np.ndarray | None,
    val_neg_edges: np.ndarray | None,
    test_pos_edges: np.ndarray | None,
    test_neg_edges: np.ndarray | None,
    rank: int,
) -> tuple[EdgeSplit | None, EdgeSplit | None]:
    if rank == 0:
        logger.info("Preparing link prediction val/test splits from provided edge arrays")
    val_split = None if val_pos_edges is None else EdgeSplit(
        pos_edges=np.asarray(val_pos_edges, dtype=np.int64),
        neg_edges=None if val_neg_edges is None else np.asarray(val_neg_edges, dtype=np.int64),
    )
    test_split = None if test_pos_edges is None else EdgeSplit(
        pos_edges=np.asarray(test_pos_edges, dtype=np.int64),
        neg_edges=None if test_neg_edges is None else np.asarray(test_neg_edges, dtype=np.int64),
    )
    if rank == 0:
        logger.info(
            "Splits ready: val_pos=%d test_pos=%d (train iterated from CSR)",
            0 if val_split is None else int(val_split.pos_edges.shape[0]),
            0 if test_split is None else int(test_split.pos_edges.shape[0]),
        )
    return val_split, test_split


def train_link_prediction_classifier(
    *,
    embedding_checkpoint_dir: str,
    graph: GraphCSR | None = None,
    indptr: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    config: LinkPredictionTrainConfig | None = None,
    val_pos_edges: np.ndarray | None = None,
    val_neg_edges: np.ndarray | None = None,
    test_pos_edges: np.ndarray | None = None,
    test_neg_edges: np.ndarray | None = None,
) -> LinkPredictionArtifacts:
    cfg = config or LinkPredictionTrainConfig()
    rank, world_size = maybe_init_distributed(cfg.backend)
    csr = make_csr_graph_view(
        graph=graph,
        indptr=indptr,
        indices=indices,
        is_directed=cfg.is_directed,
    )
    val_split, test_split = _prepare_splits(
        val_pos_edges=val_pos_edges,
        val_neg_edges=val_neg_edges,
        test_pos_edges=test_pos_edges,
        test_neg_edges=test_neg_edges,
        rank=rank,
    )
    if val_split is not None:
        _log_edge_array_details("val_split.pos_edges", val_split.pos_edges, rank=rank)
        _log_edge_array_details("val_split.neg_edges", val_split.neg_edges, rank=rank)
    if test_split is not None:
        _log_edge_array_details("test_split.pos_edges", test_split.pos_edges, rank=rank)
        _log_edge_array_details("test_split.neg_edges", test_split.neg_edges, rank=rank)
    _log_memory_snapshot("train_link_prediction_classifier:after_prepare_splits", rank=rank)
    trainer = LinkPredictionTrainer(
        csr=csr,
        embedding_checkpoint_dir=embedding_checkpoint_dir,
        val_split=val_split,
        test_split=test_split,
        config=cfg,
    )
    return trainer.fit()

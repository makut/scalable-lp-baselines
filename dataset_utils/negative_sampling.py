from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import torch

from graph_csr import GraphCSRSerializer

from .config import NegativeSamplingConfig

logger = logging.getLogger(__name__)


class NegativeSampler(Protocol):
    def sample(self, pos_edges: torch.Tensor, *, meta: dict[str, Any] | None = None) -> torch.Tensor:
        ...

    def examples_per_positive(self) -> float:
        ...


class UniformNegativeSampler:
    def __init__(
        self,
        *,
        num_nodes: int,
        neg_per_pos: int,
        seed: int,
        has_self_loops: bool,
        reject_existing_edges: bool = False,
        indptr: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        graph_csr_root: str | None = None,
        graph_csr_use_mmap: bool = True,
        graph_csr_file_endian: str = "big",
        graph_csr_allow_non_native: bool = True,
        graph_csr_chunk_bytes: int = 256 * 1024 * 1024,
        treat_as_undirected_for_lp: bool = True,
        max_attempts_per_edge: int = 1_000_000,
        **unused_kwargs: Any,
    ) -> None:
        del unused_kwargs
        self.num_nodes = int(num_nodes)
        self.neg_per_pos = int(neg_per_pos)
        self.seed = int(seed)
        self.has_self_loops = bool(has_self_loops)
        self.reject_existing_edges = bool(reject_existing_edges)
        self.graph_csr_root = None if graph_csr_root is None else str(graph_csr_root)
        self.graph_csr_use_mmap = bool(graph_csr_use_mmap)
        self.graph_csr_file_endian = str(graph_csr_file_endian)
        self.graph_csr_allow_non_native = bool(graph_csr_allow_non_native)
        self.graph_csr_chunk_bytes = int(graph_csr_chunk_bytes)
        self.treat_as_undirected_for_lp = bool(treat_as_undirected_for_lp)
        self.max_attempts_per_edge = int(max_attempts_per_edge)
        self._graph = None
        self._edge_starts = None if indptr is None else np.asarray(indptr)
        self._edge_ends = None if indices is None else np.asarray(indices)
        self._rng: np.random.Generator | None = None

    def _get_rng(self, meta: dict[str, Any] | None = None) -> np.random.Generator:
        if self._rng is None:
            worker_id = 0
            rank = 0
            if meta is not None:
                worker_id = int(meta.get("worker_id", 0))
                rank = int(meta.get("rank", 0))
            self._rng = np.random.default_rng(self.seed + 10_007 * rank + worker_id)
        return self._rng

    def examples_per_positive(self) -> float:
        return float(1 + 2 * max(0, self.neg_per_pos))

    def _ensure_graph_state(self) -> None:
        if self._edge_starts is not None and self._edge_ends is not None:
            return
        if not self.reject_existing_edges:
            return
        if not self.graph_csr_root:
            raise ValueError("reject_existing_edges=True requires indptr/indices or graph_csr_root")
        self._graph = GraphCSRSerializer.deserialize(
            self.graph_csr_root,
            use_mmap=self.graph_csr_use_mmap,
            file_endian=self.graph_csr_file_endian,
            writable=False,
            allow_non_native=self.graph_csr_allow_non_native,
            chunk_bytes=self.graph_csr_chunk_bytes,
        )
        self._edge_starts = self._graph.edge_starts.numpy()
        self._edge_ends = self._graph.edge_ends.numpy()

    def _edge_exists(self, u: int, v: int) -> bool:
        edge_starts = self._edge_starts
        edge_ends = self._edge_ends
        if edge_starts is None or edge_ends is None:
            return False
        if not self.treat_as_undirected_for_lp:
            return _has_edge_sorted(edge_starts, edge_ends, int(u), int(v))
        if _has_edge_sorted(edge_starts, edge_ends, int(u), int(v)):
            return True
        if int(u) == int(v):
            return False
        return _has_edge_sorted(edge_starts, edge_ends, int(v), int(u))

    def _sample_candidates(
        self,
        *,
        anchors: np.ndarray,
        mutate_source: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        total = int(anchors.shape[0])
        out = np.empty((total, 2), dtype=np.int64)
        filled = 0
        attempts = 0
        max_attempts = max(1, total * self.max_attempts_per_edge)
        while filled < total:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to sample {total} uniform negatives after {max_attempts} attempts"
                )
            anchor_src = int(anchors[filled, 0])
            anchor_dst = int(anchors[filled, 1])
            if mutate_source:
                src = int(rng.integers(0, self.num_nodes))
                dst = anchor_dst
            else:
                src = anchor_src
                dst = int(rng.integers(0, self.num_nodes))
            if not self.has_self_loops and src == dst:
                continue
            if self.reject_existing_edges and self._edge_exists(src, dst):
                continue
            out[filled, 0] = src
            out[filled, 1] = dst
            filled += 1
        return out

    def sample(self, pos_edges: torch.Tensor, *, meta: dict[str, Any] | None = None) -> torch.Tensor:
        pos_cpu = pos_edges.to(dtype=torch.int64, device="cpu")
        if self.neg_per_pos <= 0:
            return torch.empty((0, 2), dtype=torch.int64)

        pos_np = pos_cpu.numpy()
        rng = self._get_rng(meta)
        self._ensure_graph_state()
        repeated_src = np.repeat(pos_np[:, 0], self.neg_per_pos)
        repeated_dst = np.repeat(pos_np[:, 1], self.neg_per_pos)
        repeated_pairs = np.stack([repeated_src, repeated_dst], axis=1)
        left_neg = self._sample_candidates(
            anchors=repeated_pairs,
            mutate_source=False,
            rng=rng,
        )
        right_neg = self._sample_candidates(
            anchors=repeated_pairs,
            mutate_source=True,
            rng=rng,
        )
        neg_edges = np.concatenate([left_neg, right_neg], axis=0)
        return torch.from_numpy(neg_edges)


class UniformNonEdgeNegativeSampler:
    def __init__(
        self,
        *,
        num_nodes: int,
        neg_per_pos: int,
        seed: int,
        has_self_loops: bool,
        indptr: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        graph_csr_root: str | None = None,
        graph_csr_use_mmap: bool = True,
        graph_csr_file_endian: str = "big",
        graph_csr_allow_non_native: bool = True,
        graph_csr_chunk_bytes: int = 256 * 1024 * 1024,
        is_directed: bool = False,
        treat_as_undirected_for_lp: bool = True,
        max_attempts_per_edge: int = 1_000_000,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.neg_per_pos = int(neg_per_pos)
        self.seed = int(seed)
        self.has_self_loops = bool(has_self_loops)
        self.indptr = None if indptr is None else np.asarray(indptr)
        self.indices = None if indices is None else np.asarray(indices)
        self.graph_csr_root = None if graph_csr_root is None else str(graph_csr_root)
        self.graph_csr_use_mmap = bool(graph_csr_use_mmap)
        self.graph_csr_file_endian = str(graph_csr_file_endian)
        self.graph_csr_allow_non_native = bool(graph_csr_allow_non_native)
        self.graph_csr_chunk_bytes = int(graph_csr_chunk_bytes)
        self.is_directed = bool(is_directed)
        self.treat_as_undirected_for_lp = bool(treat_as_undirected_for_lp)
        self.max_attempts_per_edge = int(max_attempts_per_edge)
        self._graph = None
        self._rng: np.random.Generator | None = None

    def _get_rng(self, meta: dict[str, Any] | None = None) -> np.random.Generator:
        if self._rng is None:
            worker_id = 0
            rank = 0
            if meta is not None:
                worker_id = int(meta.get("worker_id", 0))
                rank = int(meta.get("rank", 0))
            self._rng = np.random.default_rng(self.seed + 10_007 * rank + worker_id)
        return self._rng

    def examples_per_positive(self) -> float:
        return float(1 + max(0, self.neg_per_pos))

    def _ensure_graph_state(self) -> None:
        if self.indptr is not None and self.indices is not None:
            return
        if not self.graph_csr_root:
            raise ValueError("uniform_nonedge negative sampling requires indptr/indices or graph_csr_root")
        self._graph = GraphCSRSerializer.deserialize(
            self.graph_csr_root,
            use_mmap=self.graph_csr_use_mmap,
            file_endian=self.graph_csr_file_endian,
            writable=False,
            allow_non_native=self.graph_csr_allow_non_native,
            chunk_bytes=self.graph_csr_chunk_bytes,
        )
        self.indptr = self._graph.edge_starts.numpy()
        self.indices = self._graph.edge_ends.numpy()

    def _edge_exists(self, u: int, v: int) -> bool:
        self._ensure_graph_state()
        if self.indptr is None or self.indices is None:
            raise RuntimeError("Graph state is not initialized")
        if not self.treat_as_undirected_for_lp:
            return _has_edge_sorted(self.indptr, self.indices, int(u), int(v))
        if _has_edge_sorted(self.indptr, self.indices, int(u), int(v)):
            return True
        if int(u) == int(v):
            return False
        return _has_edge_sorted(self.indptr, self.indices, int(v), int(u))

    def sample(self, pos_edges: torch.Tensor, *, meta: dict[str, Any] | None = None) -> torch.Tensor:
        pos_cpu = pos_edges.to(dtype=torch.int64, device="cpu")
        total = int(pos_cpu.shape[0]) * max(0, self.neg_per_pos)
        if total <= 0:
            return torch.empty((0, 2), dtype=torch.int64)

        self._ensure_graph_state()
        rng = self._get_rng(meta)
        out = np.empty((total, 2), dtype=np.int64)
        filled = 0
        attempts = 0
        max_attempts = max(1, total * self.max_attempts_per_edge)
        while filled < total:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to sample {total} uniform non-edge negatives after {max_attempts} attempts"
                )
            u = int(rng.integers(0, self.num_nodes))
            v = int(rng.integers(0, self.num_nodes))
            if not self.has_self_loops and u == v:
                continue
            if self._edge_exists(u, v):
                continue
            out[filled, 0] = u
            out[filled, 1] = v
            filled += 1
        return torch.from_numpy(out)


def _row_bounds(edge_starts: np.ndarray, num_edges: int, node_id: int) -> tuple[int, int]:
    start = int(edge_starts[int(node_id)])
    end = int(edge_starts[int(node_id) + 1]) if int(node_id) + 1 < edge_starts.size else int(num_edges)
    return start, end


def _has_edge_sorted(edge_starts: np.ndarray, edge_ends: np.ndarray, u: int, v: int) -> bool:
    lo, row_end = _row_bounds(edge_starts, int(edge_ends.size), int(u))
    hi = row_end - 1
    if lo > hi:
        return False

    while lo <= hi:
        mid = (lo + hi) // 2
        x = int(edge_ends[mid])
        if x == v:
            return True
        if x < v:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


def _one_two_hop_step(
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    src: int,
    rng: np.random.Generator,
) -> int | None:
    """Single length-2 random walk from `src`. Returns the final vertex, or
    None if the walk is stuck (src isolated or intermediate isolated).

    No filtering: the caller decides whether the result is acceptable
    (e.g. not equal to src and not already a neighbour of src).
    """
    num_edges = int(edge_ends.size)
    first_start, first_end = _row_bounds(edge_starts, num_edges, int(src))
    if first_end <= first_start:
        return None
    mid_pos = int(rng.integers(first_start, first_end))
    mid = int(edge_ends[mid_pos])
    second_start, second_end = _row_bounds(edge_starts, num_edges, mid)
    if second_end <= second_start:
        return None
    dst_pos = int(rng.integers(second_start, second_end))
    return int(edge_ends[dst_pos])


def sample_two_hop_unique_dsts(
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    *,
    src: int,
    n_walks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run `n_walks` length-2 random walks from `src` over a CSR graph whose
    per-row neighbour lists are sorted ascending. Return unique final vertices
    that differ from `src` and that have no direct edge `src → v` in the graph.

    Returns int64 ndarray of shape [K] with K <= n_walks.
    """
    if int(n_walks) <= 0:
        return np.empty((0,), dtype=np.int64)
    src_int = int(src)
    seen: set[int] = set()
    for _ in range(int(n_walks)):
        dst = _one_two_hop_step(edge_starts, edge_ends, src_int, rng)
        if dst is None or dst == src_int or dst in seen:
            continue
        if _has_edge_sorted(edge_starts, edge_ends, src_int, dst):
            continue
        seen.add(dst)
    if not seen:
        return np.empty((0,), dtype=np.int64)
    return np.fromiter(seen, dtype=np.int64, count=len(seen))


class TwoHopNegativeSampler:
    def __init__(
        self,
        *,
        graph_csr_root: str,
        neg_per_pos: int,
        seed: int,
        graph_csr_use_mmap: bool = True,
        graph_csr_file_endian: str = "big",
        graph_csr_allow_non_native: bool = True,
        graph_csr_chunk_bytes: int = 256 * 1024 * 1024,
        max_total_trials: int = 100,
        stats_log_every: int = 100_000,
        **unused_kwargs: Any,
    ) -> None:
        del unused_kwargs
        if not graph_csr_root:
            raise ValueError("graph_csr_root must be provided for TwoHopNegativeSampler")
        self.graph_csr_root = str(graph_csr_root)
        self.neg_per_pos = max(0, int(neg_per_pos))
        self.seed = int(seed)
        self.graph_csr_use_mmap = bool(graph_csr_use_mmap)
        self.graph_csr_file_endian = str(graph_csr_file_endian)
        self.graph_csr_allow_non_native = bool(graph_csr_allow_non_native)
        self.graph_csr_chunk_bytes = int(graph_csr_chunk_bytes)
        self.max_total_trials = int(max_total_trials)
        self.stats_log_every = int(stats_log_every)
        self._graph = None
        self._edge_starts: np.ndarray | None = None
        self._edge_ends: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._stats_total = 0
        self._stats_found = 0
        self._stats_first_trial = 0
        self._stats_failed = 0
        self._stats_trials_to_found = 0
        self._stats_since_log = 0

    def examples_per_positive(self) -> float:
        return float(1 + self.neg_per_pos)

    def _get_rng(self, meta: dict[str, Any] | None = None) -> np.random.Generator:
        if self._rng is None:
            worker_id = 0 if meta is None else int(meta.get("worker_id", 0))
            rank = 0 if meta is None else int(meta.get("rank", 0))
            self._rng = np.random.default_rng(self.seed + 10_007 * rank + worker_id)
        return self._rng

    def _ensure_graph_state(self) -> None:
        if self._edge_starts is not None:
            return
        self._graph = GraphCSRSerializer.deserialize(
            self.graph_csr_root,
            use_mmap=self.graph_csr_use_mmap,
            file_endian=self.graph_csr_file_endian,
            writable=False,
            allow_non_native=self.graph_csr_allow_non_native,
            chunk_bytes=self.graph_csr_chunk_bytes,
        )
        self._edge_starts = self._graph.edge_starts.numpy()
        self._edge_ends = self._graph.edge_ends.numpy()

    def _sample_one(self, src: int, rng: np.random.Generator) -> tuple[int | None, int]:
        edge_starts = self._edge_starts
        edge_ends = self._edge_ends
        if edge_starts is None or edge_ends is None:
            raise RuntimeError("Graph state is not initialized")

        num_edges = int(edge_ends.size)
        first_start, first_end = _row_bounds(edge_starts, num_edges, src)
        if first_end <= first_start:
            return None, 0

        max_trials = max(1, self.max_total_trials)
        for trial_idx in range(1, max_trials + 1):
            dst = _one_two_hop_step(edge_starts, edge_ends, src, rng)
            if dst is None or dst == src:
                continue
            if _has_edge_sorted(edge_starts, edge_ends, src, dst):
                continue
            return dst, trial_idx
        return None, max_trials

    def _record_sample_stats(self, *, found: bool, trials: int) -> None:
        self._stats_total += 1
        self._stats_since_log += 1
        if found:
            self._stats_found += 1
            self._stats_trials_to_found += int(trials)
            if int(trials) == 1:
                self._stats_first_trial += 1
        else:
            self._stats_failed += 1

        if self.stats_log_every <= 0 or self._stats_since_log < self.stats_log_every:
            return

        avg_trials = self._stats_trials_to_found / max(1, self._stats_found)
        first_trial_rate = self._stats_first_trial / max(1, self._stats_total)
        failed_rate = self._stats_failed / max(1, self._stats_total)
        logger.info(
            "Two-hop negative sampling stats total=%d found=%d failed=%d avg_trials_to_found=%.3f first_trial_rate=%.6f failed_rate=%.6f",
            self._stats_total,
            self._stats_found,
            self._stats_failed,
            avg_trials,
            first_trial_rate,
            failed_rate,
        )
        self._stats_since_log = 0

    def sample(self, pos_edges: torch.Tensor, *, meta: dict[str, Any] | None = None) -> torch.Tensor:
        pos_cpu = pos_edges.to(dtype=torch.int64, device="cpu")
        if self.neg_per_pos <= 0 or pos_cpu.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int64)

        self._ensure_graph_state()
        target_nodes = np.asarray(pos_cpu[:, 0].numpy(), dtype=np.int64)
        rng = self._get_rng(meta)

        sampled_edges: list[tuple[int, int]] = []
        for src_raw in target_nodes:
            src = int(src_raw)
            for _ in range(self.neg_per_pos):
                dst, trials = self._sample_one(src, rng)
                self._record_sample_stats(found=dst is not None, trials=trials)
                if dst is not None:
                    sampled_edges.append((src, dst))

        if not sampled_edges:
            return torch.empty((0, 2), dtype=torch.int64)
        return torch.as_tensor(sampled_edges, dtype=torch.int64)


NEGATIVE_SAMPLER_REGISTRY = {
    "uniform": UniformNegativeSampler,
    "uniform_nonedge": UniformNonEdgeNegativeSampler,
    "two_hop": TwoHopNegativeSampler,
}


def build_negative_sampler(cfg: NegativeSamplingConfig, dataset_config: Any) -> NegativeSampler:
    name = str(cfg.name)
    if name not in NEGATIVE_SAMPLER_REGISTRY:
        raise ValueError(f"Unsupported negative sampler: {name}")
    sampler_cls = NEGATIVE_SAMPLER_REGISTRY[name]
    kwargs = dict(cfg.extra_kwargs)
    if name == "uniform":
        kwargs.update(
            num_nodes=int(getattr(dataset_config, "num_nodes")),
            neg_per_pos=int(cfg.neg_per_pos),
            seed=int(cfg.seed),
            has_self_loops=bool(getattr(dataset_config, "has_self_loops", False)) and not bool(cfg.reject_self_loops),
            reject_existing_edges=bool(cfg.reject_existing_edges),
            graph_csr_root=getattr(dataset_config, "graph_csr_root", None),
            graph_csr_use_mmap=bool(getattr(dataset_config, "graph_csr_use_mmap", True)),
            graph_csr_file_endian=str(getattr(dataset_config, "graph_csr_file_endian", "big")),
            graph_csr_allow_non_native=bool(getattr(dataset_config, "graph_csr_allow_non_native", True)),
            graph_csr_chunk_bytes=int(getattr(dataset_config, "graph_csr_chunk_bytes", 256 * 1024 * 1024)),
        )
    elif name == "uniform_nonedge":
        graph_csr_root = getattr(dataset_config, "graph_csr_root", None) or kwargs.get("graph_csr_root")
        kwargs.update(
            num_nodes=int(getattr(dataset_config, "num_nodes")),
            neg_per_pos=int(cfg.neg_per_pos),
            seed=int(cfg.seed),
            has_self_loops=bool(getattr(dataset_config, "has_self_loops", False)) and not bool(cfg.reject_self_loops),
            graph_csr_root=graph_csr_root,
            graph_csr_use_mmap=bool(getattr(dataset_config, "graph_csr_use_mmap", kwargs.get("graph_csr_use_mmap", True))),
            graph_csr_file_endian=str(getattr(dataset_config, "graph_csr_file_endian", kwargs.get("graph_csr_file_endian", "big"))),
            graph_csr_allow_non_native=bool(getattr(dataset_config, "graph_csr_allow_non_native", kwargs.get("graph_csr_allow_non_native", True))),
            graph_csr_chunk_bytes=int(getattr(dataset_config, "graph_csr_chunk_bytes", kwargs.get("graph_csr_chunk_bytes", 256 * 1024 * 1024))),
        )
    elif name == "two_hop":
        graph_csr_root = getattr(dataset_config, "graph_csr_root", None)
        if graph_csr_root is None:
            raise ValueError("dataset_config.graph_csr_root must be set for two_hop negative sampling")
        kwargs.update(
            graph_csr_root=str(graph_csr_root),
            neg_per_pos=int(cfg.neg_per_pos),
            seed=int(cfg.seed),
            graph_csr_use_mmap=bool(getattr(dataset_config, "graph_csr_use_mmap", True)),
            graph_csr_file_endian=str(getattr(dataset_config, "graph_csr_file_endian", "big")),
            graph_csr_allow_non_native=bool(getattr(dataset_config, "graph_csr_allow_non_native", True)),
            graph_csr_chunk_bytes=int(getattr(dataset_config, "graph_csr_chunk_bytes", 256 * 1024 * 1024)),
        )
    return sampler_cls(**kwargs)

"""Helpers shared by the GraphCSR preprocessing scripts.

Previously copy-pasted across `sort_graph_csr_neighbors.py`,
`convert_graph_csr_to_upper.py`, `temporal_split_graph_csr.py`,
`prepare_seal_dataset.py` and `prepare_dataset.py`.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from graph_csr.io_utils import Endian


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def format_int(n: int) -> str:
    return f"{int(n):,}"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def raw_int64_memmap(path: Path, size: int, file_endian: Endian) -> np.memmap:
    dtype = np.dtype(">i8") if file_endian == "big" else np.dtype("<i8")
    return np.memmap(path, dtype=dtype, mode="w+", shape=(int(size),))


def chunk_counts(chunk_starts: np.ndarray, end_edge: int) -> np.ndarray:
    counts = np.empty(chunk_starts.shape[0], dtype=np.int64)
    if chunk_starts.size == 0:
        return counts
    if chunk_starts.size > 1:
        counts[:-1] = chunk_starts[1:] - chunk_starts[:-1]
    counts[-1] = int(end_edge) - int(chunk_starts[-1])
    if np.any(counts < 0):
        raise ValueError("Bad CSR: negative degree found in chunk")
    return counts


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def delete_dir_safely(path: Path, *, preserve_paths: Iterable[Path]) -> None:
    """Recursively delete `path` after the producing step has succeeded.

    Refuses if `path` overlaps (as ancestor, descendant, or equal) with any
    of `preserve_paths`. This is the only safeguard between this helper and
    losing the outputs you just wrote.

    Silently no-ops if `path` does not exist; raises if it is not a directory.
    """
    target = Path(path)
    if not target.exists():
        LOGGER.warning("delete_dir_safely: %s does not exist, nothing to delete", target)
        return
    if not target.is_dir():
        raise ValueError(f"delete_dir_safely: {target} is not a directory")
    target_resolved = target.resolve()
    for preserve in preserve_paths:
        if _is_subpath(target_resolved, preserve):
            raise ValueError(
                f"refusing to delete {target_resolved}: it is inside (or equal to) preserve path {Path(preserve).resolve()}"
            )
        if _is_subpath(Path(preserve), target_resolved):
            raise ValueError(
                f"refusing to delete {target_resolved}: it contains preserve path {Path(preserve).resolve()}"
            )
    LOGGER.info("Deleting input graph directory: %s", target_resolved)
    shutil.rmtree(target_resolved)

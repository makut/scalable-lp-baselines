"""
Memory-bounded state-dict serialization.

Big tensors (>= ``CHUNK_TENSOR_THRESHOLD_BYTES``) are streamed to their own
``chunk_*.bin`` file in row-aligned slices of ``chunk_bytes``. After every
chunk (and after the manifest write) we ``fsync`` and hint the kernel via
``POSIX_FADV_DONTNEED`` to drop the clean pages from the page cache, so peak
RSS + page-cache stays bounded by ``chunk_bytes`` regardless of total table
size. Small tensors stay inline in a single ``manifest.pt`` next to the
chunks. TorchRec ``ShardedTensor`` values are reduced to their local shard
via ``local_shards()[0].tensor`` before the size check.

Adapted from ``vk_gnn/chunked_state_dict.py``.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CHUNK_TENSOR_THRESHOLD_BYTES = 64 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024
MANIFEST_FILE_NAME = "manifest.pt"
CHUNK_FILE_PREFIX = "chunk_"
CHUNK_FILE_SUFFIX = ".bin"


@dataclass
class ChunkRef:
    file: str
    shape: list[int]
    dtype: str


_POSIX_FADVISE = getattr(os, "posix_fadvise", None)
_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)


def _drop_file_from_page_cache(fd: int) -> None:
    if _POSIX_FADVISE is None or _FADV_DONTNEED is None:
        return
    try:
        _POSIX_FADVISE(fd, 0, 0, _FADV_DONTNEED)
    except OSError:
        pass


def _flush_and_drop(f) -> None:
    f.flush()
    os.fsync(f.fileno())
    _drop_file_from_page_cache(f.fileno())


def _atomic_replace(write_fn, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            _flush_and_drop(f)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _extract_local_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if hasattr(value, "local_shards"):
        try:
            shards = value.local_shards()
        except Exception:
            return None
        if not shards:
            return None
        return shards[0].tensor
    if isinstance(value, torch.Tensor):
        return value
    return None


def _tensor_size_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _write_tensor_chunked(tensor: torch.Tensor, path: Path, chunk_bytes: int) -> None:
    def write(f):
        if tensor.dim() == 0 or tensor.shape[0] == 0:
            cpu = tensor.detach().contiguous().cpu()
            f.write(cpu.numpy().tobytes())
            return
        n_first = tensor.shape[0]
        row_numel = tensor.numel() // n_first
        row_bytes = max(1, row_numel * tensor.element_size())
        rows_per_chunk = max(1, chunk_bytes // row_bytes)
        for i in range(0, n_first, rows_per_chunk):
            chunk_cpu = tensor[i:i + rows_per_chunk].detach().contiguous().cpu()
            f.write(chunk_cpu.numpy().tobytes())
            del chunk_cpu
            # Cap per-file peak page-cache contribution at ~chunk_bytes,
            # otherwise the kernel may buffer the whole tensor and OOM the
            # cgroup before writeback drains.
            _flush_and_drop(f)

    _atomic_replace(write, path)


def _read_chunks_into(path: Path, target_tensor: torch.Tensor, chunk_bytes: int) -> None:
    if target_tensor.dim() == 0 or target_tensor.shape[0] == 0:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return
        cpu = torch.frombuffer(bytearray(data), dtype=target_tensor.dtype).reshape(target_tensor.shape)
        target_tensor.copy_(cpu.to(target_tensor.device))
        return

    n_first = target_tensor.shape[0]
    row_numel = target_tensor.numel() // n_first
    row_bytes = max(1, row_numel * target_tensor.element_size())
    rows_per_chunk = max(1, chunk_bytes // row_bytes)
    sub_shape = tuple(target_tensor.shape[1:])
    with open(path, "rb") as f:
        for i in range(0, n_first, rows_per_chunk):
            row_end = min(i + rows_per_chunk, n_first)
            n_rows = row_end - i
            buf = f.read(n_rows * row_bytes)
            if len(buf) != n_rows * row_bytes:
                raise RuntimeError(
                    f"Truncated chunk file {path}: expected {n_rows * row_bytes} bytes, got {len(buf)}"
                )
            cpu_chunk = torch.frombuffer(bytearray(buf), dtype=target_tensor.dtype).reshape((n_rows,) + sub_shape)
            target_tensor[i:row_end].copy_(cpu_chunk.to(target_tensor.device))
            del cpu_chunk


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[len("torch."):]
    dtype = getattr(torch, dtype_str, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Cannot resolve torch dtype from {dtype_str!r}")
    return dtype


def _collect_chunk_refs(value: Any) -> set[str]:
    if isinstance(value, ChunkRef):
        return {value.file}
    if isinstance(value, dict):
        result: set[str] = set()
        for nested in value.values():
            result.update(_collect_chunk_refs(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for nested in value:
            result.update(_collect_chunk_refs(nested))
        return result
    return set()


def _cleanup_unreferenced_chunk_files(output_dir: Path, keep_files: set[str]) -> None:
    for child in output_dir.iterdir():
        if child.name in keep_files or child.name == MANIFEST_FILE_NAME:
            continue
        if (
            child.name.startswith(CHUNK_FILE_PREFIX)
            and child.name.endswith(CHUNK_FILE_SUFFIX)
            and child.is_file()
        ):
            child.unlink()
            continue
        if (
            child.name.startswith(f".{CHUNK_FILE_PREFIX}")
            or child.name.startswith(f".{MANIFEST_FILE_NAME}.tmp.")
        ) and ".tmp." in child.name and child.is_file():
            child.unlink()


def chunked_save_state_dict(
    state_dict: Any,
    output_dir: str | Path,
    *,
    threshold: int = CHUNK_TENSOR_THRESHOLD_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_id = uuid.uuid4().hex
    counter = 0
    new_chunk_files: set[str] = set()

    def transform(value):
        nonlocal counter
        local = _extract_local_tensor(value)
        if local is not None:
            if _tensor_size_bytes(local) >= threshold:
                counter += 1
                file_name = f"{CHUNK_FILE_PREFIX}{save_id}_{counter:04d}{CHUNK_FILE_SUFFIX}"
                _write_tensor_chunked(local, output_dir / file_name, chunk_bytes=chunk_bytes)
                new_chunk_files.add(file_name)
                return ChunkRef(file=file_name, shape=list(local.shape), dtype=str(local.dtype))
            return local.detach().contiguous().cpu()
        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        if isinstance(value, tuple):
            return tuple(transform(v) for v in value)
        return value

    try:
        transformed = transform(state_dict)
        manifest_path = output_dir / MANIFEST_FILE_NAME

        def write_manifest(f):
            torch.save(transformed, f)

        _atomic_replace(write_manifest, manifest_path)
    except Exception:
        for file_name in new_chunk_files:
            chunk_path = output_dir / file_name
            if chunk_path.exists():
                chunk_path.unlink()
        raise

    _cleanup_unreferenced_chunk_files(output_dir, _collect_chunk_refs(transformed))


def load_chunked_manifest(input_dir: str | Path) -> Any:
    """Load just the small manifest (no chunk data). Returned value still
    contains ``ChunkRef`` placeholders where big tensors were offloaded."""
    return torch.load(Path(input_dir) / MANIFEST_FILE_NAME, weights_only=False, map_location="cpu")


def chunked_restore(
    saved: Any,
    target: Any,
    input_dir: str | Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Any:
    """Recursively rebuild a saved state-dict subtree.

    For ``ChunkRef`` leaves: if the matching ``target`` exposes a writable
    tensor (plain Tensor, or ShardedTensor with a local shard), chunks are
    streamed in-place into that storage. Otherwise a fresh CPU tensor of the
    saved shape/dtype is allocated and filled — used for lazy-init optimizer
    state where the template is still empty.
    """
    input_dir = Path(input_dir)

    def restore(saved_value, target_value):
        if isinstance(saved_value, ChunkRef):
            chunk_path = input_dir / saved_value.file
            target_local = _extract_local_tensor(target_value)
            if target_local is not None:
                _read_chunks_into(chunk_path, target_local, chunk_bytes=chunk_bytes)
                return target_value
            fresh = torch.empty(
                tuple(saved_value.shape),
                dtype=_resolve_dtype(saved_value.dtype),
            )
            _read_chunks_into(chunk_path, fresh, chunk_bytes=chunk_bytes)
            return fresh
        if isinstance(saved_value, torch.Tensor):
            target_local = _extract_local_tensor(target_value)
            if target_local is not None:
                target_local.copy_(saved_value.to(target_local.device))
                return target_value
            return saved_value
        if isinstance(saved_value, dict):
            target_dict = target_value if isinstance(target_value, dict) else {}
            return {k: restore(v, target_dict.get(k)) for k, v in saved_value.items()}
        if isinstance(saved_value, (list, tuple)):
            target_seq = list(target_value) if isinstance(target_value, (list, tuple)) else []
            items = [
                restore(saved_value[i], target_seq[i] if i < len(target_seq) else None)
                for i in range(len(saved_value))
            ]
            return type(saved_value)(items)
        return saved_value

    return restore(saved, target)


def chunked_shard_dir_is_loadable(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    manifest = path / MANIFEST_FILE_NAME
    if not manifest.exists():
        return False
    try:
        torch.load(manifest, weights_only=False, map_location="cpu")
    except Exception:
        return False
    return True

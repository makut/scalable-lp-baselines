from __future__ import annotations

import mmap
from pathlib import Path
from typing import BinaryIO, Literal, Tuple

import numpy as np

Endian = Literal["little", "big"]
NATIVE_ENDIAN: Endian = "little" if np.little_endian else "big"


def read_exact_into(f: BinaryIO, mv: memoryview) -> None:
    """Read exactly len(mv) bytes into mv or raise EOFError."""
    need = len(mv)
    off = 0
    while off < need:
        n = f.readinto(mv[off:])
        if n == 0:
            raise EOFError(f"Unexpected EOF: needed {need} bytes, got {off}")
        off += n


def normalize_chunk_bytes(chunk_bytes: int, elem_size: int) -> int:
    chunk_bytes = max(elem_size, int(chunk_bytes))
    return chunk_bytes - (chunk_bytes % elem_size)


def stream_read_array(
    path: Path | str,
    *,
    dtype_file: np.dtype,
    count: int,
    offset: int,
    chunk_bytes: int,
) -> np.ndarray:
    arr_file = np.empty((int(count),), dtype=dtype_file)
    mv = memoryview(arr_file).cast("B")
    with open(path, "rb", buffering=1024 * 1024) as f:
        if offset:
            f.seek(offset)
        for off in range(0, mv.nbytes, chunk_bytes):
            read_exact_into(f, mv[off : off + min(chunk_bytes, mv.nbytes - off)])
    return arr_file


def mmap_array(
    path: Path | str,
    *,
    dtype_file: np.dtype,
    count: int,
    offset: int,
    writable: bool,
) -> Tuple[np.ndarray, mmap.mmap, BinaryIO]:
    access_flags = "r+b" if writable else "rb"
    f = open(path, access_flags, buffering=0)
    access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
    length = offset + (int(count) * dtype_file.itemsize)
    mm = mmap.mmap(f.fileno(), length=length, access=access)
    arr = np.ndarray((int(count),), dtype=dtype_file, buffer=mm, offset=offset)
    return arr, mm, f


def stream_write_array(
    path: Path | str,
    *,
    arr: np.ndarray,
    out_dtype: np.dtype,
    native_dtype: np.dtype,
    file_endian: Endian,
    chunk_bytes: int,
    header_bytes: bytes = b"",
) -> None:
    elem_size = out_dtype.itemsize
    chunk_bytes = normalize_chunk_bytes(chunk_bytes, elem_size)
    elems_per_chunk = chunk_bytes // elem_size

    with open(path, "wb", buffering=1024 * 1024) as f:
        if header_bytes:
            f.write(header_bytes)

        if file_endian == NATIVE_ENDIAN and arr.dtype == native_dtype:
            mv = memoryview(arr).cast("B")
            for off in range(0, mv.nbytes, chunk_bytes):
                f.write(mv[off : off + min(chunk_bytes, mv.nbytes - off)])
            return

        tmp = np.empty((elems_per_chunk,), dtype=out_dtype)
        arr_native = arr.view(native_dtype) if arr.dtype == native_dtype else np.asarray(arr, dtype=native_dtype)

        for start in range(0, arr_native.size, elems_per_chunk):
            end = min(arr_native.size, start + elems_per_chunk)
            n = end - start
            tmp[:n] = arr_native[start:end]
            f.write(memoryview(tmp[:n]).cast("B"))

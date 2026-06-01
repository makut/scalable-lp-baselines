from __future__ import annotations

import mmap
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

from .io_utils import Endian, NATIVE_ENDIAN, mmap_array, normalize_chunk_bytes, stream_read_array, stream_write_array


class LargeInt64RawArray:
    """
    Raw int64[] without a header:
      - use_mmap=True  -> file-backed mmap
      - use_mmap=False -> load the complete array into RAM

    Java edgeStarts arrays are commonly big-endian (`ByteBuffer` default).
    """

    __slots__ = ("_arr", "_mmap", "_file", "_path", "_file_endian")

    def __init__(
        self,
        arr: np.ndarray,
        *,
        mmap_obj: Optional[mmap.mmap] = None,
        file_obj: Optional[BinaryIO] = None,
        path: Optional[Path] = None,
        file_endian: Endian = "big",
    ):
        self._arr = arr
        self._mmap = mmap_obj
        self._file = file_obj
        self._path = path
        self._file_endian = file_endian

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def size(self) -> int:
        return int(self._arr.size)

    def flush(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()

    def close(self) -> None:
        self._arr = None  # type: ignore
        if self._mmap is not None:
            try:
                self._mmap.close()
            finally:
                self._mmap = None
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __enter__(self) -> "LargeInt64RawArray":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        use_mmap: bool,
        file_endian: Endian = "big",
        writable: bool = False,
        allow_non_native: bool = True,
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> "LargeInt64RawArray":
        path = Path(path)
        size_bytes = path.stat().st_size
        if size_bytes % 8 != 0:
            raise ValueError(f"edge_starts.bin size must be multiple of 8, got {size_bytes}")
        n = size_bytes // 8

        dtype_file = np.dtype(">i8") if file_endian == "big" else np.dtype("<i8")

        if use_mmap:
            if (file_endian != NATIVE_ENDIAN) and (not allow_non_native):
                raise ValueError(
                    f"file_endian={file_endian} != native={NATIVE_ENDIAN}. "
                    f"Either allow_non_native=True or load into RAM (use_mmap=False)."
                )
            arr, mm, f = mmap_array(path, dtype_file=dtype_file, count=int(n), offset=0, writable=writable)

            if file_endian == NATIVE_ENDIAN:
                arr = arr.view(np.int64)

            return cls(arr, mmap_obj=mm, file_obj=f, path=path, file_endian=file_endian)

        chunk_bytes = normalize_chunk_bytes(chunk_bytes, 8)
        arr_file = stream_read_array(
            path,
            dtype_file=dtype_file,
            count=int(n),
            offset=0,
            chunk_bytes=chunk_bytes,
        )

        if file_endian != NATIVE_ENDIAN:
            arr_file.byteswap(inplace=True)
        arr = arr_file.view(np.int64)

        return cls(arr, file_endian=file_endian)

    def to_file(
        self,
        path: Path | str,
        *,
        file_endian: Endian = "big",
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Write raw int64[] without a header, matching Java edgeStarts."""
        path = Path(path)
        out_dtype = np.dtype(">i8") if file_endian == "big" else np.dtype("<i8")
        stream_write_array(
            path,
            arr=self._arr,
            out_dtype=out_dtype,
            native_dtype=np.dtype(np.int64),
            file_endian=file_endian,
            chunk_bytes=chunk_bytes,
            header_bytes=b"",
        )

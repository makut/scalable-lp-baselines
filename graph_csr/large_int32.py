from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

from .io_utils import (
    Endian,
    NATIVE_ENDIAN,
    mmap_array,
    normalize_chunk_bytes,
    stream_read_array,
    stream_write_array,
)


class LargeInt32Array:
    """
    Large int32 array with a selectable backend:
      - RAM (numpy.empty)      : use_mmap=False
      - mmap (file/anon)       : use_mmap=True

    File format:
      [int64 size] + [int32 * size]
    """

    __slots__ = ("size", "_arr", "_mmap", "_file", "_path", "_file_endian")

    def __init__(
        self,
        size: int,
        arr: np.ndarray,
        *,
        mmap_obj: Optional[mmap.mmap] = None,
        file_obj: Optional[BinaryIO] = None,
        path: Optional[Path] = None,
        file_endian: Endian = NATIVE_ENDIAN,
    ):
        self.size = int(size)
        self._arr = arr
        self._mmap = mmap_obj
        self._file = file_obj
        self._path = path
        self._file_endian = file_endian

    def numpy(self) -> np.ndarray:
        """Return a zero-copy view. The dtype may use non-native byte order."""
        return self._arr

    def get(self, index: int) -> int:
        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of bounds: {index}")
        return int(self._arr[index])

    def set(self, index: int, value: int) -> None:
        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of bounds: {index}")
        self._arr[index] = np.int32(value)

    @property
    def is_mmap(self) -> bool:
        return self._mmap is not None

    def flush(self) -> None:
        """Flush dirty pages for a file-backed mmap."""
        if self._mmap is not None and self._path is not None:
            self._mmap.flush()

    def close(self) -> None:
        """Release mmap and file resources."""
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

    def __enter__(self) -> "LargeInt32Array":
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
    def create(
        cls,
        size: int,
        *,
        use_mmap: bool,
        path: Optional[Path | str] = None,
        file_endian: Endian = NATIVE_ENDIAN,
        writable: bool = True,
    ) -> "LargeInt32Array":
        """
        Create a new array.

        use_mmap=False -> RAM (numpy.empty)
        use_mmap=True  ->
          - path=None: anonymous mmap
          - path is set: file-backed mmap
        """
        size = int(size)
        if size <= 0:
            raise ValueError("Size must be positive")

        nbytes = size * 4

        if not use_mmap:
            arr = np.empty((size,), dtype=np.int32)
            return cls(size, arr, file_endian=file_endian)

        if path is None:
            access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
            mm = mmap.mmap(-1, nbytes, access=access)
            arr = np.ndarray((size,), dtype=np.int32, buffer=mm)
            return cls(size, arr, mmap_obj=mm, file_endian=file_endian)

        path = Path(path)
        total = 8 + nbytes
        with open(path, "wb") as f:
            hdr_fmt = ">q" if file_endian == "big" else "<q"
            f.write(struct.pack(hdr_fmt, size))
            f.truncate(total)

        dtype = np.dtype(">i4") if file_endian == "big" else np.dtype("<i4")
        arr, mm, f = mmap_array(path, dtype_file=dtype, count=size, offset=8, writable=writable)
        return cls(int(size), arr, mmap_obj=mm, file_obj=f, path=path, file_endian=file_endian)

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
    ) -> "LargeInt32Array":
        """
        Load an array from a file.

        use_mmap=True  -> file-backed mmap with lazy OS page loading.
        use_mmap=False -> stream the complete array into RAM.
        """
        path = Path(path)

        with open(path, "rb") as f:
            hdr = f.read(8)
            if len(hdr) != 8:
                raise EOFError("File too short: missing size header")
            hdr_fmt = ">q" if file_endian == "big" else "<q"
            size = struct.unpack(hdr_fmt, hdr)[0]
            if size <= 0:
                raise ValueError(f"Bad size in header: {size}")

        nbytes = int(size) * 4
        total = 8 + nbytes

        if use_mmap:
            if (file_endian != NATIVE_ENDIAN) and (not allow_non_native):
                raise ValueError(
                    f"file_endian={file_endian} != native={NATIVE_ENDIAN}. "
                    f"Either set allow_non_native=True, or load into RAM (use_mmap=False)."
                )

            dtype = np.dtype(">i4") if file_endian == "big" else np.dtype("<i4")
            arr, mm, f = mmap_array(path, dtype_file=dtype, count=size, offset=8, writable=writable)
            if file_endian == NATIVE_ENDIAN:
                arr = arr.view(np.int32)

            return cls(int(size), arr, mmap_obj=mm, file_obj=f, path=path, file_endian=file_endian)

        chunk_bytes = normalize_chunk_bytes(chunk_bytes, 4)
        dtype_file = np.dtype(">i4") if file_endian == "big" else np.dtype("<i4")
        arr_file = stream_read_array(
            path,
            dtype_file=dtype_file,
            count=int(size),
            offset=8,
            chunk_bytes=chunk_bytes,
        )

        if file_endian != NATIVE_ENDIAN:
            arr_file.byteswap(inplace=True)
        arr = arr_file.view(np.int32)

        return cls(int(size), arr, file_endian=file_endian)

    def to_file(
        self,
        path: Path | str,
        *,
        file_endian: Endian = "big",
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """
        Write the file format:
          [int64 size] + [int32 data]
        """
        path = Path(path)
        hdr_fmt = ">q" if file_endian == "big" else "<q"
        out_dtype = np.dtype(">i4") if file_endian == "big" else np.dtype("<i4")
        header = struct.pack(hdr_fmt, int(self.size))
        stream_write_array(
            path,
            arr=self._arr,
            out_dtype=out_dtype,
            native_dtype=np.dtype(np.int32),
            file_endian=file_endian,
            chunk_bytes=chunk_bytes,
            header_bytes=header,
        )

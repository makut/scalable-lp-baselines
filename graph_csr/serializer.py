from __future__ import annotations

from pathlib import Path
from typing import Optional

from .graph import GraphCSR
from .io_utils import Endian
from .large_int32 import LargeInt32Array
from .large_int64_raw import LargeInt64RawArray


class GraphCSRSerializer:
    """
    Directory layout:
      edge_ends.bin      (LargeInt32Array format)
      edge_starts.bin    (raw int64 big-endian)
      timestamps.bin     (LargeInt32Array format) or empty
    """

    EDGE_ENDS = "edge_ends.bin"
    EDGE_STARTS = "edge_starts.bin"
    TIMESTAMPS = "timestamps.bin"

    @staticmethod
    def serialize(
        graph: GraphCSR,
        folder: Path | str,
        *,
        file_endian: Endian = "big",
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        graph.edge_ends.to_file(folder / GraphCSRSerializer.EDGE_ENDS, file_endian=file_endian, chunk_bytes=chunk_bytes)

        graph.edge_starts.to_file(folder / GraphCSRSerializer.EDGE_STARTS, file_endian=file_endian, chunk_bytes=chunk_bytes)

        ts_path = folder / GraphCSRSerializer.TIMESTAMPS
        if graph.timestamps is None:
            ts_path.write_bytes(b"")
        else:
            graph.timestamps.to_file(ts_path, file_endian=file_endian, chunk_bytes=chunk_bytes)

    @staticmethod
    def deserialize(
        folder: Path | str,
        *,
        use_mmap: bool,
        file_endian: Endian = "big",
        writable: bool = False,
        allow_non_native: bool = True,
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> GraphCSR:
        """
        use_mmap controls:
          - edge_ends (LargeInt32Array.from_file)
          - edge_starts (LargeInt64RawArray.from_file)
          - timestamps (LargeInt32Array.from_file, if the file is not empty)
        """
        folder = Path(folder)

        edge_ends_path = folder / GraphCSRSerializer.EDGE_ENDS
        edge_starts_path = folder / GraphCSRSerializer.EDGE_STARTS
        timestamps_path = folder / GraphCSRSerializer.TIMESTAMPS

        edge_ends = LargeInt32Array.from_file(
            edge_ends_path,
            use_mmap=use_mmap,
            file_endian=file_endian,
            writable=writable,
            allow_non_native=allow_non_native,
            chunk_bytes=chunk_bytes,
        )

        edge_starts = LargeInt64RawArray.from_file(
            edge_starts_path,
            use_mmap=use_mmap,
            file_endian=file_endian,
            writable=writable,
            allow_non_native=allow_non_native,
            chunk_bytes=chunk_bytes,
        )

        timestamps: Optional[LargeInt32Array] = None
        if timestamps_path.exists() and timestamps_path.stat().st_size > 0:
            timestamps = LargeInt32Array.from_file(
                timestamps_path,
                use_mmap=use_mmap,
                file_endian=file_endian,
                writable=writable,
                allow_non_native=allow_non_native,
                chunk_bytes=chunk_bytes,
            )

        return GraphCSR(edge_ends=edge_ends, edge_starts=edge_starts, timestamps=timestamps)

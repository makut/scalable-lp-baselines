from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .large_int32 import LargeInt32Array
from .large_int64_raw import LargeInt64RawArray


@dataclass
class GraphCSR:
    """CSR graph with optional per-edge int32 timestamps.

    The `timestamps` array is opaque: the only guarantee is monotonicity —
    if `timestamps[i] < timestamps[j]` then edge `i` was added strictly
    before edge `j`. Equal values mean "same time" (granularity ~1 second).
    No base offset, unit, or epoch is assumed.
    """

    edge_ends: Optional[LargeInt32Array] = None
    edge_starts: Optional[LargeInt64RawArray] = None
    timestamps: Optional[LargeInt32Array] = None

    def close(self) -> None:
        if self.edge_ends is not None:
            self.edge_ends.close()
        if self.edge_starts is not None:
            self.edge_starts.close()
        if self.timestamps is not None:
            self.timestamps.close()

    def __enter__(self) -> "GraphCSR":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        self.close()
        return False

    def node_neighbors_count(self, node_id: int) -> int:
        node_id = int(node_id)
        if node_id < 0:
            raise ValueError(f"Bad node_id: {node_id}")
        starts = self.edge_starts.numpy()
        if node_id >= starts.size:
            return 0

        start = int(starts[node_id])
        if node_id != starts.size - 1:
            end = int(starts[node_id + 1])
        else:
            end = int(self.edge_ends.size)
        return end - start

    def neighbor_by_index(self, node_id: int, index: int) -> int:
        index = int(index)
        cnt = self.node_neighbors_count(node_id)
        if index < 0 or index >= cnt:
            raise ValueError(f"Bad neighbor index: {index} (cnt={cnt})")
        starts = self.edge_starts.numpy()
        pos = int(starts[int(node_id)]) + index
        return int(self.edge_ends.get(pos))

    def neighbors_view(self, node_id: int) -> np.ndarray:
        """
        Return a fast NumPy view of the neighbor slice without a Python loop.
        The dtype may be non-native if edge_ends maps a non-native-endian file.
        """
        starts = self.edge_starts.numpy()
        node_id = int(node_id)
        if node_id < 0 or node_id >= starts.size:
            return np.empty((0,), dtype=np.int32)

        start = int(starts[node_id])
        if node_id != starts.size - 1:
            end = int(starts[node_id + 1])
        else:
            end = int(self.edge_ends.size)

        return self.edge_ends.numpy()[start:end]

    def edge_timestamp(self, node_id: int, index: int) -> Optional[int]:
        """Return the raw int32 timestamp for the `index`-th out-edge of `node_id`.

        Returns None if the graph has no timestamps. The value is opaque —
        only its relative order vs. other timestamps is meaningful.
        """
        if self.timestamps is None:
            return None
        index = int(index)
        cnt = self.node_neighbors_count(node_id)
        if index < 0 or index >= cnt:
            raise ValueError(f"Bad neighbor index: {index} (cnt={cnt})")
        starts = self.edge_starts.numpy()
        pos = int(starts[int(node_id)]) + index
        return int(self.timestamps.get(pos))

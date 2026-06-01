# GraphCSR

`GraphCSR` is a low-level CSR graph representation designed for graphs that may
not fit comfortably in RAM. Its arrays can be loaded into memory or accessed
through `mmap`.

## Directory layout

A serialized graph is a directory with three files:

```text
edge_ends.bin      [int64 size header] + [int32 neighbor ids]
edge_starts.bin    raw int64 row offsets
timestamps.bin     [int64 size header] + [int32 timestamps], or an empty file
```

`edge_starts` has one offset per node, not `num_nodes + 1` offsets. The last row
implicitly ends at `edge_ends.size`.

The `file_endian` argument controls how bytes are read and written. Big-endian
files are useful when exchanging data with Java `ByteBuffer` pipelines.

## Basic usage

```python
from graph_csr.serializer import GraphCSRSerializer

with GraphCSRSerializer.deserialize(
    "graph_csr",
    use_mmap=True,
    file_endian="big",
) as graph:
    degree = graph.node_neighbors_count(10)
    neighbors = graph.neighbors_view(10)
```

With `use_mmap=False`, arrays are read fully into RAM. With `use_mmap=True`,
the operating system loads pages lazily. If the file byte order differs from
the native byte order, mmap-backed arrays keep a non-native NumPy dtype. This
is correct but can make vectorized operations slower.

Set `allow_non_native=False` to reject non-native mmap access:

```python
graph = GraphCSRSerializer.deserialize(
    "graph_csr",
    use_mmap=True,
    file_endian="big",
    allow_non_native=False,
)
```

## Timestamps

Timestamps are opaque `int32` values. Only their relative ordering is
meaningful: if `timestamps[i] < timestamps[j]`, edge `i` is older than edge
`j`. Equal values represent the same time bucket. No epoch, unit, or base
offset is assumed.

## Serialization

```python
from graph_csr.serializer import GraphCSRSerializer

GraphCSRSerializer.serialize(graph, "graph_csr", file_endian="big")
```

Close mmap-backed graphs explicitly or use them as context managers.

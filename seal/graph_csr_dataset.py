# Based on facebookresearch/SEAL_OGB, licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_csr.serializer import GraphCSRSerializer


def _log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[GRAPH CSR DATASET {timestamp}] {message}', flush=True)


class GraphCSRData:
    def __init__(
        self,
        root,
        num_nodes,
        graph_root,
        file_endian='big',
        use_mmap=True,
        allow_non_native=True,
        chunk_bytes=256 * 1024 * 1024,
        x=None,
    ):
        self.root = root
        self.num_nodes = int(num_nodes)
        self.graph_root = graph_root
        self.file_endian = file_endian
        self.use_mmap = use_mmap
        self.allow_non_native = allow_non_native
        self.chunk_bytes = chunk_bytes
        self.x = x


def load_graph_csr_data(
    graph_root,
    *,
    file_endian='big',
    use_mmap=True,
    allow_non_native=True,
    chunk_bytes=256 * 1024 * 1024,
):
    _log(f'Loading graph metadata graph_root={graph_root}')
    with GraphCSRSerializer.deserialize(
        graph_root,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        num_nodes = int(graph.edge_starts.numpy().size)
        num_positive_edges = int(graph.edge_ends.size)

    data = GraphCSRData(
        root=str(graph_root),
        num_nodes=num_nodes,
        graph_root=str(graph_root),
        file_endian=file_endian,
        use_mmap=use_mmap,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        x=None,
    )
    meta = {'num_positive_edges': num_positive_edges}
    _log(
        f'Loaded graph metadata num_nodes={num_nodes} train_edges={num_positive_edges}'
    )
    return data, meta

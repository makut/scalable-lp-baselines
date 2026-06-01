# Based on facebookresearch/SEAL_OGB, licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.
import os
import time

import numpy as np
import scipy.sparse as ssp

try:
    from .graph_csr_kernels import (
        NUMBA_AVAILABLE,
        _build_induced_edges_neighbor_scan,
        _build_induced_edges_pairwise,
        _count_total_neighbors_numba,
        _fill_neighbors_numba,
        _fill_neighbors_with_owner_numba,
        _merge_sorted_unique_int64,
        _sample_neighbors_per_vertex_numba,
        _sorted_setdiff_int64,
        _unique_sorted_int64,
    )
except ImportError:
    from graph_csr_kernels import (
        NUMBA_AVAILABLE,
        _build_induced_edges_neighbor_scan,
        _build_induced_edges_pairwise,
        _count_total_neighbors_numba,
        _fill_neighbors_numba,
        _fill_neighbors_with_owner_numba,
        _merge_sorted_unique_int64,
        _sample_neighbors_per_vertex_numba,
        _sorted_setdiff_int64,
        _unique_sorted_int64,
    )


def _log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[GRAPH CSR DATASET {timestamp}] {message}', flush=True)


DEBUG_SUBGRAPH = os.environ.get('SEAL_GRAPHCSR_DEBUG_SUBGRAPH', '0') == '1'
SLOW_SUBGRAPH_SEC = float(os.environ.get('SEAL_GRAPHCSR_SLOW_SUBGRAPH_SEC', '5.0'))
DEBUG_COLLECT = os.environ.get('SEAL_GRAPHCSR_DEBUG_COLLECT', '0') == '1'
SLOW_COLLECT_SEC = float(os.environ.get('SEAL_GRAPHCSR_SLOW_COLLECT_SEC', '1.0'))


def _build_induced_edges_pairwise_python(graph, nodes, src_i, dst_i):
    """Python correctness fallback for non-native CSR dtypes."""
    nodes_list = nodes.tolist()
    num_nodes = len(nodes_list)
    global_to_local: dict[int, int] = {}
    for i in range(num_nodes):
        global_to_local[int(nodes_list[i])] = i

    rows_list: list[int] = []
    cols_list: list[int] = []
    for i in range(num_nodes):
        v_global = int(nodes_list[i])
        neigh = np.asarray(graph.neighbors_view(v_global), dtype=np.int64)
        for w_global in neigh.tolist():
            w_local = global_to_local.get(int(w_global), -1)
            if w_local <= i:
                continue
            if (i == src_i and w_local == dst_i) or (i == dst_i and w_local == src_i):
                continue
            rows_list.append(i)
            cols_list.append(w_local)
            rows_list.append(w_local)
            cols_list.append(i)

    if rows_list:
        rows = np.asarray(rows_list, dtype=np.int64)
        cols = np.asarray(cols_list, dtype=np.int64)
        vals = np.ones(rows.size, dtype=np.int8)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        vals = np.empty(0, dtype=np.int8)
    return rows, cols, vals


def _sample_neighbors_per_vertex_python(graph, frontier, per_vertex_k):
    """Python fallback for non-native CSR dtypes."""
    per_vertex_k = int(per_vertex_k)
    if per_vertex_k <= 0 or len(frontier) == 0:
        return np.empty(0, dtype=np.int64), 0

    chunks = []
    scanned = 0
    for vertex in frontier:
        neigh = np.asarray(graph.neighbors_view(int(vertex)), dtype=np.int64)
        degree = int(neigh.size)
        if degree == 0:
            continue
        if degree <= per_vertex_k:
            chunks.append(neigh)
            scanned += degree
        else:
            idx = np.random.randint(0, degree, size=per_vertex_k)
            chunks.append(neigh[idx])
            scanned += per_vertex_k
    if not chunks:
        return np.empty(0, dtype=np.int64), scanned
    return np.concatenate(chunks).astype(np.int64, copy=False), scanned


def _collect_neighbors_sampled(graph, frontier, per_vertex_k):
    """Collect a bounded sample of neighbors for each frontier vertex."""
    edge_starts = graph.edge_starts.numpy()
    edge_ends = graph.edge_ends.numpy()
    use_numba = (
        NUMBA_AVAILABLE
        and getattr(edge_starts.dtype, 'isnative', False)
        and getattr(edge_ends.dtype, 'isnative', False)
    )
    frontier_arr = np.asarray(frontier, dtype=np.int64)
    if use_numba:
        return _sample_neighbors_per_vertex_numba(
            edge_starts, edge_ends, int(graph.edge_ends.size), frontier_arr, int(per_vertex_k)
        )
    return _sample_neighbors_per_vertex_python(graph, frontier_arr, int(per_vertex_k))


def _collect_neighbors(graph, nodes, *, with_owner=False):
    """Collect concatenated CSR neighbors, optionally with their local owners."""
    total_t0 = time.time()
    edge_starts = graph.edge_starts.numpy()
    edge_ends = graph.edge_ends.numpy()
    use_numba_collect = (
        NUMBA_AVAILABLE
        and getattr(edge_starts.dtype, 'isnative', False)
        and getattr(edge_ends.dtype, 'isnative', False)
    )

    nodes_arr = np.asarray(nodes, dtype=np.int64)
    if use_numba_collect:
        count_t0 = time.time()
        scanned = _count_total_neighbors_numba(edge_starts, int(graph.edge_ends.size), nodes_arr)
        count_sec = time.time() - count_t0
        if with_owner:
            fill_t0 = time.time()
            neighbors, owners = _fill_neighbors_with_owner_numba(
                edge_starts, edge_ends, int(graph.edge_ends.size), nodes_arr
            )
            fill_sec = time.time() - fill_t0
            total_sec = time.time() - total_t0
            if DEBUG_COLLECT or total_sec >= SLOW_COLLECT_SEC:
                _log(
                    f'_collect_neighbors with_owner={with_owner} backend=numba '
                    f'nodes={len(nodes_arr)} scanned={int(scanned)} '
                    f'count={count_sec:.3f}s fill={fill_sec:.3f}s total={total_sec:.3f}s'
                )
            return neighbors, int(scanned), owners
        fill_t0 = time.time()
        neighbors = _fill_neighbors_numba(
            edge_starts, edge_ends, int(graph.edge_ends.size), nodes_arr
        )
        fill_sec = time.time() - fill_t0
        total_sec = time.time() - total_t0
        if DEBUG_COLLECT or total_sec >= SLOW_COLLECT_SEC:
            _log(
                f'_collect_neighbors backend=numba '
                f'nodes={len(nodes_arr)} scanned={int(scanned)} '
                f'count={count_sec:.3f}s fill={fill_sec:.3f}s total={total_sec:.3f}s'
            )
        return neighbors, int(scanned)

    warning_owner = getattr(graph, '_seal_numba_collect_warning_logged', False)
    if not warning_owner:
        reason = []
        if not NUMBA_AVAILABLE:
            reason.append('numba is not installed')
        if not getattr(edge_starts.dtype, 'isnative', False):
            reason.append(f'edge_starts dtype is non-native ({edge_starts.dtype})')
        if not getattr(edge_ends.dtype, 'isnative', False):
            reason.append(f'edge_ends dtype is non-native ({edge_ends.dtype})')
        if not reason:
            reason.append('raw CSR arrays are not numba-compatible in this environment')
        _log(
            'WARNING: _collect_neighbors is using Python fallback because '
            + '; '.join(reason)
        )
        try:
            setattr(graph, '_seal_numba_collect_warning_logged', True)
        except Exception:
            pass

    if len(nodes) == 0:
        empty = np.empty((0,), dtype=np.int64)
        if with_owner:
            return empty, 0, empty
        return empty, 0

    counts = np.empty((len(nodes),), dtype=np.int64)
    scanned = 0
    count_t0 = time.time()
    for i, node in enumerate(nodes):
        neigh = graph.neighbors_view(int(node))
        counts[i] = int(neigh.size)
        scanned += int(neigh.size)
    count_sec = time.time() - count_t0

    total = int(counts.sum())
    neighbors = np.empty((total,), dtype=np.int64)
    owners = np.empty((total,), dtype=np.int64) if with_owner else None
    write_pos = 0

    fill_t0 = time.time()
    for i, node in enumerate(nodes):
        neigh = graph.neighbors_view(int(node))
        count = int(neigh.size)
        if count == 0:
            continue
        neighbors[write_pos:write_pos + count] = np.asarray(neigh, dtype=np.int64)
        if owners is not None:
            owners[write_pos:write_pos + count] = i
        write_pos += count
    fill_sec = time.time() - fill_t0
    total_sec = time.time() - total_t0
    if DEBUG_COLLECT or total_sec >= SLOW_COLLECT_SEC:
        _log(
            f'_collect_neighbors backend=python '
            f'nodes={len(nodes_arr)} scanned={int(scanned)} '
            f'count={count_sec:.3f}s fill={fill_sec:.3f}s total={total_sec:.3f}s'
        )

    if with_owner:
        return neighbors, scanned, owners
    return neighbors, scanned


def _k_hop_subgraph_graph_csr(
    src,
    dst,
    num_hops,
    graph,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    per_vertex_oversample=1.5,
    graph_csr_use_per_vertex_sampling=True,
    graph_csr_use_pairwise_subgraph=True,
    node_features=None,
    y=1,
):
    total_t0 = time.time()
    src = int(src)
    dst = int(dst)
    node_chunks = [np.asarray([src, dst], dtype=np.int64)]
    dist_chunks = [np.asarray([0, 0], dtype=np.int64)]
    visited_sorted = _unique_sorted_int64(node_chunks[0])
    frontier = node_chunks[0]
    hop_debug = []

    use_per_vertex = (
        graph_csr_use_per_vertex_sampling
        and max_nodes_per_hop is not None
        and sample_ratio >= 1.0
    )
    target_n = int(max_nodes_per_hop) if max_nodes_per_hop is not None else 0
    alpha = float(per_vertex_oversample)

    for dist in range(1, num_hops + 1):
        hop_t0 = time.time()
        collect_t0 = time.time()
        if use_per_vertex and frontier.size > 0:
            per_vertex_k = int(np.ceil(target_n * alpha / frontier.size))
            if per_vertex_k < 1:
                per_vertex_k = 1
            neighbor_values, scanned_neighbors = _collect_neighbors_sampled(
                graph, frontier, per_vertex_k
            )
        else:
            per_vertex_k = 0
            neighbor_values, scanned_neighbors = _collect_neighbors(graph, frontier)
        collect_sec = time.time() - collect_t0
        unique_t0 = time.time()
        raw_unique = _unique_sorted_int64(neighbor_values)
        unique_sec = time.time() - unique_t0
        raw_next_size = int(raw_unique.size)
        setdiff_t0 = time.time()
        next_fringe = _sorted_setdiff_int64(raw_unique, visited_sorted)
        setdiff_sec = time.time() - setdiff_t0
        deduped_size = int(next_fringe.size)
        sample_sec = 0.0
        if sample_ratio < 1.0 and next_fringe.size > 0:
            sample_t0 = time.time()
            sample_size = int(sample_ratio * next_fringe.size)
            if sample_size > 0:
                perm = np.random.permutation(next_fringe.size)[:sample_size]
                next_fringe = np.sort(next_fringe[perm])
            else:
                next_fringe = np.empty((0,), dtype=np.int64)
            sample_sec += time.time() - sample_t0
        if max_nodes_per_hop is not None and max_nodes_per_hop < next_fringe.size:
            sample_t0 = time.time()
            perm = np.random.permutation(next_fringe.size)[:max_nodes_per_hop]
            next_fringe = np.sort(next_fringe[perm])
            sample_sec += time.time() - sample_t0
        merge_t0 = time.time()
        final_size = int(next_fringe.size)
        visited_sorted = _merge_sorted_unique_int64(visited_sorted, next_fringe)
        merge_sec = time.time() - merge_t0
        hop_total_sec = time.time() - hop_t0
        hop_debug.append({
            'hop': dist,
            'input_fringe': int(frontier.size),
            'per_vertex_k': int(per_vertex_k),
            'scanned_neighbors': scanned_neighbors,
            'raw_next_size': raw_next_size,
            'deduped_size': deduped_size,
            'final_size': final_size,
            'collect_sec': collect_sec,
            'unique_sec': unique_sec,
            'setdiff_sec': setdiff_sec,
            'sample_sec': sample_sec,
            'merge_sec': merge_sec,
            'expand_sec': hop_total_sec,
            'total_sec': hop_total_sec,
        })
        if next_fringe.size == 0:
            break
        node_chunks.append(next_fringe)
        dist_chunks.append(np.full(next_fringe.size, dist, dtype=np.int64))
        frontier = next_fringe

    nodes = np.concatenate(node_chunks)
    dists = np.concatenate(dist_chunks)
    src_i = 0
    dst_i = 1
    num_subgraph_nodes = nodes.size

    edge_build_t0 = time.time()
    if not graph_csr_use_pairwise_subgraph:
        order = np.argsort(nodes)
        sorted_nodes = nodes[order]
        sorted_local = order.astype(np.int64, copy=False)
        collect_subgraph_t0 = time.time()
        neigh_nodes, scanned_subgraph_neighbors, owner_local = _collect_neighbors(
            graph, nodes, with_owner=True
        )
        collect_subgraph_sec = time.time() - collect_subgraph_t0
        edge_lookup_t0 = time.time()
        rows, cols, vals = _build_induced_edges_neighbor_scan(
            owner_local, neigh_nodes, sorted_nodes, sorted_local, src_i, dst_i
        )
        edge_lookup_sec = time.time() - edge_lookup_t0
        edge_backend = 'legacy_neighbor_scan'
        edge_debug = (
            f'scan_subgraph_neighbors={scanned_subgraph_neighbors} '
            f'collect_subgraph={collect_subgraph_sec:.3f}s '
            f'edge_lookup={edge_lookup_sec:.3f}s'
        )
    else:
        edge_starts = graph.edge_starts.numpy()
        edge_ends = graph.edge_ends.numpy()
        use_numba_edges = (
            NUMBA_AVAILABLE
            and getattr(edge_starts.dtype, 'isnative', False)
            and getattr(edge_ends.dtype, 'isnative', False)
        )
        if use_numba_edges:
            rows, cols, vals = _build_induced_edges_pairwise(
                edge_starts, edge_ends, int(graph.edge_ends.size), nodes, src_i, dst_i
            )
        else:
            rows, cols, vals = _build_induced_edges_pairwise_python(
                graph, nodes, src_i, dst_i
            )
        pairs_checked = num_subgraph_nodes * (num_subgraph_nodes - 1) // 2
        edge_backend = 'pairwise_numba' if use_numba_edges else 'pairwise_python'
        edge_debug = f'pairs_checked={pairs_checked}'
    edge_build_sec = time.time() - edge_build_t0

    csr_t0 = time.time()
    if rows.size:
        subgraph = ssp.csr_matrix((vals, (rows, cols)), shape=(num_subgraph_nodes, num_subgraph_nodes))
    else:
        subgraph = ssp.csr_matrix((num_subgraph_nodes, num_subgraph_nodes))
    csr_sec = time.time() - csr_t0

    if node_features is not None:
        node_features = node_features[nodes.tolist()]

    total_sec = time.time() - total_t0
    if DEBUG_SUBGRAPH or total_sec >= SLOW_SUBGRAPH_SEC:
        hop_parts = []
        for stat in hop_debug:
            hop_parts.append(
                'hop={hop} in={input_fringe} k_per_v={per_vertex_k} scanned={scanned_neighbors} '
                'raw={raw_next_size} dedup={deduped_size} out={final_size} '
                'collect={collect_sec:.3f}s unique={unique_sec:.3f}s setdiff={setdiff_sec:.3f}s '
                'sample={sample_sec:.3f}s merge={merge_sec:.3f}s total={total_sec:.3f}s'.format(
                    **stat
                )
            )
        _log(
            f'Slow subgraph src={src} dst={dst} hops={num_hops} nodes={num_subgraph_nodes} '
            f'edges={rows.size} backend={edge_backend} {edge_debug} '
            f'edge_build={edge_build_sec:.3f}s csr={csr_sec:.3f}s total={total_sec:.3f}s | '
            + ' ; '.join(hop_parts)
        )

    return nodes.tolist(), subgraph, dists.tolist(), node_features, y

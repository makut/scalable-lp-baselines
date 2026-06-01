# Based on facebookresearch/SEAL_OGB, licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.
import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@njit(cache=True)
def _unique_sorted_int64(arr):
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_arr = np.sort(arr.copy())
    unique_count = 1
    for i in range(1, sorted_arr.size):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_count += 1
    out = np.empty((unique_count,), dtype=np.int64)
    out[0] = sorted_arr[0]
    write_pos = 1
    for i in range(1, sorted_arr.size):
        if sorted_arr[i] != sorted_arr[i - 1]:
            out[write_pos] = sorted_arr[i]
            write_pos += 1
    return out


@njit(cache=True)
def _sorted_setdiff_int64(lhs, rhs):
    if lhs.size == 0:
        return np.empty((0,), dtype=np.int64)
    out = np.empty(lhs.size, dtype=np.int64)
    i = 0
    j = 0
    write_pos = 0
    while i < lhs.size:
        while j < rhs.size and rhs[j] < lhs[i]:
            j += 1
        if j >= rhs.size or lhs[i] != rhs[j]:
            out[write_pos] = lhs[i]
            write_pos += 1
        i += 1
    return out[:write_pos]


@njit(cache=True)
def _merge_sorted_unique_int64(lhs, rhs):
    out = np.empty(lhs.size + rhs.size, dtype=np.int64)
    i = 0
    j = 0
    write_pos = 0
    last_written = np.int64(0)
    has_last = False

    while i < lhs.size or j < rhs.size:
        if j >= rhs.size or (i < lhs.size and lhs[i] <= rhs[j]):
            val = lhs[i]
            i += 1
            if j < rhs.size and val == rhs[j]:
                j += 1
        else:
            val = rhs[j]
            j += 1

        if (not has_last) or val != last_written:
            out[write_pos] = val
            write_pos += 1
            last_written = val
            has_last = True

    return out[:write_pos]


@njit(cache=True)
def _binary_search_range(arr, lo, hi, target):
    """Return index of target in arr[lo:hi], or -1 if it is missing."""
    left = int(lo)
    right = int(hi)
    while left < right:
        mid = (left + right) // 2
        value = arr[mid]
        if value < target:
            left = mid + 1
        elif value > target:
            right = mid
        else:
            return mid
    return -1


@njit(cache=True)
def _binary_search_int64(arr, target):
    lo = 0
    hi = arr.size
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < arr.size and arr[lo] == target:
        return lo
    return -1


@njit(cache=True)
def _count_induced_edges(owner_local, neigh_nodes, sorted_nodes, sorted_local, src_i, dst_i):
    edge_count = 0
    for idx in range(neigh_nodes.size):
        local_u = owner_local[idx]
        pos = _binary_search_int64(sorted_nodes, neigh_nodes[idx])
        if pos < 0:
            continue
        local_v = sorted_local[pos]
        if (local_u == src_i and local_v == dst_i) or (local_u == dst_i and local_v == src_i):
            continue
        edge_count += 1
    return edge_count


@njit(cache=True)
def _build_induced_edges_neighbor_scan(owner_local, neigh_nodes, sorted_nodes, sorted_local, src_i, dst_i):
    """Build induced edges using the pre-refactor adjacency-list scan."""
    edge_count = _count_induced_edges(owner_local, neigh_nodes, sorted_nodes, sorted_local, src_i, dst_i)
    rows = np.empty(edge_count, dtype=np.int64)
    cols = np.empty(edge_count, dtype=np.int64)
    vals = np.ones(edge_count, dtype=np.int8)
    write_pos = 0
    for idx in range(neigh_nodes.size):
        local_u = owner_local[idx]
        pos = _binary_search_int64(sorted_nodes, neigh_nodes[idx])
        if pos < 0:
            continue
        local_v = sorted_local[pos]
        if (local_u == src_i and local_v == dst_i) or (local_u == dst_i and local_v == src_i):
            continue
        rows[write_pos] = local_u
        cols[write_pos] = local_v
        write_pos += 1
    return rows, cols, vals


@njit(cache=True)
def _build_induced_edges_pairwise(edge_starts, edge_ends, num_edges, nodes, src_i, dst_i):
    """Build induced edges by binary-searching each selected vertex pair."""
    num_nodes = nodes.size
    edge_starts_size = edge_starts.size

    edge_count = 0
    for i in range(num_nodes):
        u_global = int(nodes[i])
        u_start = int(edge_starts[u_global])
        if u_global + 1 < edge_starts_size:
            u_end = int(edge_starts[u_global + 1])
        else:
            u_end = int(num_edges)
        u_deg = u_end - u_start
        for j in range(i + 1, num_nodes):
            v_global = int(nodes[j])
            v_start = int(edge_starts[v_global])
            if v_global + 1 < edge_starts_size:
                v_end = int(edge_starts[v_global + 1])
            else:
                v_end = int(num_edges)
            v_deg = v_end - v_start
            if u_deg <= v_deg:
                pos = _binary_search_range(edge_ends, u_start, u_end, v_global)
            else:
                pos = _binary_search_range(edge_ends, v_start, v_end, u_global)
            if pos < 0:
                continue
            if (i == src_i and j == dst_i) or (i == dst_i and j == src_i):
                continue
            edge_count += 2

    rows = np.empty(edge_count, dtype=np.int64)
    cols = np.empty(edge_count, dtype=np.int64)
    vals = np.ones(edge_count, dtype=np.int8)

    write_pos = 0
    for i in range(num_nodes):
        u_global = int(nodes[i])
        u_start = int(edge_starts[u_global])
        if u_global + 1 < edge_starts_size:
            u_end = int(edge_starts[u_global + 1])
        else:
            u_end = int(num_edges)
        u_deg = u_end - u_start
        for j in range(i + 1, num_nodes):
            v_global = int(nodes[j])
            v_start = int(edge_starts[v_global])
            if v_global + 1 < edge_starts_size:
                v_end = int(edge_starts[v_global + 1])
            else:
                v_end = int(num_edges)
            v_deg = v_end - v_start
            if u_deg <= v_deg:
                pos = _binary_search_range(edge_ends, u_start, u_end, v_global)
            else:
                pos = _binary_search_range(edge_ends, v_start, v_end, u_global)
            if pos < 0:
                continue
            if (i == src_i and j == dst_i) or (i == dst_i and j == src_i):
                continue
            rows[write_pos] = i
            cols[write_pos] = j
            write_pos += 1
            rows[write_pos] = j
            cols[write_pos] = i
            write_pos += 1

    return rows, cols, vals


@njit(cache=True)
def _count_total_neighbors_numba(edge_starts, num_edges, nodes):
    total = 0
    for i in range(nodes.size):
        node = int(nodes[i])
        start = int(edge_starts[node])
        if node + 1 < edge_starts.size:
            end = int(edge_starts[node + 1])
        else:
            end = int(num_edges)
        total += end - start
    return total


@njit(cache=True)
def _fill_neighbors_numba(edge_starts, edge_ends, num_edges, nodes):
    total = _count_total_neighbors_numba(edge_starts, num_edges, nodes)
    neighbors = np.empty((total,), dtype=np.int64)
    write_pos = 0
    for i in range(nodes.size):
        node = int(nodes[i])
        start = int(edge_starts[node])
        if node + 1 < edge_starts.size:
            end = int(edge_starts[node + 1])
        else:
            end = int(num_edges)
        for pos in range(start, end):
            neighbors[write_pos] = int(edge_ends[pos])
            write_pos += 1
    return neighbors


@njit(cache=True)
def _fill_neighbors_with_owner_numba(edge_starts, edge_ends, num_edges, nodes):
    total = _count_total_neighbors_numba(edge_starts, num_edges, nodes)
    neighbors = np.empty((total,), dtype=np.int64)
    owners = np.empty((total,), dtype=np.int64)
    write_pos = 0
    for i in range(nodes.size):
        node = int(nodes[i])
        start = int(edge_starts[node])
        if node + 1 < edge_starts.size:
            end = int(edge_starts[node + 1])
        else:
            end = int(num_edges)
        for pos in range(start, end):
            neighbors[write_pos] = int(edge_ends[pos])
            owners[write_pos] = i
            write_pos += 1
    return neighbors, owners


@njit(cache=True)
def _sample_neighbors_per_vertex_numba(edge_starts, edge_ends, num_edges, frontier, per_vertex_k):
    """Sample at most per_vertex_k neighbors from each frontier vertex."""
    frontier_size = frontier.size
    per_vertex_k = int(per_vertex_k)
    if frontier_size == 0 or per_vertex_k <= 0:
        return np.empty(0, dtype=np.int64), 0

    out = np.empty(frontier_size * per_vertex_k, dtype=np.int64)
    write_pos = 0
    scanned = 0
    edge_starts_size = edge_starts.size
    for i in range(frontier_size):
        vertex = int(frontier[i])
        start = int(edge_starts[vertex])
        if vertex + 1 < edge_starts_size:
            end = int(edge_starts[vertex + 1])
        else:
            end = int(num_edges)
        degree = end - start
        if degree == 0:
            continue
        if degree <= per_vertex_k:
            for j in range(degree):
                out[write_pos] = int(edge_ends[start + j])
                write_pos += 1
            scanned += degree
        else:
            for _ in range(per_vertex_k):
                idx = np.random.randint(0, degree)
                out[write_pos] = int(edge_ends[start + idx])
                write_pos += 1
            scanned += per_vertex_k
    return out[:write_pos], scanned

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT License.
# Modified for GraphCSR-backed training in this repository.
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.

import numpy as np
import scipy.sparse as ssp
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data


def drnl_node_labeling(adj, src, dst):
    """Double Radius Node Labeling (DRNL)."""
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    """Distance Encoding."""
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    """Distance Encoding Plus (DRNL-style temporary node removal)."""
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label="drnl"):
    """Construct a pytorch_geometric Data object from a scipy csr adjacency."""
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == "drnl":
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == "hop":
        z = torch.tensor(dists)
    elif node_label == "zo":
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == "de":
        z = de_node_labeling(adj, 0, 1)
    elif node_label == "de+":
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == "degree":
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    return Data(
        node_features,
        edge_index,
        edge_weight=edge_weight,
        y=y,
        z=z,
        node_id=node_ids,
        num_nodes=num_nodes,
    )

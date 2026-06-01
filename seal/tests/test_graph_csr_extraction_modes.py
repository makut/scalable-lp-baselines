from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from graph_csr.graph import GraphCSR
from graph_csr.large_int32 import LargeInt32Array
from graph_csr.large_int64_raw import LargeInt64RawArray
from seal import graph_csr_extraction
from seal.config import SEALExtractionConfig, load_config


def _make_graph(adjacency: dict[int, list[int]]) -> GraphCSR:
    num_nodes = max(
        [*adjacency, *(neighbor for neighbors in adjacency.values() for neighbor in neighbors)]
    ) + 1
    edge_starts = []
    edge_ends = []
    for node in range(num_nodes):
        edge_starts.append(len(edge_ends))
        edge_ends.extend(sorted(adjacency.get(node, [])))
    return GraphCSR(
        edge_ends=LargeInt32Array(size=len(edge_ends), arr=np.asarray(edge_ends, dtype=np.int32)),
        edge_starts=LargeInt64RawArray(arr=np.asarray(edge_starts, dtype=np.int64)),
    )


def _extract(
    graph: GraphCSR,
    *,
    use_per_vertex_sampling: bool = True,
    use_pairwise_subgraph: bool = True,
    max_nodes_per_hop=None,
):
    return graph_csr_extraction._k_hop_subgraph_graph_csr(
        0,
        1,
        1,
        graph,
        max_nodes_per_hop=max_nodes_per_hop,
        graph_csr_use_per_vertex_sampling=use_per_vertex_sampling,
        graph_csr_use_pairwise_subgraph=use_pairwise_subgraph,
    )


class GraphCSRExtractionModesTest(unittest.TestCase):
    def test_optimized_algorithms_are_enabled_by_default(self) -> None:
        config = SEALExtractionConfig()
        self.assertTrue(config.graph_csr_use_per_vertex_sampling)
        self.assertTrue(config.graph_csr_use_pairwise_subgraph)

    def test_flags_are_loaded_from_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "seal.yaml"
            config_path.write_text(
                """
dataset:
  graph_csr_root: /tmp/graph
seal:
  graph_csr_use_per_vertex_sampling: false
  graph_csr_use_pairwise_subgraph: false
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertFalse(config.seal.graph_csr_use_per_vertex_sampling)
        self.assertFalse(config.seal.graph_csr_use_pairwise_subgraph)

    def test_modes_produce_same_regular_undirected_subgraph(self) -> None:
        graph = _make_graph({
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2],
        })

        optimized = _extract(graph)
        legacy_subgraph = _extract(graph, use_pairwise_subgraph=False)

        self.assertEqual(optimized[0], legacy_subgraph[0])
        self.assertEqual(optimized[2], legacy_subgraph[2])
        np.testing.assert_array_equal(optimized[1].toarray(), legacy_subgraph[1].toarray())

    def test_flags_select_algorithms_independently(self) -> None:
        graph = _make_graph({
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1],
            3: [0, 1],
        })

        for use_per_vertex_sampling in (False, True):
            for use_pairwise_subgraph in (False, True):
                with self.subTest(
                    use_per_vertex_sampling=use_per_vertex_sampling,
                    use_pairwise_subgraph=use_pairwise_subgraph,
                ):
                    with (
                        mock.patch.object(
                            graph_csr_extraction,
                            "_collect_neighbors_sampled",
                            wraps=graph_csr_extraction._collect_neighbors_sampled,
                        ) as collect_sampled,
                        mock.patch.object(
                            graph_csr_extraction,
                            "_build_induced_edges_neighbor_scan",
                            wraps=graph_csr_extraction._build_induced_edges_neighbor_scan,
                        ) as build_neighbor_scan,
                    ):
                        _extract(
                            graph,
                            use_per_vertex_sampling=use_per_vertex_sampling,
                            use_pairwise_subgraph=use_pairwise_subgraph,
                            max_nodes_per_hop=1,
                        )

                    self.assertEqual(collect_sampled.call_count, int(use_per_vertex_sampling))
                    self.assertEqual(build_neighbor_scan.call_count, int(not use_pairwise_subgraph))

    def test_neighbor_scan_subgraph_preserves_self_loops(self) -> None:
        graph = _make_graph({
            0: [1, 2],
            1: [0],
            2: [0, 2],
        })

        optimized = _extract(graph)[1].toarray()
        neighbor_scan = _extract(graph, use_pairwise_subgraph=False)[1].toarray()

        self.assertEqual(optimized[2, 2], 0)
        self.assertEqual(neighbor_scan[2, 2], 1)


if __name__ == "__main__":
    unittest.main()

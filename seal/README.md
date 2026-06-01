# SEAL

This package implements dynamic SEAL link prediction over `GraphCSR`. For each
positive or negative edge, it extracts an enclosing subgraph from the
mmap-backed training graph, assigns structural labels, converts the subgraph
to a PyG graph, and trains a graph classifier.

The included trainer is intended for very large graphs where pre-materializing
all enclosing subgraphs is impractical.

## Input

SEAL consumes the prepared dataset root produced by `scripts.prepare_dataset`:

```text
my_dataset/
  train_csr/
  train_pairs_csr/
  valid_edge.npy
  valid_edge_neg.npy
  test_edge.npy
  test_edge_neg.npy
```

Set `dataset.root` to this directory. `dataset.num_nodes` may remain `0`: the
trainer infers it from the CSR graph. Set a positive value only when you want
an additional consistency check. `dataset.graph_csr_file_endian` must match
the prepared CSR graph.

## Install

Install the repository with the SEAL dependencies:

```bash
python -m pip install -e ".[seal]"
```

## Run

```bash
cp seal/configs/link_prediction_default.yaml /tmp/seal.yaml
# Edit dataset.root, dataset.graph_csr_file_endian, and output.dir.
python -m seal.seal_link_pred --config /tmp/seal.yaml
```

SEAL currently supports undirected graphs and runs as a single process. It
uses CUDA when available and falls back to CPU otherwise.

## Configuration

The main sections in the YAML file are:

| Section | Purpose |
| --- | --- |
| `dataset` | Prepared dataset root and `GraphCSR` loading options |
| `train_loader` | Batch size, workers, and training-negative sampling |
| `seal` | Enclosing-subgraph extraction controls |
| `model` | Graph-classifier architecture |
| `training` | Optimization, validation cadence, and early stopping |
| `runtime` | Optional validation and test batch limits |
| `output` | Result directory |

Available models are `DGCNN`, `SAGE`, `GCN`, and `GIN`.

The extraction options are the most important scalability controls:

| Option | Purpose |
| --- | --- |
| `num_hops` | Radius of the enclosing subgraph |
| `ratio_per_hop` | Fraction of each frontier retained during expansion |
| `max_nodes_per_hop` | Optional hard cap for each hop |
| `per_vertex_oversample` | Oversampling factor used with capped expansion |
| `graph_csr_use_per_vertex_sampling` | Use bounded per-vertex frontier sampling when a hop cap is configured. Set to `false` to restore full-frontier collection before sampling |
| `graph_csr_use_pairwise_subgraph` | Build induced subgraph edges with pairwise CSR lookup. Set to `false` to restore adjacency-list scanning |
| `node_label` | Structural node-labeling scheme, usually `drnl` |

For large or dense graphs, set `max_nodes_per_hop` before increasing
`num_hops`. Both GraphCSR extraction optimizations are enabled by default.
The switches are useful for compatibility checks and algorithm comparisons.

## Outputs

The output directory contains:

```text
output/
  cmd_input.txt
  log.txt
  tensorboard/
  run1/
    best_model.pth
  ...
```

The trainer selects `best_model.pth` by the configured validation metric and
evaluates the selected checkpoint on the test split after training.

When `model.train_node_embedding: true`, the checkpoint also stores the node
embedding table. Convert it to the shared embedding-table format with:

```bash
python -m baselines.seal.node_embedding_to_embedding_table \
  --checkpoint /data/lpp-runs/my_dataset/seal/run1/best_model.pth \
  --out-dir /data/lpp-runs/my_dataset/seal/node_embedding_table \
  --backend vanilla
```

## Diagnostics

The trainer logs host and CUDA memory usage. Additional environment variables
enable detailed timing for slow batches and subgraph extraction:

```bash
SEAL_TRAIN_DEBUG=1
SEAL_TEST_DEBUG=1
SEAL_GRAPHCSR_DEBUG_SUBGRAPH=1
SEAL_GRAPHCSR_DEBUG_COLLECT=1
```

The adapter for running the external `facebookresearch/SEAL_OGB`
implementation is separate and documented in
[`../baselines/seal/README.md`](../baselines/seal/README.md).

## Attribution and license

This internal implementation is based on and contains modified portions of
[`facebookresearch/SEAL_OGB`](https://github.com/facebookresearch/SEAL_OGB),
which is distributed under the MIT License. This repository preserves the
upstream notice in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The repository as a
whole is distributed under the Apache License, Version 2.0.

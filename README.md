# Scalable Link Prediction with Node2Vec and SEAL

This repository contains the code accompanying a study of link prediction on
very large graphs. It provides memory-mapped graph storage, temporal dataset
preparation, a distributed Node2Vec implementation, a dynamic SEAL trainer,
shared downstream evaluation, and adapters for external baselines.

The code is organized around graphs that are too large for convenient
in-memory processing. `GraphCSR` keeps node ids in `int32`, row offsets in
`int64`, and supports mmap-backed arrays with configurable byte order.

## Repository layout

| Path | Purpose |
| --- | --- |
| `graph_csr/` | Low-level mmap-friendly CSR graph representation |
| `dataset_utils/` | Shared datasets, samplers, metrics, and split loading |
| [`embedding_table_utils/`](embedding_table_utils/README.md) | Vanilla and TorchRec embedding-table backends |
| [`node2vec/`](node2vec/README.md) | Node2Vec training and downstream link prediction |
| [`embedding_lp/`](embedding_lp/README.md) | Method-agnostic classifier over frozen embeddings |
| [`seal/`](seal/README.md) | Dynamic SEAL subgraph extraction and training |
| `baselines/` | GRAPE Node2Vec and SEAL_OGB adapters |
| `scripts/` | Dataset preparation, OGB orchestration, and monitoring |

## Installation

Python 3.10 or newer is required. Create an isolated environment and install
the core package plus the extras needed for OGB and SEAL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ogb,seal,monitor,test]"
```

The default Node2Vec configuration uses the TorchRec backend for distributed
embedding tables. Install TorchRec and FBGEMM GPU separately using versions
compatible with your PyTorch, CUDA, and hardware stack. For a local
single-process smoke run, set `embedding_table_config.backend: vanilla` in the
Node2Vec YAML config.

GRAPE is optional:

```bash
python -m pip install -e ".[grape]"
```

## Dataset format

The training pipeline expects a prepared dataset root:

```text
dataset/
  train_csr/
  train_pairs_csr/
  valid_edge.npy
  valid_edge_neg.npy
  test_edge.npy
  test_edge_neg.npy
```

`train_csr` stores the graph visible during training. `train_pairs_csr` stores
canonical positive pairs with `u < v`. Evaluation edge files are NumPy arrays
with shape `[N, 2]`.

The binary `GraphCSR` format is documented in
[`graph_csr/README.md`](graph_csr/README.md).

## Prepare a dataset

Convert one of the supported OGB datasets:

```bash
python -m scripts.prepare_ogb_dataset \
  --dataset ogbn-papers100M \
  --ogb-root /data/ogb \
  --out-dir /data/lpp/ogbn-papers100M/raw \
  --file-endian big
```

Create a temporal train/validation/test split and the canonical pairs CSR:

```bash
python -m scripts.prepare_dataset \
  --graph-dir /data/lpp/ogbn-papers100M/raw/graph_csr \
  --out-root /data/lpp/ogbn-papers100M/prepared \
  --val-edges 100000 \
  --test-edges 100000 \
  --use-mmap \
  --file-endian big \
  --out-file-endian big \
  --allow-non-native
```

`scripts.prepare_ogb_dataset` supports `ogbl-citation2` and
`ogbn-papers100M`. For `ogbl-citation2`, the converter also preserves the
official OGB splits under the raw output directory. The experiment runner
below intentionally creates fresh temporal splits, so its results are not
directly comparable to the official OGB benchmark leaderboard.

## Run experiments

The orchestration script downloads data, prepares temporal splits, runs SEAL,
trains Node2Vec embeddings, evaluates the shared classifier, and runs the
GRAPE Node2Vec baseline:

Install the `grape` extra before using all default stages. To run only the
included trainers, pass `--stages download prepare seal node2vec node2vec-lp`.

```bash
python -m scripts.run_ogb_experiments \
  --ogb-root /data/ogb \
  --data-root /data/lpp/ogb \
  --runs-root /data/lpp-runs/ogb \
  --datasets ogbn-papers100M \
  --val-edges 100000 \
  --test-edges 100000
```

Inspect generated commands without starting work:

```bash
python -m scripts.run_ogb_experiments \
  --ogb-root /data/ogb \
  --data-root /data/lpp/ogb \
  --runs-root /data/lpp-runs/ogb \
  --datasets ogbl-citation2 \
  --dry-run
```

For individual runs, copy and edit the YAML templates in `node2vec/configs/`,
`seal/configs/`, and `embedding_lp/configs/`, then launch:

```bash
python -m node2vec.train --config /path/to/node2vec.yaml
python -m embedding_lp.train --config /path/to/link_prediction.yaml
python -m seal.seal_link_pred --config /path/to/seal.yaml
```

Use `torchrun` for distributed Node2Vec and downstream classifier training.
SEAL currently runs as a single process.

The parameters used for the article experiments are recorded in
[`EXPERIMENT_CONFIGS.md`](EXPERIMENT_CONFIGS.md).

## Method guides

Detailed usage notes live next to the implementations:

- [`node2vec/README.md`](node2vec/README.md): local and distributed Node2Vec,
  shared negatives, checkpoints, and resume.
- [`seal/README.md`](seal/README.md): dynamic enclosing-subgraph extraction,
  scalability controls, outputs, and diagnostics.
- [`embedding_lp/README.md`](embedding_lp/README.md): the shared downstream
  classifier for frozen embedding checkpoints.
- [`embedding_table_utils/README.md`](embedding_table_utils/README.md):
  embedding backends, checkpoint layout, and frozen loading.

## External baselines

The [`baselines/README.md`](baselines/README.md) guide documents:

- GRAPE/ensmallen Node2Vec export, training, and checkpoint conversion.
- Exporting prepared datasets to the objects expected by
  `facebookresearch/SEAL_OGB`.

## Resource monitoring

Wrap an experiment to collect CPU, RAM, and NVIDIA GPU statistics:

```bash
python scripts/watch_resources.py -- \
  torchrun --standalone --nproc_per_node=8 \
  -m node2vec.train --config /path/to/node2vec.yaml
```

## Tests

Run the lightweight unit suite with:

```bash
python -m unittest discover -v
```

SEAL imports require the `seal` extra. TorchRec-specific tests require a
working TorchRec/FBGEMM installation for the current platform.

## Citation

Please cite the accompanying article when using this code. A full citation
entry can be added once the article metadata is public.

## License

This project is released under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The internal SEAL implementation
contains portions based on the MIT-licensed
[`facebookresearch/SEAL_OGB`](https://github.com/facebookresearch/SEAL_OGB)
project. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

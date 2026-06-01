# Embedding Table Utilities

This package provides the embedding-table abstraction used by Node2Vec,
baseline conversion, and frozen-embedding link prediction.

## Backends

| Backend | Use case |
| --- | --- |
| `vanilla` | Regular `torch.nn.Embedding` for local runs and portable baseline checkpoints |
| `torchrec` | TorchRec-backed sharded tables for distributed large-graph training |

TorchRec and FBGEMM GPU must be installed separately with versions compatible
with the local PyTorch and CUDA stack.

## Configuration

Embedding tables are described by `EmbeddingTableConfig`:

```python
from embedding_table_utils import EmbeddingTableConfig

config = EmbeddingTableConfig(
    backend="vanilla",
    num_embeddings=1_000_000,
    embedding_dim=128,
    optimizer_type="adam",
    optimizer_kwargs={"learning_rate": 0.01},
)
```

For TorchRec, `sharding_type` may be `row_wise` or `column_wise`.
`compute_kernel_policy` may be `auto`, `prefer_hbm`, or `allow_uvm`.

## Checkpoint layout

Embedding tables save chunked per-rank checkpoints:

```text
embedding_table/
  rank0/
    manifest.pt
    ...
  rank1/
    manifest.pt
    ...
```

These checkpoints are designed for large tables and streamed restoration.
Distributed checkpoints must be loaded with the same world size used to write
them. They do not guarantee portability across arbitrary sharding layouts or
compute-kernel policies.

## Frozen loading

`ReadOnlyEmbeddingStore` reconstructs a frozen table directly from checkpoint
metadata:

```python
import torch
from embedding_table_utils import ReadOnlyEmbeddingStore

store = ReadOnlyEmbeddingStore.from_checkpoint(
    "/data/lpp-runs/my_dataset/node2vec/checkpoints/final_embedding_shards",
    device=torch.device("cpu"),
)
embeddings = store.lookup(torch.tensor([1, 5, 9]))
```

The loader accepts:

- An inner directory containing `rank0/manifest.pt`.
- A Node2Vec step directory containing `embedding_table/rank0/manifest.pt`.
- A Node2Vec checkpoint root containing `latest_checkpoint.txt`.

## Baseline conversion

Convert NumPy embeddings into the shared layout:

```bash
python -m baselines.convert_embeddings_to_embedding_table \
  --npy /path/to/embeddings.npy \
  --out-dir /data/lpp-runs/my_dataset/baseline_embedding_table \
  --backend vanilla
```

Use `--backend torchrec` under `torchrun` when the downstream classifier should
load a sharded table. More conversion options are documented in
[`../baselines/README.md`](../baselines/README.md).

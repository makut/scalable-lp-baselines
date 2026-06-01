# Node2Vec

This package trains Node2Vec embeddings directly from a `GraphCSR` graph. It
supports a dense PyTorch embedding table for small runs and a TorchRec-backed
sharded table for large distributed runs.

The implementation follows the standard Node2Vec random-walk parameters:

- `p` controls the return probability.
- `q` controls inward versus outward exploration.
- `walk_length`, `context_size`, and `walks_per_node` control the generated
  training windows.

## Input

Point `dataset.path` at the `train_csr/` directory produced by
`scripts.prepare_dataset`:

```text
my_dataset/
  train_csr/
  valid_edge.npy
  valid_edge_neg.npy
```

Set `dataset.split_root` to the prepared dataset root when parameter-free
validation is enabled through `training.val_eval_every`. Validation scores
edges with the embedding dot product and does not train an extra classifier.

The `dataset.file_endian` value must match the byte order used while preparing
the CSR graph.

## Run locally

Copy the template and select the `vanilla` backend for a single-process smoke
run:

```bash
cp node2vec/configs/node2vec_default.yaml /tmp/node2vec.yaml
# Edit dataset.path, dataset.split_root, output.dir, and:
# embedding_table_config.backend: vanilla
python -m node2vec.train --config /tmp/node2vec.yaml
```

The vanilla backend stores embeddings in a regular `torch.nn.Embedding`.

## Run with TorchRec

The default template selects `embedding_table_config.backend: torchrec`.
Install TorchRec and FBGEMM GPU versions compatible with the local PyTorch and
CUDA stack, then launch one process per GPU:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m node2vec.train --config /path/to/node2vec.yaml
```

Important TorchRec options:

| Option | Purpose |
| --- | --- |
| `sharding_type` | `row_wise` or `column_wise` embedding sharding |
| `compute_kernel_policy` | `auto`, `prefer_hbm`, or `allow_uvm` |
| `optimizer_type` | Embedding-table optimizer |
| `optimizer_kwargs.learning_rate` | Embedding-table learning rate |

Local TorchRec checkpoints are per-rank artifacts. Resume with the same number
of ranks and a compatible sharding layout.

## Shared negatives

`training.enable_shared_negatives` reduces embedding lookup traffic by scoring
each positive anchor against one shared pool of negative ids. This is useful
when distributed embedding lookup dominates runtime. The trade-offs and
parameter semantics are documented in
[`SHARED_NEGATIVES.md`](SHARED_NEGATIVES.md).

## Outputs

If `training.checkpoint_dir` is `null`, checkpoints are written below
`<output.dir>/checkpoints`:

```text
checkpoints/
  latest_checkpoint.txt
  epoch_.../
    metadata.json
    embedding_table/
      rank0/
      rank1/
      ...
  final_embedding_shards/
    rank0/
    rank1/
    ...
```

`final_embedding_shards/` is the checkpoint to pass to the shared downstream
classifier:

```bash
cp embedding_lp/configs/link_prediction_default.yaml /tmp/node2vec_lp.yaml
# Edit dataset.*, embeddings.checkpoint_dir, and output.dir.
python -m embedding_lp.train --config /tmp/node2vec_lp.yaml
```

The training output directory also contains `train_history.json` and
`resolved_config.json`.

## Resume

Set `training.resume_checkpoint_dir` to either:

- The checkpoint root containing `latest_checkpoint.txt`.
- A specific `epoch_...` checkpoint directory.

The stored node count, embedding dimension, and world size must match the new
run.

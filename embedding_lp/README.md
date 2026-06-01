# Frozen-Embedding Link Prediction

`embedding_lp` trains a lightweight link-prediction classifier over frozen
node embeddings. It is method-agnostic: embeddings may come from the included
Node2Vec trainer, GRAPE Node2Vec, or another method converted to the shared
checkpoint layout.

This stage makes embedding methods comparable under the same downstream
classifier and evaluation code.

## Input

The classifier requires:

- A prepared dataset root from `scripts.prepare_dataset`.
- A checkpoint readable by
  `embedding_table_utils.ReadOnlyEmbeddingStore`.
- The correct `dataset.num_nodes`.

The prepared dataset layout is:

```text
my_dataset/
  train_csr/
  train_pairs_csr/
  valid_edge.npy
  valid_edge_neg.npy
  test_edge.npy
  test_edge_neg.npy
```

Training positives are streamed from `train_pairs_csr`. Validation and test
files must contain NumPy arrays with shape `[N, 2]`.

## Run

```bash
cp embedding_lp/configs/link_prediction_default.yaml /tmp/link_prediction.yaml
# Edit dataset.root, dataset.num_nodes, embeddings.checkpoint_dir,
# dataset.graph_csr_file_endian, and output.dir.
python -m embedding_lp.train --config /tmp/link_prediction.yaml
```

For Node2Vec, point `embeddings.checkpoint_dir` at:

```text
<node2vec-output>/checkpoints/final_embedding_shards
```

The compatibility entry point below is equivalent:

```bash
python -m node2vec.train_lp --config /tmp/link_prediction.yaml
```

## Distributed evaluation

Use `torchrun` when the embedding checkpoint was produced by distributed
TorchRec training:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m embedding_lp.train --config /tmp/link_prediction.yaml
```

Per-rank embedding shards must be loaded with the same world size used to
write them.

## Configuration

The classifier supports these edge operators:

| Operator | Edge feature |
| --- | --- |
| `hadamard` | Element-wise product |
| `average` | Mean of source and destination embeddings |
| `concat` | Concatenated embeddings |
| `weighted_l1` | Absolute difference |
| `weighted_l2` | Squared difference |

`training.negative_edge_strategy` selects training negatives:

| Strategy | Behavior |
| --- | --- |
| `uniform_nonedge` | Uniform random non-edge sampling |
| `two_hop` | Two-hop sampling aligned with the prepared evaluation negatives |

The default template uses `two_hop`.

## Outputs

If `training.checkpoint_dir` is `null`, classifier checkpoints are stored
below `<output.dir>/checkpoints`:

```text
output/
  metrics_history.json
  final_metrics.json
  resolved_config.json
  checkpoints/
    latest.pt
    best.pt
    epoch_....pt
```

Set `evaluation.save_predictions: true` to write prediction archives for the
evaluated splits.

## Baseline embeddings

External embedding matrices can be converted before training:

```bash
python -m baselines.convert_embeddings_to_embedding_table \
  --npy /path/to/embeddings.npy \
  --out-dir /data/lpp-runs/my_dataset/baseline_embedding_table \
  --backend vanilla
```

See [`../baselines/README.md`](../baselines/README.md) for the baseline
workflows.

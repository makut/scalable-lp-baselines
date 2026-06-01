# Shared negatives

This repository includes an optional Node2Vec negative-sampling mode designed
for sharded embedding tables. Enable it with:

```yaml
training:
  enable_shared_negatives: true
  num_negative_samples: 1024
```

## Motivation

The conventional path samples negative windows independently for each
positive window. This matches the standard PyG-style formulation, but it can
generate a large volume of embedding lookups. For a TorchRec table, those
lookups may become distributed all-to-all traffic.

The shared-negatives path samples one uniform pool of node ids per batch and
scores every positive anchor against that pool with a matrix multiplication.
This substantially reduces lookup traffic when embeddings are sharded.

## Semantics

`num_negative_samples` has a different meaning in the two modes:

| Mode | Meaning |
| --- | --- |
| `enable_shared_negatives: false` | Number of negative windows per positive window |
| `enable_shared_negatives: true` | Size of the single negative-id pool for the batch |

Both modes sample node ids uniformly from `[0, num_nodes)`. They do not filter
false negatives. Shared negatives introduce correlation between examples, so
the pool size is an experimental hyperparameter rather than a drop-in
equivalent of the conventional count.

## Implementation outline

The sampler places either conventional negative random walks or one
`shared_neg_ids` array into a batch. The loss then performs one embedding-table
lookup for the combined ids and uses matrix multiplication for shared
negative scores.

This optimization targets the large-graph distributed setting. For a
single-device vanilla embedding table, start with the conventional mode.

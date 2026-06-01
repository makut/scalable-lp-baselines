# GRAPE Node2Vec baseline

This folder adapts the GRAPE/ensmallen Node2Vec workflow to LPP `GraphCSR`.

## 1. Export LPP train graph to GRAPE TSV

Use the `train_csr` produced by `scripts.prepare_dataset`. It should contain
only training edges, so validation/test positives do not leak into Node2Vec.

```bash
python -m baselines.grape.export_graphcsr_to_grape \
  --graph-dir /data/lpp/my_dataset/train_csr \
  --out-dir /data/lpp_baselines/grape/my_dataset \
  --file-endian big \
  --graph-kind directed
```

Outputs:

- `edges.tsv`: tab-separated `src dst` edge list.
- `metadata.json`: exact `Graph.from_csv` loading hints used by the trainer.

For the default LPP undirected symmetric `train_csr`, `--graph-kind directed`
keeps every stored direction and asks GRAPE to build an undirected graph unless
`--directed` is passed.

## 2. Train Node2Vec with GRAPE

Install GRAPE in the environment where you will train, then run:

```bash
python -m baselines.grape.train_node2vec \
  --metadata /data/lpp_baselines/grape/my_dataset/metadata.json \
  --out-emb /data/lpp_baselines/grape/my_dataset/node2vec.npy \
  --out-embedding-checkpoint /data/lpp_baselines/grape/my_dataset/node2vec_embedding_table \
  --embedding-checkpoint-backend vanilla \
  --embedding-size 128 \
  --epochs 5 \
  --iterations 10 \
  --walk-length 80 \
  --window-size 10 \
  --negative-samples 5 \
  --p 1.0 \
  --q 1.0
```

`--p` and `--q` use the standard Node2Vec notation. The trainer converts them
to GRAPE's `return_weight = 1 / p` and `explore_weight = 1 / q`.

`--out-embedding-checkpoint` is optional. When set, `--embedding-checkpoint-backend`
selects the `embedding_table_utils` layout: `vanilla` for `nn.Embedding` or
`torchrec` for TorchRec-style loading.

For a quick load smoke test:

```bash
python -m baselines.grape.train_node2vec \
  --metadata /data/lpp_baselines/grape/my_dataset/metadata.json \
  --out-emb /tmp/unused.npy \
  --load-only
```

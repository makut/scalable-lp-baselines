# External baselines

Adapters for running external baseline implementations on LPP datasets:

- `grape/`: export `GraphCSR` to GRAPE TSV and train GRAPE Node2Vec.
- `seal/`: export the prepared LPP dataset to SEAL_OGB `data.pt` and `split_edge.pt`.

Embedding outputs can be converted to the shared `embedding_table_utils`
checkpoint layout with:

```bash
python -m baselines.convert_embeddings_to_embedding_table \
  --npy /path/to/embeddings.npy \
  --out-dir /path/to/embedding_table \
  --backend vanilla
```

`--backend` selects either `vanilla` (`nn.Embedding`) or `torchrec`
(TorchRec-style). The same command also accepts `--raw-bin` for raw float32
embedding matrices and `--torch-checkpoint --tensor-key node_embedding.weight` for SEAL_OGB-style
node embeddings.

For all embedding baselines, use `train_csr` from `scripts.prepare_dataset` as
the input graph so held-out validation/test edges are not visible during
embedding training.

## Train the shared LP classifier

The downstream LP trainer is embedding-method agnostic. Point it at any
`embedding_table_utils` checkpoint produced above:

```bash
cp embedding_lp/configs/link_prediction_default.yaml /tmp/grape_lp.yaml
# Set dataset.root, dataset.num_nodes, embeddings.checkpoint_dir, and output.dir.
python -m embedding_lp.train --config /tmp/grape_lp.yaml
```

For the simplest one-GPU workflow, write a `vanilla` checkpoint and launch LP
with plain `python`. For a distributed TorchRec workflow, convert with
`--backend torchrec` under `torchrun`, choose `--sharding-type row_wise` or
`column_wise`, and train LP with the same number of ranks. The LP trainer reads
the saved topology from the checkpoint manifest.

# SEAL_OGB baseline

This folder exports an LPP prepared dataset to the PyG objects used by
`facebookresearch/SEAL_OGB`.

## 1. Export from LPP format

Input is the output of `scripts.prepare_dataset`:

```text
<dataset-root>/
  train_csr/
  train_pairs_csr/
  valid_edge.npy
  valid_edge_neg.npy
  test_edge.npy
  test_edge_neg.npy
```

Run:

```bash
python -m baselines.seal.export_lpp_to_seal_ogb \
  --dataset-root /data/lpp/my_dataset \
  --out-dir /data/lpp_baselines/seal/my_dataset \
  --file-endian big
```

Outputs:

- `data.pt`: `torch_geometric.data.Data` with the training graph in
  `data.edge_index`.
- `split_edge.pt`: OGB-style split dictionary with `train`, `valid`, and
  `test` edges.
- `arrays/*.npy`: disk-backed copies used to build the `.pt` files.
- `metadata.json`: counts and loader metadata.

## 2. Use it from the SEAL_OGB repository

Clone and install SEAL_OGB as usual, then make the exported loader visible:

```bash
cp /path/to/lpp/baselines/seal/seal_ogb_lpp_loader.py /path/to/SEAL_OGB/
```

Patch `seal_link_pred.py` in the SEAL_OGB repo:

```python
parser.add_argument('--lpp_data_dir', type=str, default=None)
```

Then replace the dataset loading block with:

```python
if args.lpp_data_dir is not None:
    from seal_ogb_lpp_loader import load_lpp_seal_ogb_export
    data, split_edge, directed, args.eval_metric, _ = load_lpp_seal_ogb_export(args.lpp_data_dir)
elif args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
else:
    path = osp.join('dataset', args.dataset)
    dataset = Planetoid(path, args.dataset)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
```

Also keep `evaluator = Evaluator(name=args.dataset)` only for real OGB
datasets. For this export, the default `--eval-metric auc` path does not need
an OGB evaluator.

Example run from the SEAL_OGB checkout:

```bash
python seal_link_pred.py \
  --lpp_data_dir /data/lpp_baselines/seal/my_dataset \
  --dataset lpp \
  --model DGCNN \
  --num_hops 1 \
  --dynamic_train \
  --dynamic_val \
  --dynamic_test \
  --num_workers 16 \
  --epochs 50
```

## 3. Convert trainable node embeddings

If the SEAL/SEAL_OGB run trains and saves a node embedding tensor, convert it to
the shared `embedding_table_utils` format:

```bash
python -m baselines.seal.node_embedding_to_embedding_table \
  --checkpoint /data/seal_runs/run1/best_model.pth \
  --out-dir /data/seal_runs/run1/node_embedding_table \
  --backend vanilla
```

Use `--backend torchrec` for a TorchRec-style checkpoint. The converter
auto-detects `node_embedding.weight`; pass `--tensor-key ...` if your patched
SEAL_OGB checkpoint uses another key.

# Node2Vec and SEAL Experiment Configurations

This document records the parameters used for the Node2Vec and SEAL
experiments. Parameters are grouped by purpose; `null` means that the setting
was not explicitly configured.

## Node2Vec

Node2Vec used random walks of length `10` with a context window of `5`. Five
walks were generated for each node. The `p = 1.0` and `q = 1.0` parameters
correspond to unbiased graph traversal.

### Training-example generation

| Parameter | Value | Description |
| --- | ---: | --- |
| `batch_size` | `8192` | Batch size for starting nodes |
| `walk_length` | `10` | Random-walk length |
| `context_size` | `5` | Context-window size |
| `walks_per_node` | `5` | Number of walks per node |
| `p` | `1.0` | Node2Vec return parameter |
| `q` | `1.0` | Node2Vec in-out parameter |

### Negative samples and loss

| Parameter | Value | Description |
| --- | ---: | --- |
| `num_negative_samples` | `2048` | Size of the shared negative-node pool |
| `enable_shared_negatives` | `true` | Use one negative pool for all positive anchors in a batch |
| `use_nce_bias` | `false` | Do not apply the NCE bias |
| `loss_reduction` | `mean` | Average the loss values |

When `enable_shared_negatives` is enabled, `num_negative_samples` is the size
of one pool for the whole batch, rather than the number of separate negative
windows for each positive example.

### Training and data loading

| Parameter | Value | Description |
| --- | ---: | --- |
| `backend` | `nccl` | Distributed-training backend |
| `num_epochs` | `100` | Number of epochs |
| `seed` | `42` | Training seed |
| `device` | `null` | Device is not explicitly configured |
| `num_sampler_workers` | `16` | Number of sampler workers |
| `pin_memory` | `false` | Disable pinned CPU memory |
| `drop_last` | `true` | Drop an incomplete final batch |
| `log_every` | `100` | Logging interval in steps |
| `show_progress` | `true` | Display training progress |
| `is_directed` | `false` | Treat the graph as undirected |

### Checkpoints

| Parameter | Value | Description |
| --- | ---: | --- |
| `checkpoint_metric` | `train_loss` | Metric used to select the best checkpoint |
| `checkpoint_metric_mode` | `min` | Lower metric values are better |
| `save_best_checkpoint` | `true` | Save the best checkpoint |
| `checkpoint_every_steps` | `null` | Periodic step-based checkpointing is not configured |
| `checkpoint_every_epoch` | `false` | Do not save after each epoch |
| `checkpoint_dir` | `null` | Checkpoint directory is not explicitly configured |
| `resume_checkpoint_dir` | `null` | Resume checkpoint is not configured |
| `tensorboard_log_dir` | `null` | TensorBoard directory is not explicitly configured |

### Validation

| Parameter | Value | Description |
| --- | ---: | --- |
| `val_eval_every` | `null` | Periodic step-based validation is disabled |
| `val_batch_size` | `65536` | Validation batch size |
| `val_num_workers` | `0` | Run validation without additional workers |
| `val_metrics` | `roc_auc`, `pr_auc`, `hits@k`, `recall@k`, `ndcg@k` | Metrics |
| `val_metrics_at_k` | `10`, `50`, `100` | Values of `k` for ranking metrics |
| `val_treat_as_undirected` | `true` | Treat validation edges as undirected |
| `val_max_batches` | `null` | Validation batch limit is not configured |

## SEAL

SEAL extracted radius-`2` local subgraphs around each node pair. Nodes were
labeled with DRNL, and a three-layer DGCNN was used as the classifier.

### Training dataset

| Parameter | Value | Description |
| --- | ---: | --- |
| `train_loader.batch_size` | `1024` | Training batch size |
| `train_loader.seed` | `42` | Data-loader seed |
| `train_loader.num_workers` | `1` | Number of data-loader workers |
| `negative_sampling.name` | `uniform` | Uniform negative sampling |
| `negative_sampling.neg_per_pos` | `1` | Number of negative examples per positive edge |
| `negative_sampling.seed` | `42` | Negative-sampling seed |

### Subgraph extraction

| Parameter | Value | Description |
| --- | ---: | --- |
| `seal.num_hops` | `2` | Radius of each local subgraph |
| `seal.node_label` | `drnl` | Double-Radius Node Labeling |
| `seal.ratio_per_hop` | `1.0` | Fraction of nodes retained at each traversal step |
| `seal.max_nodes_per_hop` | `50` | Maximum number of nodes at each traversal step |
| `seal.per_vertex_oversample` | `1.5` | Oversampling factor for neighbor sampling |
| `seal.graph_csr_use_per_vertex_sampling` | `true` | Sample neighbors separately for each node |
| `seal.graph_csr_use_pairwise_subgraph` | `true` | Build induced subgraphs with pairwise node checks |

### Model

| Parameter | Value | Description |
| --- | ---: | --- |
| `model.name` | `DGCNN` | Classifier architecture |
| `model.hidden_channels` | `32` | Hidden representation size |
| `model.num_layers` | `3` | Number of layers |
| `model.sortpool_k` | `0.6` | Fraction of nodes retained by SortPooling |

### Training and checkpoint selection

| Parameter | Value | Description |
| --- | ---: | --- |
| `training.lr` | `0.0001` | Learning rate |
| `training.epochs` | `50` | Number of epochs |
| `training.eval_every_steps` | `10000` | Validation interval in steps |
| `training.log_loss_every_steps` | `50` | Loss-logging interval in steps |
| `training.checkpoint_metric` | `hits@10` | Metric used to select the best checkpoint |
| `training.checkpoint_metric_mode` | `max` | Higher metric values are better |
| `training.early_stopping_patience` | `3` | Stop after three checks without improvement |

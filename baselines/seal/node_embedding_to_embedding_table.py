from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from baselines.common import configure_logging
from baselines.embedding_table_conversion import (
    DEFAULT_CHUNK_ROWS,
    TorchTensorEmbeddingMatrixSource,
    maybe_init_distributed_for_conversion,
    save_embedding_matrix_as_checkpoint,
)


LOGGER = logging.getLogger("seal_node_embedding_to_embedding_table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a SEAL/SEAL_OGB train_node_embedding tensor to an "
            "embedding_table_utils checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--tensor-key",
        default=None,
        help="Checkpoint tensor key. Auto-detect prefers node_embedding.weight.",
    )
    parser.add_argument("--backend", required=True, choices=["torchrec", "vanilla"])
    parser.add_argument("--checkpoint-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--table-name", default="node_table")
    parser.add_argument("--feature-name", default="node")
    parser.add_argument("--sharding-type", default="row_wise", choices=["row_wise", "column_wise"])
    parser.add_argument("--compute-kernel-policy", default="auto", choices=["auto", "prefer_hbm", "allow_uvm"])
    parser.add_argument("--device", default=None, help="Conversion device. Defaults to CPU.")
    parser.add_argument("--dist-backend", default="auto")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    maybe_init_distributed_for_conversion(
        args.backend,
        device=device,
        dist_backend=str(args.dist_backend),
    )
    source = TorchTensorEmbeddingMatrixSource.from_checkpoint(
        args.checkpoint,
        tensor_key=args.tensor_key,
    )
    result = save_embedding_matrix_as_checkpoint(
        source,
        out_dir=args.out_dir,
        backend=args.backend,
        dtype=args.checkpoint_dtype,
        table_name=args.table_name,
        feature_name=args.feature_name,
        sharding_type=args.sharding_type,
        compute_kernel_policy=args.compute_kernel_policy,
        device=device,
        step=args.step,
        chunk_rows=args.chunk_rows,
        extra_metadata={"baseline": "seal_ogb"},
    )
    if result.rank == 0:
        LOGGER.info(
            "Saved SEAL node embedding checkpoint to %s backend=%s shape=(%d, %d)",
            result.out_dir,
            result.backend,
            result.num_embeddings,
            result.embedding_dim,
        )


if __name__ == "__main__":
    main()


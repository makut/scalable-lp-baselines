from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from baselines.common import configure_logging
from baselines.embedding_table_conversion import (
    DEFAULT_CHUNK_ROWS,
    NumpyEmbeddingMatrixSource,
    TorchTensorEmbeddingMatrixSource,
    maybe_init_distributed_for_conversion,
    save_embedding_matrix_as_checkpoint,
)


LOGGER = logging.getLogger("convert_embeddings_to_embedding_table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert baseline embedding outputs (.npy, raw float bin, or torch checkpoint tensor) "
            "to an embedding_table_utils checkpoint."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--npy", type=Path, help="2D NumPy embedding matrix, e.g. GRAPE output.")
    inputs.add_argument("--raw-bin", type=Path, help="Raw float32 binary embedding matrix.")
    inputs.add_argument("--torch-checkpoint", type=Path, help="Torch checkpoint containing a 2D embedding tensor.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=["torchrec", "vanilla"])
    parser.add_argument("--checkpoint-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--table-name", default="node_table")
    parser.add_argument("--feature-name", default="node")
    parser.add_argument("--sharding-type", default="row_wise", choices=["row_wise", "column_wise"])
    parser.add_argument("--compute-kernel-policy", default="auto", choices=["auto", "prefer_hbm", "allow_uvm"])
    parser.add_argument("--device", default=None, help="Conversion device. Defaults to CPU.")
    parser.add_argument("--dist-backend", default="auto", help="Distributed backend for torchrun conversions.")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--num-nodes", type=int, default=None, help="Required for --raw-bin.")
    parser.add_argument("--dim", type=int, default=None, help="Required for --raw-bin.")
    parser.add_argument("--raw-dtype", default="<f4", help="NumPy dtype for --raw-bin; default little-endian float32.")
    parser.add_argument("--tensor-key", default=None, help="Tensor key for --torch-checkpoint.")
    parser.add_argument("--extra-metadata-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build_source(args: argparse.Namespace):
    if args.npy is not None:
        return NumpyEmbeddingMatrixSource.from_npy(args.npy)
    if args.raw_bin is not None:
        if args.num_nodes is None or args.dim is None:
            raise ValueError("--raw-bin requires --num-nodes and --dim")
        return NumpyEmbeddingMatrixSource.from_raw_binary(
            args.raw_bin,
            num_nodes=int(args.num_nodes),
            dim=int(args.dim),
            dtype=str(args.raw_dtype),
        )
    if args.torch_checkpoint is not None:
        return TorchTensorEmbeddingMatrixSource.from_checkpoint(
            args.torch_checkpoint,
            tensor_key=args.tensor_key,
        )
    raise AssertionError("argparse should enforce one input source")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source = _build_source(args)
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
    extra_metadata = {}
    if args.extra_metadata_json is not None:
        extra_metadata = json.loads(args.extra_metadata_json.read_text(encoding="utf-8"))

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
        extra_metadata=extra_metadata,
    )
    if result.rank == 0:
        LOGGER.info(
            "Saved embedding_table_utils checkpoint to %s backend=%s shape=(%d, %d) dtype=%s",
            result.out_dir,
            result.backend,
            result.num_embeddings,
            result.embedding_dim,
            result.dtype,
        )


if __name__ == "__main__":
    main()

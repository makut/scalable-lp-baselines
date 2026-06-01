from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines.common import (
    GraphKind,
    configure_logging,
    format_int,
    graph_info_from_dir,
    write_edge_index_npy_from_graphcsr,
    write_edge_pairs_npy_from_graphcsr,
)


LOGGER = logging.getLogger("export_lpp_to_seal_ogb")


def _load_edges(path: Path, *, name: str, num_nodes: int, required: bool) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required {name} edge file: {path}")
        return None
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {arr.shape} at {path}")
    if arr.size:
        min_id = int(arr.min())
        max_id = int(arr.max())
        if min_id < 0 or max_id >= num_nodes:
            raise ValueError(
                f"{name} contains node ids outside [0, {num_nodes}): "
                f"min={min_id}, max={max_id}"
            )
    return arr


def _save_edge_copy(arr: np.ndarray | None, out_path: Path) -> dict[str, Any] | None:
    if arr is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int64, shape=arr.shape)
    for start in range(0, int(arr.shape[0]), 5_000_000):
        end = min(int(arr.shape[0]), start + 5_000_000)
        mm[start:end] = np.asarray(arr[start:end], dtype=np.int64)
    mm.flush()
    return {"path": str(out_path), "count": int(arr.shape[0])}


def _torch_load_compatible(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_array_as_long_tensor(path: Path):
    import torch

    arr = np.load(path, mmap_mode="r")
    return torch.from_numpy(np.asarray(arr, dtype=np.int64)).long()


def write_seal_ogb_pt_files(
    *,
    out_dir: Path,
    data_edge_index_path: Path,
    train_edge_path: Path,
    valid_edge_path: Path | None,
    valid_edge_neg_path: Path | None,
    test_edge_path: Path | None,
    test_edge_neg_path: Path | None,
    num_nodes: int,
) -> None:
    import torch
    from torch_geometric.data import Data

    edge_index = _load_array_as_long_tensor(data_edge_index_path)
    data = Data(edge_index=edge_index, num_nodes=int(num_nodes), x=None)

    split_edge: dict[str, dict[str, Any]] = {
        "train": {"edge": _load_array_as_long_tensor(train_edge_path)},
    }
    if valid_edge_path is not None:
        split_edge["valid"] = {"edge": _load_array_as_long_tensor(valid_edge_path)}
        if valid_edge_neg_path is not None:
            split_edge["valid"]["edge_neg"] = _load_array_as_long_tensor(valid_edge_neg_path)
    if test_edge_path is not None:
        split_edge["test"] = {"edge": _load_array_as_long_tensor(test_edge_path)}
        if test_edge_neg_path is not None:
            split_edge["test"]["edge_neg"] = _load_array_as_long_tensor(test_edge_neg_path)

    torch.save(data, out_dir / "data.pt")
    torch.save(split_edge, out_dir / "split_edge.pt")

    # Catch serialization surprises while we still have the exporting env.
    _torch_load_compatible(out_dir / "data.pt")
    _torch_load_compatible(out_dir / "split_edge.pt")


def export_lpp_dataset_to_seal_ogb(
    *,
    dataset_root: Path,
    out_dir: Path,
    file_endian: str,
    use_mmap: bool,
    allow_non_native: bool,
    chunk_bytes: int,
    chunk_nodes: int,
    node_dtype: str,
    data_graph_kind: GraphKind,
    train_pairs_graph_kind: GraphKind,
    drop_selfloops: bool,
    require_test: bool,
    write_pt: bool,
    eval_metric: str,
    directed: bool,
    log_every_chunks: int,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    train_graph_dir = dataset_root / "train_csr"
    train_pairs_dir = dataset_root / "train_pairs_csr"
    if not train_graph_dir.exists():
        raise FileNotFoundError(f"Missing train graph: {train_graph_dir}")

    info = graph_info_from_dir(
        train_graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    )
    dtype = np.dtype(node_dtype)
    if dtype == np.dtype("int32") and info.num_nodes > np.iinfo(np.int32).max:
        raise ValueError(f"num_nodes={info.num_nodes} does not fit int32; use --node-dtype int64")

    LOGGER.info("Writing PyG data edge_index from %s", train_graph_dir)
    data_info, data_edges, data_dropped_selfloops = write_edge_index_npy_from_graphcsr(
        train_graph_dir,
        arrays_dir / "data_edge_index.npy",
        use_mmap=use_mmap,
        file_endian=file_endian,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        chunk_nodes=chunk_nodes,
        node_dtype=dtype,
        graph_kind=data_graph_kind,
        drop_selfloops=drop_selfloops,
        log_every_chunks=log_every_chunks,
    )
    if data_info.num_nodes != info.num_nodes:
        raise RuntimeError("Graph info changed while exporting data edge_index")

    if train_pairs_dir.exists():
        train_source_dir = train_pairs_dir
        train_source_kind = train_pairs_graph_kind
    else:
        LOGGER.warning(
            "train_pairs_csr is missing at %s; falling back to train_csr with src < dst filtering",
            train_pairs_dir,
        )
        train_source_dir = train_graph_dir
        train_source_kind = "undirected-symmetric-csr"

    LOGGER.info("Writing split_edge train positives from %s", train_source_dir)
    _, train_edges, train_dropped_selfloops = write_edge_pairs_npy_from_graphcsr(
        train_source_dir,
        arrays_dir / "train_edge.npy",
        use_mmap=use_mmap,
        file_endian=file_endian,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
        chunk_nodes=chunk_nodes,
        node_dtype=dtype,
        graph_kind=train_source_kind,
        drop_selfloops=drop_selfloops,
        log_every_chunks=log_every_chunks,
    )

    valid_edge = _load_edges(
        dataset_root / "valid_edge.npy",
        name="valid_edge",
        num_nodes=info.num_nodes,
        required=True,
    )
    valid_edge_neg = _load_edges(
        dataset_root / "valid_edge_neg.npy",
        name="valid_edge_neg",
        num_nodes=info.num_nodes,
        required=False,
    )
    test_edge = _load_edges(
        dataset_root / "test_edge.npy",
        name="test_edge",
        num_nodes=info.num_nodes,
        required=require_test,
    )
    test_edge_neg = _load_edges(
        dataset_root / "test_edge_neg.npy",
        name="test_edge_neg",
        num_nodes=info.num_nodes,
        required=require_test,
    )

    split_arrays: dict[str, Any] = {
        "train": {"path": str(arrays_dir / "train_edge.npy"), "count": int(train_edges)},
    }
    valid_meta = _save_edge_copy(valid_edge, arrays_dir / "valid_edge.npy")
    if valid_meta is not None:
        split_arrays["valid"] = valid_meta
    valid_neg_meta = _save_edge_copy(valid_edge_neg, arrays_dir / "valid_edge_neg.npy")
    if valid_neg_meta is not None:
        split_arrays.setdefault("valid", {})["neg_path"] = valid_neg_meta["path"]
        split_arrays["valid"]["neg_count"] = valid_neg_meta["count"]

    test_meta = _save_edge_copy(test_edge, arrays_dir / "test_edge.npy")
    if test_meta is not None:
        split_arrays["test"] = test_meta
    test_neg_meta = _save_edge_copy(test_edge_neg, arrays_dir / "test_edge_neg.npy")
    if test_neg_meta is not None:
        split_arrays.setdefault("test", {})["neg_path"] = test_neg_meta["path"]
        split_arrays["test"]["neg_count"] = test_neg_meta["count"]

    if write_pt:
        if "valid" not in split_arrays:
            raise ValueError("Cannot write split_edge.pt without valid_edge.npy")
        if require_test and "test" not in split_arrays:
            raise ValueError("Cannot write split_edge.pt without test_edge.npy")
        LOGGER.info("Materializing data.pt and split_edge.pt")
        write_seal_ogb_pt_files(
            out_dir=out_dir,
            data_edge_index_path=arrays_dir / "data_edge_index.npy",
            train_edge_path=arrays_dir / "train_edge.npy",
            valid_edge_path=arrays_dir / "valid_edge.npy",
            valid_edge_neg_path=(arrays_dir / "valid_edge_neg.npy") if valid_edge_neg is not None else None,
            test_edge_path=(arrays_dir / "test_edge.npy") if test_edge is not None else None,
            test_edge_neg_path=(arrays_dir / "test_edge_neg.npy") if test_edge_neg is not None else None,
            num_nodes=info.num_nodes,
        )

    meta = {
        "format": "seal_ogb_lpp_export",
        "dataset_root": str(dataset_root),
        "num_nodes": int(info.num_nodes),
        "train_csr_raw_edges": int(info.raw_edges),
        "data_edge_index_edges": int(data_edges),
        "train_positive_edges": int(train_edges),
        "valid_positive_edges": int(valid_edge.shape[0]) if valid_edge is not None else 0,
        "valid_negative_edges": int(valid_edge_neg.shape[0]) if valid_edge_neg is not None else 0,
        "test_positive_edges": int(test_edge.shape[0]) if test_edge is not None else 0,
        "test_negative_edges": int(test_edge_neg.shape[0]) if test_edge_neg is not None else 0,
        "directed": bool(directed),
        "eval_metric": eval_metric,
        "data_graph_kind": data_graph_kind,
        "train_pairs_graph_kind": train_source_kind,
        "drop_selfloops": bool(drop_selfloops),
        "dropped_selfloops": {
            "data_edge_index": int(data_dropped_selfloops),
            "train_positive_edges": int(train_dropped_selfloops),
        },
        "array_outputs": {
            "data": {
                "edge_index_path": str(arrays_dir / "data_edge_index.npy"),
                "count": int(data_edges),
            },
            "split_edge": split_arrays,
        },
        "pt_outputs": {
            "data_pt": str(out_dir / "data.pt") if write_pt else None,
            "split_edge_pt": str(out_dir / "split_edge.pt") if write_pt else None,
        },
        "elapsed_seconds": time.time() - t0,
        "notes": [
            "data.pt contains the training graph only.",
            "split_edge.pt train positives come from train_pairs_csr when available.",
            "valid/test edges are copied from the LPP dataset root.",
            "SEAL_OGB will sample train negatives itself unless you add edge_neg to split_edge['train'].",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info(
        "Done: num_nodes=%s data_edges=%s train_pos=%s valid_pos=%s test_pos=%s",
        format_int(info.num_nodes),
        format_int(data_edges),
        format_int(train_edges),
        format_int(meta["valid_positive_edges"]),
        format_int(meta["test_positive_edges"]),
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an LPP prepared dataset into data.pt/split_edge.pt for SEAL_OGB.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--file-endian", default="big", choices=["big", "little"])
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--disallow-non-native", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--chunk-nodes", type=int, default=1_000_000)
    parser.add_argument("--node-dtype", default="int32", choices=["int32", "int64"])
    parser.add_argument(
        "--data-graph-kind",
        default="directed",
        choices=["undirected-symmetric-csr", "directed", "undirected-single-edge-list"],
        help="Filtering for data.edge_index. Default keeps all stored train_csr directions.",
    )
    parser.add_argument(
        "--train-pairs-graph-kind",
        default="directed",
        choices=["undirected-symmetric-csr", "directed", "undirected-single-edge-list"],
        help="Filtering for train_pairs_csr. Default keeps all stored pairs.",
    )
    parser.add_argument("--keep-selfloops", action="store_true")
    parser.add_argument("--allow-missing-test", action="store_true")
    parser.add_argument("--skip-pt", action="store_true", help="Write only arrays/metadata, not data.pt/split_edge.pt.")
    parser.add_argument("--directed", action="store_true", help="Mark the dataset as directed in metadata.")
    parser.add_argument("--eval-metric", default="auc", choices=["auc", "hits", "mrr", "rocauc"])
    parser.add_argument("--log-every-chunks", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    export_lpp_dataset_to_seal_ogb(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        file_endian=args.file_endian,
        use_mmap=not args.no_mmap,
        allow_non_native=not args.disallow_non_native,
        chunk_bytes=args.chunk_bytes,
        chunk_nodes=args.chunk_nodes,
        node_dtype=args.node_dtype,
        data_graph_kind=args.data_graph_kind,
        train_pairs_graph_kind=args.train_pairs_graph_kind,
        drop_selfloops=not args.keep_selfloops,
        require_test=not args.allow_missing_test,
        write_pt=not args.skip_pt,
        eval_metric=args.eval_metric,
        directed=args.directed,
        log_every_chunks=args.log_every_chunks,
    )


if __name__ == "__main__":
    main()


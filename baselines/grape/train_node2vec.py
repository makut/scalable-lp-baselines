from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines.common import configure_logging
from baselines.embedding_table_conversion import (
    DEFAULT_CHUNK_ROWS,
    NumpyEmbeddingMatrixSource,
    save_embedding_matrix_as_checkpoint,
)


LOGGER = logging.getLogger("train_grape_node2vec")


def _filter_kwargs_for_signature(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _count_lines(path: Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> int:
    lines = 0
    last_byte = b""
    with path.open("rb") as f:
        while chunk := f.read(chunk_bytes):
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return lines


def _restore_node_count_for_sorted_list(meta: dict[str, Any], kwargs: dict[str, Any]) -> None:
    declared_counts = {
        int(value)
        for value in (
            kwargs.get("number_of_nodes"),
            meta.get("num_nodes"),
        )
        if value is not None
    }
    if len(declared_counts) > 1:
        raise ValueError(f"Metadata contains conflicting node counts: {sorted(declared_counts)}")

    if declared_counts:
        kwargs.setdefault("number_of_nodes", declared_counts.pop())
    elif kwargs.get("edge_list_is_sorted"):
        raise ValueError(
            "Metadata declares a sorted edge list but does not contain number_of_nodes. "
            "Add grape_from_csv_kwargs.number_of_nodes using the exact node count from "
            "the original GraphCSR. This cannot be safely inferred from edges.tsv because "
            "isolated nodes may not occur in the edge list."
        )


def _restore_edge_count_for_sorted_list(meta: dict[str, Any], kwargs: dict[str, Any], edge_path: Path) -> None:
    declared_counts = {
        int(value)
        for value in (
            kwargs.get("number_of_edges"),
            kwargs.get("edges_number"),
            meta.get("written_edges"),
        )
        if value is not None
    }
    if len(declared_counts) > 1:
        raise ValueError(f"Metadata contains conflicting edge counts: {sorted(declared_counts)}")

    if declared_counts:
        edge_count = declared_counts.pop()
    elif kwargs.get("edge_list_is_sorted"):
        LOGGER.warning(
            "Metadata declares a sorted edge list but does not contain an edge count. "
            "Counting TSV lines once; add number_of_edges and edges_number to metadata.json "
            "to skip this recovery scan on future runs."
        )
        edge_count = _count_lines(edge_path)
        if kwargs.get("edge_list_header", kwargs.get("header", True)):
            edge_count -= 1
        if edge_count < 0:
            raise ValueError(f"Edge TSV {edge_path} does not contain the declared header")
    else:
        return

    kwargs.setdefault("number_of_edges", edge_count)
    kwargs.setdefault("edges_number", edge_count)


def load_graph_from_metadata(metadata_path: Path):
    from grape import Graph

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "grape_from_csv_kwargs" not in meta:
        raise ValueError(f"{metadata_path} does not contain grape_from_csv_kwargs")

    kwargs = dict(meta["grape_from_csv_kwargs"])
    edge_path = Path(kwargs["edge_path"])
    if not edge_path.exists():
        candidate = metadata_path.parent / edge_path.name
        if candidate.exists():
            kwargs["edge_path"] = str(candidate)
            edge_path = candidate
    _restore_node_count_for_sorted_list(meta, kwargs)
    _restore_edge_count_for_sorted_list(meta, kwargs, edge_path)

    attempts: list[tuple[str, dict[str, Any]]] = [("preferred kwargs", dict(kwargs))]
    alias_kwargs = dict(kwargs)
    if "edge_list_separator" in alias_kwargs:
        alias_kwargs["separator"] = alias_kwargs.pop("edge_list_separator")
    if "edge_list_header" in alias_kwargs:
        alias_kwargs["header"] = alias_kwargs.pop("edge_list_header")
    attempts.append(("separator/header aliases", alias_kwargs))
    attempts.append(
        (
            "minimal fallback",
            {
                "directed": kwargs["directed"],
                "edge_path": kwargs["edge_path"],
                "sources_column": kwargs["sources_column"],
                "destinations_column": kwargs["destinations_column"],
                "edge_list_separator": "\t",
                "edge_list_header": kwargs.get("edge_list_header", True),
            },
        )
    )

    errors: list[str] = []
    for label, candidate in attempts:
        try:
            candidate = _filter_kwargs_for_signature(Graph.from_csv, candidate)
            LOGGER.info("Trying Graph.from_csv variant: %s", label)
            graph = Graph.from_csv(**candidate)
            return graph, meta, label
        except TypeError as exc:
            errors.append(f"{label}: {exc}")

    raise TypeError("Could not call Graph.from_csv with known signatures:\n" + "\n".join(errors))


def extract_embedding_matrix(graph, embedding_result, *, combine: str) -> np.ndarray:
    if hasattr(embedding_result, "get_all_node_embedding"):
        embeddings = embedding_result.get_all_node_embedding()
    elif hasattr(embedding_result, "get_node_embedding"):
        embeddings = embedding_result.get_node_embedding()
    else:
        embeddings = embedding_result

    if not isinstance(embeddings, (list, tuple)):
        internal = np.asarray(embeddings, dtype=np.float32)
    else:
        matrices = [np.asarray(matrix, dtype=np.float32) for matrix in embeddings]
        if combine == "first":
            internal = matrices[0]
        elif combine == "mean":
            shape = matrices[0].shape
            if any(matrix.shape != shape for matrix in matrices):
                raise ValueError("Cannot combine by mean: embedding matrices have different shapes")
            internal = np.mean(matrices, axis=0, dtype=np.float32)
        elif combine == "concat":
            rows = matrices[0].shape[0]
            if any(matrix.shape[0] != rows for matrix in matrices):
                raise ValueError("Cannot combine by concat: embedding matrices have different row counts")
            internal = np.concatenate(matrices, axis=1)
        else:
            raise ValueError(f"Unknown combine mode: {combine}")

    if hasattr(graph, "get_node_names"):
        try:
            original_ids = np.asarray(graph.get_node_names()).astype(np.int64)
            if original_ids.shape[0] == internal.shape[0]:
                if original_ids.min(initial=0) >= 0 and original_ids.max(initial=-1) < internal.shape[0]:
                    out = np.empty_like(internal)
                    out[original_ids] = internal
                    return out
        except Exception as exc:
            LOGGER.warning("Could not remap embeddings by graph.get_node_names(): %s", exc)

    return internal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRAPE Node2Vec on an exported LPP TSV edge list.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-emb", required=True, type=Path)
    parser.add_argument("--out-meta", default=None, type=Path)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--walk-length", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grape-dtype", default="f32")
    parser.add_argument("--combine-embeddings", default="first", choices=["first", "mean", "concat"])
    parser.add_argument(
        "--out-embedding-checkpoint",
        default=None,
        type=Path,
        help="Optional embedding_table_utils checkpoint directory to write from the trained embeddings.",
    )
    parser.add_argument(
        "--embedding-checkpoint-backend",
        default="vanilla",
        choices=["torchrec", "vanilla"],
        help="Backend style for --out-embedding-checkpoint.",
    )
    parser.add_argument("--embedding-checkpoint-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--embedding-checkpoint-device", default=None, help="Conversion device. Defaults to CPU.")
    parser.add_argument("--embedding-checkpoint-step", type=int, default=None)
    parser.add_argument("--embedding-checkpoint-chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    graph, input_meta, load_variant = load_graph_from_metadata(args.metadata)
    if hasattr(graph, "get_number_of_nodes"):
        LOGGER.info("GRAPE nodes: %s", graph.get_number_of_nodes())
    if hasattr(graph, "get_number_of_edges"):
        LOGGER.info("GRAPE edges: %s", graph.get_number_of_edges())

    if args.load_only:
        return

    from grape.embedders import Node2VecSkipGramEnsmallen

    if args.p == 0.0 or args.q == 0.0:
        raise ValueError("--p and --q must be non-zero because GRAPE uses 1/p and 1/q weights")

    model = Node2VecSkipGramEnsmallen(
        embedding_size=args.embedding_size,
        epochs=args.epochs,
        walk_length=args.walk_length,
        iterations=args.iterations,
        window_size=args.window_size,
        number_of_negative_samples=args.negative_samples,
        return_weight=1.0 / args.p,
        explore_weight=1.0 / args.q,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        dtype=args.grape_dtype,
        verbose=True,
    )

    LOGGER.info("Training Node2VecSkipGramEnsmallen")
    t0 = time.time()
    embedding_result = model.fit_transform(graph)
    elapsed = time.time() - t0
    x = extract_embedding_matrix(graph, embedding_result, combine=args.combine_embeddings)

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, x)

    output_meta = {
        "input_metadata": str(args.metadata),
        "load_variant": load_variant,
        "embedding": {
            "path": str(args.out_emb),
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "combine_embeddings": args.combine_embeddings,
        },
        "node2vec": {
            "embedding_size": args.embedding_size,
            "epochs": args.epochs,
            "walk_length": args.walk_length,
            "iterations": args.iterations,
            "window_size": args.window_size,
            "negative_samples": args.negative_samples,
            "p": args.p,
            "q": args.q,
            "return_weight": 1.0 / args.p,
            "explore_weight": 1.0 / args.q,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
            "dtype": args.grape_dtype,
            "train_seconds": elapsed,
        },
        "source_graph_metadata": input_meta,
    }
    if args.out_embedding_checkpoint is not None:
        checkpoint = save_embedding_matrix_as_checkpoint(
            NumpyEmbeddingMatrixSource(x, path=args.out_emb, source_type="grape_npy"),
            out_dir=args.out_embedding_checkpoint,
            backend=args.embedding_checkpoint_backend,
            dtype=args.embedding_checkpoint_dtype,
            device=args.embedding_checkpoint_device,
            step=args.embedding_checkpoint_step,
            chunk_rows=args.embedding_checkpoint_chunk_rows,
            extra_metadata={
                "baseline": "grape_node2vec",
                "embedding_metadata": output_meta["embedding"],
                "node2vec": output_meta["node2vec"],
            },
        )
        output_meta["embedding_table_checkpoint"] = {
            "path": checkpoint.out_dir,
            "backend": checkpoint.backend,
            "dtype": checkpoint.dtype,
            "shape": [checkpoint.num_embeddings, checkpoint.embedding_dim],
        }
    out_meta = args.out_meta or args.out_emb.with_suffix(args.out_emb.suffix + ".metadata.json")
    out_meta.write_text(json.dumps(output_meta, indent=2), encoding="utf-8")
    LOGGER.info("Saved embeddings: %s shape=%s dtype=%s", args.out_emb, x.shape, x.dtype)
    LOGGER.info("Saved metadata: %s", out_meta)


if __name__ == "__main__":
    main()

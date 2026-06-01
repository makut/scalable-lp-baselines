from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines.common import (
    GraphKind,
    configure_logging,
    edge_mask,
    format_int,
    iter_csr_edge_chunks,
    validate_node_id_bounds,
)
from graph_csr.serializer import GraphCSRSerializer


LOGGER = logging.getLogger("export_graphcsr_to_grape")


def _filter_kwargs_for_signature(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _validate_sorted_and_duplicates(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    validate_sorted: bool,
    validate_no_duplicates: bool,
) -> None:
    if dst.size <= 1:
        return

    same_src = src[1:] == src[:-1]
    if validate_sorted:
        descending = same_src & (dst[1:] < dst[:-1])
        if bool(descending.any()):
            idx = int(np.flatnonzero(descending)[0])
            raise ValueError(
                "CSR neighbours are not sorted, so the generated TSV cannot be "
                "declared edge_list_is_sorted=True. First offending row: "
                f"src={int(src[idx])}, dst_prev={int(dst[idx])}, dst_next={int(dst[idx + 1])}."
            )

    if validate_no_duplicates:
        duplicated = same_src & (dst[1:] == dst[:-1])
        if bool(duplicated.any()):
            idx = int(np.flatnonzero(duplicated)[0])
            raise ValueError(
                "Duplicate neighbor found inside a CSR row. The generated TSV cannot "
                "safely be declared edge_list_may_contain_duplicates=False. "
                f"First duplicate: src={int(src[idx])}, dst={int(dst[idx])}."
            )


def write_grape_edges_tsv(
    *,
    graph_dir: Path,
    out_dir: Path,
    file_endian: str,
    use_mmap: bool,
    allow_non_native: bool,
    chunk_bytes: int,
    chunk_nodes: int,
    graph_kind: GraphKind,
    directed: bool,
    drop_selfloops: bool,
    node_dtype: str,
    validate_node_ids: bool,
    validate_sorted_neighbors: bool,
    validate_no_duplicates: bool,
    header: bool,
    log_every_chunks: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_path = out_dir / "edges.tsv"
    meta_path = out_dir / "metadata.json"

    dtype = np.dtype(node_dtype)
    if dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError("node_dtype must be either int32 or int64")

    kept_edges = 0
    dropped_selfloops = 0
    min_dst: int | None = None
    max_dst: int | None = None
    t0 = time.time()

    with GraphCSRSerializer.deserialize(
        graph_dir,
        use_mmap=use_mmap,
        file_endian=file_endian,
        writable=False,
        allow_non_native=allow_non_native,
        chunk_bytes=chunk_bytes,
    ) as graph:
        starts = graph.edge_starts.numpy()
        num_nodes = int(starts.size)
        raw_edges = int(graph.edge_ends.size)

        if dtype == np.dtype("int32") and num_nodes > np.iinfo(np.int32).max:
            raise ValueError(f"num_nodes={num_nodes} does not fit int32; use --node-dtype int64")

        with edge_path.open("w", encoding="utf-8", buffering=64 * 1024 * 1024) as f:
            if header:
                f.write("src\tdst\n")

            for chunk_id, chunk in enumerate(
                iter_csr_edge_chunks(graph, chunk_nodes=chunk_nodes, node_dtype=dtype)
            ):
                if chunk.dst.size:
                    dmin = int(chunk.dst.min())
                    dmax = int(chunk.dst.max())
                    min_dst = dmin if min_dst is None else min(min_dst, dmin)
                    max_dst = dmax if max_dst is None else max(max_dst, dmax)

                if drop_selfloops:
                    dropped_selfloops += int((chunk.src == chunk.dst).sum())

                mask = edge_mask(
                    chunk.src,
                    chunk.dst,
                    graph_kind=graph_kind,
                    drop_selfloops=drop_selfloops,
                )
                src = chunk.src[mask]
                dst = chunk.dst[mask]
                _validate_sorted_and_duplicates(
                    src,
                    dst,
                    validate_sorted=validate_sorted_neighbors,
                    validate_no_duplicates=validate_no_duplicates,
                )

                if dst.size:
                    pairs = np.empty((dst.size, 2), dtype=dtype)
                    pairs[:, 0] = src
                    pairs[:, 1] = dst
                    np.savetxt(f, pairs, fmt="%d\t%d")
                    kept_edges += int(dst.size)

                if log_every_chunks > 0 and chunk_id % log_every_chunks == 0:
                    LOGGER.info(
                        "Processed rows [%s, %s) / %s; written_edges=%s",
                        format_int(chunk.node_from),
                        format_int(chunk.node_to),
                        format_int(num_nodes),
                        format_int(kept_edges),
                    )

    if validate_node_ids and raw_edges > 0:
        validate_node_id_bounds(min_dst, max_dst, num_nodes)

    stat = edge_path.stat()
    from_csv_kwargs = {
        "directed": bool(directed),
        "edge_path": str(edge_path),
        "sources_column": "src",
        "destinations_column": "dst",
        "edge_list_separator": "\t",
        "edge_list_header": bool(header),
        "edge_list_numeric_node_ids": True,
        "numeric_node_ids": True,
        "number_of_nodes": int(num_nodes),
        "number_of_edges": int(kept_edges),
        "edges_number": int(kept_edges),
        "edge_list_is_complete": bool(directed or graph_kind == "directed"),
        "edge_list_is_sorted": bool(validate_sorted_neighbors),
        "edge_list_is_correct": True,
        "edge_list_may_contain_duplicates": not bool(validate_no_duplicates),
        "may_have_singletons": True,
        "may_have_singleton_with_selfloops": not bool(drop_selfloops),
        "name": "lpp_graphcsr",
        "verbose": True,
    }

    meta = {
        "format": "grape_from_csv_tsv",
        "graph_dir": str(graph_dir),
        "edge_path": str(edge_path),
        "num_nodes": int(num_nodes),
        "raw_edges": int(raw_edges),
        "written_edges": int(kept_edges),
        "dropped_selfloops": int(dropped_selfloops),
        "graph_kind": graph_kind,
        "directed": bool(directed),
        "drop_selfloops": bool(drop_selfloops),
        "node_dtype": str(dtype),
        "header": bool(header),
        "validate_sorted_neighbors": bool(validate_sorted_neighbors),
        "validate_no_duplicates": bool(validate_no_duplicates),
        "edge_file_size_bytes": int(stat.st_size),
        "edge_file_size_gib": stat.st_size / (1024 ** 3),
        "elapsed_seconds": time.time() - t0,
        "grape_from_csv_kwargs": from_csv_kwargs,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", edge_path)
    LOGGER.info("Wrote %s", meta_path)
    return meta


def load_graph_from_metadata(metadata_path: Path):
    from grape import Graph

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    kwargs = dict(meta["grape_from_csv_kwargs"])
    edge_path = Path(kwargs["edge_path"])
    if not edge_path.exists():
        candidate = metadata_path.parent / edge_path.name
        if candidate.exists():
            kwargs["edge_path"] = str(candidate)

    attempts: list[tuple[str, dict[str, Any]]] = [
        ("preferred kwargs", dict(kwargs)),
    ]

    alias_kwargs = dict(kwargs)
    if "edge_list_separator" in alias_kwargs:
        alias_kwargs["separator"] = alias_kwargs.pop("edge_list_separator")
    if "edge_list_header" in alias_kwargs:
        alias_kwargs["header"] = alias_kwargs.pop("edge_list_header")
    attempts.append(("separator/header aliases", alias_kwargs))

    minimal = {
        "directed": kwargs["directed"],
        "edge_path": kwargs["edge_path"],
        "sources_column": kwargs["sources_column"],
        "destinations_column": kwargs["destinations_column"],
        "edge_list_separator": "\t",
        "edge_list_header": kwargs.get("edge_list_header", True),
    }
    attempts.append(("minimal fallback", minimal))

    errors: list[str] = []
    for label, candidate in attempts:
        try:
            candidate = _filter_kwargs_for_signature(Graph.from_csv, candidate)
            LOGGER.info("Trying Graph.from_csv variant: %s", label)
            return Graph.from_csv(**candidate), meta, label
        except TypeError as exc:
            errors.append(f"{label}: {exc}")

    raise TypeError("Could not call Graph.from_csv with known signatures:\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an LPP GraphCSR folder to a GRAPE/ensmallen TSV edge list.",
    )
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--file-endian", default="big", choices=["big", "little"])
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument(
        "--disallow-non-native",
        action="store_true",
        help="Fail instead of mmap-reading non-native endian CSR files.",
    )
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--chunk-nodes", type=int, default=1_000_000)
    parser.add_argument(
        "--graph-kind",
        default="directed",
        choices=["undirected-symmetric-csr", "directed", "undirected-single-edge-list"],
        help=(
            "How to filter stored CSR edges. Use 'directed' for LPP train_csr "
            "when you want all stored directions in the GRAPE file. Use "
            "'undirected-symmetric-csr' to keep only src < dst."
        ),
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Pass directed=True to GRAPE. The default is an undirected GRAPE graph.",
    )
    parser.add_argument("--keep-selfloops", action="store_true")
    parser.add_argument("--node-dtype", default="int32", choices=["int32", "int64"])
    parser.add_argument("--skip-node-id-validation", action="store_true")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--no-validate-sorted-neighbors", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--load-test", action="store_true", help="Load the TSV back with GRAPE after exporting.")
    parser.add_argument("--log-every-chunks", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    meta = write_grape_edges_tsv(
        graph_dir=args.graph_dir,
        out_dir=args.out_dir,
        file_endian=args.file_endian,
        use_mmap=not args.no_mmap,
        allow_non_native=not args.disallow_non_native,
        chunk_bytes=args.chunk_bytes,
        chunk_nodes=args.chunk_nodes,
        graph_kind=args.graph_kind,
        directed=args.directed,
        drop_selfloops=not args.keep_selfloops,
        node_dtype=args.node_dtype,
        validate_node_ids=not args.skip_node_id_validation,
        validate_sorted_neighbors=not args.no_validate_sorted_neighbors,
        validate_no_duplicates=not args.allow_duplicates,
        header=not args.no_header,
        log_every_chunks=args.log_every_chunks,
    )
    if args.load_test:
        graph, _, variant = load_graph_from_metadata(Path(meta["edge_path"]).with_name("metadata.json"))
        LOGGER.info("Loaded GRAPE graph using variant: %s", variant)
        if hasattr(graph, "get_number_of_nodes"):
            LOGGER.info("GRAPE nodes: %s", graph.get_number_of_nodes())
        if hasattr(graph, "get_number_of_edges"):
            LOGGER.info("GRAPE edges: %s", graph.get_number_of_edges())


if __name__ == "__main__":
    main()

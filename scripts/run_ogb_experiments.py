"""Run the OGB link-prediction experiment suite.

The pipeline intentionally creates a fresh temporal train/valid/test split via
``scripts.prepare_dataset`` for every selected dataset. For ``ogbl-citation2``
this is different from the official OGB benchmark split: the official split is
still saved under ``<data-root>/ogbl-citation2/raw`` by
``scripts.prepare_ogb_dataset``.

Example:
  python -m scripts.run_ogb_experiments \
      --ogb-root /data/ogb \
      --data-root /data/lpp/ogb \
      --runs-root /data/lpp-runs/ogb \
      --val-edges 100000 \
      --test-edges 100000

Use ``--torchrun-nproc-per-node N`` for distributed node2vec and downstream
classifier runs. SEAL and GRAPE remain single-process commands.
"""
from __future__ import annotations

import argparse
import copy
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from dataset_utils import load_yaml


LOGGER = logging.getLogger("run_ogb_experiments")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_DATASETS = ("ogbl-citation2", "ogbn-papers100M")
STAGES = (
    "download",
    "prepare",
    "seal",
    "node2vec",
    "node2vec-lp",
    "grape-export",
    "grape",
    "grape-lp",
)
CONFIG_STAGES = frozenset({"seal", "node2vec", "node2vec-lp", "grape-lp"})


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    raw_dir: Path
    prepared_dir: Path
    run_dir: Path

    @property
    def train_csr_dir(self) -> Path:
        return self.prepared_dir / "train_csr"

    @property
    def configs_dir(self) -> Path:
        return self.run_dir / "configs"

    @property
    def markers_dir(self) -> Path:
        return self.run_dir / ".done"

    def config_path(self, name: str) -> Path:
        return self.configs_dir / f"{name}.yaml"

    def marker_path(self, stage: str) -> Path:
        return self.markers_dir / stage

    def method_dir(self, method: str) -> Path:
        return self.run_dir / method

    def embedding_checkpoint(self, method: str) -> Path:
        if method == "grape":
            return self.method_dir(method) / "node2vec_embedding_table"
        return self.method_dir(method) / "checkpoints" / "final_embedding_shards"


def make_dataset_paths(*, dataset: str, data_root: Path, runs_root: Path) -> DatasetPaths:
    dataset_data_root = data_root / dataset
    return DatasetPaths(
        dataset=dataset,
        raw_dir=dataset_data_root / "raw",
        prepared_dir=dataset_data_root / "prepared",
        run_dir=runs_root / dataset,
    )


def _mapping_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.setdefault(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return section


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _dataset_config(*, paths: DatasetPaths, num_nodes: int, file_endian: str) -> dict[str, Any]:
    return {
        "num_nodes": int(num_nodes),
        "root": str(paths.prepared_dir),
        "graph_csr_use_mmap": True,
        "graph_csr_file_endian": str(file_endian),
        "graph_csr_allow_non_native": True,
        "mmap": True,
        "has_self_loops": False,
    }


def _reset_output_paths(config: dict[str, Any], *, out_dir: Path) -> None:
    output = _mapping_section(config, "output")
    output["dir"] = str(out_dir)


def _reset_training_paths(config: dict[str, Any], *, resume_key: str) -> None:
    training = _mapping_section(config, "training")
    training["checkpoint_dir"] = None
    training["tensorboard_log_dir"] = None
    training[resume_key] = None


def _embedding_lp_config(
    template: dict[str, Any],
    *,
    paths: DatasetPaths,
    method: str,
    num_nodes: int,
    file_endian: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    dataset = _mapping_section(config, "dataset")
    dataset.update(_dataset_config(paths=paths, num_nodes=num_nodes, file_endian=file_endian))
    embeddings = _mapping_section(config, "embeddings")
    embeddings["checkpoint_dir"] = str(paths.embedding_checkpoint(method))
    _reset_output_paths(config, out_dir=paths.method_dir(f"{method}_lp"))
    _reset_training_paths(config, resume_key="resume_checkpoint_path")
    return config


def write_run_configs(
    *,
    paths: DatasetPaths,
    num_nodes: int,
    file_endian: str,
    seal_template_path: Path,
    node2vec_template_path: Path,
    lp_template_path: Path,
) -> dict[str, Path]:
    seal = load_yaml(seal_template_path)
    seal_dataset = _mapping_section(seal, "dataset")
    seal_dataset.update(_dataset_config(paths=paths, num_nodes=num_nodes, file_endian=file_endian))
    _reset_output_paths(seal, out_dir=paths.method_dir("seal"))
    seal_training = _mapping_section(seal, "training")
    seal_training["tensorboard_log_dir"] = None

    node2vec = load_yaml(node2vec_template_path)
    node2vec_dataset = _mapping_section(node2vec, "dataset")
    node2vec_dataset.update(
        {
            "path": str(paths.train_csr_dir),
            "split_root": str(paths.prepared_dir),
            "use_mmap": True,
            "file_endian": str(file_endian),
            "is_directed": False,
        }
    )
    _reset_output_paths(node2vec, out_dir=paths.method_dir("node2vec"))
    _reset_training_paths(node2vec, resume_key="resume_checkpoint_dir")

    lp_template = load_yaml(lp_template_path)
    configs = {
        "seal": seal,
        "node2vec": node2vec,
        "node2vec_lp": _embedding_lp_config(
            lp_template,
            paths=paths,
            method="node2vec",
            num_nodes=num_nodes,
            file_endian=file_endian,
        ),
        "grape_lp": _embedding_lp_config(
            lp_template,
            paths=paths,
            method="grape",
            num_nodes=num_nodes,
            file_endian=file_endian,
        ),
    }

    result: dict[str, Path] = {}
    for name, config in configs.items():
        path = paths.config_path(name)
        _write_yaml(path, config)
        result[name] = path
    return result


def _append_flag(command: list[str], enabled: bool, flag: str) -> list[str]:
    if enabled:
        command.append(flag)
    return command


class Pipeline:
    def __init__(self, args: argparse.Namespace, paths: DatasetPaths) -> None:
        self.args = args
        self.paths = paths
        self._configs_ready = False

    def _python_module(self, module: str, *args: str) -> list[str]:
        return [self.args.python, "-m", module, *args]

    def _distributed_python_module(self, module: str, *args: str) -> list[str]:
        if self.args.torchrun_nproc_per_node <= 1:
            return self._python_module(module, *args)
        return [
            self.args.torchrun,
            "--standalone",
            f"--nproc_per_node={self.args.torchrun_nproc_per_node}",
            "-m",
            module,
            *args,
        ]

    def _ensure_configs(self) -> None:
        if self._configs_ready or self.args.dry_run:
            return
        num_nodes_path = self.paths.raw_dir / "num_nodes.txt"
        if not num_nodes_path.exists():
            raise FileNotFoundError(
                f"{num_nodes_path} does not exist. Run the download stage before model stages."
            )
        num_nodes = int(num_nodes_path.read_text(encoding="utf-8").strip())
        write_run_configs(
            paths=self.paths,
            num_nodes=num_nodes,
            file_endian=self.args.file_endian,
            seal_template_path=self.args.seal_config,
            node2vec_template_path=self.args.node2vec_config,
            lp_template_path=self.args.lp_config,
        )
        self._configs_ready = True

    def _run(self, stage: str, command: list[str]) -> None:
        marker = self.paths.marker_path(stage)
        if self.args.resume and marker.exists():
            LOGGER.info("[%s] skipping completed stage %s", self.paths.dataset, stage)
            return
        LOGGER.info("[%s] %s", self.paths.dataset, shlex.join(command))
        if self.args.dry_run:
            return
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("completed\n", encoding="utf-8")

    def run_stage(self, stage: str) -> None:
        if stage == "download":
            command = self._python_module(
                "scripts.prepare_ogb_dataset",
                "--dataset",
                self.paths.dataset,
                "--out-dir",
                str(self.paths.raw_dir),
                "--ogb-root",
                str(self.args.ogb_root),
                "--file-endian",
                self.args.file_endian,
            )
            self._run(stage, _append_flag(command, self.args.verbose, "--verbose"))
            return

        if stage == "prepare":
            command = self._python_module(
                "scripts.prepare_dataset",
                "--graph-dir",
                str(self.paths.raw_dir / "graph_csr"),
                "--out-root",
                str(self.paths.prepared_dir),
                "--val-edges",
                str(self.args.val_edges),
                "--test-edges",
                str(self.args.test_edges),
                "--use-mmap",
                "--file-endian",
                self.args.file_endian,
                "--out-file-endian",
                self.args.file_endian,
                "--allow-non-native",
            )
            self._run(stage, _append_flag(command, self.args.verbose, "--verbose"))
            return

        if stage in CONFIG_STAGES:
            self._ensure_configs()
        if stage == "seal":
            self._run(
                stage,
                self._python_module(
                    "seal.seal_link_pred",
                    "--config",
                    str(self.paths.config_path("seal")),
                ),
            )
        elif stage == "node2vec":
            self._run(
                stage,
                self._distributed_python_module(
                    "node2vec.train",
                    "--config",
                    str(self.paths.config_path("node2vec")),
                ),
            )
        elif stage == "node2vec-lp":
            self._run(
                stage,
                self._distributed_python_module(
                    "embedding_lp.train",
                    "--config",
                    str(self.paths.config_path("node2vec_lp")),
                ),
            )
        elif stage == "grape-export":
            self._run(
                stage,
                self._python_module(
                    "baselines.grape.export_graphcsr_to_grape",
                    "--graph-dir",
                    str(self.paths.train_csr_dir),
                    "--out-dir",
                    str(self.paths.method_dir("grape")),
                    "--file-endian",
                    self.args.file_endian,
                    "--graph-kind",
                    "directed",
                ),
            )
        elif stage == "grape":
            grape_dir = self.paths.method_dir("grape")
            self._run(
                stage,
                self._python_module(
                    "baselines.grape.train_node2vec",
                    "--metadata",
                    str(grape_dir / "metadata.json"),
                    "--out-emb",
                    str(grape_dir / "node2vec.npy"),
                    "--out-embedding-checkpoint",
                    str(self.paths.embedding_checkpoint("grape")),
                    "--embedding-checkpoint-backend",
                    "vanilla",
                    "--embedding-size",
                    str(self.args.grape_embedding_size),
                    "--epochs",
                    str(self.args.grape_epochs),
                    "--iterations",
                    str(self.args.grape_iterations),
                    "--walk-length",
                    str(self.args.grape_walk_length),
                    "--window-size",
                    str(self.args.grape_window_size),
                    "--negative-samples",
                    str(self.args.grape_negative_samples),
                    "--p",
                    str(self.args.grape_p),
                    "--q",
                    str(self.args.grape_q),
                ),
            )
        elif stage == "grape-lp":
            self._run(
                stage,
                self._python_module(
                    "embedding_lp.train",
                    "--config",
                    str(self.paths.config_path("grape_lp")),
                ),
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def run(self, stages: Iterable[str]) -> None:
        for stage in stages:
            self.run_stage(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, prepare, and run SEAL, node2vec, and GRAPE on OGB datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ogb-root", required=True, type=Path, help="OGB download/cache directory.")
    parser.add_argument("--data-root", required=True, type=Path, help="Raw and prepared LPP dataset root.")
    parser.add_argument("--runs-root", required=True, type=Path, help="Generated configs and experiment outputs.")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--val-edges", type=int, default=100_000)
    parser.add_argument("--test-edges", type=int, default=100_000)
    parser.add_argument("--file-endian", choices=["big", "little"], default="big")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocesses.")
    parser.add_argument("--torchrun", default="torchrun", help="torchrun executable.")
    parser.add_argument("--torchrun-nproc-per-node", type=int, default=1)
    parser.add_argument(
        "--seal-config",
        type=Path,
        default=PROJECT_ROOT / "seal" / "configs" / "link_prediction_default.yaml",
        help="SEAL YAML template. Dataset/output paths are overwritten.",
    )
    parser.add_argument(
        "--node2vec-config",
        type=Path,
        default=PROJECT_ROOT / "node2vec" / "configs" / "node2vec_default.yaml",
        help="node2vec YAML template. Dataset/output/checkpoint paths are overwritten.",
    )
    parser.add_argument(
        "--lp-config",
        type=Path,
        default=PROJECT_ROOT / "embedding_lp" / "configs" / "link_prediction_default.yaml",
        help="Shared classifier YAML template. Dataset/embedding/output paths are overwritten.",
    )
    parser.add_argument("--grape-embedding-size", type=int, default=128)
    parser.add_argument("--grape-epochs", type=int, default=5)
    parser.add_argument("--grape-iterations", type=int, default=10)
    parser.add_argument("--grape-walk-length", type=int, default=80)
    parser.add_argument("--grape-window-size", type=int, default=10)
    parser.add_argument("--grape-negative-samples", type=int, default=5)
    parser.add_argument("--grape-p", type=float, default=1.0)
    parser.add_argument("--grape-q", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages with a completion marker under <runs-root>/<dataset>/.done.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them or writing configs.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    if args.val_edges <= 0 or args.test_edges <= 0:
        raise ValueError("--val-edges and --test-edges must be positive")
    if args.torchrun_nproc_per_node <= 0:
        raise ValueError("--torchrun-nproc-per-node must be positive")

    LOGGER.warning(
        "Using custom temporal splits from scripts.prepare_dataset. "
        "For ogbl-citation2 these are not the official OGB benchmark splits."
    )
    for dataset in args.datasets:
        paths = make_dataset_paths(dataset=dataset, data_root=args.data_root, runs_root=args.runs_root)
        Pipeline(args, paths).run(args.stages)


if __name__ == "__main__":
    main()

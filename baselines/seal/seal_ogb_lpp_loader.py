from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _torch_load_compatible(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_lpp_seal_ogb_export(export_dir: str | Path):
    """Load data.pt/split_edge.pt produced by export_lpp_to_seal_ogb.py.

    Returns:
        data, split_edge, directed, eval_metric, metadata
    """
    export_dir = Path(export_dir)
    meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    data = _torch_load_compatible(export_dir / "data.pt")
    split_edge = _torch_load_compatible(export_dir / "split_edge.pt")
    return data, split_edge, bool(meta.get("directed", False)), meta.get("eval_metric", "auc"), meta


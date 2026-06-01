from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .collators import LinkPredictionCollator
from .config import TrainLoaderConfig
from .eval_data import build_train_positive_dataset
from .iterable import PositiveEdgeIterableDataset
from .negative_sampling import build_negative_sampler
from .utils import compute_positive_batch_size


def build_train_loader(
    *,
    train_loader_config: TrainLoaderConfig | dict[str, Any],
    batch_transform,
    rank: int,
    world_size: int,
    device_type: str,
) -> DataLoader:
    if isinstance(train_loader_config, dict):
        train_loader_config = TrainLoaderConfig.from_dict(train_loader_config)

    positive_edges_config = train_loader_config.positive_edges
    positive_dataset = build_train_positive_dataset(dataset_config=positive_edges_config)
    negative_sampler = build_negative_sampler(
        train_loader_config.negative_sampling,
        positive_edges_config,
    )
    iterable_dataset = PositiveEdgeIterableDataset(
        positive_dataset,
        seed=int(train_loader_config.seed),
        rank=rank,
        world_size=world_size,
        epoch=0,
    )
    pos_batch_size = compute_positive_batch_size(int(train_loader_config.batch_size), negative_sampler)
    collator = LinkPredictionCollator(
        negative_sampler=negative_sampler,
        batch_transform=batch_transform,
        static_meta={"rank": int(rank), "world_size": int(world_size)},
    )
    return DataLoader(
        iterable_dataset,
        batch_size=pos_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=int(train_loader_config.num_workers),
        pin_memory=device_type == "cuda",
        drop_last=False,
    )

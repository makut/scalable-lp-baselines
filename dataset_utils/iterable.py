from __future__ import annotations

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .utils import choose_coprime_stride, splitmix64


class PositiveEdgeIterableDataset(IterableDataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.base_dataset)
        if self.world_size <= 1:
            return int(total)
        return int((total + self.world_size - 1) // self.world_size)

    def _iter_indices(self):
        total = len(self.base_dataset)
        if total <= 0:
            return

        epoch_seed = self.seed + 1_000_003 * self.epoch + 17
        offset = splitmix64(epoch_seed) % total
        stride = choose_coprime_stride(total, epoch_seed + 1)

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker.id)
            num_workers = int(worker.num_workers)

        consumer_id = self.rank * num_workers + worker_id
        num_consumers = self.world_size * num_workers

        for seq_idx in range(consumer_id, total, num_consumers):
            yield int((offset + seq_idx * stride) % total)

    def __iter__(self):
        for idx in self._iter_indices():
            yield self.base_dataset[idx]

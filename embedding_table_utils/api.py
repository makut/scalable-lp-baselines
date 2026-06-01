from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from .checkpointing import load_local_checkpoint, save_local_checkpoint
from .config import EmbeddingTableConfig
from .optimizer_adapters import BaseOptimizerAdapter, NoOpOptimizerAdapter


class BaseEmbeddingTable(nn.Module, ABC):
    """
    Common embedding-table API for both TorchRec/DMP and vanilla backends.

    Local checkpoints created through save_local/load_local are per-rank artifacts:
    they are meant for resume with the same world size and do not guarantee portability
    across different sharding layouts or compute-kernel policies.
    """

    def __init__(self, config: EmbeddingTableConfig, *, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self._optimizer_adapter: BaseOptimizerAdapter = NoOpOptimizerAdapter()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lookup(ids)

    @abstractmethod
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def zero_grad(self) -> None:
        self._optimizer_adapter.zero_grad()

    def step(self) -> None:
        self._optimizer_adapter.step()

    def train(self, mode: bool = True) -> "BaseEmbeddingTable":
        return super().train(mode)

    def eval(self) -> "BaseEmbeddingTable":
        return super().eval()

    @abstractmethod
    def local_model_state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_local_model_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


    def local_optimizer_state_dict(self) -> dict[str, Any] | None:
        return self._optimizer_adapter.state_dict()

    def load_local_optimizer_state_dict(self, state: dict[str, Any] | None) -> None:
        self._optimizer_adapter.load_state_dict(state)

    def save_local(self, ckpt_dir: str, step: int | None = None) -> None:
        save_local_checkpoint(
            ckpt_dir=ckpt_dir,
            config=self.config,
            model_state=self.local_model_state_dict(),
            optimizer_state=self.local_optimizer_state_dict(),
            step=step,
            process_group=self.process_group,
        )

    def load_local(self, ckpt_dir: str) -> int | None:
        model_template = self.local_model_state_dict()
        optimizer_template = self.local_optimizer_state_dict()
        payload = load_local_checkpoint(
            ckpt_dir=ckpt_dir,
            process_group=self.process_group,
            model_template=model_template,
            optimizer_template=optimizer_template,
        )
        self.load_local_model_state_dict(payload["model"])
        self.load_local_optimizer_state_dict(payload.get("optimizer"))
        step = payload.get("step")
        return None if step is None else int(step)

    @property
    def process_group(self) -> torch.distributed.ProcessGroup | None:
        return None

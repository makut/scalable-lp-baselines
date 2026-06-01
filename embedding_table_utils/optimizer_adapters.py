from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch
from torch import nn

from .config import EmbeddingTableConfig


_TORCHREC_FUSED_ONLY_OPTIMIZERS = {"adam", "adagrad", "rowwise_adagrad"}


class BaseOptimizerAdapter(ABC):
    def zero_grad(self) -> None:
        return None

    def step(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any] | None:
        return None

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        return None


class NoOpOptimizerAdapter(BaseOptimizerAdapter):
    """Inference-safe optimizer adapter that intentionally performs no updates."""


class TorchRecInBackwardAdapter(NoOpOptimizerAdapter):
    """No-op adapter for optimizer paths where the update happens outside explicit step()."""


class TorchOptimizerAdapter(BaseOptimizerAdapter):
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict[str, Any] | None:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        self.optimizer.load_state_dict(state)


def _make_torch_optimizer(params: Iterable[nn.Parameter], config: EmbeddingTableConfig) -> torch.optim.Optimizer:
    optimizer_type = config.optimizer_type
    if optimizer_type is None:
        raise ValueError("optimizer_type must be set before creating a torch optimizer")

    kwargs = dict(config.optimizer_kwargs)
    if "learning_rate" in kwargs and "lr" not in kwargs:
        kwargs["lr"] = kwargs.pop("learning_rate")
    params_list = list(params)
    if optimizer_type == "sgd":
        return torch.optim.SGD(params_list, **kwargs)
    if optimizer_type == "adam":
        return torch.optim.Adam(params_list, **kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params_list, **kwargs)
    if optimizer_type in {"adagrad", "rowwise_adagrad"}:
        return torch.optim.Adagrad(params_list, **kwargs)
    raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")


def build_vanilla_optimizer_adapter(
    params: Iterable[nn.Parameter],
    config: EmbeddingTableConfig,
) -> BaseOptimizerAdapter:
    if config.optimizer_type is None:
        return NoOpOptimizerAdapter()
    return TorchOptimizerAdapter(_make_torch_optimizer(params, config))


def _lazy_import_keyed_optimizers() -> tuple[Any, Any]:
    try:
        from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise RuntimeError(
            "TorchRec keyed optimizers are unavailable. Ensure torchrec/fbgemm are installed "
            "for the current platform and runtime."
        ) from exc
    return KeyedOptimizerWrapper, CombinedOptimizer


def build_torchrec_optimizer_adapter(
    *,
    module: nn.Module,
    config: EmbeddingTableConfig,
) -> BaseOptimizerAdapter:
    if config.optimizer_type is None:
        return NoOpOptimizerAdapter()

    fused_optimizer = getattr(module, "fused_optimizer", None)
    named_params = dict(module.named_parameters())

    if fused_optimizer is not None:
        return TorchOptimizerAdapter(fused_optimizer)

    if not named_params:
        return NoOpOptimizerAdapter()

    if config.optimizer_type in _TORCHREC_FUSED_ONLY_OPTIMIZERS:
        raise RuntimeError(
            "TorchRec embedding optimizer_type="
            f"{config.optimizer_type!r} requires a fused TorchRec optimizer. "
            "Build the embedding table through the TorchRec sharded/DMP path "
            "or use backend='vanilla' for dense torch.optim training."
        )

    try:
        KeyedOptimizerWrapper, _ = _lazy_import_keyed_optimizers()
    except RuntimeError:
        return TorchOptimizerAdapter(_make_torch_optimizer(named_params.values(), config))

    factory = lambda params: _make_torch_optimizer(params, config)
    return TorchOptimizerAdapter(KeyedOptimizerWrapper(cast(Mapping[str, Any], named_params), factory))

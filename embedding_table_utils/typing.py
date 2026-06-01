from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn


StateDict = dict[str, Any]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
ProcessGroupLike = torch.distributed.ProcessGroup | None


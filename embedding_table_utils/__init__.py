from .api import BaseEmbeddingTable
from .config import EmbeddingTableConfig
from .factory import create_embedding_table
from .optimizer_adapters import (
    BaseOptimizerAdapter,
    NoOpOptimizerAdapter,
    TorchOptimizerAdapter,
    TorchRecInBackwardAdapter,
)
from .read_only import ReadOnlyEmbeddingStore
from .torchrec_backend import TorchRecShardedEmbeddingTable
from .vanilla_backend import VanillaEmbeddingTable

__all__ = [
    "BaseEmbeddingTable",
    "BaseOptimizerAdapter",
    "EmbeddingTableConfig",
    "NoOpOptimizerAdapter",
    "ReadOnlyEmbeddingStore",
    "TorchOptimizerAdapter",
    "TorchRecInBackwardAdapter",
    "TorchRecShardedEmbeddingTable",
    "VanillaEmbeddingTable",
    "create_embedding_table",
]

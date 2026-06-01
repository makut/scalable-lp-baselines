"""Compatibility CLI for downstream link prediction over frozen embeddings."""

from embedding_lp.train import _build_lp_config, main

__all__ = ["_build_lp_config", "main"]


if __name__ == "__main__":
    main()

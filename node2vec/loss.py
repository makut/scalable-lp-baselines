"""Skip-gram NEG loss for node2vec.

Two negative-sampling modes are supported:

* Per-anchor (default): `neg_rw` is a `[B, context_size]` tensor with the same
  layout as `pos_rw` (column 0 = anchor, columns 1.. = context). Every row
  contributes `context_size - 1` independent negative pairs.
* Shared (`shared_neg_ids` is set instead of `neg_rw`): a single 1-D pool of
  random node ids is sampled once per batch and *every* positive anchor in the
  batch is scored against the entire pool via a single matmul. This collapses
  the negative-side embedding-lookup volume from `O(Bp * N * C)` to `O(N)`.

In both modes the model is called once with all required ids concatenated to
avoid redundant lookups.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def node2vec_loss(
    model: nn.Module,
    pos_rw: torch.Tensor,
    neg_rw: torch.Tensor | None = None,
    *,
    num_nodes: int,
    use_nce_bias: bool = False,
    reduction: str = "mean",
    shared_neg_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if pos_rw.shape[0] == 0:
        raise ValueError("pos_rw must be non-empty")
    if pos_rw.shape[1] < 2:
        raise ValueError("context_size must be >= 2")
    if (neg_rw is None) == (shared_neg_ids is None):
        raise ValueError("Pass exactly one of `neg_rw` or `shared_neg_ids`")

    Bp, L = pos_rw.shape
    K = L - 1

    pos_anchor = pos_rw[:, 0]
    pos_context = pos_rw[:, 1:].reshape(-1)

    if shared_neg_ids is not None:
        return _shared_negatives_loss(
            model,
            pos_anchor=pos_anchor,
            pos_context=pos_context,
            shared_neg_ids=shared_neg_ids.reshape(-1),
            Bp=Bp,
            K=K,
            num_nodes=num_nodes,
            use_nce_bias=use_nce_bias,
            reduction=reduction,
        )

    assert neg_rw is not None
    if neg_rw.shape[0] == 0:
        raise ValueError("neg_rw must be non-empty")
    if neg_rw.shape[1] < 2:
        raise ValueError("context_size must be >= 2")
    Bn, _ = neg_rw.shape
    neg_anchor = neg_rw[:, 0]
    neg_context = neg_rw[:, 1:].reshape(-1)

    all_ids = torch.cat([pos_anchor, pos_context, neg_anchor, neg_context], dim=0)
    emb = model(all_ids)

    offset = 0
    pos_anchor_emb = emb[offset : offset + Bp]
    offset += Bp
    pos_ctx_emb = emb[offset : offset + Bp * K]
    offset += Bp * K
    neg_anchor_emb = emb[offset : offset + Bn]
    offset += Bn
    neg_ctx_emb = emb[offset : offset + Bn * K]

    pos_ctx_emb = pos_ctx_emb.view(Bp, K, -1)
    neg_ctx_emb = neg_ctx_emb.view(Bn, K, -1)

    pos_logits = (pos_anchor_emb.unsqueeze(1) * pos_ctx_emb).sum(dim=-1)
    neg_logits = (neg_anchor_emb.unsqueeze(1) * neg_ctx_emb).sum(dim=-1)

    if use_nce_bias:
        pos_logits = pos_logits - math.log(float(num_nodes))
        neg_logits = neg_logits - math.log(float(num_nodes))

    loss_pos = -F.logsigmoid(pos_logits).reshape(-1)
    loss_neg = -F.logsigmoid(-neg_logits).reshape(-1)

    if reduction == "mean":
        return loss_pos.mean() + loss_neg.mean()
    if reduction == "sum":
        return loss_pos.sum() + loss_neg.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _shared_negatives_loss(
    model: nn.Module,
    *,
    pos_anchor: torch.Tensor,
    pos_context: torch.Tensor,
    shared_neg_ids: torch.Tensor,
    Bp: int,
    K: int,
    num_nodes: int,
    use_nce_bias: bool,
    reduction: str,
) -> torch.Tensor:
    N = int(shared_neg_ids.numel())
    if N == 0:
        raise ValueError("shared_neg_ids must be non-empty")

    all_ids = torch.cat([pos_anchor, pos_context, shared_neg_ids], dim=0)
    emb = model(all_ids)

    offset = 0
    pos_anchor_emb = emb[offset : offset + Bp]
    offset += Bp
    pos_ctx_emb = emb[offset : offset + Bp * K].view(Bp, K, -1)
    offset += Bp * K
    shared_neg_emb = emb[offset : offset + N]

    pos_logits = (pos_anchor_emb.unsqueeze(1) * pos_ctx_emb).sum(dim=-1)
    # Every positive anchor is scored against every shared negative id.
    neg_logits = pos_anchor_emb @ shared_neg_emb.transpose(0, 1)

    if use_nce_bias:
        pos_logits = pos_logits - math.log(float(num_nodes))
        neg_logits = neg_logits - math.log(float(num_nodes))

    loss_pos = -F.logsigmoid(pos_logits).reshape(-1)
    loss_neg = -F.logsigmoid(-neg_logits).reshape(-1)

    if reduction == "mean":
        return loss_pos.mean() + loss_neg.mean()
    if reduction == "sum":
        return loss_pos.sum() + loss_neg.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")

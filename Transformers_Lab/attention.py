"""Attention mechanisms built from scratch in PyTorch.

Worksheet: Transformer.pdf + AI_lab_transformers.ipynb.

Concepts covered (from the hands-on section of the PDF):
  1. Scaled dot-product attention
  2. Self-attention
  3. Cross-attention
  4. Multi-head attention
  5. Causal / masked attention

The implementations are deliberately explicit so that the formulas
match the lecture notation.  For a given set of queries Q, keys K and
values V,

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Self-attention is the special case where Q, K, V all come from the
same input X by linear projections.  Cross-attention takes Q from one
sequence and (K, V) from another.  Causal attention adds a mask that
prevents a position from attending to the future.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output, attention_weights).

    Shapes:
        q: (..., L_q, d_k)
        k: (..., L_k, d_k)
        v: (..., L_k, d_v)
        mask: broadcastable to (..., L_q, L_k); True (or 1) means KEEP;
              False (or 0) means mask out.  Pass None to disable.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights


# ---------------------------------------------------------------------------
# 2. Self-attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Single-head self-attention: Q, K, V are projections of the same X."""

    def __init__(self, d_model: int, d_k: int | None = None) -> None:
        super().__init__()
        d_k = d_k or d_model
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return out


# ---------------------------------------------------------------------------
# 3. Cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Q comes from the decoder side, K and V come from the encoder side."""

    def __init__(self, d_model: int, d_k: int | None = None) -> None:
        super().__init__()
        d_k = d_k or d_model
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(
        self,
        x_dec: torch.Tensor,
        x_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.w_q(x_dec)
        k = self.w_k(x_enc)
        v = self.w_v(x_enc)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return out


# ---------------------------------------------------------------------------
# 4. Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention from "Attention Is All You Need"."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) -> (B, num_heads, L, d_k)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, L, d_k) -> (B, L, d_model)
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if mask is not None and mask.dim() == 2:
            # (L_q, L_k) -> (1, 1, L_q, L_k) for broadcasting across heads.
            mask = mask[None, None, :, :]
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        out = self._merge_heads(out)
        return self.w_o(out), weights


# ---------------------------------------------------------------------------
# 5. Causal mask
# ---------------------------------------------------------------------------

def causal_mask(length: int) -> torch.Tensor:
    """Lower-triangular keep-mask: True means `allowed to attend`.

    Shape: (length, length).  Position i can only attend to positions j <= i.
    """
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(0)

    print("=== 1. Scaled dot-product attention on a toy example ===")
    # 3 queries, 4 keys, d_k = 5, d_v = 6.
    q = torch.randn(3, 5)
    k = torch.randn(4, 5)
    v = torch.randn(4, 6)
    out, w = scaled_dot_product_attention(q, k, v)
    print(f"  output shape: {tuple(out.shape)}  (expected (3, 6))")
    print(f"  attention row-sums: {w.sum(-1).tolist()}  (each should be 1)")

    print("\n=== 2. Self-attention module ===")
    x = torch.randn(2, 7, 16)  # batch=2, seq_len=7, d_model=16
    sa = SelfAttention(d_model=16)
    y = sa(x)
    print(f"  input {tuple(x.shape)} -> output {tuple(y.shape)}")

    print("\n=== 3. Cross-attention module ===")
    x_dec = torch.randn(2, 5, 16)  # decoder has shorter sequence
    x_enc = torch.randn(2, 9, 16)  # encoder has longer sequence
    ca = CrossAttention(d_model=16)
    y = ca(x_dec, x_enc)
    print(f"  decoder {tuple(x_dec.shape)}, encoder {tuple(x_enc.shape)} -> output {tuple(y.shape)}")

    print("\n=== 4. Multi-head attention ===")
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    y, w = mha(x, x, x)
    print(f"  output {tuple(y.shape)}  (same as input)")
    print(f"  weights {tuple(w.shape)}  (B, heads, L_q, L_k)")

    print("\n=== 5. Causal mask ===")
    mask = causal_mask(5)
    print("  mask (True = attend):")
    for row in mask.int().tolist():
        print("   ", row)
    # With a causal mask, position 0 can only attend to position 0, so its
    # output must equal V[0] (the value at position 0) ignoring projections.
    q = torch.randn(1, 5, 8)
    k = torch.randn(1, 5, 8)
    v = torch.randn(1, 5, 8)
    out, w = scaled_dot_product_attention(q, k, v, mask=mask[None, :, :])
    # Position 0's attention weights should be exactly [1, 0, 0, 0, 0].
    print(f"  position 0 attention row: {w[0, 0].tolist()}")
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    print("  -> position 0 attends only to itself; future tokens are masked")


if __name__ == "__main__":
    main()

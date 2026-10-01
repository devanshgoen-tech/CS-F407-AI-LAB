"""Tests for the from-scratch attention primitives."""

from __future__ import annotations

import math

import torch

from attention import (
    CrossAttention,
    MultiHeadAttention,
    SelfAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from positional_encoding import SinusoidalPositionalEncoding


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_scaled_dot_product_shapes() -> None:
    print("test_scaled_dot_product_shapes")
    q = torch.randn(2, 3, 5)      # (B, L_q, d_k)
    k = torch.randn(2, 4, 5)      # (B, L_k, d_k)
    v = torch.randn(2, 4, 7)      # (B, L_k, d_v)
    out, w = scaled_dot_product_attention(q, k, v)
    check("output shape", tuple(out.shape) == (2, 3, 7))
    check("weights shape", tuple(w.shape) == (2, 3, 4))
    check("weights rows sum to 1", torch.allclose(w.sum(-1), torch.ones(2, 3), atol=1e-6))


def test_identity_attention() -> None:
    print("test_identity_attention")
    # Make each query identical to its matching key; expect ~one-hot weights
    # concentrated on the diagonal.
    torch.manual_seed(0)
    L, d = 4, 32
    k = torch.randn(1, L, d) * 5          # large magnitudes to sharpen softmax
    q = k.clone()
    v = torch.arange(L, dtype=torch.float).view(1, L, 1).expand(1, L, 3)
    out, w = scaled_dot_product_attention(q, k, v)
    diag_mass = w.diagonal(dim1=-2, dim2=-1).mean().item()
    check("diagonal attention mass > 0.9", diag_mass > 0.9)
    check("output follows v order", torch.allclose(out[0, :, 0], torch.arange(L, dtype=torch.float), atol=0.3))


def test_causal_mask_blocks_future() -> None:
    print("test_causal_mask_blocks_future")
    torch.manual_seed(0)
    L, d = 5, 8
    q = k = torch.randn(1, L, d)
    v = torch.randn(1, L, d)
    mask = causal_mask(L)[None, :, :]
    _, w = scaled_dot_product_attention(q, k, v, mask=mask)
    # Position i must have zero weight on positions j > i.
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    check("no attention leaks to the future", float(w[0][upper].abs().max()) < 1e-6)
    check("row 0 is one-hot on index 0", torch.allclose(w[0, 0], torch.tensor([1.0, 0, 0, 0, 0])))


def test_self_attention_shape() -> None:
    print("test_self_attention_shape")
    x = torch.randn(2, 7, 16)
    y = SelfAttention(16)(x)
    check("self-attention output shape matches input", tuple(y.shape) == (2, 7, 16))


def test_cross_attention_shape() -> None:
    print("test_cross_attention_shape")
    x_dec = torch.randn(2, 5, 16)
    x_enc = torch.randn(2, 9, 16)
    y = CrossAttention(16)(x_dec, x_enc)
    check("cross-attention preserves decoder length", tuple(y.shape) == (2, 5, 16))


def test_multihead_decomposition() -> None:
    print("test_multihead_decomposition")
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 6, 16)
    y, w = mha(x, x, x)
    check("multi-head output shape", tuple(y.shape) == (2, 6, 16))
    check("weights carry a heads dimension", tuple(w.shape) == (2, 4, 6, 6))
    check("per-head softmax rows sum to 1", torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-6))


def test_positional_encoding_properties() -> None:
    print("test_positional_encoding_properties")
    d, L = 16, 50
    pe = SinusoidalPositionalEncoding(d_model=d, max_len=L)
    table = pe.pe
    check("PE table shape", tuple(table.shape) == (L, d))
    check("PE[0, even] = 0", torch.allclose(table[0, 0::2], torch.zeros(d // 2)))
    check("PE[0, odd]  = 1", torch.allclose(table[0, 1::2], torch.ones(d // 2)))
    # Adjacent positions should be more similar than far-apart positions.
    def cs(a, b): return float(torch.dot(a, b) / (a.norm() * b.norm()))
    check("adjacent PEs more similar than distant ones",
          cs(table[0], table[1]) > cs(table[0], table[40]))


def test_self_attention_is_permutation_invariant_without_PE() -> None:
    print("test_self_attention_is_permutation_invariant_without_PE")
    # Self-attention without positional information commutes with
    # permutations of the sequence (the key property that motivates PE).
    torch.manual_seed(0)
    sa = SelfAttention(d_model=8)
    x = torch.randn(1, 4, 8)
    perm = torch.tensor([3, 1, 0, 2])
    y1 = sa(x)[0][perm]
    y2 = sa(x[:, perm, :])[0]
    check("permuting input permutes output identically",
          torch.allclose(y1, y2, atol=1e-6))


if __name__ == "__main__":
    for t in (
        test_scaled_dot_product_shapes,
        test_identity_attention,
        test_causal_mask_blocks_future,
        test_self_attention_shape,
        test_cross_attention_shape,
        test_multihead_decomposition,
        test_positional_encoding_properties,
        test_self_attention_is_permutation_invariant_without_PE,
    ):
        t()
    print("\nAll tests passed.")

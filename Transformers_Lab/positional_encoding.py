"""Sinusoidal positional encoding (Vaswani et al., 2017).

Worksheet: Transformer.pdf, hands-on concept #5.

For position pos (0-indexed) and dimension i (0-indexed),

    PE[pos, 2k]   = sin(pos / 10000^{2k / d_model})
    PE[pos, 2k+1] = cos(pos / 10000^{2k / d_model})

The encoding is added to the token embeddings before the first
attention layer so that the model can distinguish token order (vanilla
self-attention is permutation-invariant).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model); returns x + PE[:L]."""
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


def main() -> None:
    torch.manual_seed(0)
    d_model, max_len = 16, 50
    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

    print("=== Positional encoding properties ===")
    table = pe.pe  # (max_len, d_model)
    print(f"  shape: {tuple(table.shape)}")

    # 1. Even dims are sin, odd dims are cos: at pos=0 all sins = 0, cosines = 1.
    print(f"  PE[0, 0::2] (sin at pos=0) all zero? {torch.allclose(table[0, 0::2], torch.zeros(d_model // 2))}")
    print(f"  PE[0, 1::2] (cos at pos=0) all one?  {torch.allclose(table[0, 1::2], torch.ones(d_model // 2))}")

    # 2. The encoding has bounded norm independent of position.
    norms = table.norm(dim=1)
    print(f"  ||PE[p]||_2 range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")

    # 3. Adjacent positions have higher cosine similarity than distant ones.
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    print(f"  cos_sim(PE[0], PE[1])  = {cos_sim(table[0], table[1]):.4f}")
    print(f"  cos_sim(PE[0], PE[10]) = {cos_sim(table[0], table[10]):.4f}")
    print(f"  cos_sim(PE[0], PE[40]) = {cos_sim(table[0], table[40]):.4f}")

    # 4. Added to a token embedding, PE preserves shape.
    x = torch.randn(2, 7, d_model)
    y = pe(x)
    print(f"  input {tuple(x.shape)} -> PE-added {tuple(y.shape)}")


if __name__ == "__main__":
    main()

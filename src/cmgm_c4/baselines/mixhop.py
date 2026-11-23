"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MixHopPropagation(nn.Module):
    """
    MTGNN-style mix-hop propagation layer.

    Two steps per hop k:
      1. Propagation:  H(k) = β·H_in + (1-β)·Â·H(k-1)
      2. Selection:    H_out = Σ_k H(k)·W(k)

    Args:
        in_dim:  Input feature dimension
        out_dim: Output feature dimension
        K:       Number of hops (default: 2)
        beta:    Self-information retention (default: 0.05)
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 2, beta: float = 0.05):
        super().__init__()
        self.K = K
        self.beta = beta
        self.Ws = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(K + 1)
        ])

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_hat = A + torch.eye(A.size(0), device=A.device)
        deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_norm = A_hat / deg

        H = x
        H_in = x
        out = self.Ws[0](H)

        for k in range(1, self.K + 1):
            H = self.beta * H_in + (1 - self.beta) * (A_norm @ H)
            out = out + self.Ws[k](H)

        return out

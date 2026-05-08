"""
Structure-Guided Graph Attention Module
=========================================
Multi-head graph attention that uses the conditional causal adjacency matrix
as structural guidance, replacing standard self-attention with causality-aware
message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StructureGuidedGraphAttention(nn.Module):
    """
    Graph attention layer guided by the conditional causal adjacency matrix.

    Unlike standard GAT which learns attention weights freely, this module
    constrains attention to the causal graph structure: attention scores are
    masked by the causal adjacency matrix A, so only causally relevant
    neighbors contribute to each node's update.

    Args:
        d_model (int): Node feature dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   Node features, shape (B, N, d_model).
            adj: Causal adjacency matrix, shape (B, N, N).
                 adj[b, i, j] = causal weight from j to i.

        Returns:
            out: Updated node features, shape (B, N, d_model).
        """
        B, N, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Multi-head projections
        Q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=H)
        K = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=H)
        V = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=H)

        # Scaled dot-product attention scores
        scale = D ** -0.5
        scores = torch.einsum('bhid,bhjd->bhij', Q, K) * scale  # (B, H, N, N)

        # Apply causal adjacency as structural mask
        # adj: (B, N, N) -> (B, 1, N, N) for broadcasting over heads
        causal_mask = adj.unsqueeze(1)  # (B, 1, N, N)
        # Zero-out non-causal edges (log-domain masking)
        log_mask = torch.log(causal_mask.clamp(min=1e-9))
        scores = scores + log_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.einsum('bhij,bhjd->bhid', attn, V)  # (B, H, N, D)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        # Residual + LayerNorm
        x = self.norm(x + out)
        x = self.ffn_norm(x + self.ffn(x))

        return x

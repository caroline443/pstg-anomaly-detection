"""
PSTG Model with Conditional Causal Graph
==========================================
Main model integrating:
  1. Multi-scale patch embedding
  2. Progressive reasoning stages with conditional causal graph (patent innovation)
  3. Structure-guided graph attention
  4. Multi-step prediction head
"""

import torch
import torch.nn as nn
from einops import rearrange

from .patch_embed import MultiScalePatchEmbedding
from .causal_graph import ConditionalCausalGraph
from .graph_attn import StructureGuidedGraphAttention


class PSTGModel(nn.Module):
    """
    Progressive Spatiotemporal Graph model with Conditional Causal Graph.

    Architecture:
        Input (B, C, T)
            ↓
        MultiScalePatchEmbedding  →  (B, C, N_p, d)
            ↓  reshape to spatiotemporal nodes
        (B, N, d)  where N = C × N_p
            ↓
        [For each layer l = 1..n_L]:
            ConditionalCausalGraph(hidden_states_l)  →  A_l  (B, N, N)
            StructureGuidedGraphAttention(x, A_l)    →  x_l  (B, N, d)
            ↓
        Prediction head  →  (B, C, F)

    Args:
        n_channels (int): Number of telemetry channels C.
        seq_len (int): Input sequence length T.
        pred_len (int): Prediction horizon F.
        patch_sizes (list[int]): Multi-scale patch sizes.
        d_model (int): Model dimension.
        n_heads (int): Attention heads.
        n_layers (int): Number of progressive reasoning layers.
        causal_hidden_dim (int): Hidden dim for causal estimator.
        causal_lag (int): Lag order for Granger causality.
        sparsity_k (int): Top-k edges per node.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        pred_len: int,
        patch_sizes: list = None,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        causal_hidden_dim: int = 256,
        causal_lag: int = 5,
        sparsity_k: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.d_model = d_model

        main_p = patch_sizes[0]
        self.n_patches = seq_len // main_p
        self.n_nodes = n_channels * self.n_patches  # N = C × N_p

        # 1. Multi-scale patch embedding
        self.patch_embed = MultiScalePatchEmbedding(patch_sizes, d_model, n_channels)

        # 2. Progressive layers: each has its own causal graph + graph attention
        self.causal_graphs = nn.ModuleList([
            ConditionalCausalGraph(
                n_nodes=self.n_nodes,
                d_model=d_model,
                hidden_dim=causal_hidden_dim,
                lag=causal_lag,
                sparsity_k=sparsity_k,
            )
            for _ in range(n_layers)
        ])
        self.graph_attns = nn.ModuleList([
            StructureGuidedGraphAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 3. Lag feature extractor (shared across layers)
        self.lag = causal_lag
        self.lag_proj = nn.Linear(main_p * causal_lag, causal_lag)

        # 4. Prediction head: map node features back to channel predictions
        self.pred_head = nn.Sequential(
            nn.Linear(d_model * self.n_patches, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

        self.dropout = nn.Dropout(dropout)

    def _extract_lag_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract lag-order features for each channel from raw input.

        Args:
            x: (B, C, T)
        Returns:
            lag_feat: (B, N, lag) where N = C × N_p
        """
        B, C, T = x.shape
        p = self.seq_len // self.n_patches
        lag = self.lag

        # Use the last `lag` patches as lag features per channel
        lag_len = min(lag * p, T)
        lag_raw = x[:, :, -lag_len:]  # (B, C, lag*p)
        # Pad if needed
        if lag_raw.shape[-1] < lag * p:
            pad = torch.zeros(B, C, lag * p - lag_raw.shape[-1], device=x.device)
            lag_raw = torch.cat([pad, lag_raw], dim=-1)

        lag_feat_ch = self.lag_proj(lag_raw)  # (B, C, lag)
        # Expand to all patches (same lag features per patch within a channel)
        lag_feat = lag_feat_ch.unsqueeze(2).expand(B, C, self.n_patches, lag)
        lag_feat = rearrange(lag_feat, 'b c n l -> b (c n) l')  # (B, N, lag)
        return lag_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input telemetry, shape (B, C, T).

        Returns:
            pred: Multi-step predictions, shape (B, C, F).
        """
        B, C, T = x.shape

        # Step 1: Multi-scale patch embedding
        patch_emb = self.patch_embed(x)          # (B, C, N_p, d)
        h = rearrange(patch_emb, 'b c n d -> b (c n) d')  # (B, N, d)
        h = self.dropout(h)

        # Extract lag features (fixed from input, not updated per layer)
        lag_feat = self._extract_lag_features(x)  # (B, N, lag)

        # Step 2: Progressive reasoning with conditional causal graph
        for l in range(self.n_layers):
            # Compute conditional causal adjacency at this stage
            adj = self.causal_graphs[l](h, lag_feat)   # (B, N, N)
            # Update node representations via structure-guided attention
            h = self.graph_attns[l](h, adj)             # (B, N, d)

        # Step 3: Prediction head
        # Reshape back to (B, C, N_p * d) and predict
        h = rearrange(h, 'b (c n) d -> b c (n d)', c=C, n=self.n_patches)
        pred = self.pred_head(h)  # (B, C, F)

        return pred

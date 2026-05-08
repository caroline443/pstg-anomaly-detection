"""
Conditional Causal Graph Module
================================
Patent Innovation: At each progressive reasoning stage, the Granger causality
adjacency matrix is dynamically updated conditioned on the current node hidden
states, rather than using a static correlation-based graph.

Key difference from PSTG (Entropy 2026):
  - PSTG: static attention-based adjacency matrix (correlation proxy)
  - Ours:  per-stage Granger causality matrix conditioned on hidden states h_l

Reference:
  Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models
  and Cross-spectral Methods. Econometrica, 37(3), 424-438.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalCausalGraph(nn.Module):
    """
    Conditional Granger Causality Graph.

    At reasoning stage l, given node hidden states H_l ∈ R^{N x d},
    estimate a sparse causal adjacency matrix A_l ∈ R^{N x N} where
    A_l[i,j] represents the causal influence of node j on node i,
    conditioned on the current hidden representation.

    Args:
        n_nodes (int): Number of spatiotemporal nodes N.
        d_model (int): Hidden state dimension d.
        hidden_dim (int): Intermediate dimension for causal estimator MLP.
        lag (int): Lag order for Granger causality approximation.
        sparsity_k (int): Top-k edges to retain per node (sparse graph).
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        hidden_dim: int = 256,
        lag: int = 5,
        sparsity_k: int = 10,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.lag = lag
        self.sparsity_k = sparsity_k

        # Causal query/key projections conditioned on hidden states
        # Q_i: "what does node i need to predict?"
        # K_j: "what causal signal does node j provide?"
        self.query_proj = nn.Linear(d_model, hidden_dim)
        self.key_proj = nn.Linear(d_model, hidden_dim)

        # Lag-aware feature extractor: compress lag-step history into a vector
        self.lag_encoder = nn.Sequential(
            nn.Linear(lag, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Causal strength estimator: combines query, key, and lag features
        self.causal_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lag_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conditional causal adjacency matrix.

        Args:
            hidden_states: Node hidden states at current stage,
                           shape (B, N, d_model).
            lag_features:  Lag-order temporal features per node,
                           shape (B, N, lag).

        Returns:
            adj: Sparse causal adjacency matrix, shape (B, N, N).
                 adj[b, i, j] = causal strength from node j to node i.
        """
        B, N, _ = hidden_states.shape

        # Project hidden states to query/key spaces
        Q = self.query_proj(hidden_states)   # (B, N, H)
        K = self.key_proj(hidden_states)     # (B, N, H)

        # Encode lag features
        lag_enc = self.lag_encoder(lag_features)  # (B, N, H)

        # Expand for pairwise computation: (B, N, N, H)
        Q_exp = Q.unsqueeze(2).expand(B, N, N, -1)   # query node i
        K_exp = K.unsqueeze(1).expand(B, N, N, -1)   # key node j
        L_exp = lag_enc.unsqueeze(1).expand(B, N, N, -1)  # lag of node j

        # Concatenate and estimate causal strength
        pair_feat = torch.cat([Q_exp, K_exp, L_exp], dim=-1)  # (B, N, N, 3H)
        causal_logits = self.causal_mlp(pair_feat).squeeze(-1)  # (B, N, N)

        # Top-k sparsification: keep only k strongest causal edges per node
        adj = self._topk_sparse(causal_logits)

        return adj

    def _topk_sparse(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k sparsification and softmax normalization.

        Args:
            logits: Raw causal scores, shape (B, N, N).

        Returns:
            adj: Sparse normalized adjacency, shape (B, N, N).
        """
        k = min(self.sparsity_k, self.n_nodes)
        # Zero out all but top-k entries per row
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:].expand_as(logits)
        mask = logits >= threshold
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        adj = F.softmax(masked_logits, dim=-1)
        # Replace NaN rows (all -inf) with uniform distribution
        adj = torch.nan_to_num(adj, nan=1.0 / self.n_nodes)
        return adj

"""
Multi-Scale Patch Embedding Module
====================================
Extracts hierarchical semantic features from multi-channel time series
using patches of different sizes, then fuses them adaptively.
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Single-scale patch embedding for one patch size."""

    def __init__(self, patch_size: int, d_model: int, n_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series, shape (B, C, T).
        Returns:
            patches: Patch embeddings, shape (B, C, n_patches, d_model).
        """
        B, C, T = x.shape
        p = self.patch_size
        n_patches = T // p
        # Truncate to multiple of patch_size
        x = x[:, :, :n_patches * p]
        # Reshape into patches
        x = rearrange(x, 'b c (n p) -> b c n p', p=p)
        # Project each patch
        out = self.proj(x)          # (B, C, n_patches, d_model)
        out = self.norm(out)
        return out


class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patch embedding with adaptive fusion.

    Embeds the input time series at multiple patch granularities and
    fuses them via a learnable attention-weighted combination.

    Args:
        patch_sizes (list[int]): List of patch sizes, e.g. [25, 50, 125].
        d_model (int): Output embedding dimension.
        n_channels (int): Number of input telemetry channels.
    """

    def __init__(self, patch_sizes: list, d_model: int, n_channels: int):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)
        self.embedders = nn.ModuleList([
            PatchEmbedding(p, d_model, n_channels) for p in patch_sizes
        ])
        # Adaptive fusion weights (one scalar per scale)
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series, shape (B, C, T).
        Returns:
            fused: Fused patch embeddings at the finest scale,
                   shape (B, C, n_patches_main, d_model).
        """
        main_p = self.patch_sizes[0]  # finest scale (p_main)
        B, C, T = x.shape
        n_main = T // main_p

        weights = torch.softmax(self.scale_weights, dim=0)
        fused = None

        for i, (embedder, p) in enumerate(zip(self.embedders, self.patch_sizes)):
            emb = embedder(x)  # (B, C, n_p, d_model)
            # Upsample coarser scales to match finest scale via repeat
            ratio = p // main_p
            if ratio > 1:
                emb = emb.repeat_interleave(ratio, dim=2)
            # Trim to n_main
            emb = emb[:, :, :n_main, :]
            if fused is None:
                fused = weights[i] * emb
            else:
                fused = fused + weights[i] * emb

        return fused  # (B, C, n_main, d_model)

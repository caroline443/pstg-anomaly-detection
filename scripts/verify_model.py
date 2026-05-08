"""
Quick sanity check: verify the model runs without errors.
Usage: python scripts/verify_model.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.pstg import PSTGModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = PSTGModel(
    n_channels=25,
    seq_len=250,
    pred_len=10,
    patch_sizes=[25, 50, 125],
    d_model=512,
    n_heads=4,
    n_layers=2,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

x = torch.randn(4, 25, 250).to(device)
out = model(x)
print(f"Input:  {tuple(x.shape)}")
print(f"Output: {tuple(out.shape)}  (expected: (4, 25, 10))")
assert out.shape == (4, 25, 10), "Shape mismatch!"
print("✓ Model verification passed.")

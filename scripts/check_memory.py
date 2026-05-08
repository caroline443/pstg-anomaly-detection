"""
Check GPU memory usage for different model configs.
Usage: python scripts/check_memory.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.pstg import PSTGModel

if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(1)

configs = [
    dict(d_model=128, causal_hidden_dim=64,  batch=16),
    dict(d_model=256, causal_hidden_dim=128, batch=16),
    dict(d_model=256, causal_hidden_dim=128, batch=32),
    dict(d_model=512, causal_hidden_dim=256, batch=16),
    dict(d_model=512, causal_hidden_dim=256, batch=32),
]

for c in configs:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = PSTGModel(
            n_channels=25, seq_len=250, pred_len=10,
            d_model=c['d_model'],
            causal_hidden_dim=c['causal_hidden_dim'],
        ).cuda()
        x = torch.randn(c['batch'], 25, 250).cuda()
        out = model(x)
        loss = out.mean()
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"d_model={c['d_model']:3d}  causal_hidden={c['causal_hidden_dim']:3d}  batch={c['batch']:2d}  显存={mem:.2f} GB  [OK]")
        del model, x, out, loss
    except RuntimeError as e:
        print(f"d_model={c['d_model']:3d}  causal_hidden={c['causal_hidden_dim']:3d}  batch={c['batch']:2d}  [OOM] {e}")
        torch.cuda.empty_cache()

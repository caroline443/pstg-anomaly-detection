# PSTG: Progressive Spatiotemporal Graph for Spacecraft Anomaly Detection

This repository contains the experimental code for the patent invention disclosure based on the PSTG framework (Entropy 2026). The core innovation is replacing static correlation-based adjacency matrices with **progressive conditional causal graphs** — where the Granger causality adjacency matrix is dynamically updated conditioned on node hidden states at each reasoning stage.

## Core Innovation

> **Conditional Causal Graph in Progressive Reasoning**
>
> At each progressive reasoning stage, instead of using a fixed correlation-based adjacency matrix, we condition the Granger causality estimation on the current node hidden states. This allows the graph structure to evolve with the model's internal representation, capturing dynamic causal dependencies that static graphs miss.

## Project Structure

```
pstg-anomaly-detection/
├── src/pstg/
│   ├── __init__.py
│   ├── model.py          # Main PSTG model with conditional causal graph
│   ├── causal_graph.py   # Conditional Granger causality module
│   ├── patch_embed.py    # Multi-scale patch embedding
│   ├── graph_attn.py     # Structure-guided graph attention
│   └── threshold.py      # Dynamic thresholding for anomaly detection
├── data/
│   └── README.md         # Dataset instructions (ESA-AD, SMAP, MSL)
├── experiments/
│   └── run_experiment.py # Main experiment runner
├── configs/
│   └── default.yaml      # Default hyperparameters
├── scripts/
│   └── preprocess.py     # Data preprocessing utilities
├── requirements.txt
└── README.md
```

## Datasets

- **ESA-AD**: European Space Agency anomaly detection benchmark (primary)
- **SMAP / MSL**: NASA public spacecraft telemetry datasets (secondary validation)

## Baselines

The following methods are used for comparison (following PSTG paper, Entropy 2026):

| Method | Category | Reference |
|--------|----------|-----------|
| Telemanom | LSTM-based (NASA) | Hundman et al., KDD 2018 |
| iTransformer | Inverted Transformer | Liu et al., ICLR 2024 |
| PatchTST | Patch-based Transformer | Nie et al., ICLR 2023 |
| Crossformer | Cross-dim Transformer | Zhang & Yan, ICLR 2023 |
| DLinear | Linear decomposition | Zeng et al., AAAI 2023 |
| TSMixer | MLP-Mixer | Chen et al., TMLR 2023 |
| WPMixer | Multi-resolution Mixer | Murad et al., AAAI 2025 |
| FreTS | Frequency-domain MLP | Yi et al., NeurIPS 2023 |
| TimeFilter | Spatiotemporal graph | Hu et al., arXiv 2025 |

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Preprocess data
python scripts/preprocess.py --dataset smap --data_dir ./data

# Run experiment
python experiments/run_experiment.py --config configs/default.yaml
```

## Citation

If you use this code, please cite the original PSTG paper:

```bibtex
@article{chen2026pstg,
  title={Progressive Spatiotemporal Graph Modeling for Spacecraft Anomaly Detection},
  author={Chen, Zihan and Li, Zewen and Cao, Yuge and Wang, Yue and Chang, Hsi},
  journal={Entropy},
  volume={28},
  pages={426},
  year={2026},
  publisher={MDPI}
}
```

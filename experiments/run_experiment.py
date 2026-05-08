"""
Main Experiment Runner
========================
Trains and evaluates the PSTG model with conditional causal graph.

Usage:
    python experiments/run_experiment.py --config configs/default.yaml
    python experiments/run_experiment.py --config configs/default.yaml --dataset smap
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pstg import PSTGModel
from src.pstg.threshold import DynamicThreshold


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_windows(data: np.ndarray, window: int, stride: int = 1):
    """Slide a window over the time axis."""
    T, C = data.shape
    windows = []
    for i in range(0, T - window + 1, stride):
        windows.append(data[i:i + window])
    return np.stack(windows)  # (n_windows, window, C)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_errors(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean absolute error per time step."""
    return np.abs(preds - targets).mean(axis=(1, 2))  # (n_windows,)


def evaluate(pred_labels: np.ndarray, true_labels: np.ndarray):
    """Compute precision, recall, F0.5 score."""
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    beta = 0.5
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, f'F{beta}': f_beta}


def run(config: dict, data_dir: str, dataset: str, device: torch.device):
    mc = config['model']
    tc = config['training']
    dc = config['data']
    ac = config['anomaly_detection']

    processed_dir = os.path.join(data_dir, f'{dataset}_processed')
    entity_files = sorted([
        f.replace('_train.npy', '')
        for f in os.listdir(processed_dir) if f.endswith('_train.npy')
    ])

    all_results = []

    for entity_id in entity_files:
        print(f"\n{'='*50}")
        print(f"Entity: {entity_id}")

        train_data = np.load(os.path.join(processed_dir, f'{entity_id}_train.npy'))
        test_data = np.load(os.path.join(processed_dir, f'{entity_id}_test.npy'))
        true_labels = np.load(os.path.join(processed_dir, f'{entity_id}_labels.npy'))

        n_channels = train_data.shape[1]
        W = dc['window_length']
        F = dc['prediction_length']

        # Build sliding windows: input (W,C) -> target (F,C)
        train_wins = make_windows(train_data, W + F, stride=dc['stride'])
        train_x = torch.FloatTensor(train_wins[:, :W, :]).permute(0, 2, 1)  # (n, C, W)
        train_y = torch.FloatTensor(train_wins[:, W:, :]).permute(0, 2, 1)  # (n, C, F)

        test_wins = make_windows(test_data, W + F, stride=1)
        test_x = torch.FloatTensor(test_wins[:, :W, :]).permute(0, 2, 1)
        test_y = torch.FloatTensor(test_wins[:, W:, :]).permute(0, 2, 1)

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=tc['batch_size'], shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=ac['test_batch_size'], shuffle=False
        )

        # Build model
        model = PSTGModel(
            n_channels=n_channels,
            seq_len=W,
            pred_len=F,
            patch_sizes=mc['patch_sizes'],
            d_model=mc['d_model'],
            n_heads=mc['n_heads'],
            n_layers=mc['n_layers'],
            causal_hidden_dim=mc['causal_hidden_dim'],
            causal_lag=mc['causal_lag'],
            sparsity_k=int(n_channels * mc['sparsity_ratio']),
            dropout=mc['dropout'],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc['learning_rate'],
            weight_decay=tc['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc['T_max'], eta_min=tc['eta_min']
        )
        criterion = nn.MSELoss()

        # Train
        for epoch in tqdm(range(tc['epochs']), desc=f'Training {entity_id}'):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{tc['epochs']}  loss={loss:.4f}")

        # Predict and detect anomalies
        preds, targets = predict(model, test_loader, device)
        errors = compute_errors(preds, targets)

        detector = DynamicThreshold(
            smoothing_base=ac['smoothing_base'],
            test_batch_size=ac['test_batch_size'],
            tuning_percentage=ac['tuning_percentage'],
        )
        pred_labels_win = detector.detect(errors)

        # Map window-level labels back to time steps
        pred_labels_ts = np.zeros(len(test_data), dtype=int)
        for i, label in enumerate(pred_labels_win):
            if label == 1:
                pred_labels_ts[i:i + W + F] = 1

        # Trim to match true_labels length
        min_len = min(len(pred_labels_ts), len(true_labels))
        metrics = evaluate(pred_labels_ts[:min_len], true_labels[:min_len])
        print(f"  Results: {metrics}")
        all_results.append({'entity': entity_id, **metrics})

    # Summary
    print(f"\n{'='*50}")
    print("Summary:")
    for r in all_results:
        print(f"  {r['entity']}: P={r['precision']:.3f} R={r['recall']:.3f} F0.5={r['F0.5']:.3f}")

    avg_f = np.mean([r['F0.5'] for r in all_results])
    print(f"\nAverage F0.5: {avg_f:.3f}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dataset', choices=['smap', 'msl'], default='smap')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    run(config, args.data_dir, args.dataset, device)


if __name__ == '__main__':
    main()

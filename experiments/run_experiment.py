"""
Main Experiment Runner
========================
Trains and evaluates the PSTG model with conditional causal graph.

Usage:
    python experiments/run_experiment.py --config configs/default.yaml
    python experiments/run_experiment.py --config configs/default.yaml --dataset smap
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pstg import PSTGModel
from src.pstg.threshold import DynamicThreshold


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{run_name}.log')

    logger = logging.getLogger('pstg')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Log file: {log_path}')
    return logger


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_windows(data: np.ndarray, window: int, stride: int = 1):
    T, C = data.shape
    windows = []
    for i in range(0, T - window + 1, stride):
        windows.append(data[i:i + window])
    return np.stack(windows)


# ── Training / evaluation ──────────────────────────────────────────────────────

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
    return np.abs(preds - targets).mean(axis=(1, 2))


def evaluate(pred_labels: np.ndarray, true_labels: np.ndarray):
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    beta = 0.5
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return {'precision': float(precision), 'recall': float(recall), f'F{beta}': float(f_beta)}


# ── Main run ───────────────────────────────────────────────────────────────────

def run(config: dict, data_dir: str, dataset: str, device: torch.device,
        log_dir: str, run_name: str):
    logger = setup_logger(log_dir, run_name)
    logger.info(f'Dataset: {dataset}  Device: {device}')
    logger.info(f'Config: {json.dumps(config, indent=2)}')

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
    csv_path = os.path.join(log_dir, f'{run_name}_results.csv')

    # Write CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s'])
        writer.writeheader()

    for entity_id in entity_files:
        logger.info(f"\n{'='*50}")
        logger.info(f'Entity: {entity_id}')
        t0 = time.time()

        train_data = np.load(os.path.join(processed_dir, f'{entity_id}_train.npy'))
        test_data  = np.load(os.path.join(processed_dir, f'{entity_id}_test.npy'))
        true_labels = np.load(os.path.join(processed_dir, f'{entity_id}_labels.npy'))

        n_channels = train_data.shape[1]
        W = dc['window_length']
        F = dc['prediction_length']

        train_wins = make_windows(train_data, W + F, stride=dc['stride'])
        train_x = torch.FloatTensor(train_wins[:, :W, :]).permute(0, 2, 1)
        train_y = torch.FloatTensor(train_wins[:, W:, :]).permute(0, 2, 1)

        test_wins = make_windows(test_data, W + F, stride=1)
        test_x = torch.FloatTensor(test_wins[:, :W, :]).permute(0, 2, 1)
        test_y = torch.FloatTensor(test_wins[:, W:, :]).permute(0, 2, 1)

        # Skip entities with too few training windows
        if len(train_x) < tc['batch_size']:
            logger.info(f'  [skip] only {len(train_x)} training windows < batch_size {tc["batch_size"]}, skipping.')
            continue

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=tc['batch_size'], shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=ac['test_batch_size'], shuffle=False
        )

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
            sparsity_k=max(1, int(n_channels * mc['sparsity_ratio'])),
            dropout=mc['dropout'],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc['learning_rate'], weight_decay=tc['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc['T_max'], eta_min=tc['eta_min']
        )
        criterion = nn.MSELoss()

        # Train
        for epoch in tqdm(range(tc['epochs']), desc=f'{entity_id}', leave=False):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f'  Epoch {epoch+1}/{tc["epochs"]}  loss={loss:.4f}')

        # Evaluate
        preds, targets = predict(model, test_loader, device)
        errors = compute_errors(preds, targets)

        # Compute train errors for adaptive threshold calibration
        train_preds, train_targets = predict(model, train_loader, device)
        train_errors = compute_errors(train_preds, train_targets)

        detector = DynamicThreshold(
            smoothing_base=ac['smoothing_base'],
            test_batch_size=ac['test_batch_size'],
            tuning_percentage=ac['tuning_percentage'],
            use_adaptive=ac.get('use_adaptive', True),
        )
        pred_labels_win = detector.detect(errors, train_errors=train_errors)
        logger.info(f'  Adaptive tuning_p={detector.calibrated_p:.4f}')

        pred_labels_ts = np.zeros(len(test_data), dtype=int)
        for i, label in enumerate(pred_labels_win):
            if label == 1:
                pred_labels_ts[i:i + W + F] = 1

        min_len = min(len(pred_labels_ts), len(true_labels))
        metrics = evaluate(pred_labels_ts[:min_len], true_labels[:min_len])
        elapsed = time.time() - t0

        logger.info(
            f'  Results: P={metrics["precision"]:.3f}  R={metrics["recall"]:.3f}'
            f'  F0.5={metrics["F0.5"]:.3f}  ({elapsed:.0f}s)'
        )

        row = {'entity': entity_id, **metrics, 'time_s': round(elapsed, 1)}
        all_results.append(row)

        # Append to CSV immediately (safe even if run is interrupted)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s'])
            writer.writerow(row)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info('Summary:')
    for r in all_results:
        logger.info(f"  {r['entity']:8s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F0.5={r['F0.5']:.3f}")

    avg_f = float(np.mean([r['F0.5'] for r in all_results]))
    logger.info(f'\nAverage F0.5: {avg_f:.3f}')
    logger.info(f'Results saved to: {csv_path}')

    # Save full JSON summary
    summary_path = os.path.join(log_dir, f'{run_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'avg_F0.5': avg_f, 'results': all_results, 'config': config}, f, indent=2)
    logger.info(f'Summary saved to: {summary_path}')

    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='configs/default.yaml')
    parser.add_argument('--dataset', choices=['smap', 'msl'], default='smap')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_dir', default='logs')
    args = parser.parse_args()

    run_name = f'{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    config   = load_config(args.config)
    device   = torch.device(args.device)

    run(config, args.data_dir, args.dataset, device, args.log_dir, run_name)


if __name__ == '__main__':
    main()
